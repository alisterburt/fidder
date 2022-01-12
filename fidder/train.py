import sys
from copy import deepcopy
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
import typer
from rich.console import Console
from rich.progress import Progress, BarColumn
from torch import optim
from torch.utils.data import DataLoader, random_split

from fidder.dataset import FidderDataSet
from fidder.unet.model import UNet
from fidder.validate import validate
from fidder.unet.utils.dice_score import dice_loss

DEFAULT_TRAINING_DATA_DIR = Path('training_data')
DEFAULT_OUTPUT_DIR = Path('output')
MODEL_NAME = 'model.pth'
INTERRUPTED_NAME = 'INTERRUPTED.pth'
CHECKPOINT_DIR_NAME = 'model_checkpoints'
LOG_FILE_NAME = 'log'

console = Console(record=True)
cli = typer.Typer()

option_args = {'prompt': True, 'prompt_required': True}


@cli.command()
def train_unet(
        training_data: Path = typer.Option(DEFAULT_TRAINING_DATA_DIR, **option_args),
        output_directory: Path = typer.Option(DEFAULT_OUTPUT_DIR, **option_args),
        epochs: int = typer.Option(20, **option_args),
        batch_size: int = typer.Option(4, **option_args),
        learning_rate: float = typer.Option(0.00001, **option_args),
        val_percent: float = typer.Option(20, **option_args),
        save_checkpoint: bool = typer.Option(True, **option_args),
        mixed_precision: bool = typer.Option(False, **option_args)
):
    # 1. Create output folder structure
    output_directory = Path(output_directory)
    model_file = output_directory / MODEL_NAME
    log_file = output_directory / LOG_FILE_NAME
    checkpoint_directory = output_directory / CHECKPOINT_DIR_NAME

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    # 2. Create model and push to device
    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    console.log(f'Network:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    console.log(f'Using device {device}')

    # 3. Create dataset
    dataset = FidderDataSet(training_data)

    # 4. Split into train / validation partitions
    n_val = int(len(dataset) * (val_percent / 100))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    val_set = deepcopy(val_set)  # train and val set reference the same dataset
    train_set.dataset.train()
    val_set.dataset.eval()

    # 5. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    console.log(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {mixed_precision}
    """)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    # 5. Train model
    try:
        model = training_loop(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_directory=checkpoint_directory,
            save_checkpoint=save_checkpoint,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            mixed_precision=mixed_precision,
        )
        console.log('Training completed!')

        console.log(f'Saving model...')
        torch.save(model.state_dict(), str(model_file))
        console.log(f'Model saved to {model_file}')

        console.save_text(f'{log_file}.txt')
        console.save_html(f'{log_file}.html')

        return model

    except KeyboardInterrupt:
        interrupted_file = checkpoint_directory / INTERRUPTED_NAME
        torch.save(model.state_dict(), str(interrupted_file))
        console.log(
            f'Saved model from interrupted training run at {interrupted_file}'
        )
        console.save_text(f'{log_file}.txt')
        console.save_html(f'{log_file}.html')
        sys.exit(0)


def training_loop(
        model: UNet,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_directory: Path,
        save_checkpoint: bool,
        epochs: int,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        grad_scaler: torch.cuda.amp.GradScaler,
        mixed_precision: bool):
    global_step = 0

    with Progress(
            "[progress.description]{task.description} {task.completed:>3.0f}  / {task.total:>3.0f}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "[progress.loss]{task.fields[loss]}",
            console=console
    ) as progress_tracker:
        epoch_pbar = progress_tracker.add_task(
            f"[magenta]Epoch",
            total=epochs,
            loss=f"Validation Dice score: "
        )
        loss = 99999
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            batch_pbar = progress_tracker.add_task("[green]Image", total=len(
                train_loader.dataset), loss=f"         Running loss: {loss:.4f}")

            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks_one_hot = einops.rearrange(
                    F.one_hot(true_masks, model.n_classes), 'b h w c -> b c h w'
                ).float()

                with torch.cuda.amp.autocast(enabled=mixed_precision):
                    masks_pred = model(images)  # (b, c, h, w)
                    normalised_predictions = F.softmax(masks_pred, dim=1).float()

                    cross_entropy_loss_value = F.cross_entropy(masks_pred, true_masks)
                    dice_loss_value = dice_loss(
                        normalised_predictions, true_masks_one_hot, multiclass=True
                    )

                    loss = cross_entropy_loss_value + dice_loss_value

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                progress_tracker.update(batch_pbar, advance=images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                progress_tracker.update(batch_pbar, loss=f"         Running loss:"
                                                         f" {loss.item():.4f}")

            # Evaluation round
            val_score = validate(model, val_loader, device)
            scheduler.step(val_score)

            console.log(f'Training loss: {loss:.4f}')
            console.log(f'Validation Dice score: {val_score:.4f}')
            progress_tracker.update(batch_pbar, visible=False)
            progress_tracker.update(epoch_pbar, advance=1, loss=f"Validation Dice score: "
                                                                f"{val_score:.4f}")

            # Save model checkpoints between epochs
            if save_checkpoint:
                checkpoint_file = checkpoint_directory / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save(model.state_dict(), str(checkpoint_file))
                console.log(f'Checkpoint for epoch {epoch + 1} saved')

    return model
