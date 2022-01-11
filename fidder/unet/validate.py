import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils.dice_score import multiclass_dice_coeff, dice_coeff


def validate(model, dataloader, device):
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    # with Progress(
    #         "[progress.description]{task.description} {task.completed:} / {task.total}",
    #         BarColumn(),
    #         "[progress.percentage]{task.percentage:>3.0f}%",
    #         TimeRemainingColumn(),
    #         console=console
    # ) as progress_tracker:
    for batch in dataloader:
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = model(image)

            # convert to one-hot format
            if model.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1,
                                                                                        2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    model.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches