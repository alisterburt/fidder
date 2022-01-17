from pathlib import Path

import numpy as np
import torch
import typer
from rich.console import Console
from tiler import Tiler, Merger
import einops
import mrcfile
from .unet.model import UNet
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .dataset import NETWORK_IMAGE_DIMENSIONS, DOWNSAMPLE_SHORT_EDGE_LENGTH

console = Console(record=True)
cli = typer.Typer()

option_args = {'prompt': True, 'prompt_required': True}


@cli.command()
def load_predict_save(
        tilt_series: Path = typer.Option(..., **option_args),
        model_weights: Path = typer.Option(..., **option_args),
        output_filename: Path = typer.Option(..., **option_args),
        batch_size: int = 3,
):
    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    console.log(f'Network:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    console.log(f'Using device {device}')

    model_state = torch.load(model_weights, map_location=device)
    model.load_state_dict(model_state)
    console.log(f'Weights loaded from {model_weights}')

    model.eval()

    with mrcfile.open(tilt_series, permissive=True) as mrc:
        tilt_series_data = torch.tensor(mrc.data)

    fiducial_probabilities = predict_tilt_series(
        tilt_series=tilt_series_data,
        model=model,
        device=device,
        batch_size=batch_size
    )
    fiducial_probabilities = np.asarray(fiducial_probabilities, dtype=np.float16)
    fiducial_mask = fiducial_probabilities > 0.3
    mrcfile.new(output_filename, data=np.array(fiducial_mask, dtype=np.int16),
                overwrite=True).close()
    return fiducial_probabilities


def predict_tilt_series(tilt_series: torch.Tensor, model, device: torch.device, batch_size=10):
    tilt_series_probabilities = torch.zeros(size=tilt_series.shape, dtype=torch.float16)
    for idx, tilt_image in enumerate(tilt_series):
        tilt_image_probabilities = predict_single_image(
            tilt_image, model=model, device=device, batch_size=batch_size
        )
        tilt_series_probabilities[idx, ...] = tilt_image_probabilities
    return tilt_series_probabilities


def predict_single_image(
        image: torch.Tensor,
        model,
        device: torch.device,
        batch_size: int = 1
):
    original_image_dimensions = (image.shape[-2], image.shape[-1])

    # add batching and channel dimensions
    image = einops.rearrange(image, 'h w -> 1 1 h w')
    downsampled_image = TF.resize(image, size=DOWNSAMPLE_SHORT_EDGE_LENGTH)
    downsampled_image = einops.rearrange(
        downsampled_image, '1 1 h w -> h w'
    )

    # set up tiling
    tiler = Tiler(
        data_shape=downsampled_image.shape,
        tile_shape=NETWORK_IMAGE_DIMENSIONS,
        overlap=0.5
    )
    merger = Merger(tiler)

    # iterate over tiles, normalise and predict
    for batch_idx, tiles in tiler(np.array(downsampled_image.cpu()), batch_size=batch_size):
        # normalise tiles and coerce to (b, c, h, w)
        tiles = torch.tensor(tiles, device=device).float().contiguous()  # (b, h, w)
        per_tile_mean = einops.rearrange(torch.mean(tiles, dim=[-2, -1]), 'b -> b 1 1')
        per_tile_std = einops.rearrange(torch.std(tiles, dim=[-2, -1]), 'b -> b 1 1')
        tiles = (tiles - per_tile_mean) / per_tile_std
        tiles = einops.rearrange(tiles, 'b h w -> b 1 h w')

        # predict
        predicted_masks = model(tiles)
        fiducial_probabilities = F.softmax(predicted_masks, dim=1)[:, 1, ...].cpu()
        fiducial_probabilities = np.array(fiducial_probabilities.detach(), dtype=np.float16)

        # add to merger for tile merging
        merger.add_batch(
            data=fiducial_probabilities,
            batch_id=batch_idx,
            batch_size=batch_size,
        )

    merged_tiles = torch.tensor(merger.merge(unpad=True))
    merged_tiles = einops.rearrange(merged_tiles, 'h w -> 1 1 h w')
    mask = TF.resize(merged_tiles, size=original_image_dimensions)
    mask = einops.rearrange(mask, '1 1 h w -> h w')
    return mask




