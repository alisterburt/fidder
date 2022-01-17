import einops
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.restoration import inpaint


def fill_mask_tilt_series(tilt_series: np.ndarray, mask: np.ndarray):
    tilt_series_filled = np.copy(tilt_series)
    for idx, (tilt_image, tilt_mask) in enumerate(zip(tilt_series, mask)):
        filled_tilt_image = fill_mask_single_image(
            tilt_image=tilt_image, mask=tilt_mask
        )
        tilt_series_filled[idx, ...] = filled_tilt_image
    return tilt_series_filled


def fill_mask_single_image(tilt_image: np.ndarray, mask: np.ndarray):
    # labels, n_labels = ndi.label(mask)
    # filled_image = np.copy(tilt_image)
    #
    # for label_idx in range(n_labels):
    #     filled_image = fill_single_label(
    #         tilt_image=filled_image, label_image=labels, label_id=label_idx
    #     )
    ##
    filled_image = np.copy(tilt_image)
    idx_to_fill = tuple(
        einops.rearrange(np.argwhere(mask == 1), 'n c -> c n')
    )
    idx_background = np.argwhere(mask == 0)

    n_pixels_to_fill = len(idx_to_fill[0])
    n_pixels_background = idx_background.shape[0]

    fill_selection = np.random.choice(
        np.arange(n_pixels_background), size=n_pixels_to_fill, replace=False
    )
    idx_fill_values = tuple(
        einops.rearrange(idx_background[fill_selection], 'n c -> c n')
    )
    fill_values = tilt_image[idx_fill_values]
    filled_image[idx_to_fill] = fill_values
    return filled_image


def fill_single_label(tilt_image: np.ndarray, label_image: np.ndarray, label_id: int):
    label_idx = np.argwhere(label_image == label_id)
    n_pixels_to_fill = label_idx.shape[0]
    label_idx = tuple(label_idx.T)  # can now be used to index into tilt_image

    edt = distance_transform_edt(label_image != label_id)

    n_fill_values = 0
    distance_threshold = 5

    while n_fill_values < n_pixels_to_fill:
        fill_value_idx_pool = np.argwhere(edt < distance_threshold & label_image == 0)
        n_fill_values = fill_value_idx_pool.shape[0]
        distance_threshold *= 2

    fill_value_idx = np.random.choice(
        np.arange(n_fill_values), size=n_pixels_to_fill, replace=False
    )
    fill_values = fill_value_idx_pool[fill_value_idx]

    tilt_image[label_idx] = fill_values
    return tilt_image
