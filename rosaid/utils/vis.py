from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt


def show_image(
    image: np.ndarray,
    title: Optional[str] = None,
    fig_size: Tuple[int, int] = (10, 10),
    axis: bool = False,
):
    """
    A function to plot an image.

    Args:
    image: Image to plot.
    title: Title of the image.
    fig_size: Size of the figure.
    cmap: Colormap for the image.
    axis: Whether to show axis or not.

    Returns:
    fig: Figure object.

    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, vmax=255, vmin=0)
    if title:
        ax.set_title(title)
    if not axis:
        ax.axis("off")
    plt.show()
    return fig


def subplot_images(
    image: List[np.ndarray],
    titles: Optional[List[str]] = None,
    fig_size: Tuple[int, int] = (10, 10),
    order: Optional[Tuple[int, int]] = None,
    axis: bool = False,
    show: bool = False,
    cmap: Optional[str] = None,
    sup_title: Optional[str] = None,
    ret_fig: bool = True,
):
    """
    Plot a list of images in a subplot grid.

    Args:
        image (List[np.ndarray]): List of images (2D or 3D arrays).
        titles (List[str], optional): Titles for each image.
        fig_size (Tuple[int, int]): Size of the entire figure.
        order (Tuple[int, int], optional): (rows, cols) in grid. If None, it's auto-computed.
        axis (bool): Whether to show axes.
        show (bool): Whether to show the figure immediately.
        cmap (str, optional): Colormap to use for grayscale images.

    Returns:
        fig (plt.Figure): Matplotlib figure object.
    """

    n_images = len(image)
    if n_images == 0:
        print("⚠️ No images to plot.")
        return None

    # Determine grid size automatically if not provided
    if order is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = order
        if rows == -1:
            rows = int(np.ceil(n_images / cols))
        elif cols == -1:
            cols = int(np.ceil(n_images / rows))

    fig, axs = plt.subplots(rows, cols, figsize=fig_size)
    axs = np.atleast_2d(axs).flatten()

    for i in range(rows * cols):
        ax = axs[i]
        if i < n_images:
            img = image[i]
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                ax.imshow(img.squeeze(), cmap=cmap or "gray", vmin=0, vmax=255)
            else:
                ax.imshow(img, vmin=0, vmax=255)
            if titles and i < len(titles):
                ax.set_title(titles[i])
        else:
            ax.axis("off")
        if not axis:
            ax.axis("off")
    if sup_title:
        plt.suptitle(sup_title)

    plt.tight_layout()
    if show:
        plt.show()
    if ret_fig:
        return fig
