import numpy as np

from rosaid.core.defs import NormalImageType


def get_filtered_image(image: np.ndarray):
    """
    First, delete all the columns that is complete black.
    """
    filtered_image = image.copy()
    non_zero_cols = image.sum(axis=0) != 0
    filtered_image = image[:, non_zero_cols].copy()

    return filtered_image


def image_to_normalized_frequency_image(filtered_image: np.ndarray):
    """
    Requires a filtered image as input.
    """

    new_image = []

    for row in range(filtered_image.shape[0]):
        row_data = filtered_image[row, :]
        if sum(row_data) == 0:
            continue

        # find leftmost and rightmost non zero pixels
        leftmost = np.argmax(row_data != 0)
        rightmost = len(row_data) - np.argmax(row_data[::-1] != 0)
        row_data_cropped = row_data[leftmost:rightmost]
        # now calculate frequency of each pixel value
        unique, counts = np.unique(row_data_cropped, return_counts=True)
        # normalze the frequency to 0-255
        # if unique[0] == 0:
        #     counts[0] = 0  # ignore black pixel frequency
        counts = counts / counts.max() * 255

        # fill new_image row with pixel values according to frequency
        new_row = np.zeros(256)
        new_row[unique.astype(int)] = counts
        new_image.append(new_row)
    new_image = np.array(new_image)

    return new_image


def sess_zscore_image(filtered_image: np.ndarray) -> np.ndarray:
    new_image = []

    for row in range(filtered_image.shape[0]):
        row_data = filtered_image[row]
        if row_data.sum() == 0:
            continue

        leftmost = np.argmax(row_data != 0)
        rightmost = len(row_data) - np.argmax(row_data[::-1] != 0)
        row_data_cropped = row_data[leftmost:rightmost]
        unique, counts = np.unique(row_data_cropped, return_counts=True)

        norm_counts = counts / counts.max() * 255
        norm_row = np.zeros(256)
        norm_row[unique.astype(int)] = norm_counts
        new_image.append(norm_row)

    freq_map = np.array(new_image)
    freq_mean = freq_map.mean(axis=1)
    freq_std = freq_map.std(axis=1)
    freq_zscore = np.nan_to_num(
        (freq_map - freq_mean[:, None]) / (freq_std[:, None] + 1e-8)
    )

    def safe_norm(img):
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max == img_min or not np.isfinite(img_max - img_min):
            return np.zeros_like(img)
        return (img - img_min) / (img_max - img_min) * 255

    mean_img = safe_norm(freq_zscore.mean(axis=0).reshape(16, 16))
    std_img = safe_norm(freq_zscore.std(axis=0).reshape(16, 16))
    median_img = safe_norm(np.median(freq_zscore, axis=0).reshape(16, 16))

    return np.stack([mean_img, std_img, median_img], axis=2)


def sess_zscore_gram_image(filtered_image: np.ndarray):
    new_image = []

    for row in range(filtered_image.shape[0]):
        row_data = filtered_image[row]
        if row_data.sum() == 0:
            continue

        leftmost = np.argmax(row_data != 0)
        rightmost = len(row_data) - np.argmax(row_data[::-1] != 0)
        row_data_cropped = row_data[leftmost:rightmost]
        unique, counts = np.unique(row_data_cropped, return_counts=True)

        norm_counts = counts / counts.max() * 255
        norm_row = np.zeros(256)
        norm_row[unique.astype(int)] = norm_counts
        new_image.append(norm_row)

    freq_map = np.array(new_image)
    freq_mean = freq_map.mean(axis=1)
    freq_std = freq_map.std(axis=1)
    freq_zscore = np.nan_to_num(
        (freq_map - freq_mean[:, None]) / (freq_std[:, None] + 1e-8)
    )

    def safe_norm(img):
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max == img_min or not np.isfinite(img_max - img_min):
            return np.zeros_like(img)
        return (img - img_min) / (img_max - img_min) * 255

    mean_img = safe_norm(freq_zscore.mean(axis=0)).reshape(-1, 1)
    std_img = safe_norm(freq_zscore.std(axis=0)).reshape(-1, 1)
    median_img = safe_norm(np.median(freq_zscore, axis=0)).reshape(-1, 1)

    # gram of each img: 256, 256
    stkd = np.stack([mean_img, std_img, median_img], axis=2)
    gram = stkd * stkd.transpose(1, 0, 2)  # 256, 256, 3
    mat_gram = np.dot(stkd.reshape(256, 3), stkd.reshape(256, 3).T)  # 256, 256

    # min max both and to uint8
    gram_min, gram_max = gram.min(), gram.max()
    gram = (gram - gram_min) / (gram_max - gram_min) * 255
    mat_gram_min, mat_gram_max = mat_gram.min(), mat_gram.max()
    mat_gram = (mat_gram - mat_gram_min) / (mat_gram_max - mat_gram_min) * 255
    return gram, mat_gram


def gram_image(image: np.ndarray) -> np.ndarray:
    """Compute Gram matrix of the image"""
    pixels = image
    # Compute Gram matrix
    gram_matrix = np.dot(pixels.T, pixels)
    gram_matrix = (
        (gram_matrix - gram_matrix.min())
        / (gram_matrix.max() - gram_matrix.min())
        * 255
    )

    return gram_matrix


def gram_image_3d(image: np.ndarray) -> np.ndarray:
    """Compute Gram matrix of the image"""
    pixels = image

    orig_gram = np.dot(pixels.T, pixels)
    row_rev_gram = np.dot(pixels.T, pixels)[::-1, :]
    col_rev_gram = np.dot(pixels.T, pixels)[:, ::-1]
    gram_3d = np.stack([orig_gram, row_rev_gram, col_rev_gram], axis=2)
    gram_matrix = gram_3d

    gram_matrix = (
        (gram_matrix - gram_matrix.min()) / (gram_matrix.max() - gram_matrix.min())
    ) * 255

    return gram_matrix


def get_normal_image(
    gray_image: np.ndarray,
    image_type: NormalImageType,
    filter_first_nonzero_columns: bool = True,
    float_precision: bool = False,
) -> np.ndarray:
    """Get normal image based on the specified type."""
    if float_precision:
        gray_image = gray_image.astype(np.float32)

    if filter_first_nonzero_columns:
        first_nonzero_col = np.argmax(gray_image.sum(axis=0) != 0)
        gray_image = gray_image[:, first_nonzero_col:]

    if image_type == NormalImageType.ORIGINAL or image_type == 0:
        output = gray_image
    elif image_type == NormalImageType.FILTERED or image_type == 1:
        output = get_filtered_image(gray_image)
    elif image_type == NormalImageType.NORMALIZED or image_type == 2:
        filtered_img = get_filtered_image(gray_image)
        output = image_to_normalized_frequency_image(filtered_img)
    elif image_type == NormalImageType.FILTERED_GRAM or image_type == 3:
        filtered_img = get_filtered_image(gray_image)
        output = gram_image(filtered_img)
    elif image_type == NormalImageType.NORMALIZED_GRAM or image_type == 4:
        filtered_img = get_filtered_image(gray_image)
        norm_img = image_to_normalized_frequency_image(filtered_img)
        output = gram_image(norm_img)
    elif image_type == NormalImageType.ZSCORE or image_type == 5:
        filtered_img = get_filtered_image(gray_image)
        output = sess_zscore_image(filtered_img)
    elif image_type == NormalImageType.ZGRAM1D or image_type == 6:
        filtered_img = get_filtered_image(gray_image)
        _, gram_img = sess_zscore_gram_image(filtered_img)
        output = gram_img
    elif image_type == NormalImageType.ZGRAM3D or image_type == 7:
        filtered_img = get_filtered_image(gray_image)
        zscore_img, gram_img = sess_zscore_gram_image(filtered_img)
        output = zscore_img
    elif image_type == NormalImageType.UNFILTERED_GRAM or image_type == 8:
        output = gram_image(gray_image)
    elif image_type == NormalImageType.UNFILTERED_GRAM3D or image_type == 9:
        output = gram_image_3d(gray_image)
    elif image_type == NormalImageType.FILTERED_GRAM3D or image_type == 10:
        filtered_img = get_filtered_image(gray_image)
        output = gram_image_3d(filtered_img)
    else:
        raise ValueError(f"Unsupported normal image type: {image_type}")
    return output


def show_test_image(image_path: str | None = None):
    import os

    import cv2

    from rosaid.utils.vis import subplot_images

    if image_path is None:
        image_path = r"rosaid\assets\figures\subcriberflood.png"
    if not os.path.exists(image_path):
        raise ValueError("Image path does not exist")

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images = []
    titles = []

    images.append(gray_image)
    titles.append("Gray Image")

    # Show the images
    for img_type in NormalImageType:
        norm_img = get_normal_image(gray_image, img_type, float_precision=True)
        images.append(norm_img)
        titles.append(img_type.name)
    subplot_images(
        images, titles, fig_size=(20, 15), order=(1, -1), show=True, ret_fig=True
    ).savefig("float_precision.png")

    images = []
    titles = []

    images.append(gray_image)
    titles.append("Gray Image")

    # Show the images
    for img_type in NormalImageType:
        norm_img = get_normal_image(gray_image, img_type)
        images.append(norm_img)
        titles.append(img_type.name)
    subplot_images(
        images, titles, fig_size=(20, 15), order=(1, -1), show=True, ret_fig=True
    ).savefig("int_precision.png")


if __name__ == "__main__":
    show_test_image()
