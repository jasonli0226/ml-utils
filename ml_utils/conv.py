import numpy as np

__all__ = ["im2col"]


def im2col(image: np.ndarray, kernel: np.ndarray, stride=1) -> np.ndarray[np.float32]:
    """
    Implementing im2col operation to reshape the matrix

    Args:
    ---------
        image (np.ndarray): Input image (H x W x C)
        kernel (np.ndarray): Kernel for convolution (H x W)
        stride (int, optional): Stride for convolution. Defaults to 1.

    Returns:
    ---------
        np.ndarray[np.float32]: Result of im2col algorithm

    Examples:
    ---------
    ```
    import numpy as np

    kernel = np.array([
        [-1, -1, 2],
        [-1, 2, -1],
        [2, -1, -1],
    ])

    lower_mat = im2col(image, kernel)
    ```
    """
    hh, ww = kernel.shape
    h, w, c = image.shape
    new_h = (h - hh) // stride + 1
    new_w = (w - ww) // stride + 1
    col = np.zeros([new_h * new_w, c * hh * ww], np.float32)

    for i in range(new_h):
        for j in range(new_w):
            patch = image[
                i * stride : i * stride + hh, j * stride : j * stride + ww, ...
            ]
            col[i * new_w + j, :] = np.reshape(patch, (1, 1, -1))
    return col


def conv_with_im2col(
    image: np.ndarray, kernel: np.ndarray, padding=False
) -> np.ndarray[np.float32]:
    """
    Convolution with im2col operation
    Assume stride is 1

    Args:
    ---------
        image (np.ndarray): Input image (H x W x C)
        kernel (np.ndarray): Kernel for convolution (H x W)
        padding (boolean, optional): Apply zero padding for the input

    Returns:
    ---------
        np.ndarray[np.float32]: Convolution result (H x W)
    """
    assert image.ndim == 3

    in_mat = image
    fh, fw = kernel.shape
    h, w, _ = image.shape

    if padding is True:
        new_h = h
        new_w = w
        w_pad = (fw - 1) // 2
        h_pad = (fh - 1) // 2
        in_mat = np.pad(image, ((h_pad, h_pad), (w_pad, w_pad), (0, 0)))
    else:
        new_h = (h - fh) // 1 + 1
        new_w = (w - fw) // 1 + 1

    lower_mat = im2col(in_mat, kernel)
    reshaped_kernel = kernel.reshape((fh * fw, 1))

    return lower_mat.dot(reshaped_kernel).reshape((new_h, new_w))
