import numpy as np

labels = {0: [255, 172, 255],
          1: [190, 255, 0],
          2: [112, 48, 160],
          3: [224, 224, 224],
          4: [0, 0, 0]
}


def encoded_mask(mask: np.ndarray):
    row, col = mask.shape[0:2]
    r, g, b = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
    single_mask = np.zeros_like(r)
    single_mask[single_mask == 0] = 4

    for i in range(row):
        for j in range(col):
            for k, pixel in labels.items():
                if (r[i][j] == pixel[0]) and (g[i][j] == pixel[1]) and (b[i][j] == pixel[2]):
                    single_mask[i][j] = k

    return single_mask


def decoded_mask(mask: np.ndarray):
    # mask should be mapped with np.argmax() already
    row, col = mask.shape[0:2]
    b = np.zeros_like(mask)
    g = np.zeros_like(mask)
    r = np.zeros_like(mask)
    b[b == 0] = 4
    g[g == 0] = 4
    r[r == 0] = 4
    for i in range(row):
        for j in range(col):
            for k, pixel in labels.items():
                if mask[i][j] == k:
                    b[i][j] = pixel[0]
                    g[i][j] = pixel[1]
                    r[i][j] = pixel[2]
    single_mask = np.zeros((row, col, 3), dtype=np.uint8)
    single_mask[:, :, 0] = b
    single_mask[:, :, 1] = g
    single_mask[:, :, 2] = r

    return single_mask
