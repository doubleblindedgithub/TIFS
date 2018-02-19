import numpy as np
import cv2

noise = lambda mu, sig, n: np.tile(
    np.abs(np.random.normal(mu, sig, n)), (3, 1)
).T.astype(np.uint8)

def shred(image, n_strips=30, apply_noise=False, sig=200):
    
    strips = []
    h, w, _ = image.shape
    ws, r = divmod(w, n_strips)
    acc = 0
    for i in range(n_strips):
        d = ws + int(r > 0)
        strips.append(image[:, acc : acc + d, :])
        r -= 1
        acc += d
    
    if apply_noise:
        for strip in strips:
            strip[:, 0, :] = cv2.add(strip[:, 0, :], noise(0, sig, h))
            strip[:, 1, :] = cv2.add(strip[:, 1, :], noise(0, 0.75 * sig, h))
            strip[:, 2, :] = cv2.add(strip[:, 2, :], noise(0, 0.5 * sig, h))
            strip[:, -1, :] = cv2.add(strip[:, -1, :], noise(0, sig, h))
            strip[:, -2, :] = cv2.add(strip[:, -2, :], noise(0, 0.75 * sig, h))
            strip[:, -3, :] = cv2.add(strip[:, -3, :], noise(0, 0.5 * sig, h))
    return strips