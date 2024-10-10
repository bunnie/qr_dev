import cv2
import numpy as np

from ref_images import *

# coverts the ascii reference images into PNGs
ascii_img = [qr_a, qr_b, qr_c, qr_d, qr_e, qr_f]

for (i, ascii) in enumerate(ascii_img):
    image = []
    for c in ascii.split(','):
        if c != '':
            image.append(int(c, 16))

    gray = np.zeros((120, 160), np.uint8)
    for y in range(120):
        for x in range(160):
            gray[y][x] = image[x + y * 160]

    cropped = np.zeros((120, 128), np.uint8)
    cropped = gray[:120,:128]
    cv2.imshow('image', cropped)
    cv2.waitKey(0)
    cv2.imwrite(f'test{i}.png', cropped)
