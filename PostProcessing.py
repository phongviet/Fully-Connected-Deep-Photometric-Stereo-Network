import numpy as np
import cv2
import os

def load_image(in_path):
    images = []
    for r, d, f in os.walk(in_path):
        for file in f:
            filepath = os.path.join(r, file)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
    return images

def masking(normal_map, in_path):
    I = np.asarray(load_image(in_path))

    mask = np.zeros_like(I[0])
    for img in I:
        mask = cv2.bitwise_or(mask, img)
    mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)[1]

    mask = cv2.medianBlur(mask, 9)

    normal_map = cv2.bitwise_and(normal_map, normal_map, mask=mask)
    return normal_map