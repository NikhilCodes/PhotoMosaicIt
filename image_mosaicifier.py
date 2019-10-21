import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

STRIDE = 10
dataset_files = glob('dataset/*')
image_pool = np.zeros((len(dataset_files), STRIDE, STRIDE, 3))
avg_colors = np.zeros((len(dataset_files), 3))
i_ = 0
for file in tqdm(dataset_files):
    image = cv2.resize(cv2.imread(file, cv2.IMREAD_COLOR), (STRIDE, STRIDE))
    image_pool[i_] = image
    avg_colors[i_] = np.sum(np.sum(image, axis=0), axis=0) / (STRIDE ** 2)
    i_ += 1


def render(input_image):
    *DIM, _ = input_image.shape

    if DIM[0] < 1000 or DIM[1] < 1000:
        DIM[1] = 1024#*= 10
        DIM[0] = 720#*= 10

    DIM[0] -= DIM[0] % STRIDE
    DIM[1] -= DIM[1] % STRIDE
    input_image = cv2.resize(input_image, (DIM[1], DIM[0]))

    for i in range(STRIDE, DIM[1] + STRIDE, STRIDE):
        for j in range(STRIDE, DIM[0] + STRIDE, STRIDE):
            sub_image = input_image[j - STRIDE:j, i - STRIDE:i]
            sub_image_avg_color = np.sum(np.sum(sub_image, axis=0), axis=0) / (STRIDE ** 2)
            distance = np.sum(abs(avg_colors - sub_image_avg_color), axis=1)
            closest_pool_image = image_pool[np.argmin(distance)]
            input_image[j - STRIDE:j, i - STRIDE:i] = closest_pool_image

    return input_image
