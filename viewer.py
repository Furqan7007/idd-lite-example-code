import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np
from glob import glob

image_paths = glob('idd20k_lite/leftImg8bit/train/*/*_image.jpg')
label_paths = [p.replace('leftImg8bit', 'gtFine').replace(
    '_image.jpg', '_label.png') for p in image_paths]

# Assigning some RGB colors for the 7 + 1 (Misc) classes
colors = np.array([[128, 64, 18], [244, 35, 232], [220, 20, 60], [0, 0, 230],
                   [220, 190, 40], [70, 70, 70], [70, 130, 180], [0, 0, 0]
                   ], dtype=np.int)

print(len(image_paths))

for i in range(len(image_paths)):
    print(image_paths[i], label_paths[i])
    image_frame = imread(image_paths[i])
    # print(image_frame.shape)
    label_map = imread(label_paths[i])
    color_image = np.zeros(
        (label_map.shape[0], label_map.shape[1], 3), dtype=np.int)
    for i in range(7):
        color_image[label_map == i] = colors[i]

    color_image[label_map == 255] = colors[7]
    plt.imshow(image_frame)
    plt.imshow(color_image, alpha=0.8)
    plt.show()
