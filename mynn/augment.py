import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import shift


def rotate_image(image, angle):
    image = image.reshape(28, 28)
    rotated = rotate(image, angle, reshape=False)
    return rotated.reshape(-1)

def translate_image(image, dx, dy):
    image = image.reshape(28, 28)
    translated = shift(image, [dy, dx])
    return translated.reshape(-1)


def data_augment(images, labels, methods):
    aug_images = []
    aug_labels = []

    for image, label in zip(images, labels):
        aug_images.append(image)
        aug_labels.append(label)
        for i in range(len(methods)):
            if methods[i] == 'rotate':
                new_image = rotate_image(image, 30)
            elif methods[i] == 'translate':
                new_image = translate_image(image, 1, 1)
            else: 
                print(f'Incorrect Method:{methods[i]}')
                break

            aug_images.append(new_image)
            aug_labels.append(label)

    return np.array(aug_images), np.array(aug_labels)