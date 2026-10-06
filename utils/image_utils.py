import os
import cv2
import numpy as np
import tensorflow as tf


def load_images(filenames, image_folder_path, image_size=(224, 224)):
    """Load, resize, and convert images to float32 RGB arrays.

    Images are returned as float32 [0–255] without normalisation —
    EfficientNetB0 includes a built-in rescaling layer.
    """
    images = []
    for filename in filenames:
        img_path = os.path.join(image_folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype("float32")
            images.append(img)
        else:
            print(f"Warning: image {filename} could not be loaded.")
    return np.array(images)


def augment_images(images, labels, copies_per_class):
    """Generate augmented copies for each image according to copies_per_class.

    Ten geometrically and photometrically distinct transforms are available;
    each label draws from the first N of them (no exact duplicates within a class).

    Args:
        images: numpy array of float32 images.
        labels: array-like of integer class labels aligned with images.
        copies_per_class: dict mapping class int -> number of augmented copies (0 = skip).

    Returns:
        Tuple (augmented_images, augmented_labels) as numpy arrays.
    """
    def make_transforms(img):
        return [
            tf.image.flip_left_right(img),
            tf.image.flip_up_down(img),
            tf.image.flip_left_right(tf.image.flip_up_down(img)),
            tf.image.rot90(img, k=1),
            tf.image.rot90(img, k=2),
            tf.image.rot90(img, k=3),
            tf.image.random_brightness(img, max_delta=0.2),
            tf.image.random_contrast(img, lower=0.8, upper=1.2),
            tf.image.flip_left_right(tf.image.rot90(img, k=1)),
            tf.image.flip_left_right(tf.image.rot90(img, k=3)),
        ]

    augmented_images, augmented_labels = [], []
    for img, label in zip(images, labels):
        n_copies = copies_per_class.get(int(label), 0)
        if n_copies > 0:
            t = make_transforms(img)
            for i in range(n_copies):
                augmented_images.append(t[i])
                augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)
