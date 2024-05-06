import os
import numpy as np
import cv2
from scipy.ndimage import rotate
import random


# Gaussian blur
def gaussian(img):
    gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    return gaussian


# Grayscale
def grayscale(img):
    means = np.mean(img, axis=2)
    img[:, :, :] = means[:, :, np.newaxis]
    return img


# Gaussian blur and grayscale
def graygaussian(img):
    means = np.mean(img, axis=2)
    img[:, :, :] = means[:, :, np.newaxis]
    gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    return gaussian


# Rotation 
def rotation(img, deg):
    degree = random.randint(-deg, deg)
    img = rotate(img, degree, reshape=False)
    return img


def create_modified_data(img, letter, index, start):
    letter_path = f'{letter}/{letter}'

    for func in (gaussian, graygaussian, grayscale, rotation):
        if func != rotation:
            modified_img = func(img)
        else:
            modified_img = func(img, 20)
        picture_index = str(index + start)
        output_path = f'asl_alphabet_modified/{letter_path}{picture_index}.jpg'
        cv2.imwrite(output_path, modified_img)
        start += 250 # Gaussian starts at 251, graygaussian at 501,
                     # grayscale at 751, and rotation at 1001.


def main():
    file_path = 'asl_alphabet_train'
    output_dir = 'asl_alphabet_modified'
    os.makedirs(output_dir, exist_ok=True)

    modified_data_start = 251

    for root, dirs, files in os.walk(file_path):
        if root == 'asl_alphabet_train': # results in undefined behavior
            continue
        path = root.split('/')
        cur_letter = path[-1]
        os.makedirs(f'asl_alphabet_modified/{cur_letter}', exist_ok=True)

        print(f'Starting data modification on {cur_letter}')

        # Add original pictures to the modified data folder.
        for file in files:
            if file == '.DS_Store': # results in undefined behavior
                continue
            cur_file_path = f'{file_path}/{cur_letter}/{file}'
            new_file_path = f'{output_dir}/{cur_letter}/{file}'
            img = cv2.imread(cur_file_path)
            
            cv2.imwrite(new_file_path, img)

        # Add modified pictures to the modified data folder.
        i = 0
        for file in files:
            if file == '.DS_Store': # results in undefined behavior
                continue
            if i == 50: # only modify 50 of the original pictures.
                break

            cur_file_path = f'{file_path}/{cur_letter}/{file}'
            img = cv2.imread(cur_file_path)

            picture_index = int(file.lstrip(cur_letter).rstrip('.jpg'))

            create_modified_data(img, cur_letter, picture_index, modified_data_start)

            i += 1


if __name__ == '__main__':
    main()
