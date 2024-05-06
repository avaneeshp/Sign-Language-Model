import os
import numpy as np
import torch
from imageio.v3 import imread
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class ImageStandardizer(object):
    def __init__(self):
        super().__init__()
        self.image_mean = None
        self.image_std = None

    def fit(self, X):
        self.image_mean = np.mean(X, axis=(0, 1, 2))
        self.image_std = np.std(X, axis=(0, 1, 2))

    def transform(self, X):
        standardized_X = (X - self.image_mean) / self.image_std
        return standardized_X


class ASLDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image, label = self.X[idx], self.y[idx]
        image = torch.from_numpy(image).float()
        return image, label


def load_and_split_data(asl_image_path, test_size=0.2, val_size=0.1, random_seed=83):
    X, y = [], []
    label_map = {}
    label_idx = 0

    valid_extensions = {'.jpg', '.png'}

    for label in sorted(os.listdir(asl_image_path)):
        directory_path = os.path.join(asl_image_path, label)
        if not os.path.isdir(directory_path):
            continue

        if label not in label_map:
            label_map[label] = label_idx
            label_idx += 1
        
        for image_name in sorted(os.listdir(directory_path)):
            ext = os.path.splitext(image_name)[1].lower()
            if ext not in valid_extensions:
                continue
            image_path = os.path.join(directory_path, image_name)
            image = imread(image_path)
            X.append(image)
            y.append(label_map[label])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=random_seed)

    return ASLDataset(X_train, y_train), ASLDataset(X_val, y_val), ASLDataset(X_test, y_test), label_map

def load_data():
    # Original Data, no modification
    asl_image_path = "./asl_alphabet_train"
    train_dataset, val_dataset, test_dataset, label_map = load_and_split_data(asl_image_path)

    # Original Data + modification
    # modified_images_path = "./asl_alphabet_modified"
    # train_dataset, val_dataset, test_dataset, label_map = load_and_split_data(modified_images_path)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(train_dataset.X)
    train_dataset.X = standardizer.transform(train_dataset.X)
    val_dataset.X = standardizer.transform(val_dataset.X)
    test_dataset.X = standardizer.transform(test_dataset.X)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    train_dataset.X = train_dataset.X.transpose(0, 3, 1, 2)
    val_dataset.X = val_dataset.X.transpose(0, 3, 1, 2)
    test_dataset.X = test_dataset.X.transpose(0, 3, 1, 2)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #was 32
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, standardizer


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    train_loader, val_loader, test_loader, standardizer = load_data()
