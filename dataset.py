import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2


def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    resized_image = cv2.resize(image, (width, height))

    return resized_image


def resize_point(original_point, scale_percent):
    # Scale the coordinates of the point
    x, y = original_point
    scaled_x = int(x * scale_percent / 100)
    scaled_y = int(y * scale_percent / 100)

    return scaled_x, scaled_y


class PetDataset(Dataset):
    def __init__(self, txt, root_dir, transform=None, scale_percent=100):
        self.labels = pd.read_csv(txt)
        self.root_dir = root_dir
        self.transform = transform
        self.scale_percent = scale_percent

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir, f'image_{idx}.jpg')
        image_name_parts = self.labels.iloc[idx, 0].split('_')
        breed_name = "_".join(image_name_parts[:-1])  # Join all parts except the last one
        img_name = os.path.join(self.root_dir, f'{breed_name}_{image_name_parts[-1]}')
        image = cv2.imread(img_name) #this works

        # Resize the image
        image = resize_image(image, self.scale_percent)

        # Convert nose coordinates to a tensor
        nose_str = self.labels.iloc[idx, 1][1:-1]
        original_nose = list(map(int, nose_str.split(',')))
        nose = resize_point(original_nose, self.scale_percent)
        nose = torch.tensor(nose, dtype=torch.float32)

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, nose