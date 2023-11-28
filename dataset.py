import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch



class CustomDataset(Dataset):
    def __init__(self):

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return image, label

    def resize_image(image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        resized_image = cv2.resize(image, (width, height))

        return resized_image

    def resize_point(original_point, scale_percent):
        # Scale teh coordinates of the point
        x, y = original_point
        scaled_x = int(x * scale_percent / 100)
        scaled_y = int(x * scale_percent / 100)