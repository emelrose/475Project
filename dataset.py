import fnmatch
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2


def resize_image(image, target_size):
    resized_image = cv2.resize(image, target_size)
    return resized_image


def create_dataloader(batch_size, mode):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if mode == "Train":
        dataset = PetDataset(txt='./data/train_noses.3.txt', root_dir='./data/images/', transform=transform,
                             target_size=(640, 640))
        shuffle = True

    else:
        dataset = PetDataset(txt='./data/test_noses.txt', root_dir='./data/images/', transform=transform,
                             target_size=(640, 640))
        shuffle = False

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def scale_point(original_point, image, target_size):
    # Get the original image size
    original_image_size = (image.shape[1], image.shape[0])

    # Calculate scale factors for both dimensions
    scale_factor_x = target_size[1] / original_image_size[1]
    scale_factor_y = target_size[0] / original_image_size[0]

    # Scale the coordinates of the point
    x, y = original_point
    scaled_x = int(x * scale_factor_x)
    scaled_y = int(y * scale_factor_y)

    return scaled_x, scaled_y


def draw_keypoints(image, nose):
    for point in nose:
        x, y = point
        cv2.circle(image, (x, y), 2, (0, 0, 255), 2)  # Draw a red circle around the keypoints
        # cv2.circle(image, center_coordinates, radius, color, thickness)
    return image


class PetDataset(Dataset):
    def __init__(self, txt, root_dir, transform=None, target_size=(640, 640)):
        self.labels = pd.read_csv(txt)
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.img_files = []

        for file in os.listdir(self.root_dir):
            try:
                if fnmatch.fnmatch(file, '*.jpg'):
                    img_path = os.path.join(self.root_dir, file)
                    cv2.imread(img_path)  # Try reading the image to check for corruption
            except (IOError, cv2.error) as e:
                print(f"File is corrupted/unreadable {file}. Skipped this one. Error: {e}")
                continue
            self.img_files.append(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name_parts = self.labels.iloc[idx, 0].split('_')
        breed_name = "_".join(image_name_parts[:-1])
        img_name = os.path.join(self.root_dir, f'{breed_name}_{image_name_parts[-1]}')
        # Read the image
        image = cv2.imread(img_name)

        #image = cv2.imread(img_name)


        # Resize the image
        image = resize_image(image, self.target_size)

        #if img_name not in self.img_files:
        #    print(f"Skipping corrupted file: {img_name}")
        #    os.remove(img_name)

        nose_str = self.labels.iloc[idx, 1][1:-1]
        original_nose = list(map(int, nose_str.split(',')))
        nose = scale_point(original_nose, image, self.target_size)
        nose = torch.tensor(nose, dtype=torch.float32)

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, nose

    def visualize_keypoints(self, image, nose):
        img_with_keypoints = draw_keypoints(self, nose)
        return img_with_keypoints