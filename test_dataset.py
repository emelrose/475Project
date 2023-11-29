import cv2
import torch
from torchvision import transforms
from dataset import PetDataset
import torch.utils.data


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


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Set scale_percent to the desired value for resizing
    train_dataset = PetDataset(txt='./data/train_noses.3.txt', root_dir='./data/images/', transform=transform,
                               scale_percent=50)
    test_dataset = PetDataset(txt='./data/test_noses.txt', root_dir='./data/images/', transform=transform, scale_percent=50)
    # Create DataLoader for training
    train_batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # Create DataLoader for testing
    test_batch_size = 32
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Test DataLoader by printing the first batch
    for batch in train_dataloader:
        images, noses = batch
        print("Train Batch:")
        print("Images Shape:", images.shape)
        print("Noses Shape:", noses.shape)
        break  # Only print the first batch for brevity

    for batch in test_dataloader:
        images, noses = batch
        print("Test Batch:")
        print("Images Shape:", images.shape)
        print("Noses Shape:", noses.shape)
        break  # Only print the first batch for brevity