import cv2
import torch
from torchvision import transforms
from dataset import PetDataset
import torch.utils.data
import os
import matplotlib.pyplot as plt

def draw_keypoints(image, noses):
    # Convert the tensor to a numpy array and extract coordinates
    x = int(noses[0].item())
    y = int(noses[1].item())

    # Draw a red circle around the keypoints
    image = cv2.circle(image, (x, y), 2, (0, 0, 255), 2)

    return image


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Set scale_percent to the desired value for resizing
    train_dataset = PetDataset(txt='./data/train_noses.3.txt', root_dir='./data/images/', transform=transform,
                               target_size=(640, 640))
    test_dataset = PetDataset(txt='./data/test_noses.txt', root_dir='./data/images/', transform=transform,
                              target_size=(640, 640))
    # Create DataLoader for testing
    test_batch_size = 32
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Save the first few images to the current working directory
    output_dir = './output_images'
    os.makedirs(output_dir, exist_ok=True)


        # Test DataLoader by printing the first batch
    for batch in test_dataloader:
        images, noses = batch
        print("Test Batch:")
        print("Images Shape:", images.shape)
        print("Noses Shape:", noses.shape)
        break  # Only print the first batch for brevity

    for images, noses in test_dataloader:
        #images, noses = batch
        i = 0

        for image, nose in zip(images, noses):
          i += 1
          image_i = image.numpy()
          image_i = cv2.imread(img_name)
          print(image_i)
          img_with_keypoints = draw_keypoints(image_i, nose)


        # for i in range(images.size(0)): #so per batch
        #   # Access the i-th image and nose from the batch
        #   image_i = images[i].permute(1, 2, 0).numpy()
        #   image_i = transforms.ToPILImage()(images[i])
        #   nose_i = noses[i]

        #   # Draw keypoints on the image
        #   img_with_keypoints = draw_keypoints(image_i, nose_i)
          # Save the image with keypoints
          img_name = f"output_image_{i}_with_keypoints.png"
          img_path = os.path.join(output_dir, img_name)
          cv2.imwrite(img_path, cv2.cvtColor(img_with_keypoints, cv2.COLOR_RGB2BGR))
          print(f"Saved {img_path}")

        if i == 32:  # Break after processing the first batch for brevity
            break