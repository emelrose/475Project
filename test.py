def draw_keypoints(image, noses):
    # Convert the tensor to a numpy array and extract coordinates
    x = int(noses[0].item())
    y = int(noses[1].item())

    # Draw a red circle around the keypoints
    cv2.circle(image, (x, y), 2, (0, 0, 255), 2)

    return image

    # Example Usage

    for batch_idx, batch in enumerate(test_dataloader):
        images, noses = batch

        # Convert tensor to numpy array and change channel order
        # mages = images.permute(0, 2, 3, 1).numpy()

        # Draw keypoints on the images
        # for i in range(images.shape[0]):
        #     img_with_keypoints = draw_keypoints(images[i].copy(), noses[i].numpy())

        #     # Save the image with keypoints
        #     img = transforms.ToPILImage()(img_with_keypoints)
        #     img_name = f"output_image_{batch_idx * test_batch_size + i}.png"
        #     img_path = os.path.join(output_dir, img_name)
        #     img.save(img_path)
        #     print(f"Saved {img_path}")

        for i in range(images.size(0)):  # so per batch
            # Access the i-th image and nose from the batch
            image_i = images[i].permute(1, 2, 0).numpy()
            image_i = transforms.ToPILImage()(images[i])
            nose_i = noses[i]

            # Draw keypoints on the image
            img_with_keypoints = draw_keypoints(image_i, nose_i)
            # Save the image with keypoints
            img_name = f"output_image_{i}_with_keypoints.png"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, cv2.cvtColor(img_with_keypoints, cv2.COLOR_RGB2BGR))
            print(f"Saved {img_path}")

        if batch_idx == 0:  # Break after processing the first batch for brevity
            break