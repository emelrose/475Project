import statistics
from pathlib import Path
import dataset
import os
import argparse
import torch
from model import VetNet


# determines the center point of the bounding box returned by yolo
def get_output_center(x_min, y_min, x_max, y_max):
    # find the length and width of the returned bounding box
    x_length = x_max - x_min
    y_length = y_max - y_min

    # find the center co-ordinate of bounding box
    x_center = x_min + x_length / 2
    y_center = y_min + y_length / 2

    return x_center, y_center


# calculate euclidian distance of 2 points
# def euclidian_distance(model_outputs, label_outputs):
#     x1, x2 = model_outputs[0,0], label_outputs[0]
#     y1, y2 = model_outputs[0,1], label_outputs[1]
#     x_component = (x2-x1)**2
#     y_component = (y2-y1)**2
#     distance = math.sqrt(x_component + y_component)
#     return distance

def euclidian_distance(model_outputs, true_labels):  # Also calculates euclidean distance
    # Assuming model_outputs and true labels are tensors of shape (batch_size, 2)
    return torch.norm(model_outputs - true_labels, dim=1)


def distance_statistics(distances):
    minimum_distance = min(distances)
    maximum_distance = max(distances)
    mean_distance = statistics.mean(distances)
    standard_deviation = statistics.stdev(distances, mean_distance)

    return minimum_distance, maximum_distance, mean_distance, standard_deviation


# function to build optimizer and lr_scheduler
def build_train_elements(model, learning_rate, lr_scaling_factor, scheduler_patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=lr_scaling_factor,
        patience=scheduler_patience,
    )
    return optimizer, scheduler


def train(batch_size, num_epochs, model, device, learning_rate, lr_scaling_factor, scheduler_patience):
    print("Training")
    data_loader = dataset.create_dataloader(batch_size, "Train")
    model = model.to(device=device)
    optimizer, scheduler = build_train_elements(
        model,
        learning_rate,
        lr_scaling_factor,
        scheduler_patience
    )

    # t_distances = []
    for epoch_number in range(num_epochs):
        epoch_distances = []

        for batch in data_loader:
            images, noses = batch
            optimizer.zero_grad()
            batch_images = images.to(device=device)
            batch_labels = noses.to(device=device)
            outputs = model(batch_images)

            # Calculate Euclidean_distance
            distance = euclidian_distance(outputs, batch_labels)
            epoch_distances.extend(distance.tolist())  # per batch, convert to list

        torch.cuda.empty_cache()  # Clear GPU memory

        # Convert list to tensor, per epoch
        epoch_distances = torch.tensor(epoch_distances)
        print(f"Epoch {epoch_number + 1}, Mean Distance: {epoch_distances.mean().item()}")

        print("Epoch training Complete")
        print(f"")


def evaluate(batch_size, model, device):
    print("Evaluating")
    model.eval()
    distances = []
    dataloader = dataset.create_dataloader(batch_size, "Test")

    with torch.no_grad():
        for batch in dataloader:
            images, noses = batch
            batch_images = images.to(device=device)
            batch_labels = noses.to(device=device)

            # Forward pass
            outputs = model(batch_images)

            # Calculate Euclidean_distance
            distance = euclidian_distance(outputs, batch_labels)
            distances.extend(distance.tolist())  # per batch, convert to list

    # Convert list to tensor, per epoch
    distances_tensor = torch.tensor(distances, device=device)

    # Calculate statistics based on distance
    min_distance, max_distance, mean_distance, std_distance = distance_statistics(distances)
    print(
        f"Min Distance: {min_distance}, Max Distance: {max_distance}, Mean Distance: {mean_distance}, Std Distance: {std_distance}")

    return min_distance, max_distance, mean_distance, std_distance


def main(args):
    print("Loading ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')
    save_dir = Path(args.s)
    save_dir.mkdir(exist_ok=True, parents=True)
    model = VetNet().to(device=device)

    if args.mode == "train":
        model.train()
        train(batch_size=args.b, num_epochs=args.e, model=model, device=device,
              learning_rate=args.gamma, lr_scaling_factor=0.1, scheduler_patience=4)
        print("training complete!")
    elif args.mode == "test":
        evaluate(batch_size=args.b, model=model, device=device)


if __name__ == '__main__':
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Lab 5 testing")
    # Basic options
    parser.add_argument('-l', type=str)  # was -vgg
    parser.add_argument('-p', type=str)
    # training options
    parser.add_argument('-s', default='./experiments',  # was -save_dir
                        help='Directory to save the model')
    parser.add_argument('-log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('-gamma', type=float, default=1e-4, help='gamma')
    parser.add_argument('-lr_decay', type=float, default=5e-5, help='lr_decay')
    parser.add_argument('-e', type=int, default=160000)  # was max_iter
    parser.add_argument('-b', type=int, default=8)  # was -batch_size
    parser.add_argument('-mode', type=str)
    # parser.add_argument('pth_num', type=int, help='Epoch file number')
    args = parser.parse_args()

    main(args)
