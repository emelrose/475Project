import math
import statistics
import dataset
import torch



#determines the center point of the bounding box returned by yolo
def get_output_center(x_min, y_min, x_max, y_max):
    #find the length and width of the returned bounding box
    x_length = x_max - x_min
    y_length = y_max - y_min

    #find the center co-ordinate of bounding box
    x_center = x_min + x_length / 2
    y_center = y_min + y_length / 2

    return x_center, y_center


#calcualte euclidian distance of 2 points
def euclidian_distance(model_outputs, label_outputs):
    x1, x2 = model_outputs[0,0], label_outputs[0]
    y1, y2 = model_outputs[0,1], label_outputs[1]
    x_component = (x2-x1)**2
    y_component = (y2-y1)**2
    distance = math.sqrt(x_component + y_component)
    return distance


def distance_statistics(distances):
    minimum_distance = min(distances)
    maximum_distance = max(distances)
    mean_distance = statistics.mean(distances)
    standard_deviation = statistics.stdev(distances, mean_distance)

    return minimum_distance, maximum_distance, mean_distance, standard_deviation

#function to build optimizer and lr_scheduler
def build_train_elements(model, learning_rate, lr_scaling_factor, scheduler_patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=lr_scaling_factor,
        patience=scheduler_patience,
    )
    return optimizer, scheduler


def train(batch_size, num_epochs, model, device, learning_rate, lr_scaling_factor, scheduler_patience):

    data_loader = dataset.create_dataloader(batch_size, "Train")
    model = model.to(device=device)
    optimizer, scheduler = build_train_elements(
        model,
        learning_rate,
        lr_scaling_factor,
        scheduler_patience
    )

    t_distances = []
    for epoch_number in range(num_epochs):
        epoch_loss = 0
        epoch_distances = []
        #loss = euclidian_distance()
        for batch_images, batch_labels in data_loader:
            batch_distances = []
            optimizer.zero_grad()
            batch_images = batch_images.to(device=device)
            batch_labels = batch_labels.to(device=device)
            outputs = model(batch_images)

            print(outputs.data[0,0])



            #for distance in batch_distances:
            #    epoch_distances.append(distance)

        print("Epoch training Complete")
        print(f"")
