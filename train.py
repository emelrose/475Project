import math
import statistics


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
def euclidian_distance(x1, y1, x2, y2):
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
