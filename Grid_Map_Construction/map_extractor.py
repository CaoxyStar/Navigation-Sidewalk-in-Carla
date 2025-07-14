import yaml
import argparse
from tqdm import tqdm

import cv2
import carla
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt


def get_grid_map(map_name):
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world(map_name)

    # Load the map configuration
    with open('Grid_Map_Construction/map_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    general_config = config['General']
    map_config = config[map_name]

    x_min, x_max = map_config['x_range']
    y_min, y_max = map_config['y_range']
    resolution = general_config['resolution']
    erode_kernel_size = map_config['erode_kernel_size']
    erode_iterations = map_config['erode_iterations']
    dilate_kernel_size = map_config['dilate_kernel_size']
    dilate_iterations = map_config['dilate_iterations']
    close_kernel_size = map_config['close_kernel_size']
    close_iterations = map_config['close_iterations']

    # Get the world and map in CARLA
    world = client.get_world()
    map = world.get_map()

    # Get the unpassable areas
    x_range = int((x_max - x_min) / resolution)
    y_range = int((y_max - y_min) / resolution)

    grid_map = np.ones((x_range, y_range), dtype=np.uint8) * 255
    for i in tqdm(range(0, x_range * y_range)):
        x = i // y_range
        y = i % y_range
        location = carla.Location(x * resolution + x_min, y * resolution + y_min, 0)
        waypoint = map.get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking)
        if waypoint is not None:
            grid_map[x, y] = 0
    
    # Erode the grid map
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    grid_map = cv2.erode(grid_map, erode_kernel, iterations=erode_iterations)

    # Get the passable areas
    crosswalk_map = map.get_crosswalks()
    for i in range(0, len(crosswalk_map), 5):
        img_points = []
        img_x = []
        img_y = []
        for j in range(4):
            point = [int((crosswalk_map[i+j].y - y_min) / resolution), int((crosswalk_map[i+j].x - x_min) / resolution)]
            img_points.append(point)
            img_x.append(point[0])
            img_y.append(point[1])
        
        max_x, min_x, max_y, min_y = max(img_x), min(img_x), max(img_y), min(img_y)
        if max_x > y_range or min_x < 0 or max_y > x_range or min_y < 0:
            continue

        poly = patches.Polygon(np.array(img_points))
        x, y = np.meshgrid(np.arange(y_range), np.arange(x_range))
        mask = poly.get_path().contains_points(np.c_[x.flatten(), y.flatten()]).reshape(x_range, y_range)
        grid_map[mask] = 255

    # Dilate the grid map
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    grid_map = cv2.dilate(grid_map, dilate_kernel, iterations=dilate_iterations)

    # Closing process
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    grid_map = cv2.morphologyEx(grid_map, cv2.MORPH_CLOSE, kernel=close_kernel, iterations=close_iterations)

    # Save the grid map
    np.save(f'Maps/{map_name}/{map_name}_grid_map.npy', grid_map)

    plt.imshow(grid_map, cmap='gray')
    plt.gca().invert_yaxis()
    plt.savefig(f'Maps/{map_name}/{map_name}_grid_map.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Grid Map')
    parser.add_argument('--map', type=str, required=True, help='please specify the map name')
    args = parser.parse_args()
    get_grid_map(args.map)
    print('Grid map for %s is generated successfully!' % args.map)