# EdgeSimPy components
from edge_sim_py import *

# Python libraries
import subprocess
from sklearn.cluster import KMeans
from random import seed, choice
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import json
import copy
import os
#################################

# Enforcing Matplotlib and NetworkX versions that don't produce visualization errors
if matplotlib.__version__ != '3.5.0':
    subprocess.check_call(["pip", "install", "matplotlib==3.5.0"])

if nx.__version__ != '2.6.3':
    subprocess.check_call(["pip", "install", "networkx==2.6.3"])

def uniform(n_items: int, valid_values: list, shuffle_distribution: bool = True) -> list:
    """Creates a list of size "n_items" with values from "valid_values" according to the uniform distribution.
    By default, the method shuffles the created list to avoid unbalanced spread of the distribution.

    Args:
        n_items (int): Number of items that will be created.
        valid_values (list): List of valid values for the list of values.
        shuffle_distribution (bool, optional): Defines whether the distribution is shuffled or not. Defaults to True.

    Raises:
        Exception: Invalid "valid_values" argument.

    Returns:
        uniform_distribution (list): List of values arranged according to the uniform distribution.
    """
    if not isinstance(valid_values, list) or isinstance(valid_values, list) and len(valid_values) == 0:
        raise Exception("You must inform a list of valid values within the 'valid_values' attribute.")

    # Number of occurrences that will be created of each item in the "valid_values" list
    distribution = [int(n_items / len(valid_values)) for _ in range(0, len(valid_values))]

    # List with size "n_items" that will be populated with "valid_values" according to the uniform distribution
    uniform_distribution = []

    for i, value in enumerate(valid_values):
        for _ in range(0, int(distribution[i])):
            uniform_distribution.append(value)

    # Computing leftover randomly to avoid disturbing the distribution
    leftover = n_items % len(valid_values)
    for i in range(leftover):
        random_valid_value = random.choice(valid_values)
        uniform_distribution.append(random_valid_value)

    # Shuffling distribution values in case 'shuffle_distribution' parameter is True
    if shuffle_distribution:
        random.shuffle(uniform_distribution)

    return uniform_distribution

def display_topology(topology: object, output_filename: str = "topology"):
    # Customizing visual representation of topology
    positions = {}
    labels = {}
    colors = []
    sizes = []

    # Gathering the coordinates of edge servers
    edge_server_coordinates = [edge_server.coordinates for edge_server in EdgeServer.all()]

    # Defining list of color options
    color_options = []
    for _ in range(EdgeServer.count()):
        color_options.append([random.random(), random.random(), random.random()])

    for node in topology.nodes():
        positions[node] = node.coordinates
        labels[node] = node.id
        node_size = 500 if any(user.coordinates == node.coordinates for user in User.all()) else 100
        sizes.append(node_size)

        if node.coordinates in edge_server_coordinates:
            colors.append("red")
            # colors.append([primary_color_value / 3 for primary_color_value in color_options[node_clusters[node.id - 1]]])
        else:
            colors.append("black")
            # colors.append(color_options[node_clusters[node.id - 1]])

    # Configuring drawing scheme
    nx.draw(
        topology,
        pos=positions,
        node_color=colors,
        node_size=sizes,
        labels=labels,
        font_size=6,
        font_weight="bold",
        font_color="whitesmoke",
    )

    # Saving a topology image in the disk
    plt.savefig(f"{output_filename}.png", dpi=120)



# Application -> provisioned
def application_to_dict(self) -> dict:
    """Method that overrides the way the object is formatted to JSON."
    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    dictionary = {
        "attributes": {
            "id": self.id,
            "label": self.label,
            # "provisioned": self.provisioned,
        },
        "relationships": {
            "services": [{"class": type(service).__name__, "id": service.id} for service in self.services],
            "users": [{"class": type(user).__name__, "id": user.id} for user in self.users],
        },
    }
    return dictionary



# User -> providers_trust
def user_to_dict(self) -> dict:
    """Method that overrides the way User objects are formatted to JSON."
    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    access_patterns = {}
    for app_id, access_pattern in self.access_patterns.items():
        access_patterns[app_id] = {"class": access_pattern.__class__.__name__, "id": access_pattern.id}

    dictionary = {
        "attributes": {
            "id": self.id,
            "coordinates": self.coordinates,
            "coordinates_trace": self.coordinates_trace,
            "delays": copy.deepcopy(self.delays),
            "delay_slas": copy.deepcopy(self.delay_slas),
            "communication_paths": copy.deepcopy(self.communication_paths),
            "making_requests": copy.deepcopy(self.making_requests),
        },
        "relationships": {
            "access_patterns": access_patterns,
            "mobility_model": self.mobility_model.__name__,
            "applications": [{"class": type(app).__name__, "id": app.id} for app in self.applications],
            "base_station": {"class": type(self.base_station).__name__, "id": self.base_station.id},
        },
    }
    return dictionary




# EdgeServer -> infrastructure_provider
def edge_server_to_dict(self) -> dict:
    """Method that overrides the way EdgeServer objects are formatted to JSON."

    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    dictionary = {
        "attributes": {
            "id": self.id,
            "available": self.available,
            "model_name": self.model_name,
            "cpu": self.cpu,
            "memory": self.memory,
            "disk": self.disk,
            "cpu_demand": self.cpu_demand,
            "cpu_cycle": self.cpu_cycle,
            "memory_demand": self.memory_demand,
            "disk_demand": self.disk_demand,
            "coordinates": self.coordinates,
            "max_concurrent_layer_downloads": self.max_concurrent_layer_downloads,
            "active": self.active,
            "power_model_parameters": self.power_model_parameters,
        },
        "relationships": {
            "power_model": self.power_model.__name__ if self.power_model else None,
            "base_station": {"class": type(self.base_station).__name__, "id": self.base_station.id} if self.base_station else None,
            "network_switch": {"class": type(self.network_switch).__name__, "id": self.network_switch.id}
            if self.network_switch
            else None,
            "services": [{"class": type(service).__name__, "id": service.id} for service in self.services],
            "container_layers": [{"class": type(layer).__name__, "id": layer.id} for layer in self.container_layers],
            "container_images": [{"class": type(image).__name__, "id": image.id} for image in self.container_images],
            "container_registries": [{"class": type(reg).__name__, "id": reg.id} for reg in self.container_registries],
        },
    }
    return dictionary


# Service -> privacy_requirement
def service_to_dict(self) -> dict:
    """Method that overrides the way Service objects are formatted to JSON."

    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    dictionary = {
        "attributes": {
            "id": self.id,
            "label": self.label,
            "state": self.state,
            "_available": self._available,
            "cpu_demand": self.cpu_demand,
            "cpu_cycles_demand": self.cpu_cycles_demand,
            "memory_demand": self.memory_demand,
            "image_digest": self.image_digest,
        },
        "relationships": {
            "application": {"class": type(self.application).__name__, "id": self.application.id},
            "server": {"class": type(self.server).__name__, "id": self.server.id} if self.server else None,
        },
    }
    return dictionary


def is_central_node(coord, all_coords, central_nodes):
    # Define the six possible neighbor directions in a hexagonal grid
    directions = [(2, 0), (-2, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    for direction in directions:
        neighbor = (coord[0] + direction[0], coord[1] + direction[1])

        # Check if the neighbor exists in the grid
        if neighbor not in all_coords:
            return False

        # Check if the neighbor is already a central node
        if neighbor in central_nodes:
            return False

    return True

def hex_distance(coord1, coord2):
    # Convert axial coordinates to cube coordinates
    x1, y1 = coord1
    z1 = -x1 - y1
    x2, y2 = coord2
    z2 = -x2 - y2

    # Calculate the distance using the cube coordinates
    return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))

def y_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2

    return (abs(y1 - y2))


# Defining a seed value to enable reproducibility
seed(1)

# Creating list of map coordinates
# MAP_SIZE = 10
MAP_SIZE = 5
map_coordinates = hexagonal_grid(x_size=MAP_SIZE, y_size=MAP_SIZE)
### List to hold the central node coordinates // central nodes are considered to be the base-stations with edge-servers
central_node_coords = []
# List to store indices of elements to be removed
indices_to_remove = []

### prunning the coordinate to have complete hexagons
### Iterate over the list in reverse order to avoid skipping elements when removing
for i in range(len(map_coordinates) - 1, -1, -1):
    if map_coordinates[i][0] == 0:
        del map_coordinates[i]

## Find central nodes
for coord in map_coordinates:
    if is_central_node(coord, map_coordinates, central_node_coords):
        central_node_coords.append(coord)

# Iterate over the map_coordinates list in reverse order to safely determine elements to remove
for i in range(len(map_coordinates) - 1, -1, -1):
    not_adjacent = True
    for central_coord in central_node_coords:
        if (hex_distance(map_coordinates[i], central_coord) <= 2) and (
                y_distance(map_coordinates[i], central_coord) < 2):
            not_adjacent = False

    # If the coordinate is not adjacent to any central node, mark it for removal
    if not_adjacent:
        indices_to_remove.append(i)

## Now remove the items outside of the loop
for i in indices_to_remove:
    del map_coordinates[i]



# Creating 'base stations' for providing wireless connectivity to users
# Creating 'network switches' for wired connectivity between base-stations
for coordinates in map_coordinates:
    # Creating the base station object
    base_station = BaseStation()
    # wireless delay between users and connected base-station (was 0, change to 2)
    base_station.wireless_delay = 2
    base_station.coordinates = coordinates
    # print(f"base_station {base_station.id} coordinates {base_station.coordinates}")

    # Creating network switch object using the "sample_switch()" generator, which embeds built-in power consumption specs
    network_switch = sample_switch()
    base_station._connect_to_network_switch(network_switch=network_switch)


# Creating a partially-connected mesh network topology
partially_connected_hexagonal_mesh(
    network_nodes=NetworkSwitch.all(),
    link_specifications=[
        # {"number_of_objects": 195, "delay": 1, "bandwidth": 12.5},   ### for (10x10)
        {"number_of_objects": 49, "delay": 1, "bandwidth": 10},      ### for (5x5)
    ],
)

### edge_servers are the ones introduced in edge_servers folder including:
## edge_server.model_name = "E5430"
## edge_server.model_name = "E5507"
## edge_server.model_name = "E5645"
## edge_server.model_name = "NVIDIA Jetson Nano"
## edge_server.model_name = "NVIDIA Jetson TX2"
## edge_server.model_name = "Raspberry Pi 4"
# print(jetson_nano().model_name)


#### Connecting base-stations (central BS) to edge-servers
edge_server_functions = [jetson_tx2, jetson_tx2, e5430, e5645]
# Create an empty list to store base stations that match the criteria
base_stations_central_nodes_without_servers = []

for i in range(len(central_node_coords)):
    # selecting equal edge servers from list 'edge_server_functions'
    # j = i % 4  ## for 10x10
    j = i      ## for 5x5
    # Creating the edge server object
    edge_server = edge_server_functions[j]()

    # Defining the maximum number of layers that the edge server can pull simultaneously
    edge_server.max_concurrent_layer_downloads = 3

    # Specifying the edge server's power model
    edge_server.power_model = LinearServerPowerModel

    # Connecting the edge server to a specific base station
    base_stations_central_nodes_without_servers = [
        base_station
        for base_station in BaseStation.all()
        if not base_station.edge_servers and base_station.coordinates in central_node_coords
    ]
    base_station = random.choice(base_stations_central_nodes_without_servers)
    base_station._connect_to_edge_server(edge_server=edge_server)


####################################################################
### Reading specifications of tools' images from container images ##
####################################################################
current_dir_dataset = os.path.dirname(__file__)
read_file = os.path.join(current_dir_dataset, "container_images.json")

with open("container_images.json", "r", encoding="UTF-8") as read_file:
    container_image_specifications = json.load(read_file)

### Manually add "registry" image specification from DockerHub in container_images.json to be used by container registries
container_registry_image = {
    "name": "registry",
    "digest": "sha256:12120425f07de11a1b899e418d4b0ea174c8d4d572d45bdb640f93bc7ca06a3d",
    "layers": [
        {
            "digest": "sha256:ce7f800efff9a5cfddf4805e3887943b4a7bd97cf83140587336261130ece03b",
            "size": 17.5,
        },
        {
            "digest": "sha256:30609d4f10ddcd7a9a6287abb4100a991745eb7c9db7cdeb1140ac547634a204",
            "size": 7.35,
        },
    ],
}
container_image_specifications.append(container_registry_image)



# Defining service image specifications
service_image_specifications = [
    ###########################
    #### Operating Systems ####
    ###########################
    {"state": 0, "image_name": "ubuntu"},
    {"state": 0, "image_name": "alpine"},
    ###########################
    #### Language Runtimes ####
    ###########################
    {"state": 0, "image_name": "python"},
    ##############################
    #### Generic Applications ####
    ##############################
    {"state": 0, "image_name": "nginx"},  # Web server
    {"state": 0, "image_name": "envoy"},  # Web server
    ###########################
    #### Databases/Cache ####
    ###########################
    {"state": 0, "image_name": "redis"}, # Crowd Counting - highly optimized for low-latency access and high-throughput operations.
    {"state": 0, "image_name": "aerospike"}, # Face Recognition - high-throughput & low-latency streaming data where can manage large-scale streaming data
    {"state": 0, "image_name": "kafka"}, # ML Model Training & Developing - handle large volumes of data with low-latency access, retraining models with new data / deploying models needs real-time-access to large datasets
    ##############################
    #### Machine Learning Framework ####
    ##############################
    {"state": 0, "image_name": "pytorch"},  # Deep learning framework
    {"state": 0, "image_name": "tensorflow"},  # Deep learning framework
    ##############################
    #### Machine Learning Models ####
    ##############################
    {"state": 0, "image_name": "yolov8"},  # Object detection model
    {"state": 0, "image_name": "mobilenetssd"},  # Lightweight object detection model
]


# Adding a "latest" tag to all container images
condensed_images_metadata = []
for container_image in container_image_specifications:
    container_image["tag"] = "latest"
    condensed_images_metadata.append(
        {
            "name": container_image["name"],
            "tag": container_image["tag"],
            "layers": container_image["layers"],
        }
    )


container_registry_specifications = [
    {
        "number_of_objects": 1,
        "cpu_demand": 1,
        "memory_demand": 25.4,
        "images": condensed_images_metadata,
    }
]

# Parsing the specifications for container images and container registries
container_registries = create_container_registries(
    container_registry_specifications=container_registry_specifications,
    container_image_specifications=container_image_specifications,
)


# Defining the initial placement for container images and container registries
worst_fit_registries(container_registry_specifications=container_registries, servers=EdgeServer.all())


# Defining user placement and applications/services specifications
def random_user_placement():
    """ Method that determines the coordinates of a given user randomly.
    Returns:
        coordinates (tuple): Random user coordinates.
    """
    coordinates = choice(map_coordinates)
    return coordinates

# """
# Required memory of each service (megabyte):
# ubuntu size: 78.1
# nginx size: 187.8
# redis size: 116.92
# kafka size: 650
# alpine size: 7.8
# aerospike size: 210.9
# envoy size: 205
# python size: 1020.1
# pytorch size: 7603.9
# tensorflow size: 1857.61
# yolov8 size: 14171.45
# mobilenetssd size: 1473.5
# """

# Defining applications/services specifications
"""
app(crowd counting) -> "number_of_objects": 228, "number_of_services": 5 (alpine, python, nginx, redis, mobilenetssd)
app(face recognition) -> "number_of_objects": 25, "number_of_services": 5 (ubuntu, python, envoy, aerospike, yolov8)
app(ml model training&development of crowd counting) -> "number_of_objects": 7, "number_of_services": 6 (ubuntu, python, envoy, kafka, pytorch, mobilenetssd)
app(ml model training&development of face recognition) -> "number_of_objects": 3, "number_of_services": 6 (ubuntu, python, envoy, kafka, tensorflow, yolov8)
"""

# ## for 10x10
# application_specifications = [
#     {"label_of_app": 'crowd counting', "number_of_objects": 152, "number_of_services": 5,
#      "name_of_services": ['alpine', 'python', 'nginx', 'redis', 'mobilenetssd']},
#     {"label_of_app": 'face recognition', "number_of_objects": 16, "number_of_services": 5,
#      "name_of_services": ['ubuntu', 'python', 'envoy', 'aerospike', 'yolov8']},
#     {"label_of_app": 'crowd counting ml dev', "number_of_objects": 4, "number_of_services": 6,
#      "name_of_services": ['ubuntu', 'python', 'envoy', 'kafka', 'pytorch', 'mobilenetssd']},
#     {"label_of_app": 'face recognition ml dev', "number_of_objects": 2, "number_of_services": 6,
#      "name_of_services": ['ubuntu', 'python', 'envoy', 'kafka', 'tensorflow', 'yolov8']},
# ]
# for 5x5
application_specifications = [
    {"label_of_app": 'crowd counting', "number_of_objects": 44, "number_of_services": 5,
     "name_of_services": ['alpine', 'python', 'nginx', 'redis', 'mobilenetssd']},
    {"label_of_app": 'face recognition', "number_of_objects": 6, "number_of_services": 5,
     "name_of_services": ['ubuntu', 'python', 'envoy', 'aerospike', 'yolov8']},
    {"label_of_app": 'crowd counting ml dev', "number_of_objects": 1, "number_of_services": 6,
     "name_of_services": ['ubuntu', 'python', 'envoy', 'kafka', 'pytorch', 'mobilenetssd']},
    {"label_of_app": 'face recognition ml dev', "number_of_objects": 1, "number_of_services": 6,
     "name_of_services": ['ubuntu', 'python', 'envoy', 'kafka', 'tensorflow', 'yolov8']},
]

# Define deadline values for each application
valid_values_for_apps = {
    'crowd counting': [22, 23],       # 1 for proc in edge server
    'face recognition': [44, 46],       # 2 for proc in edge server
    'crowd counting ml dev': [2800, 4000],  # 5 for proc in edge server
    'face recognition ml dev': [5600, 8000], # 7 for proc in edge server
}

# Generate deadline for each application
delay_slas = []
for app_spec in application_specifications:
    app_label = app_spec["label_of_app"]
    n_items = app_spec["number_of_objects"]
    valid_values = valid_values_for_apps.get(app_label, [2, 3])  # Default to [2, 3] if not found
    delay_slas.extend(uniform(n_items=n_items, valid_values=valid_values, shuffle_distribution=True))


## Defining service demands
service_demand_values = [
    {"label": 'ubuntu', "cpu": 1, "memory": 350, "cpu_cycles_demand": (25_000 * 350)},
    {"label": 'alpine', "cpu": 1, "memory": 60, "cpu_cycles_demand": (55_00 * 60)},
    {"label": 'nginx', "cpu": 1, "memory": 60, "cpu_cycles_demand": (35_00 * 60)},
    {"label": 'redis', "cpu": 1, "memory": 60, "cpu_cycles_demand": (15_000 * 60)},
    {"label": 'kafka', "cpu": 1, "memory": 350, "cpu_cycles_demand": (25_000 * 350)},
    {"label": 'aerospike', "cpu": 1, "memory": 250, "cpu_cycles_demand": (20_000 * 250)},
    {"label": 'envoy', "cpu": 1, "memory": 150, "cpu_cycles_demand": (25_000 * 150)},
    {"label": 'python', "cpu": 1, "memory": 60, "cpu_cycles_demand": (30_000 * 60)},
    {"label": 'pytorch', "cpu": 1, "memory": 1024, "cpu_cycles_demand": (3_000_000 * 1024)},
    {"label": 'tensorflow', "cpu": 1, "memory": 1024, "cpu_cycles_demand": (3_000_000 * 1024)},
    {"label": 'yolov8', "cpu": 1, "memory": 800, "cpu_cycles_demand": (250_000 * 800)},
    {"label": 'mobilenetssd', "cpu": 1, "memory": 450, "cpu_cycles_demand": (90_000 * 450)},
]

# 'number_of_services' considers all service demand combinations based on 'application_specifications'
number_of_services = (
    sum([app_spec["number_of_objects"] * app_spec["number_of_services"] for app_spec in application_specifications]) * 2
)

for app_spec in application_specifications:
    for _ in range(app_spec["number_of_objects"]):
        # Creating an application
        app = Application()

        # Creating the user that access the application
        user = User()

        user.communication_paths[str(app.id)] = None
        user.delays[str(app.id)] = None
        user.delay_slas[str(app.id)] = delay_slas[user.id - 1]

        # Defining user's coordinates and connecting him to a base station
        user.mobility_model = random_mobility
        user._set_initial_position(coordinates=random_user_placement(), number_of_replicates=0)

        # Defining user's access pattern
        CircularDurationAndIntervalAccessPattern(
            user=user,
            app=app,
            start=1,
            duration_values=[float("inf")],
            interval_values=[0],
        )

        # Defining the relationship attributes between the user and the application
        user.applications.append(app)
        app.users.append(user)

        # Creating the services that compose the application
        for service_index in range(app_spec["number_of_services"]):
            # Gathering information on the service image based on the specified 'name' and 'tag' parameters
            service_image = next((img for img in ContainerImage.all() if img.name == app_spec['name_of_services'][service_index]), None)

            service_image_cpu_cycles_demand = 0
            service_image_memory = 0
            service_image_cpu = 0

            for service in service_demand_values:
                if service['label'] == service_image.name:
                    # print(f'cpu cycles demand of {service_image.name}: {service["cpu_cycles_demand"]}')
                    service_image_cpu_cycles_demand = service["cpu_cycles_demand"]

                    # print(f'memory of {service_image.name}: {service["memory"]}')
                    service_image_memory = service["memory"]

                    # print(f'cpu of {service_image.name}: {service["cpu"]}')
                    service_image_cpu = service["cpu"]

            # Creating the service object
            service = Service(
                image_digest=service_image.digest,
                cpu_demand=None,
                memory_demand=None,
                cpu_cycles_demand=None,
                label=service_image.name,
                state=0,
            )

            service.cpu_demand = service_image_cpu
            service.memory_demand = service_image_memory
            service.cpu_cycles_demand = service_image_cpu_cycles_demand

            # Connecting the application to its new service
            app.connect_to_service(service)


##########################
#### DATASET ANALYSIS ####
##########################
# Calculating the network delay between users and edge servers (useful for defining reasonable delay SLAs)
users = []
for user in User.all():
    user_metadata = {"object": user, "all_delays": []}
    edge_servers = []
    for edge_server in EdgeServer.all():
        path = nx.shortest_path(
            G=Topology.first(), source=user.base_station.network_switch, target=edge_server.network_switch, weight="delay"
        )
        user_metadata["all_delays"].append(Topology.first().calculate_path_delay(path=path))
    user_metadata["min_delay"] = min(user_metadata["all_delays"])
    user_metadata["max_delay"] = max(user_metadata["all_delays"])
    user_metadata["avg_delay"] = sum(user_metadata["all_delays"]) / len(user_metadata["all_delays"])
    user_metadata["delays"] = {}
    for delay in sorted(list(set(user_metadata["all_delays"]))):
        user_metadata["delays"][delay] = user_metadata["all_delays"].count(delay)

    users.append(user_metadata)


# Calculating the infrastructure occupation and information about the services
edge_server_memory_capacity = 0
service_cpu_demand = 0
service_memory_demand = 0
service_cpu_cycles_demand_demand = 0
edge_server_cpu_cycle_capacity = 0
avg_cpu_cycles_demand = 0
avg_edge_server_cpu_cycle_capacity = 0

for edge_server in EdgeServer.all():
    edge_server_memory_capacity += edge_server.memory
    edge_server_cpu_cycle_capacity += (edge_server.cpu_cycle * edge_server.cpu)  ## clock frequency of each core multiply to number of cores

avg_edge_server_cpu_cycle_capacity = (edge_server_cpu_cycle_capacity / len(EdgeServer.all()))
# print(f"\navg_edge_server_cpu_cycle_capacity:{avg_edge_server_cpu_cycle_capacity}")

for service in Service.all():
    service_memory_demand += service.memory_demand
    service_cpu_cycles_demand_demand += service.cpu_cycles_demand

avg_cpu_cycles_demand = round((service_cpu_cycles_demand_demand / len(Service.all())), 1)
# print(f"\navg_cpu_cycles_demand:{avg_cpu_cycles_demand}")

avg_execution_time = avg_cpu_cycles_demand / avg_edge_server_cpu_cycle_capacity
# print(f"\navg_execution_time:{avg_execution_time}")

overall_memory_occupation = round((service_memory_demand / edge_server_memory_capacity) * 100, 1)
overall_clock_frequency_occupation = round((service_cpu_cycles_demand_demand / edge_server_cpu_cycle_capacity) * 100, 1)

print("\n\n==== INFRASTRUCTURE OCCUPATION OVERVIEW ====")
print(f"Edge Servers: {EdgeServer.count()}")
print(f"\tTotal CPU computational capacity: {edge_server_cpu_cycle_capacity} Hz")  ## regarding number of cores and their clock frequency
print(f"\tTotal RAM capacity: {edge_server_memory_capacity} MB")

print(f"\nEdge users: {User.count()}")
total_CLS_edge_users = 0  ## Critical Sensitivity
total_HLS_edge_users = 0  ## High Sensitivity
total_MLS_edge_users = 0  ## Moderate Sensitivity
total_LLS_edge_users = 0  ## Low Sensitivity
for a in Application.all():
    if ((list(a.users[0].delay_slas.values())[0] == 22) or (list(a.users[0].delay_slas.values())[0] == 23)):
        total_CLS_edge_users = total_CLS_edge_users + 1
        for app, values in valid_values_for_apps.items():
            if ((22) or (23)) in values:
                CLS_app_name = app
    elif ((list(a.users[0].delay_slas.values())[0] == 44) or (list(a.users[0].delay_slas.values())[0] == 46)):
        total_HLS_edge_users = total_HLS_edge_users + 1
        for app, values in valid_values_for_apps.items():
            if ((44) or (46)) in values:
                HLS_app_name = app
    elif ((list(a.users[0].delay_slas.values())[0] == 2800) or (list(a.users[0].delay_slas.values())[0] == 4000)):
        total_MLS_edge_users = total_MLS_edge_users + 1
        for app, values in valid_values_for_apps.items():
            if ((2800) or (4000)) in values:
                MLS_app_name = app
    elif ((list(a.users[0].delay_slas.values())[0] == 5600) or (list(a.users[0].delay_slas.values())[0] == 8000)):
        total_LLS_edge_users = total_LLS_edge_users + 1
        for app, values in valid_values_for_apps.items():
            if ((5600) or (8000)) in values:
                LLS_app_name = app
print(f"\t{CLS_app_name} [Critical Sensitivity]: {total_CLS_edge_users} ({round(((total_CLS_edge_users/User.count())*100),2)}%)")
print(f"\t{HLS_app_name} [High Sensitivity]: {total_HLS_edge_users} ({round(((total_HLS_edge_users/User.count())*100),2)}%)")
print(f"\t{MLS_app_name} [Moderate Sensitivity]: {total_MLS_edge_users} ({round(((total_MLS_edge_users/User.count())*100),2)}%)")
print(f"\t{LLS_app_name} [Low Sensitivity]: {total_LLS_edge_users} ({round(((total_LLS_edge_users/User.count())*100),2)}%)")

print(f"\nServices of {User.count()} edge users: {Service.count()}")

print(f"\tTotal CPU Cycles Demands: {service_cpu_cycles_demand_demand}")
total_cpu_cycles_CLS = 0  ## Critical Sensitivity
total_cpu_cycles_HLS = 0  ## High Sensitivity
total_cpu_cycles_MLS = 0  ## Moderate Sensitivity
total_cpu_cycles_LLS = 0  ## Low Sensitivity
for a in Application.all():
    if ((list(a.users[0].delay_slas.values())[0] == 22) or (list(a.users[0].delay_slas.values())[0] == 23)):
        for i in a.services:
            total_cpu_cycles_CLS = total_cpu_cycles_CLS + i.cpu_cycles_demand
    elif ((list(a.users[0].delay_slas.values())[0] == 44) or (list(a.users[0].delay_slas.values())[0] == 46)):
        for i in a.services:
            total_cpu_cycles_HLS = total_cpu_cycles_HLS + i.cpu_cycles_demand
    elif ((list(a.users[0].delay_slas.values())[0] == 2800) or (list(a.users[0].delay_slas.values())[0] == 4000)):
        for i in a.services:
            total_cpu_cycles_MLS = total_cpu_cycles_MLS + i.cpu_cycles_demand
    elif ((list(a.users[0].delay_slas.values())[0] == 5600) or (list(a.users[0].delay_slas.values())[0] == 8000)):
        for i in a.services:
            total_cpu_cycles_LLS = total_cpu_cycles_LLS + i.cpu_cycles_demand
print(f"\t\t[Critical Sensitivity]: {total_cpu_cycles_CLS} ({round(((total_cpu_cycles_CLS/service_cpu_cycles_demand_demand)*100),2)}%)")
print(f"\t\t[High Sensitivity]: {total_cpu_cycles_HLS} ({round(((total_cpu_cycles_HLS/service_cpu_cycles_demand_demand)*100),2)}%)")
print(f"\t\t[Moderate Sensitivity]: {total_cpu_cycles_MLS} ({round(((total_cpu_cycles_MLS/service_cpu_cycles_demand_demand)*100),2)}%)")
print(f"\t\t[Low Sensitivity]: {total_cpu_cycles_LLS} ({round(((total_cpu_cycles_LLS/service_cpu_cycles_demand_demand)*100),2)}%)")

print(f"\tTotal RAM Demand: {service_memory_demand} MB")
total_memory_CLS = 0  ## Critical Sensitivity
total_memory_HLS = 0  ## High Sensitivity
total_memory_MLS = 0  ## Moderate Sensitivity
total_memory_LLS = 0  ## Low Sensitivity
for a in Application.all():
    if ((list(a.users[0].delay_slas.values())[0] == 22) or (list(a.users[0].delay_slas.values())[0] == 23)):
        for i in a.services:
            total_memory_CLS = total_memory_CLS + i.memory_demand
    elif ((list(a.users[0].delay_slas.values())[0] == 44) or (list(a.users[0].delay_slas.values())[0] == 46)):
        for i in a.services:
            total_memory_HLS = total_memory_HLS + i.memory_demand
    elif ((list(a.users[0].delay_slas.values())[0] == 2800) or (list(a.users[0].delay_slas.values())[0] == 4000)):
        for i in a.services:
            total_memory_MLS = total_memory_MLS + i.memory_demand
    elif ((list(a.users[0].delay_slas.values())[0] == 5600) or (list(a.users[0].delay_slas.values())[0] == 8000)):
        for i in a.services:
            total_memory_LLS = total_memory_LLS + i.memory_demand
print(f"\t\t[Critical Sensitivity]: {total_memory_CLS} ({round(((total_memory_CLS/service_memory_demand)*100),2)}%)")
print(f"\t\t[High Sensitivity]: {total_memory_HLS} ({round(((total_memory_HLS/service_memory_demand)*100),2)}%)")
print(f"\t\t[Moderate Sensitivity]: {total_memory_MLS} ({round(((total_memory_MLS/service_memory_demand)*100),2)}%)")
print(f"\t\t[Low Sensitivity]: {total_memory_LLS} ({round(((total_memory_LLS/service_memory_demand)*100),2)}%)")

##########################
### Exporting scenario ###
##########################
"""
If you want to export your dataset,
    please uncomment the following lines & change the dataset name to your preferred name
"""
# Application._to_dict = application_to_dict
# User._to_dict = user_to_dict
# EdgeServer._to_dict = edge_server_to_dict
# Service._to_dict = service_to_dict
# ComponentManager.export_scenario(save_to_file=True, file_name="dataset1")
# display_topology(Topology.first())