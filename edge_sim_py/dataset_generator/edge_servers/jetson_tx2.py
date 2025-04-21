"""Contains a function that creates NVIDIA Jetson TX2 edge server objects with the capacity/power specifications provided in [1].

[1] Süzen, Ahmet Ali, Burhan Duman, and Betül Şen. "Benchmark Analysis of Jetson TX2, Jetson nano and
    raspberry Pi using Deep-CNN." In 2020 International Congress on Human-Computer Interaction,
    Optimization and Robotic Applications (HORA), pp. 1-5. IEEE, 2020.

[2] Belloch, Jose A., et al. "Multicore implementation of a multichannel parallel graphic equalizer."
    The Journal of Supercomputing 78.14 (2022): 15715-15729.

        [2] is used for determining the cpu frequency of NVIDIA Jetson TX2
"""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer


def jetson_tx2() -> object:
    """Creates an edge server object with the specifications of NVIDIA Jetson TX2 single-board computers provided in [1].

    Returns:
        edge_server (object): Created edge server object.
    """
    edge_server = EdgeServer()
    edge_server.model_name = "NVIDIA Jetson TX2"

    # Computational capacity (CPU in cores, RAM memory in megabytes, and disk in megabytes)
    edge_server.cpu = 4
    edge_server.memory = 8192
    edge_server.disk = 32768
    edge_server.cpu_cycle = 1479 * (10 ** 6)  ## GHz = 10^9 / MHz = 10^6 ##

    # Power-related attributes
    edge_server.power_model_parameters = {
        "max_power_consumption": 15,
        "static_power_percentage": 7.5 / 15,
    }

    return edge_server
