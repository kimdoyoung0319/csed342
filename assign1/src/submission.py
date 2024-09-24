# 20200429 Doyoung Kim

def get_route_backtrack(parent, start_station_id, end_station_id):
    route = []
    current_station_id = end_station_id
    while current_station_id is not None:
        route.insert(0, current_station_id)
        current_station_id = parent[current_station_id]
    if route[0] != start_station_id:
        return None
    return route


def calc_heuristic(station1, station2, type='random'):
    if type == 'random':
        return calc_heuristic_random()
    elif type == 'euclidean':
        return calc_heuristic_euclidean(station1, station2)
    elif type == 'simple':
        return calc_heuristic_euclidean(station1, station2)
    else:
        raise ValueError("Invalid heuristic type.")


def calc_heuristic_random():
    import random
    return random.random() * 300


def calc_heuristic_euclidean(station1, station2):
    import math

    # Extract coordinates of the stations
    lon_a, lat_a = station1.coordinate
    lon_b, lat_b = station2.coordinate

    # Convert degrees to radians for cosine calculation
    avg_lat = math.radians((lat_a + lat_b) / 2.0)

    # Constants
    km_per_degree_lat = 111  # Approximately 111 km per degree latitude
    km_per_degree_lon = 111 * math.cos(avg_lat)  # Adjust for longitude

    # Differences in coordinates
    delta_lat = lat_a - lat_b
    delta_lon = lon_a - lon_b

    # Convert degree differences to kilometers
    delta_y = delta_lat * km_per_degree_lat
    delta_x = delta_lon * km_per_degree_lon

    # Euclidean distance
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2) * 1000  # convert to meters

    # Assume average speed of 60 km/h
    speed = 60 * 1000 / 60  # m/min
    travel_time = distance / speed

    return travel_time


def calc_simple_heuristic_euclidean(station1, station2):
    import math
    return math.sqrt((station1.coordinate[0] - station2.coordinate[0]) ** 2 +
                     (station1.coordinate[1] - station2.coordinate[1]) ** 2)


def find_route_bfs(dict_graph, start_station_id, end_station_id):
    """
    Find the route between two stations using Breadth-First Search (BFS).
    :param dict_graph: station graph represented as a dictionary
    :param start_station_id: ID of the start station
    :param end_station_id: ID of the end station
    :return: route and total_time.
    route is a list of station IDs from start to end, or None if no route is found.
    total_time is the accumulated travel_time, or None if no route is found.
    """
    # Dictionary to store the parent node of each station
    parent = {start_station_id: None}
    # Dictionary to store the travel time to reach each station
    travel_time = {start_station_id: 0}

    # Initialize the queue with the start station
    queue = [start_station_id]
    visited = set()

    while queue:
        current_station_id = queue.pop(0)
        visited.add(current_station_id)

        # Check if we have reached the end station
        if current_station_id == end_station_id:
            break

        # Iterate over the routes from the current station
        for route in dict_graph[current_station_id].routes:
            next_station_id = route.station_to.id
            # Calculate the total travel time to reach the next station
            new_travel_time = travel_time[current_station_id] + route.travel_time

            # Update the travel time and parent station if
            # 1. the new time is less than the current time or 2. the station has not been visited
            if next_station_id not in travel_time or new_travel_time < travel_time[next_station_id]:
                travel_time[next_station_id] = new_travel_time
                parent[next_station_id] = current_station_id

            # Add the next station to the queue if it has not been visited
            if next_station_id not in visited:
                queue.append(next_station_id)

    # Reconstruct the route by backtracking from the end station to the start station
    route = get_route_backtrack(parent, start_station_id, end_station_id)
    if route is None:
        return None, None

    # Calculate the total travel time
    total_time = travel_time[end_station_id]

    return route, total_time


def find_route_dfs(dict_graph, start_station_id, end_station_id, max_depth=1000):
    """
    Find the route between two stations using Depth-First Search (DFS).
    The depth of the search should not exceed the max_depth.
    :param dict_graph: station graph represented as a dictionary
    :param start_station_id: ID of the start station
    :param end_station_id: ID of the end station
    :param max_depth: maximum depth for the search
    :return: route and total_time.
    route is a list of station IDs from start to end, or None if no route is found.
    total_time is the accumulated travel_time, or None if no route is found.
    """
    # Dictionary to store the parent node of each station
    parent = {start_station_id: None}
    # Dictionary to store the travel time to reach each station
    travel_time = {start_station_id: 0}

    # BEGIN_YOUR_ANSWER

    # Initialize the queue with the start station
    stack = [start_station_id]
    visited = set()

    while stack:
        current_station_id = stack.pop(0)
        visited.add(current_station_id)

        # Check if we have reached the end station
        if current_station_id == end_station_id:
            break

        # Iterate over the routes from the current station
        for route in dict_graph[current_station_id].routes:
            next_station_id = route.station_to.id
            # Calculate the total travel time to reach the next station
            new_travel_time = travel_time[current_station_id] + route.travel_time

            # Update the travel time and parent station if
            # 1. the new time is less than the current time or 2. the station has not been visited
            if next_station_id not in travel_time or new_travel_time < travel_time[next_station_id]:
                travel_time[next_station_id] = new_travel_time
                parent[next_station_id] = current_station_id

            # Add the next station to the stack if it has not been visited
            if next_station_id not in visited:
                stack.insert(0, next_station_id)

    # END_YOUR_ANSWER

    # Reconstruct the route by backtracking from the end station to the start station
    route = get_route_backtrack(parent, start_station_id, end_station_id)
    if route is None:
        return None, None

    # Calculate the total travel time
    total_time = travel_time[end_station_id]

    return route, total_time


def find_route_iterative_deepening(dict_graph, start_station_id, end_station_id, max_depth=1000):
    """
    Find the route between two stations using Iterative Deepening Search (IDS).
    :param dict_graph: station graph represented as a dictionary
    :param start_station_id: ID of the start station
    :param end_station_id: ID of the end station
    :param max_depth: maximum depth for the search
    :return: route and total_time.
    route is a list of station IDs from start to end, or None if no route is found.
    total_time is the accumulated travel_time, or None if no route is found.
    Hint: Use the find_route_dfs function with increasing depth.
    """
    route, total_time = None, None

    # BEGIN_YOUR_ANSWER



    # END_YOUR_ANSWER

    return route, total_time


def find_route_uniform_cost(dict_graph, start_station_id, end_station_id):
    """
    Find the route between two stations using Uniform Cost Search.
    :param dict_graph: station graph represented as a dictionary
    :param start_station_id: ID of the start station
    :param end_station_id: ID of the end station
    :return: route and total_time.
    route is a list of station IDs from start to end, or None if no route is found.
    total_time is the accumulated travel_time, or None if no route is found.
    """
    # Dictionary to store the parent node of each station
    parent = {start_station_id: None}
    # Dictionary to store the travel time to reach each station
    travel_time = {start_station_id: 0}

    # BEGIN_YOUR_ANSWER



    # END_YOUR_ANSWER

    # Reconstruct the route by backtracking from the end station to the start station
    route = get_route_backtrack(parent, start_station_id, end_station_id)
    if route is None:
        return None, None

    # Calculate the total travel time
    total_time = travel_time[end_station_id]

    return route, total_time


def find_route_greedy(dict_graph, start_station_id, end_station_id, heuristic_type='simple'):
    """
    Find the route between two stations using Greedy Search with the specified heuristic.
    :param dict_graph: station graph represented as a dictionary
    :param start_station_id: ID of the start station
    :param end_station_id: ID of the end station
    :param heuristic_type: type of heuristic to use ('random' or 'euclidean')
    :return: route and total_time.
    route is a list of station IDs from start to end, or None if no route is found.
    total_time is the accumulated travel_time, or None if no route is found.
    """
    # Dictionary to store the parent node of each station
    parent = {start_station_id: None}
    # Dictionary to store the travel time to reach each station
    travel_time = {start_station_id: 0}

    # BEGIN_YOUR_ANSWER



    # END_YOUR_ANSWER

    # Reconstruct the route by backtracking from the end station to the start station
    route = get_route_backtrack(parent, start_station_id, end_station_id)
    if route is None:
        return None, None

    # Calculate the total travel time
    total_time = travel_time[end_station_id]

    return route, total_time


def find_route_a_star(dict_graph, start_station_id, end_station_id, heuristic_type='simple'):
    """
    Find the route between two stations using A* Search with the specified heuristic.
    :param dict_graph: station graph represented as a dictionary
    :param start_station_id: ID of the start station
    :param end_station_id: ID of the end station
    :param heuristic_type: type of heuristic to use ('random' or 'euclidean')
    :return: route and total_time.
    route is a list of station IDs from start to end, or None if no route is found.
    total_time is the accumulated travel_time, or None if no route is found.
    """
    # Dictionary to store the parent node of each station
    parent = {start_station_id: None}
    # Dictionary to store the travel time to reach each station
    travel_time = {start_station_id: 0}

    # BEGIN_YOUR_ANSWER



    # END_YOUR_ANSWER

    # Reconstruct the route by backtracking from the end station to the start station
    route = get_route_backtrack(parent, start_station_id, end_station_id)
    if route is None:
        return None, None

    # Calculate the total travel time
    total_time = travel_time[end_station_id]

    return route, total_time
