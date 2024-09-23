import networkx as nx
from station import Station, Route


def load_graph(file_path):
    """
    Load the Seoul Metro graph from a nx_graphML file.
    """
    nx_graph = nx.read_gml(file_path)
    return nx_graph


def augment_graph_with_travel_times(nx_graph, type='uniform', travel_time=2):
    """
    Add travel times to the edges in the graph.
    """
    for u, v, data in nx_graph.edges(data=True):
        if type == 'uniform':
            # Assign a uniform travel time to all edges
            data['travel_time'] = travel_time
        elif type == 'distance':
            import math
            # Assign travel time based on the distance between stations
            pos_u = nx_graph.nodes[u]['pos']
            pos_v = nx_graph.nodes[v]['pos']
            # calculate distance between two coordinates
            R = 6371e3  # earth radius in meters
            lat1 = math.radians(pos_u[1])
            lat2 = math.radians(pos_v[1])
            delta_lat = math.radians(pos_v[1] - pos_u[1])
            delta_lon = math.radians(pos_v[0] - pos_u[0])
            a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c
            # Assume average speed of 60 km/h
            speed = 60 * 1000 / 60  # m/min
            travel_time = (distance / speed) + 1  # add 1 minute for stopping at the station
            data['travel_time'] = travel_time
        else:
            raise ValueError("Invalid travel time type. Must be 'uniform' or 'distance'.")
    return nx_graph


def add_transfer_edges(nx_graph, transfer_time=5):
    """
    Add edges representing transfers between different lines at the same station.
    """
    # Stations to ignore for transfers
    ignore_station = ['신촌', '양평']
    # Find stations with multiple lines (same station_name)
    station_lines = {}
    for node, data in nx_graph.nodes(data=True):
        station_name = data['station_name']
        line_no = data['line_no']
        if station_name not in station_lines or station_name in ignore_station:
            station_lines[station_name] = []
        station_lines[station_name].append((node, line_no))

    # Add transfer edges between nodes with the same station_name but different line_no
    for station, nodes in station_lines.items():
        if len(nodes) > 1:
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    u, line_u = nodes[i]
                    v, line_v = nodes[j]
                    if line_u != line_v:
                        # Add an edge with transfer time
                        nx_graph.add_edge(u, v, travel_time=transfer_time)
    return nx_graph


def convert_nx(nx_graph):
    station_dict_graph = {}
    for node in nx_graph.nodes(data=True):
        station = Station(node[0], node[1]['station_name'], node[1]['eng_station_name'],
                          node[1]['line_no'], node[1]['pos'])
        station_dict_graph[node[0]] = station

    for edge in nx_graph.edges(data=True):
        route = Route(station_dict_graph[edge[0]], station_dict_graph[edge[1]], edge[2]['travel_time'])
        station_dict_graph[edge[0]].add_route(route)
        route = Route(station_dict_graph[edge[1]], station_dict_graph[edge[0]], edge[2]['travel_time'])
        station_dict_graph[edge[1]].add_route(route)

    return station_dict_graph


def convert_simpledict(simple_dict):
    station_dict_graph = {}
    for node_id, data in simple_dict.items():
        station = Station(node_id, None, None, None, data['coord'])
        station_dict_graph[node_id] = station

    for node_id, data in simple_dict.items():
        for node_to, travel_time in data.items():
            if node_to == 'coord':
                continue
            route = Route(station_dict_graph[node_id], station_dict_graph[node_to], travel_time)
            station_dict_graph[node_id].add_route(route)

    return station_dict_graph


def find_station_by_name(dict_graph, station_name):
    station_id_list = []
    for station_id, station in dict_graph.items():
        if station.station_name == station_name or station.eng_station_name.lower() == station_name:
            station_id_list.append(station_id)
    return station_id_list
