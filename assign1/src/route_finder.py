from submission import *
from graph_utils import *
from plot_utils import visualize_route
from station import set_access_count_zero, get_access_count


def main(args):
    algorithm = args.algorithm
    heuristic_type = args.heuristic_type
    travel_time_type = args.travel_time_type
    travel_time = args.travel_time
    transfer_time = args.transfer_time
    visualize = args.visualize

    file_path = 'metro_graph.gml'
    nx_graph = load_graph(file_path)
    nx_graph = augment_graph_with_travel_times(nx_graph, type=travel_time_type, travel_time=travel_time)
    nx_graph = add_transfer_edges(nx_graph, transfer_time=transfer_time)
    dict_graph = convert_nx(nx_graph)

    start_station_name = input("Enter the start station name (such as 'seoul station'): ").lower()
    start_station_id_list = find_station_by_name(dict_graph, start_station_name)
    if not start_station_id_list:
        print("Start station '{}' not found in the graph.".format(start_station_name))
        return

    goal_station_name = input("Enter the goal station name (such as 'gangnam'): ").lower()
    goal_station_id_list = find_station_by_name(dict_graph, goal_station_name)
    if not goal_station_id_list:
        print("Goal station '{}' not found in the graph.".format(goal_station_name))
        return

    best_route_id_list = None
    best_total_time = None
    best_node_access_count = None
    for start_station_id in start_station_id_list:
        for goal_station_id in goal_station_id_list:
            set_access_count_zero()
            if algorithm == 'bfs':
                route_id_list, total_time = find_route_bfs(dict_graph, start_station_id, goal_station_id)
            elif algorithm == 'dfs':
                route_id_list, total_time = find_route_dfs(dict_graph, start_station_id, goal_station_id)
            elif algorithm == 'uniform_cost':
                route_id_list, total_time = find_route_uniform_cost(dict_graph, start_station_id, goal_station_id)
            elif algorithm == 'iterative_deepening':
                route_id_list, total_time = find_route_iterative_deepening(dict_graph, start_station_id, goal_station_id)
            elif algorithm == 'greedy_heuristic':
                route_id_list, total_time = find_route_greedy(dict_graph, start_station_id, goal_station_id, heuristic_type)
            elif algorithm == 'a_star':
                route_id_list, total_time = find_route_a_star(dict_graph, start_station_id, goal_station_id, heuristic_type)
            else:
                raise ValueError("Invalid algorithm type. Must be 'bfs', 'dfs', 'uniform_cost', 'iterative_deepening', "
                                 "'greedy_heuristic', or 'a_star'.")
            if total_time is not None and (best_total_time is None or total_time < best_total_time):
                best_route_id_list = route_id_list
                best_total_time = total_time
                best_node_access_count = get_access_count()


    if best_route_id_list and best_total_time is not None:
        print("Fastest route from '{}' to '{}':".format(start_station_name, goal_station_name))
        print(" -> ".join(str(dict_graph[route_id]) for route_id in best_route_id_list))
        print("Total travel time: {:.2f} minutes".format(best_total_time))
        print("Number of node visits: {}".format(best_node_access_count))

        if visualize:
            visualize_route(nx_graph, best_route_id_list)

        return best_route_id_list
    else:
        print("Unable to find a route between the specified stations.")
        return None


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--algorithm', type=str, default='bfs')
    args.add_argument('--heuristic_type', type=str, default='euclidean')
    args.add_argument('--travel_time_type', type=str, default='uniform')
    args.add_argument('--travel_time', type=int, default=5)
    args.add_argument('--transfer_time', type=int, default=5)
    args.add_argument('--visualize', type=bool, default=True)
    args = args.parse_args()
    main(args)
