from math import isclose
from submission import *
from graph_utils import convert_simpledict
from station import set_access_count_zero, get_access_count
from test_cases import *

def print_error(message):
    print(f'\033[31m{message}\033[0m')


def print_total_success(message):
    print(f'\033[92m{message}\033[0m')


def print_total_error(message):
    print(f'\033[91m{message}\033[0m')


def evaluate(algorithm='DFS'):
    test_result = True

    test_cases = {'DFS': dfs_basic_test,
                  'Iterative Deepening': ids_basic_test,
                  'Uniform Cost': ucs_basic_test,
                  'Greedy': greedy_basic_test,
                  'A*': astar_basic_test}
    algorithms = {'DFS': find_route_dfs,
                  'Iterative Deepening': find_route_iterative_deepening,
                  'Uniform Cost': find_route_uniform_cost,
                  'Greedy': find_route_greedy,
                  'A*': find_route_a_star}

    for i, test_case in enumerate(test_cases[algorithm]):
        simple_dict_graph = test_case['dict_graph']
        start_node_id = test_case['start_node_id']
        end_node_id = test_case['end_node_id']
        dict_graph = convert_simpledict(simple_dict_graph)

        set_access_count_zero()
        route, total_time = algorithms[algorithm](dict_graph, start_node_id, end_node_id)
        node_access_count = get_access_count()

        if (route is None and test_case['route'] is None or route in test_case['route']
                and isclose(total_time, test_case['total_time'])):
            continue
        if route not in test_case['route'] or total_time != test_case['total_time']:
            test_result = False
            print_error(f'[{algorithm}] Test {i + 1} failed: '
                        f'expected ({test_case["route"]}, {test_case["total_time"]}), got ({route}, {total_time})')
        elif node_access_count > test_case['node_access_count_threshold']:
            test_result = False
            print_error(f'[{algorithm}] Test {i + 1} failed: '
                        f'node access count is too high: {node_access_count}')

    return test_result


if __name__ == "__main__":
    all_passed = True

    test_list = ['DFS', 'Iterative Deepening', 'Uniform Cost', 'Greedy', 'A*']
    for test in test_list:
        print(f"Running {test} tests...")
        results = evaluate(algorithm=test)
        if not results:
            all_passed = False

    if all_passed:
        print_total_success("[✓] All tests passed.")
    else:
        print_total_error("[✗] Some tests failed. Please check the error messages above.")
