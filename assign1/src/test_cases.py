dfs_basic_test = [
    {
        # Test 1
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 1, 'coord': (0, 0)},
            1: {0: 1, 3: 1, 'coord': (0, 0)},
            2: {0: 1, 'coord': (0, 0)},
            3: {1: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 1, 3]],
        'total_time': 2,
        'node_access_count_threshold': 5
    },
    {
        # Test 2
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 1, 'coord': (0, 0)},
            1: {0: 1, 2: 1, 'coord': (0, 0)},
            2: {0: 1, 1: 1, 3: 1, 'coord': (0, 0)},
            3: {2: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 2, 3], [0, 1, 2, 3]],
        'total_time': 2,
        'node_access_count_threshold': 7
    },
    {
        # Test 3
        # Input:
        'dict_graph': {
            0: {1: 1, 3: 1, 'coord': (0, 0)},
            1: {0: 1, 2: 1, 'coord': (0, 0)},
            2: {1: 1, 3: 1, 'coord': (0, 0)},
            3: {0: 1, 2: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 2,
        # Expected output:
        'route': [[0, 1, 2], [0, 3, 2]],
        'total_time': 2,
        'node_access_count_threshold': 6
    },
]

dfs_hidden_test = [
    {
        # Hidden Test 1
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 1, 'coord': (0, 0)},
            1: {0: 1, 3: 1, 'coord': (0, 0)},
            2: {0: 1, 4: 1, 'coord': (0, 0)},
            3: {1: 1, 5: 1, 'coord': (0, 0)},
            4: {2: 1, 5: 1, 'coord': (0, 0)},
            5: {3: 1, 4: 1, 6: 1, 'coord': (0, 0)},
            6: {5: 1, 7: 1, 8: 1, 'coord': (0, 0)},
            7: {6: 1, 9: 1, 'coord': (0, 0)},
            8: {6: 1, 9: 1, 'coord': (0, 0)},
            9: {7: 1, 8: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 9,
    },
    {
        # Hidden Test 2
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 1, 'coord': (0, 0)},
            1: {0: 1, 3: 1, 'coord': (0, 0)},
            2: {0: 1, 10: 1, 'coord': (0, 0)},
            3: {1: 1, 4: 1, 'coord': (0, 0)},
            4: {3: 1, 5: 1, 'coord': (0, 0)},
            5: {4: 1, 6: 1, 'coord': (0, 0)},
            6: {5: 1, 7: 1, 'coord': (0, 0)},
            7: {6: 1, 8: 1, 'coord': (0, 0)},
            8: {7: 1, 9: 1, 'coord': (0, 0)},
            9: {8: 1, 10: 1, 'coord': (0, 0)},
            10: {9: 1, 2: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 10,
    },
    {
        # Hidden Test 3
        # Input:
        'dict_graph': {
            0: {1: 1, 'coord': (0, 0)},
            1: {0: 1, 2: 1, 5: 1, 'coord': (0, 0)},
            2: {1: 1, 3: 1, 'coord': (0, 0)},
            3: {2: 1, 4: 1, 'coord': (0, 0)},
            4: {3: 1, 5: 1, 'coord': (0, 0)},
            5: {4: 1, 1: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 5,
    },
    {
        # Hidden Test 4
        # Input:
        'dict_graph': {i: {2*i+1: 1, 2*i+2: 1, 'coord': (0, 0)} for i in range(15) if 2*i+2 < 31}
        | {i: {'coord': (0, 0)} for i in range(15, 31)},
        'start_node_id': 0,
        'end_node_id': 30,
    },
    {
        # Hidden Test 5
        # Input:
        'dict_graph': {
            0: {1: 1, 'coord': (0, 0)},
            1: {0: 1, 2: 1, 'coord': (0, 0)},
            2: {1: 1, 'coord': (0, 0)},
            3: {4: 1, 'coord': (0, 0)},
            4: {3: 1, 5: 1, 'coord': (0, 0)},
            5: {4: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 5,
    },
]


ids_basic_test = [
    {
        # Test 1
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 1, 'coord': (0, 0)},
            1: {0: 1, 3: 1, 'coord': (0, 0)},
            2: {0: 1, 3: 1, 'coord': (0, 0)},
            3: {1: 1, 2: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 1, 3], [0, 2, 3]],
        'total_time': 2,
        'node_access_count_threshold': 10
    },
    {
        # Test 2
        # Input:
        'dict_graph': {
            0: {1: 2, 'coord': (0, 0)},
            1: {0: 2, 2: 2, 3: 2, 'coord': (0, 0)},
            2: {1: 2, 4: 2, 'coord': (0, 0)},
            3: {1: 2, 4: 2, 'coord': (0, 0)},
            4: {2: 2, 3: 2, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 4,
        # Expected output:
        'route': [[0, 1, 2, 4], [0, 1, 3, 4]],
        'total_time': 6,
        'node_access_count_threshold': 15
    },
    {
        # Test 3
        # Input:
        'dict_graph': {
            0: {1: 3, 'coord': (0, 0)},
            1: {0: 3, 2: 3, 'coord': (0, 0)},
            2: {1: 3, 3: 3, 'coord': (0, 0)},
            3: {2: 3, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 1, 2, 3]],
        'total_time': 9,
        'node_access_count_threshold': 10
    },
]

ids_hidden_test = [
    {
        # Hidden Test 1
        # Input:
        'dict_graph': {
            0: {1: 1, 'coord': (0, 0)},
            1: {0: 1, 2: 1, 5: 1, 'coord': (0, 0)},
            2: {1: 1, 3: 1, 'coord': (0, 0)},
            3: {2: 1, 4: 1, 'coord': (0, 0)},
            4: {3: 1, 'coord': (0, 0)},
            5: {1: 1, 6: 1, 'coord': (0, 0)},
            6: {5: 1, 7: 1, 'coord': (0, 0)},
            7: {6: 1, 4: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 4,
    },
    {
        # Hidden Test 2
        # Input:
        'dict_graph': {i: {i + 1: 1, 'coord': (0, 0)} for i in range(0, 10)} | {10: {'coord': (0, 0)}},
        'start_node_id': 0,
        'end_node_id': 10,
    },
    {
        # Hidden Test 3
        # Input:
        'dict_graph': {
            0: {1: 2, 'coord': (0, 0)},
            1: {0: 2, 2: 2, 'coord': (0, 0)},
            2: {1: 2, 3: 2, 'coord': (0, 0)},
            3: {2: 2, 4: 2, 'coord': (0, 0)},
            4: {3: 2, 5: 2, 'coord': (0, 0)},
            5: {4: 2, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 5,
    },
    {
        # Hidden Test 4
        # Input:
        'dict_graph': (
                {i: {i + 1: 1, i + 2: 1, 'coord': (0, 0)} for i in range(0, 8)} |
                {8: {9: 1, 'coord': (0, 0)}, 9: {'coord': (0, 0)}}
        ),
        'start_node_id': 0,
        'end_node_id': 9,
    },
    {
        # Hidden Test 5
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 1, 3: 1, 'coord': (0, 0)},
            1: {0: 1, 'coord': (0, 0)},
            2: {0: 1, 'coord': (0, 0)},
            3: {0: 1, 4: 1, 'coord': (0, 0)},
            4: {3: 1, 5: 1, 'coord': (0, 0)},
            5: {4: 1, 'coord': (0, 0)},
        },
        'start_node_id': 1,
        'end_node_id': 5,
    },
]


ucs_basic_test = [
    {
        # Test 1
        # Input:
        'dict_graph': {
            0: {1: 2, 2: 5, 'coord': (0, 0)},
            1: {0: 2, 3: 3, 'coord': (0, 0)},
            2: {0: 5, 3: 2, 'coord': (0, 0)},
            3: {1: 3, 2: 2, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 1, 3], [0, 2, 3]],
        'total_time': 5,
        'node_access_count_threshold': 10
    },
    {
        # Test 2
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 2, 'coord': (0, 0)},
            1: {0: 1, 3: 2, 'coord': (0, 0)},
            2: {0: 2, 3: 1, 'coord': (0, 0)},
            3: {1: 2, 2: 1, 4: 1, 'coord': (0, 0)},
            4: {3: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 4,
        # Expected output:
        'route': [[0, 1, 3, 4], [0, 2, 3, 4]],
        'total_time': 4,
        'node_access_count_threshold': 12
    },
    {
        # Test 3
        # Input:
        'dict_graph': {
            0: {1: 3, 2: 2, 'coord': (0, 0)},
            1: {0: 3, 3: 4, 'coord': (0, 0)},
            2: {0: 2, 3: 1, 'coord': (0, 0)},
            3: {1: 4, 2: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 2, 3]],
        'total_time': 3,
        'node_access_count_threshold': 10
    },
]

ucs_hidden_test = [
    {
        # Hidden Test 1
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 3, 'coord': (0, 0)},
            1: {0: 1, 3: 4, 'coord': (0, 0)},
            2: {0: 3, 4: 2, 'coord': (0, 0)},
            3: {1: 4, 5: 1, 'coord': (0, 0)},
            4: {2: 2, 5: 5, 'coord': (0, 0)},
            5: {3: 1, 4: 5, 6: 1, 'coord': (0, 0)},
            6: {5: 1, 7: 2, 'coord': (0, 0)},
            7: {6: 2, 8: 1, 'coord': (0, 0)},
            8: {7: 1, 9: 1, 'coord': (0, 0)},
            9: {8: 1, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 9,
    },
    {
        # Hidden Test 2
        # Input:
        'dict_graph': (
            {i: {i+1: 1, 'coord': (0, 0)} for i in range(0, 10)} |
            {10: {'coord': (0, 0)}}
        ),
        'start_node_id': 0,
        'end_node_id': 10,
    },
    {
        # Hidden Test 3
        # Input:
        'dict_graph': {
            0: {1: 5, 2: 2, 'coord': (0, 0)},
            1: {0: 5, 3: 2, 'coord': (0, 0)},
            2: {0: 2, 3: 4, 'coord': (0, 0)},
            3: {1: 2, 2: 4, 4: 1, 'coord': (0, 0)},
            4: {3: 1, 5: 3, 'coord': (0, 0)},
            5: {4: 3, 'coord': (0, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 5,
    },
    {
        # Hidden Test 4
        # Input:
        'dict_graph': (
            {i: {i+1: i+1, 'coord': (0, 0)} for i in range(0, 10)} |
            {10: {'coord': (0, 0)}}
        ),
        'start_node_id': 0,
        'end_node_id': 10,
    },
    {
        # Hidden Test 5
        # Input:
        'dict_graph': {
            0: {1: 3, 2: 3, 3: 3, 'coord': (0, 0)},
            1: {0: 3, 'coord': (0, 0)},
            2: {0: 3, 'coord': (0, 0)},
            3: {0: 3, 4: 3, 'coord': (0, 0)},
            4: {3: 3, 5: 3, 'coord': (0, 0)},
            5: {4: 3, 'coord': (0, 0)},
        },
        'start_node_id': 1,
        'end_node_id': 5,
    },
]


greedy_basic_test = [
    {
        # Test 1
        # Input:
        'dict_graph': {
            0: {1: 1, 2: 1, 'coord': (0, 0)},
            1: {0: 1, 3: 1, 'coord': (1, 0)},
            2: {0: 1, 3: 1, 'coord': (0, 1)},
            3: {1: 1, 2: 1, 'coord': (1, 1)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 1, 3], [0, 2, 3]],
        'total_time': 2,
        'node_access_count_threshold': 6
    },
    {
        # Test 2
        # Input:
        'dict_graph': {
            0: {1: 2, 2: 2, 'coord': (0, 0)},
            1: {0: 2, 3: 2, 'coord': (2, 0)},
            2: {0: 2, 3: 2, 'coord': (0, 2)},
            3: {1: 2, 2: 2, 4: 2, 'coord': (2, 2)},
            4: {3: 2, 'coord': (1, 3)},
        },
        'start_node_id': 0,
        'end_node_id': 4,
        # Expected output:
        'route': [[0, 2, 3, 4]],
        'total_time': 6,
        'node_access_count_threshold': 10
    },
    {
        # Test 3
        # Input:
        'dict_graph': {
            0: {1: 2, 2: 1.414, 'coord': (0, 0)},
            1: {0: 2, 3: 2.828, 'coord': (2, 0)},
            2: {0: 1.414, 3: 1.414, 'coord': (1, 1)},
            3: {1: 2.828, 2: 1.414, 'coord': (2, 2)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 2, 3]],
        'total_time': 2.828,
        'node_access_count_threshold': 8
    },
]

greedy_hidden_test = [
    {
        # Hidden Test 1
        # Input:
        'dict_graph': {
            0: {1: 1.0, 2: 3.162, 'coord': (0, 0)},
            1: {0: 1.0, 3: 2.236, 'coord': (1, 0)},
            2: {0: 3.162, 4: 2.236, 'coord': (3, 0)},
            3: {1: 2.236, 5: 1.414, 'coord': (1, 2)},
            4: {2: 2.236, 5: 2.828, 'coord': (3, 2)},
            5: {3: 1.414, 4: 2.828, 6: 1.0, 'coord': (2, 3)},
            6: {5: 1.0, 7: 1.0, 'coord': (2, 4)},
            7: {6: 1.0, 8: 1.0, 'coord': (2, 5)},
            8: {7: 1.0, 9: 1.0, 'coord': (2, 6)},
            9: {8: 1.0, 'coord': (2, 7)},
        },
        'start_node_id': 0,
        'end_node_id': 9,
    },
    {
        # Hidden Test 2
        # Input:
        'dict_graph': (
            {i: {i+1: 1.0, 'coord': (i, 0)} for i in range(0, 10)} |
            {10: {'coord': (10, 0)}}
        ),
        'start_node_id': 0,
        'end_node_id': 10,
    },
    {
        # Hidden Test 3
        # Input:
        'dict_graph': {
            0: {1: 2.236, 4: 3.605, 'coord': (0, 0)},
            1: {0: 2.236, 2: 2.236, 'coord': (1, 2)},
            2: {1: 2.236, 3: 2.0, 'coord': (2, 4)},
            3: {2: 2.0, 5: 4.472, 'coord': (4, 4)},
            4: {0: 3.605, 5: 10.0, 'coord': (3, 2)},
            5: {3: 4.472, 4: 10.0, 'coord': (4, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 5,
    },
    {
        # Hidden Test 4
        # Input:
        'dict_graph': {
            0: {1: 1.0, 2: 1.0, 'coord': (0, 0)},
            1: {0: 1.0, 3: 1.414, 'coord': (0, 1)},
            2: {0: 1.0, 3: 1.0, 'coord': (1, 0)},
            3: {1: 1.0, 2: 1.414, 4: 1.0, 'coord': (1, 1)},
            4: {3: 1.0, 5: 1.0, 'coord': (1, 2)},
            5: {4: 1.0, 6: 1.0, 'coord': (1, 3)},
            6: {5: 1.0, 7: 1.0, 'coord': (1, 4)},
            7: {6: 1.0, 'coord': (1, 5)},
        },
        'start_node_id': 0,
        'end_node_id': 7,
    },
    {
        # Hidden Test 5
        # Input:
        'dict_graph': {
            0: {1: 1.0, 2: 1.0, 'coord': (0, 0)},
            1: {0: 1.0, 'coord': (1, 0)},
            2: {0: 1.0, 'coord': (0, 1)},
            3: {4: 1.0, 'coord': (2, 2)},
            4: {3: 1.0, 5: 1.0, 'coord': (2, 3)},
            5: {4: 1.0, 'coord': (2, 4)},
        },
        'start_node_id': 1,
        'end_node_id': 5,
    },
]


astar_basic_test = [
    {
        # Test 1
        # Input:
        'dict_graph': {
            0: {1: 1.0, 2: 1.0, 'coord': (0, 0)},
            1: {0: 1.0, 3: 1.414, 4: 1.0, 'coord': (1, 0)},
            2: {0: 1.0, 3: 1.0, 'coord': (0, 1)},
            3: {1: 1.414, 2: 1.0, 'coord': (1, 1)},
            4: {1: 1.0, 'coord': (2, 0)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 2, 3]],
        'total_time': 2.0,
        'node_access_count_threshold': 6
    },
    {
        # Test 2
        # Input:
        'dict_graph': {
            0: {1: 2.0, 2: 2.0, 'coord': (0, 0)},
            1: {0: 2.0, 3: 2.828, 'coord': (2, 0)},
            2: {0: 2.0, 3: 2.0, 'coord': (0, 2)},
            3: {1: 2.828, 2: 2.0, 4: 1.414, 'coord': (2, 2)},
            4: {3: 1.414, 'coord': (3, 3)},
        },
        'start_node_id': 0,
        'end_node_id': 4,
        # Expected output:
        'route': [[0, 2, 3, 4]],
        'total_time': 5.414,
        'node_access_count_threshold': 10
    },
    {
        # Test 3
        # Input:
        'dict_graph': {
            0: {1: 3.0, 2: 2.236, 'coord': (0, 0)},
            1: {0: 3.0, 3: 2.236, 'coord': (3, 0)},
            2: {0: 2.236, 3: 1.0, 'coord': (1, 2)},
            3: {1: 2.236, 2: 1.0, 'coord': (2, 2)},
        },
        'start_node_id': 0,
        'end_node_id': 3,
        # Expected output:
        'route': [[0, 2, 3]],
        'total_time': 3.236,
        'node_access_count_threshold': 8
    },
]

astar_hidden_test = [
    {
        # Hidden Test 1
        # Input:
        'dict_graph': {
            0: {1: 1.0, 2: 2.0, 'coord': (0, 0)},
            1: {0: 1.0, 3: 2.0, 'coord': (1, 0)},
            2: {0: 2.0, 3: 1.0, 'coord': (0, 2)},
            3: {1: 2.0, 2: 1.0, 4: 1.0, 'coord': (1, 2)},
            4: {3: 1.0, 5: 1.0, 'coord': (2, 2)},
            5: {4: 1.0, 6: 1.0, 'coord': (3, 2)},
            6: {5: 1.0, 7: 1.0, 'coord': (4, 2)},
            7: {6: 1.0, 8: 1.0, 'coord': (5, 2)},
            8: {7: 1.0, 9: 1.0, 'coord': (6, 2)},
            9: {8: 1.0, 'coord': (7, 2)},
        },
        'start_node_id': 0,
        'end_node_id': 9,
    },
    {
        # Hidden Test 2
        # Input:
        'dict_graph': (
            {i: {i+1: 1.0, 'coord': (i, 0)} for i in range(0, 15)} |
            {15: {'coord': (15, 0)}}
        ),
        'start_node_id': 0,
        'end_node_id': 15,
    },
    {
        # Hidden Test 3
        # Input:
        'dict_graph': {
            0: {1: 1.0, 2: 2.236, 'coord': (0, 0)},
            1: {0: 1.0, 3: 2.0, 'coord': (1, 0)},
            2: {0: 2.236, 3: 1.0, 'coord': (0, 2)},
            3: {1: 2.0, 2: 1.0, 4: 1.414, 'coord': (1, 2)},
            4: {3: 1.414, 5: 2.828, 'coord': (2, 3)},
            5: {4: 2.828, 6: 1.0, 'coord': (4, 3)},
            6: {5: 1.0, 'coord': (5, 3)},
        },
        'start_node_id': 0,
        'end_node_id': 6,
    },
    {
        # Hidden Test 4
        # Input:
        'dict_graph': (
            {i: {i+1: 1.0, i+2: 1.414, 'coord': (i, 0)} for i in range(0, 8)} |
            {8: {9: 1.0, 'coord': (8, 0)}, 9: {'coord': (9, 0)}}
        ),
        'start_node_id': 0,
        'end_node_id': 9,
    },
    {
        # Hidden Test 5
        # Input:
        'dict_graph': {
            0: {1: 2.0, 2: 2.0, 3: 3.0, 'coord': (0, 0)},
            1: {0: 2.0, 'coord': (1, 0)},
            2: {0: 2.0, 'coord': (0, 1)},
            3: {0: 3.0, 4: 1.0, 'coord': (2, 2)},
            4: {3: 1.0, 5: 1.0, 'coord': (3, 2)},
            5: {4: 1.0, 'coord': (4, 2)},
        },
        'start_node_id': 1,
        'end_node_id': 5,
    },
]
