class Station:
    node_access_count = 0

    def __init__(self, id, station_name, eng_station_name, line_no, coordinate):
        self.id = id
        self.station_name = station_name
        self.eng_station_name = eng_station_name
        self.line_no = line_no
        self.coordinate = coordinate
        self.routes = []

    def add_route(self, route):
        self.routes.append(route)

    def __str__(self):
        return f"{self.station_name}, {self.eng_station_name} (Line {self.line_no})"

    def __getattribute__(self, item):
        if item == 'routes':
            Station.node_access_count += 1
        return super().__getattribute__(item)


def set_access_count_zero():
    Station.node_access_count = 0


def get_access_count():
    return Station.node_access_count


class Route:
    def __init__(self, station_from, station_to, travel_time):
        self.station_from = station_from
        self.station_to = station_to
        self.travel_time = travel_time
