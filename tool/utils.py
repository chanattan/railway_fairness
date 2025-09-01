from collections import deque
from city import *
from line import *
import heapq
from itertools import count


def __shortest_path(graph, start, goal):
    visited = set()
    queue = deque([(start, [])])  # (current_node, path_so_far)
    
    while queue:
        current, path = queue.popleft()
         
        if current == goal:
            return path
        
        visited.add(current)
        for (neighbor, line) in graph.get(current, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [line]))
                visited.add(neighbor)
    
    return None

def __shortest_time_path(graph, start, goal):
    counter = count() # tie breaker for heap
    queue = [(0, start, next(counter), [])]  # (total_time_so_far, current_city, path_so_far)
    visited = set()

    while queue:
        time, city, _, path = heapq.heappop(queue)

        if city in visited:
            continue
        visited.add(city)

        #path = path + [graph.get]
        if city == goal:
            return path, time

        for neighbor, line in graph.get(city, []):
            travel_time = line.time
            if neighbor not in visited:
                heapq.heappush(queue, (time + travel_time, neighbor, next(counter), path + [line]))

    return None, float('inf')

def get_shortest_path(city_a: City, city_b: City, transitions: dict[City, list[(City, Line)]], time = False):
    """
        Returns the shortest path (in terms of number of edges) between two cities (city_a and city_b).
        If the boolean time is set to true, then it returns the shortest path in terms of time,
        e.g., there may exist a path of three edges which has a cumulative time bigger than another
        path of four edges.
        NB: the considered graph is non-oriented and connected.
    """
    #print("city a", city_a)
    #print("city b", city_b)
    #print("transitions", list(map(lambda x: (x[0], list(map(lambda y: y[0], x[1]))), transitions.items())))
    if not time:
        path = __shortest_path(transitions, city_a.id, city_b.id)
    else:
        path, t = __shortest_time_path(transitions, city_a.id, city_b.id) # Path and time
    return (path, t) if time else path

def all_paths(graph, city_a, city_b):
    """
    DFS to retrieve all paths without loop
    """
    start = city_a.id
    goal = city_b.id
    paths = []
    def dfs(current, visited, path):
        if current == goal:
            paths.append(path)
            return
        visited.add(current)
        for neighbor, line in graph.get(current, []):
            if neighbor not in visited:
                dfs(neighbor, visited.copy(), path + [line])
    
    dfs(start, set(), [])
    return paths

def test():
    from city import City
    a,b,c=City(-10, -10), City(0, 0), City(10, 10)#0,1,2
    city_graph = {a: [c], b:[c], c:[a,b]}
    path = __shortest_path(city_graph, a, b)
    print(list(map(lambda x: x.id, path)))  # Output: ['A', 'B', 'D', 'E']