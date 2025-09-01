import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
import pandas as pd
import json

def load_wikidata_regions(path="query.json", pop_threshold=100000):
    """
    Load cities with population and coordinates from a Wikidata query file,
    and return them as region strings usable by OSM (e.g., "Lyon, France").
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    regions = []
    for entry in data:
        name = entry.get("cityLabel")
        pop_str = entry.get("population", "0")
        try:
            pop = int(pop_str)
        except ValueError:
            pop = 0

        if name and pop >= pop_threshold:
            region = f"{name}, France"
            regions.append(region)

    return list(set(regions))

# From OSM extraction
def get_cities(region="France", pop_threshold=150000):
    print("Extraction des villes...")
    tags = {"place": ["city", "town"]}
    cities = ox.features_from_place(region, tags)
    cities = cities[["geometry", "name", "population"]].dropna(subset=["geometry", "name"])

    def parse_pop(pop):
        try:
            return int(str(pop).replace(",", ""))
        except:
            return 0

    cities["pop_int"] = cities["population"].apply(parse_pop)
    cities = cities[cities["pop_int"] >= pop_threshold]
    cities = cities.to_crs(epsg=2154)
    cities["centroid"] = cities.geometry.centroid
    cities = cities.reset_index(drop=True)
    print(f"{len(cities)} villes extraites avec population >= {pop_threshold}.")
    return cities

def get_rail_segments(region="Île-de-France, France"):
    print("Extraction des segments ferroviaires")
    tags = {"railway": "rail"}
    gdf = ox.features_from_place(region, tags)
    gdf = gdf.to_crs(epsg=2154)
    print(f"{len(gdf)} segments récupérés.")
    return gdf

def get_rail_segments_geojson(path="railways_france.geojson"):
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=2154)
    print(f"{len(gdf)} segments ferroviaires chargés.")
    return gdf

# Aggregate lines to city
def nearest_city(point, city_points, city_names):
    dists = [point.distance(city_point) for city_point in city_points]
    idx = np.argmin(dists)
    return city_names[idx]

def build_city_graph(rail_gdf, cities, buffer_km=10, line_buffer_m=200):
    print("Construction du graphe ferroviaire")
    # Separate rail lines and stations
    rail_lines = rail_gdf[rail_gdf["railway"] == "rail"]
    stations = rail_gdf[rail_gdf["railway"] == "station"]

    if rail_lines.empty or stations.empty:
        print("Erreur : aucune ligne ou station trouvée.")
        return nx.Graph()

    # Ensure cities are unique by name and have centroids
    cities = cities.drop_duplicates(subset="name").copy()
    if "centroid" not in cities.columns:
        cities["centroid"] = cities.geometry.centroid

    # Associate stations to closest cities within buffer_km
    station_to_city = {}
    for idx, station in stations.iterrows():
        pt = station.geometry
        dists = cities["centroid"].distance(pt)
        min_dist = dists.min()
        if min_dist <= buffer_km * 1000:
            city_name = cities.loc[dists.idxmin()]["name"]
            station_to_city[idx] = city_name

    print(f"{len(station_to_city)} gares associées à une ville (≤{buffer_km} km).")

    # Filter only mapped stations and assign city name
    stations_with_city = stations.loc[station_to_city.keys()].copy()
    stations_with_city["city"] = stations_with_city.index.map(station_to_city)

    # Build edges: for each rail line, find all cities whose stations intersect with it
    # Pour chaque ligne, on trouve les 2 villes les plus proches aux extrémités de la ligne
    edges = set()

    # Préparation : centroids + noms en cache
    city_points = list(cities["centroid"])
    city_names = list(cities["name"])

    for _, line_row in rail_lines.iterrows():
        geom = line_row.geometry
        if not isinstance(geom, LineString):
            continue

        # Points de départ et fin
        start_pt = Point(geom.coords[0])
        end_pt = Point(geom.coords[-1])

        # Ville la plus proche (sans limite de distance)
        city_start = nearest_city(start_pt, city_points, city_names)
        city_end = nearest_city(end_pt, city_points, city_names)

        # Évite les boucles
        if city_start != city_end:
            edges.add(tuple(sorted((city_start, city_end))))

    # Build graph
    G = nx.Graph()
    for _, row in cities.iterrows():
        G.add_node(row["name"], pos=(row["centroid"].x, row["centroid"].y))

    for u, v in edges:
        if u in G.nodes and v in G.nodes:
            G.add_edge(u, v)

    print(f"Graphe final : {G.number_of_nodes()} villes, {G.number_of_edges()} connexions.")
    return G

def plot_city_graph(G):
    print("Visualisation du graphe simplifié")
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_size=50, node_color='red', edge_color='gray', with_labels=True, font_size=8)
    plt.title("Graphe ferroviaire simplifié entre villes")
    plt.savefig("simplified_graph.png", dpi=300, bbox_inches='tight')
    plt.show()

def compute_demand_matrix(cities, alpha=1.5):
    coords = np.array([(pt.y, pt.x) for pt in cities["centroid"]])
    populations = cities["pop_int"].values.reshape(-1, 1)

    n = len(cities)
    demand_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                demand_matrix[i, j] = 0
            else:
                dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j])) # in m
                #print("dist", dist)
                demand_matrix[i, j] = (populations[i, 0] * populations[j, 0]) / ((dist ** alpha) if (dist ** alpha) > 0 else 1)

    return demand_matrix

def add_edge_attributes(G, cities, average_speed_kmh=20):
    city_pos = {row["name"]: (row["centroid"].y, row["centroid"].x) for _, row in cities.iterrows()}

    for u, v in G.edges():
        if u not in city_pos or v not in city_pos:
            print(f"Node {u} or {v} missing position.")
            continue
        coord_u = city_pos[u]
        coord_v = city_pos[v]
        distance_factor = 1.5
        dist_km = (np.linalg.norm(np.array(coord_u) - np.array(coord_v)) * distance_factor) / 1000 # correction pour distance réelle plus longue avec distance_factor
        travel_time_h = dist_km / average_speed_kmh
        G.edges[u, v]["distance_km"] = dist_km
        G.edges[u, v]["travel_time_h"] = travel_time_h

def assign_demand_to_nodes(G, cities, demand_matrix):
    node_index = {name: i for i, name in enumerate(cities["name"])}
    for node in G.nodes():
        i = node_index.get(node, None)
        if i is None:
            G.nodes[node]["demand"] = {}
        else:
            demand_row = demand_matrix[i]
            G.nodes[node]["demand"] = {
                cities["name"].iloc[j]: demand_row[j]
                for j in range(len(cities))
                if j != i
            }

import json

def export_graph_json(G, cities, demand_matrix, filename="graph.json", width=800, height=500):
    print("Export du graphe en JSON")

    pos = nx.get_node_attributes(G, 'pos')
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    def normalize(values, new_min, new_max):
        old_min = min(values)
        old_max = max(values)
        scale = (new_max - new_min) / (old_max - old_min) if old_max > old_min else 1
        return [(v - old_min) * scale + new_min for v in values]

    xs_norm = normalize(xs, 0, width)
    ys_norm = normalize(ys, 0, height)
    node_to_id = {node: i for i, node in enumerate(G.nodes())}
    city_pop_map = dict(zip(cities["name"], cities["pop_int"]))

    cities_json = []
    for node, x_norm, y_norm in zip(G.nodes(), xs_norm, ys_norm):
        cities_json.append({
            "id": node_to_id[node],
            "x": int(x_norm),
            "y": int(y_norm),
            "name": node,
            "pop": city_pop_map.get(node, 0)
        })

    # Railways list
    lines_json = []
    for u, v, attr in G.edges(data=True):
        city_a = node_to_id[u]
        city_b = node_to_id[v]

        # Travel time to minutes
        time = int(round(attr.get("travel_time_h", 0) * 60))

        dist = int(round(attr.get("distance_km", 0)))
        # For now we don't have input on waitings
        wait = 0

        # Demand between city_a and city_b (symetric)
        demand_a = G.nodes[u].get("demand", {}).get(v, 0)
        demand = int(round(demand_a))

        lines_json.append({
            "city_a": city_a,
            "city_b": city_b,
            "time": time,
            "dist": dist,
            "wait": wait,
            "demand": demand
        })

    export_data = {
        "cities": cities_json,
        "lines": lines_json,
        "demand_matrix": demand_matrix.tolist()
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Export terminé : {filename}")

def main():
    pop_threshold = 250000 # 25000 is good

    print(f"Chargement des villes avec population >={pop_threshold} en France")
    regions = load_wikidata_regions(pop_threshold=pop_threshold)
    cities_all = []
    for region in regions:
        print(f"Chargement pour {region} ...")
        c = get_cities(region, pop_threshold=pop_threshold)
        cities_all.append(c)
    cities = pd.concat(cities_all).reset_index(drop=True)
    valid_city_names = [entry['cityLabel'] for entry in json.load(open("query.json"))]
    cities = cities[cities["name"].isin(valid_city_names)]

    rail_gdf = get_rail_segments_geojson()

    G = build_city_graph(rail_gdf, cities)

    demand_matrix = compute_demand_matrix(cities)
    add_edge_attributes(G, cities, average_speed_kmh=80)
    assign_demand_to_nodes(G, cities, demand_matrix)

    plot_city_graph(G)

    export_graph_json(G, cities, demand_matrix, filename="/Users/csok/Downloads/graph.json")

    print("Graphe exporté")

if __name__ == "__main__":
    main()

