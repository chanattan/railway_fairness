from pyrosm import OSM
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import numpy as np
import shapely
from shapely.strtree import STRtree
import matplotlib.pyplot as plt

# === 1. Charger les lignes ferroviaires ===

def load_railways(pbf_path, bbox=None):
    osm = OSM(pbf_path, bounding_box=bbox)
    all_rail = osm.get_data_by_custom_criteria(
        custom_filter={"railway": ["rail"]},
        osm_keys_to_keep=["railway"],#, "service", "usage"],
        filter_type="keep",
        keep_nodes=False
    )

    # Convertir CRS en Lambert 93
    all_rail = all_rail.to_crs(epsg=2154)

    # Filtrer les vraies lignes interurbaines (pas tram, métro, etc.)
    clean_rail = all_rail[
        (~all_rail["railway"].isin(["subway", "tram", "light_rail"]))# &
        #(~all_rail["service"].isin(["yard", "siding", "spur", "crossover", "platform"])) &
        #(~all_rail["usage"].isin(["military", "tourism"]))
    ]

    print(f"✅ {len(clean_rail)} segments ferroviaires conservés.")
    #print("clean_rail", clean_rail)
    return clean_rail

# === 2. Construire le graphe ferroviaire complet ===

def build_rail_graph(rail_gdf):
    G = nx.Graph()
    for line in rail_gdf.geometry:
        if isinstance(line, LineString):
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = tuple(coords[i])
                p2 = tuple(coords[i + 1])
                dist = Point(p1).distance(Point(p2))
                G.add_edge(p1, p2, weight=dist)
    return G

def plot_full_graph(G_full):
    plt.figure(figsize=(12, 12))
    pos = {n: (n[0], n[1]) for n in G_full.nodes}
    nx.draw(
        G_full, pos,
        node_size=1, node_color="blue",
        edge_color="lightgray", width=0.3
    )
    plt.title("Graphe ferroviaire complet (G_full)")
    plt.axis("equal")
    plt.show()

def plot_rail_lines(rail_gdf, cities=None):
    fig, ax = plt.subplots(figsize=(12, 12))

    # tracer les lignes ferroviaires
    rail_gdf.plot(ax=ax, color="gray", linewidth=0.5)

    # tracer les villes sur le même axe
    if cities is not None:
        cities.plot(ax=ax, color="red", markersize=20, zorder=5)

        # optionnel : labels des villes
        for _, row in cities.iterrows():
            ax.text(row.geometry.x, row.geometry.y, row["name"],
                    fontsize=6, ha="center", va="bottom")

    ax.set_title("Réseau ferroviaire et villes")
    ax.set_axis_off()
    plt.show()

# === 3. Charger les villes et les associer au réseau ===

def load_cities(cities_fp):
    cities = gpd.read_file(cities_fp)
    cities = cities.rename(columns={"cityLabel": "name"})
    if cities.crs is None:
        cities.set_crs(epsg=4326, inplace=True)
    cities = cities.to_crs(epsg=2154)
    return cities

def associate_cities_to_graph(cities_gdf, G_full, buffer_m=5000):
    if len(G_full.nodes) > 0:
        print("Type of G_full node", type(list(G_full.nodes)[0]))
    else:
        print("G_full.nodes is empty!")
    nodes = [Point(n) for n in G_full.nodes]
    tree = STRtree(nodes)
    node_lookup = {id(pt): pt.coords[0] for pt in nodes}

    city_to_node = {}
    for _, row in cities_gdf.iterrows():
        city_name = row["name"]
        city_geom = row.geometry

        # Assurer que city_geom est un Point
        if city_geom.geom_type != 'Point':
            city_geom = city_geom.centroid

        try:
            # nearest retourne un geometry, ici on donne bien un Point
            if not isinstance(city_geom, Point):
                city_geom = city_geom.centroid
            print("Nearest_pt check", type(city_geom), city_geom)
            idx = tree.nearest(city_geom)   
            nearest_pt = nodes[idx]
            print("Type of nearest_pt:", type(nearest_pt), nearest_pt)

            if nearest_pt is None:
                print(f"❌ Aucun point proche pour {city_name}")
                continue

            # distance en mètres (puisque CRS est Lambert 93)
            print("Distance test:", city_geom.distance(nearest_pt))
            dist = city_geom.distance(nearest_pt)
            if dist <= buffer_m:
                node_coord = node_lookup[id(nearest_pt)]
                city_to_node[city_name] = node_coord
            else:
                print(f"⚠️ {city_name} ignorée (hors portée réseau)")
        except Exception as e:
            print(f"❌ Erreur STRtree pour {city_name} : {e}")
            continue

    print(f"✅ {len(city_to_node)} villes associées à un nœud du réseau ferré.")
    return city_to_node

# === 4. Graphe entre villes (ville-à-ville) ===

def build_city_graph(G_full, city_node_map):
    G_city = nx.Graph()
    # Ajouter toutes les villes comme noeuds
    for city in city_node_map:
        G_city.add_node(city)

    # Pour chaque arête dans G_full, vérifier si elle relie des villes
    edges_added = set()
    for u, v, data in G_full.edges(data=True):
        # Regarder si u ou v correspondent à un noeud proche d'une ville
        cities_u = [city for city, node in city_node_map.items() if node == u]
        cities_v = [city for city, node in city_node_map.items() if node == v]

        # Si u et v sont chacun associés à une ville différente, créer l'arête ville-ville
        for city_u in cities_u:
            for city_v in cities_v:
                if city_u != city_v:
                    # Pour éviter doublons
                    if (city_u, city_v) not in edges_added and (city_v, city_u) not in edges_added:
                        G_city.add_edge(city_u, city_v, weight=data['weight'])
                        edges_added.add((city_u, city_v))

    return G_city

# === 5. Affichage rapide ===

def plot_city_graph(G_city, cities_gdf):
    pos = {row["name"]: (row.geometry.x, row.geometry.y)
           for _, row in cities_gdf.iterrows()
           if row["name"] in G_city.nodes}
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(G_city, pos, alpha=0.4)
    nx.draw_networkx_nodes(G_city, pos, node_size=20, node_color="red")
    nx.draw_networkx_labels(G_city, pos, font_size=6)
    plt.title("Réseau ferroviaire interurbain entre villes")
    plt.axis("off")
    plt.show()

# === MAIN ===

if __name__ == "__main__":
    print("Shapely version", shapely.__version__)
    # ⚠️ Remplace par les chemins réels
    pbf_path = "resources/france-rail.osm.pbf"
    cities_fp = "resources/villes_france.geojson"

    cities = load_cities(cities_fp)

    cities_wgs84 = cities.to_crs(epsg=4326)
    minx, miny, maxx, maxy = cities_wgs84.total_bounds
    bbox = [minx, miny, maxx, maxy]
    print("BBOX utilisé:", bbox)
    rail = load_railways(pbf_path, bbox)
    G_full = build_rail_graph(rail)
    plot_rail_lines(rail, cities)
    components = list(nx.connected_components(G_full))
    sizes = [len(c) for c in components]
    #print("Tailles des composantes :", sorted(sizes, reverse=True))
    #print("Number connected components", nx.number_connected_components(G_full))

    city_node_map = associate_cities_to_graph(cities, G_full, buffer_m=5000)
    G_city = build_city_graph(G_full, city_node_map)

    print(f"📊 Graphe ville-à-ville : {G_city.number_of_nodes()} villes, {G_city.number_of_edges()} connexions.")
    plot_city_graph(G_city, cities)

# 18:40