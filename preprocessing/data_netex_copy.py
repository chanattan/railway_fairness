import xml.etree.ElementTree as ET
import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt

# Ce fichier contient des fonctions pour charger et traiter des données NeTEx, ainsi que pour construire un graphe de villes basé sur les arrêts de transport.
# Il est un équivalent de data_test.py mais pour le format NeTEx.
# Pour l'instant, il ne fonctionne pas car il produit 0 liaisons entre villes.
# Pour corriger cela, il faut s'assurer que les arrêts sont correctement extraits et que les relations entre les villes et les arrêts sont établies.

# ----------------------
# Fonction haversine
# ----------------------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    dlon, dlat = radians(lon2 - lon1), radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ----------------------
# Chargement des villes
# ----------------------
def build_ssp_map(root):
    ssp_map = {}

    # Extraire le namespace du document
    ns_uri = root.tag[root.tag.find("{")+1:root.tag.find("}")]
    ns = {"ns": ns_uri}

    # 1. Direct QuayRef dans ScheduledStopPoint
    for ssp in root.findall(".//ns:ScheduledStopPoint", ns):
        ssp_id = ssp.attrib.get("id")
        quay_ref = ssp.find("ns:QuayRef", ns)
        if quay_ref is not None:
            ssp_map[ssp_id] = quay_ref.attrib.get("ref")
        # Certains fichiers SNCF utilisent StopPlaceRef au lieu de QuayRef
        stopplace_ref = ssp.find("ns:StopPlaceRef", ns)
        if stopplace_ref is not None:
            ssp_map[ssp_id] = stopplace_ref.attrib.get("ref")

    # 2. Via StopAssignment
    for sa in root.findall(".//ns:StopAssignment", ns):
        ssp_ref = sa.find("ns:ScheduledStopPointRef", ns)
        quay_ref = sa.find("ns:QuayRef", ns)
        if ssp_ref is not None and quay_ref is not None:
            ssp_map[ssp_ref.attrib.get("ref")] = quay_ref.attrib.get("ref")
        # Certains fichiers SNCF utilisent StopPlaceRef au lieu de QuayRef
        stopplace_ref = sa.find("ns:StopPlaceRef", ns)
        if ssp_ref is not None and stopplace_ref is not None:
            ssp_map[ssp_ref.attrib.get("ref")] = stopplace_ref.attrib.get("ref")

    # 3. Via ScheduledStopPointInJourneyPattern (si dispo)
    for sjp in root.findall(".//ns:ScheduledStopPointInJourneyPattern", ns):
        ssp_ref = sjp.find("ns:ScheduledStopPointRef", ns)
        quay_ref = sjp.find("ns:QuayRef", ns)
        if ssp_ref is not None and quay_ref is not None:
            ssp_map[ssp_ref.attrib.get("ref")] = quay_ref.attrib.get("ref")
        # Ajout : StopPlaceRef dans ScheduledStopPointInJourneyPattern
        stopplace_ref = sjp.find("ns:StopPlaceRef", ns)
        if ssp_ref is not None and stopplace_ref is not None:
            ssp_map[ssp_ref.attrib.get("ref")] = stopplace_ref.attrib.get("ref")

    # 4. Via PassengerStopAssignment (mapping clé pour SNCF)
    for psa in root.findall(".//{*}PassengerStopAssignment"):
        ssp_ref = psa.find("{*}ScheduledStopPointRef")
        stopplace_ref = psa.find("{*}StopPlaceRef")
        if ssp_ref is not None and stopplace_ref is not None:
            ssp_map[ssp_ref.attrib.get("ref")] = stopplace_ref.attrib.get("ref")

    print(f"DEBUG: Mapping SSP → Quay/StopPlace construit : {len(ssp_map)} entrées")
    if len(ssp_map) < 10:
        print("EXTRAIT du mapping:", list(ssp_map.items())[:10])
    return ssp_map

def load_cities(cities_fp):
    cities = gpd.read_file(cities_fp)
    cities = cities.rename(columns={"cityLabel": "name"})
    if cities.crs is None:
        cities.set_crs(epsg=4326, inplace=True)
    return cities

# ----------------------
# Parse du fichier NeTEx
# ----------------------
def load_netex_bis(netex_file):
    """
    Version adaptée : extrait les séquences de trajets à partir des balises ServiceJourneyPattern/pointsInSequence
    et ajoute explicitement les ScheduledStopPoint dans le DataFrame stops.
    """
    tree = ET.parse(netex_file)
    root = tree.getroot()

    # ---- Extraire arrêts (StopPlace + Quay) ----
    stops_data = []
    for sp in root.findall(".//{*}StopPlace"):
        stop_id = sp.attrib.get("id")
        name = sp.findtext("{*}Name")
        lat, lon = None, None
        centroid = sp.find(".//{*}Centroid/{*}Location")
        if centroid is not None:
            lat = centroid.findtext("{*}Latitude")
            lon = centroid.findtext("{*}Longitude")
        if lat and lon:
            stops_data.append({
                "stop_id": stop_id,
                "stop_name": name,
                "stop_lat": float(lat),
                "stop_lon": float(lon)
            })
        for quay in sp.findall(".//{*}Quay"):
            quay_id = quay.attrib.get("id")
            quay_name = quay.findtext("{*}Name")
            centroid = quay.find(".//{*}Centroid/{*}Location")
            if centroid is not None:
                lat = centroid.findtext("{*}Latitude")
                lon = centroid.findtext("{*}Longitude")
                if lat and lon:
                    stops_data.append({
                        "stop_id": quay_id,
                        "stop_name": quay_name,
                        "stop_lat": float(lat),
                        "stop_lon": float(lon)
                    })
    for quay in root.findall(".//{*}Quay"):
        quay_id = quay.attrib.get("id")
        quay_name = quay.findtext("{*}Name")
        centroid = quay.find(".//{*}Centroid/{*}Location")
        if centroid is not None:
            lat = centroid.findtext("{*}Latitude")
            lon = centroid.findtext("{*}Longitude")
            if lat and lon:
                stops_data.append({
                    "stop_id": quay_id,
                    "stop_name": quay_name,
                    "stop_lat": float(lat),
                    "stop_lon": float(lon)
                })
    # ---- Extraire explicitement les ScheduledStopPoint ----
    for ssp in root.findall(".//{*}ScheduledStopPoint"):
        ssp_id = ssp.attrib.get("id")
        name = ssp.findtext("{*}Name")
        centroid = ssp.find(".//{*}Centroid/{*}Location")
        lat, lon = None, None
        if centroid is not None:
            lat = centroid.findtext("{*}Latitude")
            lon = centroid.findtext("{*}Longitude")
        if lat and lon:
            stops_data.append({
                "stop_id": ssp_id,
                "stop_name": name,
                "stop_lat": float(lat),
                "stop_lon": float(lon)
            })
    stops = pd.DataFrame(stops_data).drop_duplicates("stop_id")

    # ---- Construire correspondance ScheduledStopPoint -> Quay ----
    ssp_map = build_ssp_map(root)

    # La même extraction qu'en dessous mais pour les différents types de balises
    # ---- Extraire séquences de voyage (JourneyPart) ----
    stop_times_data = []
    for jp in root.findall(".//{*}JourneyPart"):
        jp_id = jp.attrib.get("id")
        from_ref = jp.find("{*}FromStopPointRef")
        to_ref = jp.find("{*}ToStopPointRef")
        if from_ref is None or to_ref is None:
            continue

        from_raw = from_ref.attrib.get("ref")
        to_raw = to_ref.attrib.get("ref")

        from_id = ssp_map.get(from_raw, from_raw)
        to_id = ssp_map.get(to_raw, to_raw)

        stop_times_data.append({
            "trip_id": jp_id,
            "stop_id": from_id,
            "stop_sequence": 1,
            "arrival_time": None,
            "departure_time": None
        })
        stop_times_data.append({
            "trip_id": jp_id,
            "stop_id": to_id,
            "stop_sequence": 2,
            "arrival_time": None,
            "departure_time": None
        })
    
    # Maintenant, pour les ServiceJourney seulement
    # ---- Extraire séquences de voyage (ServiceJourney) ----
    for sj in root.findall(".//{*}ServiceJourney"):
        sj_id = sj.attrib.get("id")
        for stop_point in sj.findall(".//{*}StopPoint"):
            ssp_ref = stop_point.find("{*}ScheduledStopPointRef")
            if ssp_ref is not None:
                stop_id = ssp_map.get(ssp_ref.attrib.get("ref"), ssp_ref.attrib.get("ref"))
                stop_times_data.append({
                    "trip_id": sj_id,
                    "stop_id": stop_id,
                    "stop_sequence": 1,  # Séquence par défaut, peut être ajustée plus tard
                    "arrival_time": None,
                    "departure_time": None
                })
    
    # Maintenant, pour les JourneyPattern
    for jp in root.findall(".//{*}JourneyPattern"):
        jp_id = jp.attrib.get("id")
        for stop_point in jp.findall(".//{*}StopPoint"):
            ssp_ref = stop_point.find("{*}ScheduledStopPointRef")
            if ssp_ref is not None:
                stop_id = ssp_map.get(ssp_ref.attrib.get("ref"), ssp_ref.attrib.get("ref"))
                stop_times_data.append({
                    "trip_id": jp_id,
                    "stop_id": stop_id,
                    "stop_sequence": 1,  # Séquence par défaut, peut être ajustée plus tard
                    "arrival_time": None,
                    "departure_time": None
                })

    # ---- Extraire séquences de voyage (ServiceJourneyPattern) ----
    #stop_times_data = []
    for sjp in root.findall(".//{*}ServiceJourneyPattern"):
        jp_id = sjp.attrib.get("id")
        points = []
        for spjp in sjp.findall(".//{*}pointsInSequence/{*}StopPointInJourneyPattern"):
            ssp_ref = spjp.find("{*}ScheduledStopPointRef")
            if ssp_ref is not None:
                ssp_id = ssp_ref.attrib.get("ref")
                stop_id = ssp_map.get(ssp_id, ssp_id)
                points.append(stop_id)
        for seq, stop_id in enumerate(points):
            stop_times_data.append({
                "trip_id": jp_id,
                "stop_id": stop_id,
                "stop_sequence": seq + 1,
                "arrival_time": None,
                "departure_time": None
            })

    stop_times_columns = ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]
    if stop_times_data:
        stop_times = pd.DataFrame(stop_times_data, columns=stop_times_columns)
    else:
        stop_times = pd.DataFrame(columns=stop_times_columns)

    print(f"DEBUG: {len(stops)} arrêts extraits, {len(stop_times)} séquences de voyage extraites via ServiceJourneyPattern")
    if stop_times.empty:
        print("WARNING: stop_times is empty. Check if ServiceJourneyPattern extraction et ScheduledStopPoint extraction sont corrects.")

    return stops, stop_times

def load_netex(netex_file):
    tree = ET.parse(netex_file)
    root = tree.getroot()

    # ---- Extraire arrêts (StopPlace) ----
    stops_data = []
    for sp in root.findall(".//{*}StopPlace"):
        stop_id = sp.attrib.get("id")
        name = sp.findtext("{*}Name")
        lat, lon = None, None
        for loc in sp.findall(".//{*}Centroid/{*}Location"):
            lat = float(loc.findtext("{*}Latitude"))
            lon = float(loc.findtext("{*}Longitude"))
        if lat is None or lon is None:
            for quay in sp.findall(".//{*}Quay/{*}Centroid/{*}Location"):
                lat = float(quay.findtext("{*}Latitude"))
                lon = float(quay.findtext("{*}Longitude"))
        if lat and lon:
            stops_data.append({"stop_id": stop_id, "stop_name": name,
                               "stop_lat": lat, "stop_lon": lon})
    stops = pd.DataFrame(stops_data)

    # ---- Construire correspondance ScheduledStopPoint -> StopPlace ----
    ssp_map = build_ssp_map(root)

    print(f"DEBUG: Mapping ScheduledStopPoint → StopPlace : {len(ssp_map)} entrées")
    if len(ssp_map) < 10:
        print("EXTRAIT du mapping:", list(ssp_map.items())[:10])

    # ---- Extraire séquences à partir des JourneyPart ----
    stop_times_data = []
    for jp in root.findall(".//{*}JourneyPart"):
        jp_id = jp.attrib.get("id")
        from_ref = jp.find("{*}FromStopPointRef")
        to_ref = jp.find("{*}ToStopPointRef")
        if from_ref is None or to_ref is None:
            continue

        from_raw = from_ref.attrib.get("ref")
        to_raw = to_ref.attrib.get("ref")

        from_id = ssp_map.get(from_raw, from_raw)
        to_id = ssp_map.get(to_raw, to_raw)

        stop_times_data.append({
            "trip_id": jp_id,
            "stop_id": from_id,
            "stop_sequence": 1,
            "arrival_time": None,
            "departure_time": None
        })
        stop_times_data.append({
            "trip_id": jp_id,
            "stop_id": to_id,
            "stop_sequence": 2,
            "arrival_time": None,
            "departure_time": None
        })

    stop_times = pd.DataFrame(stop_times_data)

    print(f"DEBUG: {len(stops)} arrêts, {len(stop_times)} correspondances extraites via JourneyPart")

    return stops, stop_times

# ----------------------
# Associer villes ↔ stations
# ----------------------
def associate_cities_to_stations(cities, stops_gdf, buffer_km=10):
    """
    Associe chaque ville à l'arrêt géolocalisé le plus proche (Quay, StopPlace, ScheduledStopPoint avec coordonnées).
    Si un stop n'a pas de coordonnées, il est ignoré.
    """
    city_to_station = {}
    stops_gdf = stops_gdf.dropna(subset=["stop_lat", "stop_lon", "geometry"])  # Ignore stops sans coordonnées
    stops_proj = stops_gdf.to_crs(epsg=2154)
    for _, row in cities.iterrows():
        city_point = gpd.GeoSeries([row.geometry.centroid], crs=cities.crs).to_crs(epsg=2154).iloc[0]
        stops_proj["dist"] = stops_proj.geometry.distance(city_point)
        nearest = stops_proj.iloc[stops_proj["dist"].idxmin()]
        dist_km = nearest["dist"] / 1000
        if dist_km <= buffer_km:
            city_to_station[row["name"]] = nearest["stop_id"]
    return city_to_station

def expand_city_station_map(city_station_map, cities, stops_gdf, buffer_km=5):
    """
    Pour chaque ville, associe tous les arrêts géolocalisés dans un buffer autour de la ville à cette ville.
    """
    stop_to_city = {}
    stops_gdf = stops_gdf.dropna(subset=["stop_lat", "stop_lon", "geometry"])  # Ignore stops sans coordonnées
    stops_proj = stops_gdf.to_crs(epsg=2154)
    for city, main_stop_id in city_station_map.items():
        city_geom = cities.loc[cities["name"] == city].geometry.iloc[0].centroid
        city_point = gpd.GeoSeries([city_geom], crs=cities.crs).to_crs(epsg=2154).iloc[0]
        stops_proj["dist"] = stops_proj.geometry.distance(city_point)
        nearby = stops_proj[stops_proj["dist"] <= buffer_km * 1000]
        for stop_id in nearby["stop_id"]:
            stop_to_city[stop_id] = city
    return stop_to_city

# ----------------------
# Construire le graphe
# ----------------------
def build_city_graph(city_station_map, stops, stop_times, stop_to_city):
    G_city = nx.Graph()
    for city in city_station_map:
        G_city.add_node(city)

    merged = stop_times.sort_values(["trip_id", "stop_sequence"])

    for trip_id, group in merged.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        for i in range(len(group) - 1):
            s1, s2 = group.iloc[i], group.iloc[i + 1]
            c1 = stop_to_city.get(s1.stop_id)
            c2 = stop_to_city.get(s2.stop_id)
            if not c1 or not c2 or c1 == c2:
                continue
            st1 = stops.loc[stops.stop_id == s1.stop_id].iloc[0]
            st2 = stops.loc[stops.stop_id == s2.stop_id].iloc[0]
            dist = haversine(st1.stop_lon, st1.stop_lat, st2.stop_lon, st2.stop_lat)
            duration = None  # tu pourras ajouter si tu veux gérer les heures
            if G_city.has_edge(c1, c2):
                if dist < G_city[c1][c2]["distance"]:
                    G_city[c1][c2]["distance"] = dist
            else:
                G_city.add_edge(c1, c2, distance=dist)
    return G_city

def build_city_graph_from_netex(netex_fp, cities_fp, buffer_km_station=10, buffer_km_expand=5):
    """
    Construit un graphe de villes à partir d'un fichier Netex, où les noeuds sont les villes
    (matching via les fonctions existantes) et une arête existe s'il y a un train entre deux villes consécutives.
    """
    # Charger Netex
    stops, stop_times = load_netex_bis(netex_fp)
    # Charger les villes
    cities = load_cities(cities_fp)
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326"
    )
    city_station_map = associate_cities_to_stations(cities, stops_gdf, buffer_km=buffer_km_station)
    stop_to_city = expand_city_station_map(city_station_map, cities, stops_gdf, buffer_km=buffer_km_expand)
    G_city = build_city_graph(city_station_map, stops, stop_times, stop_to_city)
    return G_city, cities

# ----------------------
# Visualisation
# ----------------------
def plot_city_graph(G_city, cities):
    pos = {row["name"]: (row.geometry.x, row.geometry.y)
           for _, row in cities.iterrows() if row["name"] in G_city.nodes}
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(G_city, pos, alpha=0.3)
    nx.draw_networkx_nodes(G_city, pos, node_size=20, node_color="red")
    nx.draw_networkx_labels(G_city, pos, font_size=6)
    plt.show()

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    netex_fp = "resources/netex/france/sncf-netex.xml"
    cities_fp = "resources/villes_france.geojson"

    # Charger Netex et afficher diagnostic
    stops, stop_times = load_netex_bis(netex_fp)
    print("Exemples de stop_id dans stop_times:", stop_times['stop_id'].unique()[:10])
    print("Exemples de stop_id dans stops:", stops['stop_id'].unique()[:10])
    n_match = sum(stop_times['stop_id'].isin(stops['stop_id']))
    print(f"Nombre de stop_id de stop_times présents dans stops : {n_match} / {len(stop_times)}")

    # Charger les villes
    cities = load_cities(cities_fp)
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326"
    )
    city_station_map = associate_cities_to_stations(cities, stops_gdf, buffer_km=10)
    stop_to_city = expand_city_station_map(city_station_map, cities, stops_gdf, buffer_km=5)
    G_city = build_city_graph(city_station_map, stops, stop_times, stop_to_city)
    print(f"{G_city.number_of_nodes()} villes, {G_city.number_of_edges()} liaisons directes.")
    plot_city_graph(G_city, cities)

    # --- Check Rennes ↔ Brest ---
    ville1 = "Rennes"
    ville2 = "Brest"
    if ville1 in G_city.nodes and ville2 in G_city.nodes:
        if G_city.has_edge(ville1, ville2):
            print(f"Il existe une arête directe entre {ville1} et {ville2}.")
        else:
            print(f"Pas d'arête directe entre {ville1} et {ville2}.")
            try:
                path = nx.shortest_path(G_city, ville1, ville2)
                print(f"Plus court chemin ({len(path)-1} arêtes) : {' -> '.join(path)}")
            except nx.NetworkXNoPath:
                print(f"Aucun chemin entre {ville1} et {ville2} dans le graphe.")
    else:
        print(f"{ville1} ou {ville2} n'est pas dans le graphe des villes.")
