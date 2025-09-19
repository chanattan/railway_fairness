"""
    This file takes part of the data enrichment, after the graph generation.
    This pertains to real travel distances, and demand matrix (which could also be done in data_main.py).

    From Google Route API, get the travel distance between two points.
    A trip is found using a departure or arrival time, including the day, month, and year, as well as the transport mode.
    The API should return only one trip. If multiple trips are found, the first one is kept.

    Documentation:
    - https://developers.google.com/maps/documentation/routes/get-started
    - https://developers.google.com/maps/documentation/routes/reference/rest/v2/computeRoutes
"""
import requests
import json
import os
from shapely.geometry import Point
import math
from pyproj import Geod
from datetime import datetime, timedelta

""" Distances à vol d'oiseau et demandes """

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Distance à vol d'oiseau en mètres (WGS84).
    """
    geod = Geod(ellps="WGS84")
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist

def euclidean_distance(x1, y1, x2, y2):
    """
    Distance euclidienne (utile si coords en mètres projetés, ex: EPSG:2056).
    """
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def compute_fly_distances(graph_path, country='france'):
    """
    Calcule les distances à vol d'oiseau entre toutes les paires de villes.
    Écrit le dictionnaire des distances dans le JSON du graphe.
    
    Parameters
    ----------
    graph_path : str
        Chemin vers le fichier JSON du graphe
    country : str
        "france" -> EPSG:4326 (lat/lon)
        "suisse" -> EPSG:2056 (mètres)
    
    Returns
    -------
    dict : distances[i-j] = dist en m
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    distances = {}

    for i, src in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            target = nodes[j]

            if country.lower() == 'france':
                # coords en lon/lat (EPSG:2056)
                dist_m = haversine_distance(src["x"], src["y"], target["x"], target["y"])
            elif country.lower() == 'switzerland':
                # coords projetées en m (EPSG:4326)
                dist_m = euclidean_distance(src["x"], src["y"], target["x"], target["y"])
            else:
                raise ValueError("Pays non supporté. Utiliser 'france' ou 'switzerland'.")

            distances[f"{src['id']}-{target['id']}"] = dist_m

    # écriture dans le graphe
    graph["distances"] = distances

    return distances

def demand_function(src_pop, target_pop, distance):
    """
        On utilise fondée sur le modèle de gravitation, mais en économie.
        TODO: trouver des paramètres qui conviennent, pour chaque pays.
    """
    alpha = 1
    return (src_pop * target_pop) / (distance ** alpha)

def compute_demands(graph_path):
    """
    Calcule les demandes entre villes = (pop_i * pop_j) / distance (loi gravitaire)
    Écrit le dictionnaire des demandes dans le JSON du graphe.
    Cette fonction est à appeler après le calcul des distances à vol d'oiseau.
    
    Parameters
    ----------
    graph_path : str
        Chemin vers le fichier JSON du graphe
    
    Returns
    -------
    dict : demands[i-j] = float où i-j est le texte du couple (i,j).
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    
    nodes = {n["id"]: n for n in graph["nodes"]}
    distances = graph.get("distances", {})

    demands = {}
    for key, dist in distances.items():
        src_id, target_id = key.split("-")
        src_pop = int(nodes[src_id]["population"])
        target_pop = int(nodes[target_id]["population"])

        if dist > 0:
            demand_val = demand_function(src_pop, target_pop, dist)
        else:
            demand_val = 0

        demands[key] = demand_val

    # On réecrit dans le graphe
    graph["demands"] = demands

    return demands

""" Distances réelles """

def read_graph(graph_json="preprocessing/enriched_graphs/france_railway_network.json"):
    """
    Lit un graphe généré par data_main.py, pour le traitement des arêtes avec les requêtes.
    """
    with open(graph_json, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    edges = graph["edges"]

    node_lookup = {node["id"]: node for node in nodes}

    return graph, nodes, edges, node_lookup

def extract_train_segments(route_json):
    """
    Extrait uniquement les segments TRAIN d'une réponse computeRoutes de Google.
    """
    main_segment, max_distance = None, 0 # Garder le segment le plus long : c'est le trajet principal (train).

    routes = route_json.get("routes", [])
    if not routes:
        return 0, 0, []

    for leg in routes[0].get("legs", []):
        for step in leg.get("steps", []):
            if step.get("travelMode") == "TRANSIT":
                transit_details = step.get("transitDetails", {})
                distance = step.get("distanceMeters", 0)
                
                # Les durées arrivent en texte ("2 heures 10 minutes") et en secondes dans step.duration
                # mais parfois step.duration n'est pas directement exposé
                # Ici on peut estimer à partir des horaires si disponibles :
                dep = transit_details.get("stopDetails", {}).get("departureTime")
                arr = transit_details.get("stopDetails", {}).get("arrivalTime")
                
                duration_s = 0
                if dep and arr:
                    from datetime import datetime
                    fmt = "%Y-%m-%dT%H:%M:%SZ"
                    try:
                        dep_dt = datetime.strptime(dep, fmt)
                        arr_dt = datetime.strptime(arr, fmt)
                        duration_s = int((arr_dt - dep_dt).total_seconds())
                    except Exception:
                        pass
                else:
                    print(f"/!\ stopDetails may be null for route {routes[0]}.")

                if distance > max_distance:
                    max_distance = distance
                    main_segment = {
                        "distance_m": distance,
                        "duration_s": duration_s,
                        "departure": dep,
                        "arrival": arr
                    }

    return main_segment

def compute_route_between(com1, com2, api_key, departure_time=None):
    """
    Calcule une route entre deux agglomérations, avec Google Routes API.
    La route peut être calculée avec un horaire de départ : utilisé pour prendre les meilleurs trajets.
    """
    # Ou utiliser les coordonnées si disponibles
    if 'geometry' in com1 and 'geometry' in com2:
        origin_coords = com1['geometry']
        dest_coords = com2['geometry']
        
        data = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": origin_coords.y,
                        "longitude": origin_coords.x
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": dest_coords.y,
                        "longitude": dest_coords.x
                    }
                }
            },
            "travelMode": "TRANSIT",
            #"departureTime": departure_time,  # Format RFC3339 UTC "Zulu" time yyyy-mm-ddThh:mm:ssZ
            "computeAlternativeRoutes": False,
            "transitPreferences": {
                "routingPreference": "FEWER_TRANSFERS",
                "allowedTravelModes": ["TRAIN"]
            },
            "units": "METRIC"
        }
        
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'routes.legs.duration,routes.legs.steps.distanceMeters,routes.legs.steps.transitDetails.stopDetails.arrivalTime,routes.legs.steps.transitDetails.stopDetails.departureTime,routes.legs.steps.localizedValues,routes.legs.steps.travelMode,routes.distanceMeters,routes.duration,routes.legs.steps.startLocation,routes.legs.steps.endLocation'
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                print(f"\rRoute calculée avec succès (code {response.status_code})", end='')
                print("\nRéponse :", response.json())
                res = extract_train_segments(response.json())
                #print("Résultat extrait :", res)
                return res
            else:
                raise ValueError(f"Erreur {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Erreur de requête: {e}")
    
    else:
        raise ValueError("Les informations de géométrie sont manquantes pour une ou les deux aires.")

# Fonction utilitaire pour calculer des routes par batch
def calculate_routes_batch(graph_path, max_requests=1):
    """
    Calcule des routes entre aires par batch pour éviter les limites d'API
    
    Parameters:
    -----------
    edges : list
        Liste des lignes de train du graphe
    mode : str
        Mode de transport
    max_requests : int
        Nombre maximum de requêtes
    
    """
    graph, nodes, edges, node_lookup = read_graph(graph_path)
    failed_edges = []
    API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
    if not API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")

    for i, edge in enumerate(edges):
        if i >= max_requests:
            print(f"Limite de {max_requests} requêtes atteinte")
            break

        # Data
        src_id = edge["source"]
        target_id = edge["target"]
        if edge["depart"] == target_id:
            src_id, target_id = target_id, src_id
        src_node = node_lookup[src_id]
        target_node = node_lookup[target_id]

        src_coords = Point(src_node["x"], src_node["y"])
        target_coords = Point(target_node["x"], target_node["y"])

        com_src = {'name': src_node["name"], 'geometry': src_coords}
        com_target = {'name': target_node["name"], 'geometry': target_coords}

        departure_time = edge["departure_time"]
        #print(f"For line {edge}, src: {src_id}, target: {target_id}, com_src {com_src}, com_target {com_target}, departure_time {departure_time}")

        try:
            print(f'\rCalcul du trajet entre {src_node["name"]} et {target_node["name"]}...', end='')
            result = compute_route_between(com_src, com_target, api_key=API_KEY, departure_time=departure_time)
            if result:
                edge["route_result"] = result
            else:
                failed_edges.append((src_node["name"], target_node["name"]))
        except Exception as e:
            failed_edges.append((src_node["name"], target_node["name"]))
            print(f"\n/!\ Impossible de calculer la route {src_node['name']} -> {target_node['name']} : {e}")

        print(f"\r{com_src['name']} ({src_coords}) -> {com_target['name']} ({target_coords})", end='')
    
    # Save graph
    # Write into same file graph_path as a new attribute into the corresponding edge.
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    
    if failed_edges and False:
        print("\nEdges non calculés :")
        for s, t in failed_edges:
            print(f"  {s} -> {t}", end="; ")
    
    print(f"{i} requêtes effectuées, résultats dans le fichier {graph_path}.")

def examples():
    """
    print(" === Test de extract_train_segments ===")
    sample_route_json = {
        'routes': [{
            'legs': [{
                'duration': '9412s',
                'steps': [
                    {'distanceMeters': 12, 'localizedValues': {'distance': {'text': '12 m'}, 'staticDuration': {'text': '1 minute'}}, 'travelMode': 'WALK'},
                    {'distanceMeters': 41, 'localizedValues': {'distance': {'text': '41 m'}, 'staticDuration': {'text': '1 minute'}}, 'travelMode': 'WALK'},
                    {'distanceMeters': 2288, 'localizedValues': {'distance': {'text': '2,3 km'}, 'staticDuration': {'text': '6 minutes'}}, 'transitDetails': {'stopDetails': {'arrivalTime': '2025-09-06T04:52:00Z', 'departureTime': '2025-09-06T04:46:00Z'}}, 'travelMode': 'TRANSIT'},
                    {'distanceMeters': 85, 'localizedValues': {'distance': {'text': '85 m'}, 'staticDuration': {'text': '2 minutes'}}, 'travelMode': 'WALK'},
                    {'distanceMeters': 432199, 'localizedValues': {'distance': {'text': '432 km'}, 'staticDuration': {'text': '2 heures 10 minutes'}}, 'transitDetails': {'stopDetails': {'arrivalTime': '2025-09-06T07:10:00Z', 'departureTime': '2025-09-06T05:00:00Z'}}, 'travelMode': 'TRANSIT'},
                    {'distanceMeters': 168, 'localizedValues': {'distance': {'text': '0,2 km'}, 'staticDuration': {'text': '3 minutes'}}, 'travelMode': 'WALK'},
                    {'distanceMeters': 1727, 'localizedValues': {'distance': {'text': '1,7 km'}, 'staticDuration': {'text': '4 minutes'}}, 'transitDetails': {'stopDetails': {'arrivalTime': '2025-09-06T07:21:00Z', 'departureTime': '2025-09-06T07:17:00Z'}}, 'travelMode': 'TRANSIT'},
                    {'distanceMeters': 36, 'localizedValues': {'distance': {'text': '36 m'}, 'staticDuration': {'text': '1 minute'}}, 'travelMode': 'WALK'},
                    {'distanceMeters': 33, 'localizedValues': {'distance': {'text': '33 m'}, 'staticDuration': {'text': '1 minute'}}, 'travelMode': 'WALK'}
                ]
            }],
            'distanceMeters': 436589,
            'duration': '9466s'
        }]
    }
    main_segment = extract_train_segments(sample_route_json)
    print("Segment principal extrait :", main_segment)
    # On pourrait également vérifier que la distance obtenue est bien supérieure ou égale à la distance à vol d'oiseau.
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
    
    # test aires
    paris_info = {
        'nom_aire': 'Chambéry',
        'geometry': Point(5.909, 45.583)
    }
    lyon_info = {
        'nom_aire': 'Ambérieu-en-Bugey', 
        'geometry': Point(5.373, 45.961)
    }
    
    print("=== Tests de compute_route_between_aires ===")

    # Route en transport en commun avec heure de départ
    print("\n2. Transport en commun (demain 8h):")
    #departure_time = str(datetime.now() + timedelta(days=1, hours=8-datetime.now().hour, minutes=-datetime.now().minute)).replace(" ", "T").split(".")[0] + "Z"
    #departure_time = "2025-08-16T05:41:00Z"
    result2 = compute_route_between(paris_info, lyon_info, api_key=api_key)#, departure_time=departure_time)
    if result2:
        print("Result", result2)

if __name__ == "__main__":
    france_graph = 'preprocessing/enriched_graphs/france_railway_network.json'
    swiss_graph = 'preprocessing/enriched_graphs/switzerland_railway_network.json'

    examples()

    print("FIN DU PROGRAMME !")
