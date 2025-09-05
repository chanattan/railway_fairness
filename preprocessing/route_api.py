"""
    From Google Route API, get the travel distance between two points.
    A trip is found using a departure or arrival time, including the day, month, and year, as well as the transport mode.
    The API should return only one trip. If multiple trips are found (rare case), the first one is kept with a warning.

    Documentation:
    - https://developers.google.com/maps/documentation/routes/get-started
    - https://developers.google.com/maps/documentation/routes/reference/rest/v2/computeRoutes
"""
import requests
import json
import os

def extract_train_segments(route_json):
    """
    Extrait uniquement les segments TRAIN d'une réponse computeRoutes de Google.
    """
    main_segment, max_distance = None, 0 # Garder le segment le plus long : trajet principal

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

                if distance > max_distance:
                    max_distance = distance
                    main_segment = {
                        "distance_m": distance,
                        "duration_s": duration_s,
                        "departure": dep,
                        "arrival": arr
                    }

    return main_segment

def compute_route_between_aires(aire1_info, aire2_info, api_key, departure_time=None):
    """
    Calcule une route entre deux aires d'attraction.
    """
    print(f"Calcul de la route entre {aire1_info['nom_aire']} et {aire2_info['nom_aire']}")
    # Ou utiliser les coordonnées si disponibles
    if 'geometry' in aire1_info and 'geometry' in aire2_info:
        origin_coords = aire1_info['geometry']
        dest_coords = aire2_info['geometry']
        
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
            "departureTime": departure_time,  # Format RFC3339 UTC "Zulu" time yyyy-mm-ddThh:mm:ssZ
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
                print(f"Route calculée avec succès (code {response.status_code})")
                #print("Réponse :", response)
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
def calculate_routes_batch(aires_dict, mode="transit", max_requests=100):
    """
    Calcule des routes entre aires par batch pour éviter les limites d'API
    
    Parameters:
    -----------
    aires_dict : dict
        Dictionnaire des aires
    mode : str
        Mode de transport
    max_requests : int
        Nombre maximum de requêtes
    
    Returns:
    --------
    dict : Résultats {(code_aire1, code_aire2): route_info}
    """
    routes_results = {}
    request_count = 0
    
    aires_codes = list(aires_dict.keys())
    
    for i, code1 in enumerate(aires_codes):
        for code2 in aires_codes[i+1:]:  # Éviter les doublons
            if request_count >= max_requests:
                print(f"Limite de {max_requests} requêtes atteinte")
                break
            
            aire1 = aires_dict[code1]
            aire2 = aires_dict[code2]
            
            result = compute_route_between_aires(aire1, aire2, mode=mode)
            if result:
                routes_results[(code1, code2)] = result
                print(f"Route {code1} -> {code2}: {result['duration_hours']:.1f}h")
            
            request_count += 1
        
        if request_count >= max_requests:
            break
    
    print(f"Calculé {len(routes_results)} routes avec {request_count} requêtes")
    return routes_results

if __name__ == "__main__":
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
    from datetime import datetime, timedelta
    
    # Récupérer la clé API
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")

    # Exemple d'aires (remplacez par vos vraies données)
    paris_info = {
        'nom_aire': 'Aire d\'attraction de Paris',
        'geometry': type('Point', (), {'x': 2.352, 'y': 48.857})()
    }
    lyon_info = {
        'nom_aire': 'Aire d\'attraction de Lyon', 
        'geometry': type('Point', (), {'x': 4.836, 'y': 45.764})()
    }
    
    print("=== Tests de compute_route_between_aires ===")
    
    # 1. Route en transport en commun sans heure spécifique
    print("\n1. Transport en commun (maintenant):")
    result1 = compute_route_between_aires(paris_info, lyon_info, api_key=api_key)
    if result1:
        print(f"   Distance: {(int(float(result1['distance_m'])/1000)):.1f} km")
        print(f"   Durée: {(int(float(result1['duration_s'])/3600)):.1f}h")

    # 2. Route en transport en commun avec heure de départ
    print("\n2. Transport en commun (demain 8h):")
    departure_time = str(datetime.now() + timedelta(days=1, hours=8-datetime.now().hour, minutes=-datetime.now().minute)).replace(" ", "T").split(".")[0] + "Z"
    result2 = compute_route_between_aires(paris_info, lyon_info, api_key=api_key, departure_time=departure_time)
    if result2:
        print(f"   Distance: {result2['distance_m']:.1f} km")
        print(f"   Durée: {result2['duration_s']:.1f}h")

    # 5. Usage avec vos vraies aires (exemple)
    print("\n5. Usage avec aires_dict:")
    print("# Exemple d'usage avec vos données:")
    print("# paris_aire = aires_dict['001']")
    print("# lyon_aire = aires_dict['069']")
    print("# route = compute_route_between_aires(paris_aire, lyon_aire, mode='transit')")
    print("# if route:")
    print("#     print(f'Paris -> Lyon: {route[\"duration_hours\"]:.1f}h')")