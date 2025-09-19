import config
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from shapely.geometry import Point
from datetime import datetime, timedelta

# haversine distance en km
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    dlon, dlat = radians(lon2 - lon1), radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def __build_city_graph(cities, stops, stop_times, trips, stops_to_gares, gares_to_cities, agglomerations):
    import networkx as nx
    import pandas as pd

    G_city = nx.Graph()
    # Ensemble des agglomérations valides
    valid_agglos = set(cities[config.key])
    print(f"[DEBUG] Nombre de communes valides :", len(valid_agglos))

    # Ajouter les noeuds
    for _, row in cities.iterrows():
        city = row[config.key]
        name = row['nom_aire' if config.by_agg else config.name_attr]
        G_city.add_node(city, name=name, insee='code_insee')

    merged = stop_times.merge(trips, on="trip_id").sort_values(["trip_id", "stop_sequence"])

    stops_without_gare = 0
    gares_without_city = 0

    for trip_id, group in merged.groupby("trip_id"):
        group = group.sort_values("stop_sequence")

        # Construire la séquence d'agglomérations avec les infos distances/durations
        agglos_sequence = []
        stops_sequence = []  # pour récupérer coordonnées/temps
        uic_sequence = []

        for _, s in group.iterrows():
            gare = stops_to_gares.get(s.stop_id, {}).get('code_uic', None)
            if gare is None:
                stops_without_gare += 1
                continue
            city_codeinsee = gares_to_cities.get(gare, {}).get('code_insee', None)
            if city_codeinsee is None:
                gares_without_city += 1
                continue

            aggl = agglomerations.get(city_codeinsee, {}).get('code_aire', None)
            if aggl is None:
                continue

            # Ajouter la ville et le stop correspondant
            agglos_sequence.append(aggl)
            stops_sequence.append(s)
            uic_sequence.append(gare)

        if '14118' in agglos_sequence and '35238' in agglos_sequence and False:
            print("Found Caen - Rennes global trip")
            print("Seq:", agglos_sequence)
            #print("Stops:", stops_sequence)
            for i, insee in enumerate(agglos_sequence):
                print(f"   {insee} ({gares_to_cities[uic_sequence[i]][name_attr]}) - valid: {insee in valid_agglos}")

        # Filtrer seulement les agglos présentes dans cities_df et supprimer doublons consécutifs
        reduced_seq = []
        reduced_stops = []
        for idx, aggl in enumerate(agglos_sequence):
            if aggl in valid_agglos:
                if not reduced_seq or aggl != reduced_seq[-1]:  # éviter doublons consécutifs
                    reduced_seq.append(aggl)
                    reduced_stops.append(stops_sequence[idx])
        
        if '14118' in reduced_seq and '35238' in reduced_seq:
            print("Found Caen - Rennes global r trip")
            print("Seq:", reduced_seq)
            print("Stops:", reduced_stops)

        # Créer les arêtes entre agglomérations consécutives de la séquence réduite
        for i in range(len(reduced_seq) - 1):
            c1, c2 = reduced_seq[i], reduced_seq[i + 1]
            s1, s2 = reduced_stops[i], reduced_stops[i + 1]
            if ((c1 == '14118' and c2 == '35238') or (c1 == '35238' and c2 == '14118')) and False:
                print("Found Caen - Rennes")
                print(f"c1 {c1} c2 {c2} s1 {s1} s2 {s2}")


            t1 = pd.to_datetime(s1.departure_time, errors="coerce")
            t2 = pd.to_datetime(s2.arrival_time, errors="coerce")
            if pd.isna(t1) or pd.isna(t2):
                continue
            duration = (t2 - t1).seconds / 60

            if G_city.has_edge(c1, c2):
                if dist < G_city[c1][c2]["distance"]:
                    G_city[c1][c2]["distance"] = dist
                if duration < G_city[c1][c2]["duration"]:
                    G_city[c1][c2]["duration"] = duration
            else:
                G_city.add_edge(c1, c2, distance=dist, duration=duration)
    G_city.remove_nodes_from(list(nx.isolates(G_city)))
    print(f"[DEBUG] {stops_without_gare} arrêts sans gare associée, {gares_without_city} gares sans ville associée.")
    return G_city

def __plot_city_graph(G_city, cities):
    pos = {str(row[config.key]): (row.geometry.x, row.geometry.y)
           for _, row in cities.iterrows() if row[config.key] in G_city.nodes}

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(G_city, pos, alpha=0.3)
    nx.draw_networkx_nodes(G_city, pos, node_size=20, node_color="red")
    labels = nx.get_node_attributes(G_city, "name")
    nx.draw_networkx_labels(G_city, pos, labels, font_size=6)
    plt.show(block=False)

def extract_train_type(stop_modes, stop_id):
    """
    Extrait le type de train depuis le stop_id
    Retourne: 'TER', 'TGV', 'INTERCITES', 'ICE', 'UNKNOWN' pour la France.
    """
    if config.country == 'france':
        # On vérifie le route type pour préfiltrer, ensuite on regarde le type précisément.
        # 0 - Tram, 1 - Subway, 2 - Rail, 3 - Bus, 4 - Ferry, ...
        #print("DEBUG --- STOP modes :", stop_id, stop_modes.get(stop_id, set()))
        if 2 not in stop_modes[stop_id]:
            # C'est pas un train
            return 'CAR' # Default filtré
        if 'OCECar' in stop_id:
            return 'CAR'
        elif 'OCETrain TER' in stop_id:
            return 'TER'
        elif 'OCETGV INOUI' in stop_id:
            return 'TGV'
        elif 'OCEINTERCITES' in stop_id:
            return 'INTERCITES'
        elif 'OCEICE' in stop_id:
            return 'ICE'
        else:
            return 'UNKNOWN'
    elif config.country == 'switzerland':
        # Pour la Suisse, le type est directement dans le route type
        # Le type doit être dans [101, 102, 103, 106] pour être un train
        if 101 in stop_modes[stop_id]:
            return 'HIGH-SPEED'
        elif 102 in stop_modes[stop_id]:
            return 'INTERCITY' # / EUROCITY
        elif 103 in stop_modes[stop_id]:
            return 'INTER-REGIONAL'
        elif 106 in stop_modes[stop_id]:
            return 'REGIONAL'
        else:
            return 'CAR' # Default filtré
    else:
        raise ValueError(f"Pays inconnu pour extraction type train: {config.country}")

import re, pytz
from datetime import datetime, timedelta

def debug_trip(trip_id, s1, s2, stops_to_gares, gares_to_cities, calendar_dict, aggl, country='france'):
    # s1/s2 doivent fournir:
    # - departure_time (str, format GTFS "HH:MM:SS" possibly >=24)
    # - departure_time_td (pd.Timedelta) OPTIONAL for duration calc
    # - arrival_time (str) and arrival_time_td
    
    g1 = stops_to_gares.get(s1.stop_id, {}).get('code_uic', None)
    g2 = stops_to_gares.get(s2.stop_id, {}).get('code_uic', None) if s2 is not None else None
    if g1 is not None:
        v1 = gares_to_cities.get(g1, {}).get(config.name_attr, None)
    else:
        v1 = ""
    if g2 is not None:
        v2 = gares_to_cities.get(g2, {}).get(config.name_attr, None)
    else:
        v2 = ""
    if (trip_id == "OCESN13104F4535914:2025-06-17T23:52:01Z"):
        print(f"AGGL {aggl} ---")
    else:
        return
    print(f"> Gare S1 : {g1} ; Ville S1 : {v1}")
    if v2 is not None:
        print(f"> Gare S2 : {g2} ; Ville S2 : {v2}")

    # 1) date choisie (calendar) et fallback au trip_id si présent
    date_from_calendar = calendar_dict.get(trip_id)
    m = re.search(r'(\d{4}-\d{2}-\d{2})T', trip_id)
    date_from_tripid = m.group(1).replace('-', '') if m else None

    print("trip_id:", trip_id)
    print("date_from_calendar:", date_from_calendar)
    print("date_from_tripid:", date_from_tripid)

    # 2) raw times
    print("s1.departure_time (raw):", s1.departure_time)
    if s2 is not None:
        print("s2.arrival_time   (raw):", s2.arrival_time)

    # 3) timedeltas if available
    try:
        td1 = s1.departure_time_td
        if s2 is not None:
            td2 = s2.arrival_time_td
        print("s1.departure_time_td:", td1, " seconds:", td1.total_seconds())
        if s2 is not None:
            print("s2.arrival_time_td  :", td2, " seconds:", td2.total_seconds())
    except Exception:
        print("No *_td fields available (ok if you didn't create them).")
    if s2 is None:
        return
    # 4) chosen date logic (prefer trip_id date if present)
    chosen_date = date_from_tripid or date_from_calendar
    if chosen_date is None:
        print("No date found for this trip -> cannot compute UTC departure.")
        return

    print("Chosen date (YYYYMMDD):", chosen_date)

    # 5) build local datetimes with GTFS hour >=24 handling
    def make_local_dt(date_yyyymmdd, time_str, country):
        hrs, mins, secs = map(int, time_str.split(":"))
        extra_days, hour = divmod(hrs, 24)
        base = datetime.strptime(date_yyyymmdd, "%Y%m%d")
        dt_local = base + timedelta(days=extra_days, hours=hour, minutes=mins, seconds=secs)
        tz = pytz.timezone({'france':'Europe/Paris','switzerland':'Europe/Zurich'}.get(country,'Europe/Paris'))
        dt_loc = tz.localize(dt_local)
        return dt_loc

    dt1_local = make_local_dt(chosen_date, s1.departure_time, country)
    dt2_local = make_local_dt(chosen_date, s2.arrival_time, country)

    print("dt1_local:", dt1_local.isoformat(), " offset:", dt1_local.utcoffset())
    print("dt1_UTC  :", dt1_local.astimezone(pytz.UTC).isoformat())
    print("dt2_local:", dt2_local.isoformat(), " offset:", dt2_local.utcoffset())
    print("dt2_UTC  :", dt2_local.astimezone(pytz.UTC).isoformat())

    # 6) duration via datetimes (robuste)
    delta = dt2_local - dt1_local
    secs = delta.total_seconds()
    print("duration via datetimes (s):", secs, " -> minutes:", secs/60)


def build_city_graph_with_trips(cities, stop_modes, stops, stop_times, trips, calendar, stops_to_gares, gares_to_cities, communes, aires):
    """
        Crée un graphe ville-à-ville à partir des données GTFS.
        Version modifiée qui stocke les trip_ids et autres infos dans les arêtes.
    """
    import networkx as nx
    import pandas as pd

    G_city = nx.Graph()

    # Ensemble des agglomérations valides
    valid_agglos = set(cities[config.key])
    valid_cities = set(cities['code_insee' if config.country == 'france' else 'code_commune'])
    print(f"[DEBUG] Nombre d'agglomérations valides :", len(valid_agglos))

    # Convertir les temps en timedelta pour faciliter les calculs
    stop_times["arrival_time_td"] = pd.to_timedelta(stop_times["arrival_time"])
    stop_times["departure_time_td"] = pd.to_timedelta(stop_times["departure_time"])

    # Ajouter les noeuds (unités urbaines ou villes)
    for _, row in cities.iterrows():
        if config.country == 'france':
            city = row[config.key]
            name = row['nom_aire' if config.by_agg else config.name_attr]
            #print("Aires :::::", aires.get(row['code_aire'], row) if config.key == 'code_aire' else row)
            #print("DEBUG city:", city, name, row['code_insee'], aires[row['code_aire']]['population'] if config.key == 'code_aire' else row['population'])
            pos = (aires[row['code_aire']]['geometry'].x, aires[row['code_aire']]['geometry'].y) if config.by_agg else (row.geometry.x, row.geometry.y)
            population = aires.get(row['code_aire'], {}).get('population', row['population']) if config.by_agg else row['population']
        elif config.country == 'switzerland':
            city = row[config.key]
            name = row[config.name_attr]
            # Reprojection en 4326 pour l'affichage
            cities = cities.to_crs(epsg=4326)
            pos = (row.geometry.x, row.geometry.y)
            population = int(row['population'])
        else:
            raise ValueError(f"Pays inconnu pour construction graphe: {config.country}")
        
        G_city.add_node(city,
                        name=name,
                        code=row[config.key],
                        population=population,
                        x=pos[0],
                        y=pos[1])

    # Traitement des données rapide
    merged = stop_times.merge(trips, on="trip_id").sort_values(["trip_id", "stop_sequence"])
    stops_indexed = stops.set_index('stop_id')

    stops_without_gare = 0
    gares_without_city = 0
    filtered_cars = 0
    import sys

    # Dictionnaire pour stocker les trip_ids par arête
    edge_trips = {}

    # Pour un trajet (trip_id) on construit les lignes qui en découlent (pour chaque paire de villes consécutives) : d'où l'usage de continue au lieu de break plus tard.
    for trip_id, group in merged.groupby("trip_id"):
        group = group.sort_values("stop_sequence")

        # Construire la séquence d'agglomérations avec les infos distances/durations
        agglos_sequence = []
        stops_sequence = []
        uic_sequence = []
        train_types = []
        saved_aglo = []

        for s in group.itertuples(index=False):
            print(f"\r[DEBUG] Processing trip {trip_id} stop {s.stop_id}...", end="")
            train_type = extract_train_type(stop_modes, s.stop_id)

            if train_type == 'CAR':
                #filtered_cars += 1
                break # On peut ignorer les voyages car

            gare = stops_to_gares.get(s.stop_id, {}).get('code_uic', None)
            if gare is None:
                stops_without_gare += 1
                continue # Pas d'affectation gare, on peut ignorer tant qu'il y a au moins une gare affectée (sinon tout est ignoré et le trip n'est pas construit).
            
            city_code = gares_to_cities.get(gare, {}).get(config.city_codes[config.country], None)
            if city_code is None:
                gares_without_city += 1
                continue # Si la ville pour ce stop n'est pas trouvée, c'est que la ville est filtrée (peu de chance que ce soit les données, car on fait un goulot d'étranglement sur les valeurs na avant).

            agglos_sequence.append(city_code)
            stops_sequence.append(s)
            uic_sequence.append(gare)
            train_types.append(train_type)

        # Filtrer seulement les agglos présentes dans cities_df et supprimer doublons consécutifs
        # On ne garde que le dernier item qui est consécutif avec le précédent : c'est le dernier arrêt capturé.
        last_idx = {}
        for i, aggl in enumerate(agglos_sequence):
            if aggl in valid_cities:
                last_idx[aggl] = i

        # then we build reduced lists, keeping only elements at last occurrences (furthest stops)
        reduced_seq = []
        reduced_stops = []
        reduced_types = []
        for i, aggl in enumerate(agglos_sequence):
            if aggl in valid_cities and last_idx[aggl] == i:
                reduced_seq.append(aggl)
                reduced_stops.append(stops_sequence[i])
                reduced_types.append(train_types[i])

        # Créer les metadata pour les arêtes entre communes consécutives
        for i in range(len(reduced_seq) - 1):
            c1, c2 = reduced_seq[i], reduced_seq[i + 1]
            s1, s2 = reduced_stops[i], reduced_stops[i + 1]
            type1, type2 = reduced_types[i], reduced_types[i + 1]

            # Prioriser le type le plus spécifique (non UNKNOWN)
            segment_type = type1 if type1 != 'UNKNOWN' else type2

            # Récupérer stops originaux pour distance/duration
            st1 = stops_indexed.loc[s1.stop_id]
            st2 = stops_indexed.loc[s2.stop_id]
            # Calcul distance géodésique pour éviter pb projection
            # Voir https://geopy.readthedocs.io/en/stable/#module-geopy
            # Create geometries with stop_lon and stop_lat
            g1 = stops_to_gares.get(s1.stop_id, {}).get('geometry', None)
            g2 = stops_to_gares.get(s2.stop_id, {}).get('geometry', None)
            if g1 is None or g2 is None:
                continue
            name = stops_to_gares.get(s1.stop_id, {}).get(config.name_attr, 'Unknown')
            
            # Cette première version servait à avoir les distances rapidement, cela est réecrit dans data_enricher.py.
            dist = haversine(st1.stop_lon, st1.stop_lat, st2.stop_lon, st2.stop_lat)

            t1 = s1.departure_time_td
            t2 = s2.arrival_time_td
            if pd.isna(t1) or pd.isna(t2):
                continue
            # Calcul du temps de trajet, en comptant les trains qui sont sur deux jours (tard le soir et arrivée tôt le lendemain matin).
            date_str = calendar.get(trip_id)
            if date_str is None:
                unknown_date += 1
                duration = (t2 - t1).total_seconds() / 60 # On calcule juste une durée, peut-être qu'elle n'est pas correcte (correction après)
            else:
                # date du service
                base_date = datetime.strptime(str(date_str), "%Y%m%d")

                # departure_time / arrival_time sont en str "HH:MM:SS"
                h1, m1, ss1 = map(int, s1.departure_time.split(":"))
                h2, m2, ss2 = map(int, s2.arrival_time.split(":"))

                dt1 = base_date + timedelta(hours=h1, minutes=m1, seconds=ss1)
                dt2 = base_date + timedelta(hours=h2, minutes=m2, seconds=ss2)

                # si heure > 23 timedelta gère automatiquement en ajoutant le jour
                duration = (dt2 - dt1).total_seconds() / 60
            #debug_trip(trip_id, s1, s2, stops_to_gares, gares_to_cities, calendar, None)

            # On conserve le trip_id à titre de référence de trajet mais également le departure time (date complète)
            # afin de faire les requêtes sur les distances réelles dans data_enricher.py.
            # get_departure_utc renvoie la date adaptée pour une requête à Google
            departure_time = config.get_departure_utc(trip_id, s1.departure_time, calendar, config.country)

            # On sauvegarde les attributs pour l'arête entre communes pour un post-traitement
            edge_key = tuple(sorted([c1, c2]))
            if edge_key not in edge_trips:
                edge_trips[edge_key] = []
            edge_trips[edge_key].append({
                'trip_id': trip_id,
                'route_id': s1.route_id,
                'stop1': st1.stop_name, # Le trajet fait stop1 -> stop2.
                'stop2': st2.stop_name,
                'departure': s1.departure_time_td,
                'arrival': s2.arrival_time_td,
                'train_type': segment_type
            })

            # Une fois l'arête prête, on crée la véritable arête entre agrégats si requis.
            if config.by_agg:
                aggl1 = communes.get(c1, {}).get(config.key, None)
                aggl2 = communes.get(c2, {}).get(config.key, None)
                if aggl1 is None or aggl2 is None:
                    continue # Une des deux extrêmités n'a pas de correspondance, la ligne n'existe pas à l'échelle des agrégats.
                if aggl1 not in valid_agglos or aggl2 not in valid_agglos:
                    continue # On ne garde que les agglomérations non-filtrées
                if aggl1 == aggl2:
                    continue # Il s'agit d'une ligne interne à l'agrégat.
                # La ligne est valide.

            n1 = c1 if not config.by_agg else aggl1
            n2 = c2 if not config.by_agg else aggl2

            # Attributs retenus dans l'arête
            if G_city.has_edge(n1, n2):
                #if dist < G_city[c1][c2]["distance"]:
                #prendre le plus court trajet en temps
                if duration < G_city[n1][n2]["duration"]:
                    G_city[n1][n2]["duration"] = duration
                    G_city[n1][n2]["departure_time"] = departure_time
                    G_city[n1][n2]["distance"] = dist
                    G_city[n1][n2]["trip_id"] = trip_id  # trip le plus rapide
                    G_city[n1][n2]["depart"] = n1
                    G_city[n1][n2]["depart_com"] = c1
                    G_city[n1][n2]["arrival_com"] = c2
            else:
                G_city.add_edge(n1, n2, distance=dist, duration=duration, trip_id=trip_id, departure_time=departure_time, depart=n1, depart_com=c1, arrival_com=c2)

    print("[DEBUG] Enrichissement des arêtes avec les trajets...")
    # Ajouter les trip_ids aux arêtes
    for u, v, data in G_city.edges(data=True):
        edge_key = tuple(sorted([data['depart_com'], data['arrival_com']]))
        if edge_key in edge_trips:
            trips_data = edge_trips[edge_key]
            
            # Améliorer les types UNKNOWN si possible
            known_types = [trip['train_type'] for trip in trips_data if trip['train_type'] != 'UNKNOWN']
            if known_types:
                # Remplacer les UNKNOWN par le type le plus fréquent parmi les connus
                most_common_type = max(set(known_types), key=known_types.count)
                for trip in trips_data:
                    if trip['train_type'] == 'UNKNOWN':
                        trip['train_type'] = most_common_type
            
            # Calculer les statistiques par type de train
            train_type_counts = {}
            for trip in trips_data:
                train_type = trip['train_type']
                train_type_counts[train_type] = train_type_counts.get(train_type, 0) + 1
            
            G_city[u][v]['trips'] = trips_data
            G_city[u][v]['nb_trips'] = len(trips_data)
            G_city[u][v]['train_types'] = train_type_counts
            
            # Type dominant pour cette arête
            dominant_type = max(train_type_counts, key=train_type_counts.get)
            G_city[u][v]['dominant_train_type'] = dominant_type

            # Cleaning
            del data['depart_com']
            del data['arrival_com']

    print(f"[DEBUG] Isolated nodes: {len(list(nx.isolates(G_city)))}")
    G_city.remove_nodes_from(list(nx.isolates(G_city)))
    print(f"[DEBUG] TERMINÉ - {stops_without_gare} arrêts sans gare associée, {gares_without_city} gares sans ville associée.")
    sys.stdout.flush()
    return G_city
    

def plot_interactive_city_graph(G_city, cities, aires):
    """
    Crée un graphe interactif avec Plotly - Version avec points virtuels pour hover des arêtes
    Basée sur la solution du forum Plotly
    """
    # Position des nœuds
    if config.country == 'france':
        pos = {str(row[config.key]): (aires[row['code_aire']]['geometry'].x, aires[row['code_aire']]['geometry'].y) if config.by_agg else (row.geometry.x, row.geometry.y)
            for _, row in cities.iterrows() if row[config.key] in G_city.nodes}
    elif config.country == 'switzerland':
        # Conversion vers 4326 pour l'affichage
        cities = cities.to_crs(epsg=4326)
        pos = {str(row[config.key]): (row.geometry.x, row.geometry.y)
               for _, row in cities.iterrows() if row[config.key] in G_city.nodes}
    else:
        raise ValueError(f"Pays inconnu pour affichage graphe: {config.country}")
    
    # Debug: vérifier si les arêtes ont des informations trips
    edges_with_trips = 0
    total_edges = 0
    for edge in G_city.edges():
        total_edges += 1
        if 'trips' in G_city[edge[0]][edge[1]]:
            edges_with_trips += 1
    print(f"[DEBUG] {edges_with_trips}/{total_edges} arêtes ont des informations trips")

    # Préparer les données des arêtes selon la méthode du forum
    edge_x = []
    edge_y = []
    edge_hover_x = []  # Points virtuels pour hover
    edge_hover_y = []
    edge_hover_text = []
    edge_widths = []
    
    for edge in G_city.edges(data=True):
        node1, node2, edge_data = edge
        if node1 in pos and node2 in pos:
            x0, y0 = pos[node1]
            x1, y1 = pos[node2]
            
            # Coordonnées des arêtes (avec None pour séparer)
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Point virtuel au milieu pour le hover
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            edge_hover_x.append(mid_x)
            edge_hover_y.append(mid_y)
            
            # Récupérer les informations de l'arête
            distance = edge_data.get('distance', 0)
            duration = edge_data.get('duration', 0)
            nb_trips = edge_data.get('nb_trips', 0)
            trips_info = edge_data.get('trips', [])
            
            city1_name = G_city.nodes[node1]['name']
            city2_name = G_city.nodes[node2]['name']
            
            # Créer le texte de hover
            hover_text = f"<b>{city1_name} ↔ {city2_name}</b><br>"
            hover_text += f"Distance: {distance:.1f} km<br>"
            hover_text += f"Durée: {duration:.0f} min<br>"
            hover_text += f"<b>Trajets ({nb_trips}):</b><br>"
            
            if trips_info:
                # Grouper les trajets par route_id pour plus de clarté
                routes = {}
                for trip in trips_info:
                    route_id = trip.get('route_id', 'Unknown')
                    if route_id not in routes:
                        routes[route_id] = []
                    routes[route_id].append(trip)
                
                for route_id, route_trips in list(routes.items())[:3]:  # Max 3 routes
                    hover_text += f"<br><b>Route {route_id}:</b><br>"
                    for i, trip in enumerate(route_trips[:2]):  # Max 2 trips par route
                        hover_text += f"• {trip.get('departure', 'N/A')} → {trip.get('arrival', 'N/A')}<br>"
                        hover_text += f"  {trip.get('stop1', 'N/A')} → {trip.get('stop2', 'N/A')}<br>"
                    if len(route_trips) > 2:
                        hover_text += f"... +{len(route_trips)-2} autres<br>"
                if len(routes) > 3:
                    hover_text += f"<br>... et {len(routes)-3} autres routes<br>"
            
            edge_hover_text.append(hover_text)
            
            # Largeur basée sur le nombre de trajets
            width = max(1, min(6, 1 + nb_trips / 10))
            edge_widths.append(width)

    # Trace des arêtes (lignes sans hover)
    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        mode='lines',
        line=dict(width=2, color='rgba(255, 69, 0, 0.6)'),
        hoverinfo='none',  # Pas de hover sur les lignes
        showlegend=False,
        name=""
    )
    
    # Trace des points virtuels pour le hover des arêtes (solution du forum)
    edge_hover_trace = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode='markers',
        marker=dict(
            size=0.5,  # Taille très petite comme dans l'exemple
            color='rgba(0,0,0,0)'  # Transparent
        ),
        text=edge_hover_text,
        hovertemplate='%{text}<extra></extra>',
        hoverinfo='text',
        showlegend=False,
        name="",
        # Permettre la sélection du texte dans le hover
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font=dict(size=11, family="monospace"),  # Police monospace pour faciliter la sélection
            align="left"
        )
    )

    # Créer la trace des nœuds
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_colors = []

    for node in G_city.nodes():
        if node in pos:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G_city.nodes[node]
            node_name = node_data['name']
            node_text.append(node_name)
            
            adjacencies = list(G_city.neighbors(node))
            node_colors.append(len(adjacencies))
            
            # Info détaillée pour le hover des nœuds
            node_info_text = f"<b>{node_name}</b><br>"
            node_info_text += f"Code: {node}<br>"
            node_info_text += f"Population: {node_data['population']}<br>"
            node_info_text += f"Connexions: {len(adjacencies)}<br>"
            if adjacencies:
                node_info_text += "<b>Connecté à:</b><br>"
                for neighbor in adjacencies[:5]:  # Max 5 voisins affichés
                    neighbor_name = G_city.nodes[neighbor]['name']
                    edge_data = G_city[node][neighbor]
                    nb_trips = edge_data.get('nb_trips', 0)
                    node_info_text += f"• {neighbor_name} ({nb_trips} trajets)<br>"
                if len(adjacencies) > 5:
                    node_info_text += f"... et {len(adjacencies)-5} autres<br>"
            
            node_info.append(node_info_text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=15,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                thickness=15, 
                title="Nb connexions"
            ),
            line=dict(width=2, color='white')
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color="black"),
        hoverinfo='text',
        hovertext=node_info,
        hovertemplate='%{hovertext}<extra></extra>',
        showlegend=False,
        name="Villes",
        # Améliorer le hover des nœuds aussi
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="darkblue",
            font=dict(size=11, family="monospace"),
            align="left"
        )
    )

    # Créer la figure selon la méthode du forum
    fig = go.Figure(
        data=[edge_trace, node_trace, edge_hover_trace],  # Ordre important !
        layout=go.Layout(
            title=dict(
                text='Graphe ferroviaire de France', 
                font=dict(size=16),
                x=0.5
            ),
            showlegend=False,
            hovermode='closest',  # Important pour la détection du hover
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Survolez les arêtes pour voir les détails des trajets",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240, 240, 240, 0.1)'
        )
    )

    print(f"[DEBUG] Figure créée avec {len(edge_hover_x)} points virtuels pour le hover des arêtes")
    
    fig.show()
    return fig