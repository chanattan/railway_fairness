import xml.etree.ElementTree as ET
import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt
import threading
import json
import os
import re

# Ce fichier contient des fonctions pour charger et traiter des données NeTEx, ainsi que pour construire un graphe de villes basé sur les arrêts de transport.
# Il est un équivalent de data_test.py mais pour le format NeTEx.
# Pour l'instant, il ne fonctionne pas car il produit 0 liaisons entre villes.
# Pour corriger cela, il faut s'assurer que les arrêts sont correctement extraits et que les relations entre les villes et les arrêts sont établies.

# TODO list
# 1. Vérifier que les arrêts sont correctement extraits du fichier NeTEx.
# 2. Filtrer le fichier des villes ou aggréger les communes avec la ville principale, e.g., Villeurbanne -> Lyon.
# 3. Vérifier les extractions des séquences de trajets et s'assurer qu'elles sont correctes.
# 4. Trouver un moyen d'extraire les bonnes distances de trajets, par exemple en utilisant Google Route API pour chaque ligne directe.
# 5. Ajouter des tests unitaires pour les fonctions de chargement et de traitement des données.
# 6. Affichage des lignes en GIS (bon relief, routes, etc.).
# 7. Optimiser la performance.
# 8. Export dans le bon format. 

"""
    Avec cette nouvelle version du code, voici les problèmes majeurs :
    1. Le fichier des communes actuel ne considère pas les pôles principaux comme des communes.
    2. Il n'est pas sûr que lors de la création des arêtes, l'association gare -> ville soit correcte.
    3. Il est affiché des villes post-filtrage qui n'ont pas nécessairement de gare associée (peut-être).
    4. Le fichier des villes ne contient pas de population et encore moins les coordonnées géographiques exactes.
    5. Le code commence à être trop complexe et pourrait être simplifié.

    Voici en théorie, pour la France, ce dont est besoin pour construire un graphe ferroviaire :
    - Un fichier GeoJSON des agglomérations françaises avec les colonnes suivantes :
        - com_code : code INSEE de la commune
        - com_name : nom de la commune
        - geometry : géométrie de la commune (centroïde ou polygone)
        - population : population de la commune (optionnel, mais utile pour filtrer les villes importantes)
        Cela concerne les villes qui sont des pôles principaux.
    - Un fichier GeoJSON des gares de voyageurs avec les colonnes suivantes :
        - codeinsee : code INSEE de la commune associée à la gare
        - gare_name : nom de la gare
        - geometry : géométrie de la gare (point)
    - Un fichier NeTEx contenant les séquences de trajets, avec les arrêts géolocalisés.

    Globalement, la logique est la suivante :
    1. Charger les villes depuis le GeoJSON des communes françaises. Filtrer.
    2. Charger les gares depuis le GeoJSON des gares de voyageurs.
    3. Charger le fichier NeTEx et extraire les séquences de trajets.
    4. Créer une correspondance entre les gares et les arrêts du NeTEx.
    5. Créer une correspondance entre les gares vers une ville.
    Cela permet pour un trajet direct (sans arrêt intermédiaire) d'avoir les stops donc les gares donc les villes qui sont les noeuds.
    6. Construire le graphe des villes avec les arêtes basées sur les trajets directs entre gares.
"""

name_attr = "nom_standard" # Nom de la colonne pour le nom de la ville dans les GeoDataFrame

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
import json
import geopandas as gpd
from shapely.geometry import Point

def load_cities_bis(csv_path, pop_threshold=50000):
    # Lecture CSV
    dtype = {
        'code_insee': str,
        'nom_standard': str,
        'dep_code': str,
        'population': float,
        'latitude_centre': float,
        'longitude_centre': float,
        'latitude_mairie': float,
        'longitude_mairie': float,
        'canton_code': str,
        'epci_code': str,
        'code_unite_urbaine': str,
        'nom_unite_urbaine': str
    }
    df = pd.read_csv(csv_path, dtype=dtype)

    df['com_code'] = df['code_insee'].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Filtrage sur population
    df = df[df['population'] >= pop_threshold]

    # replace the NaN latitude_centre and longitude_centre are NaN for a city, use latitude_mairie and longitude_mairie
    df['latitude_centre'].fillna(df['latitude_mairie'], inplace=True)
    df['longitude_centre'].fillna(df['longitude_mairie'], inplace=True)

    # si latitude_centre et longitude_centre sont toujours NaN, print un message de debug avec le nombre de villes concernées
    if df['latitude_centre'].isna().any() or df['longitude_centre'].isna().any():
        print(f"[DEBUG] {df['latitude_centre'].isna().sum()} villes ont latitude_centre NaN, {df['longitude_centre'].isna().sum()} villes ont longitude_centre NaN")

    # Oublier les colonnes inutiles
    df = df[['com_code', name_attr, 'dep_code', 'population', 'latitude_centre', 'longitude_centre']]

    # Création géométrie
    geometry = [Point(xy) for xy in zip(df['longitude_centre'], df['latitude_centre'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Debug
    print(f"[DEBUG] {len(gdf)} communes chargées après filtre population ≥ {pop_threshold}")
    for city in ["Paris", "Toulouse"]:
        if city in gdf['nom_standard'].values:
            print(f"✅ {city} trouvée")
            # print the row
            print(gdf[gdf['nom_standard'] == city])
        else:
            print(f"❌ {city} manquante")

    return gdf

def load_cities(cities_fp, pop_threshold=100000):
    """
    Charge le GeoJSON des communes françaises et adapte les colonnes au format attendu.
    Filtre les villes avec population > pop_threshold (défaut 100000).
    """
    import json
    import geopandas as gpd
    from shapely.geometry import shape
    #import json

    with open(cities_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Clés racine du JSON :", data.keys())
    print("Type de data :", type(data))

    if "features" in data:
        print("Nombre de features :", len(data["features"]))
    elif isinstance(data, list):
        print("Nombre d'éléments (liste racine) :", len(data))
    else:
        for k, v in data.items():
            if isinstance(v, list):
                print(f"Clé '{k}' contient {len(v)} éléments")


    features = data.get('features', [])
    rows = []

    for feat in features:
        props = feat.get('properties', {})

        # Code INSEE
        code = props.get('com_code') or props.get('insee')
        if isinstance(code, list):
            code = code[0]

        # Nom
        name = props.get(name_attr) or props.get('nom') or props.get('name')
        if isinstance(name, list):
            name = name[0]

        # Population
        pop = props.get('population')
        if isinstance(pop, list):
            pop = pop[0]

        # Code département
        dep = props.get('dep_code') or props.get('department')
        if isinstance(dep, list):
            dep = dep[0]

        # Géométrie
        geom = None
        if feat.get('geometry') and feat['geometry'] is not None:
            try:
                geom = shape(feat['geometry'])
            except Exception:
                pass  # on laisse geom = None
        elif 'latitude_centre' in props and 'longitude_centre' in props:
            try:
                geom = Point(float(props['longitude_centre']), float(props['latitude_centre']))
            except Exception:
                pass

        rows.append({
            'com_code': code,
            name_attr: name,
            'geometry': geom,
            'population': pop,
            'dep_code': dep
        })

    print(f"[DEBUG] Nombre de communes sans géométrie : {sum(1 for r in rows if r['geometry'] is None)}")
    print("Type de rows :", type(rows))
    print("Exemple ligne 0 :", rows[0])
    print("Toutes les clés :")
    from collections import Counter
    print(Counter(tuple(sorted(r.keys())) for r in rows))

    cities = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')

    # Vérification finale
    if cities['com_code'].isnull().any():
        print('[WARNING] Certaines communes n’ont pas de code INSEE.')
    if cities[name_attr].isnull().any():
        print('[WARNING] Certaines communes n’ont pas de nom.')

    # --- Vérification présence Paris / Toulouse ---
    for city in ["Paris", "Toulouse"]:
        if city in cities[name_attr].values:
            print(f"[OK] {city} est présent dans les données.")
        else:
            print(f"[ERREUR] {city} est absent des données.")

    # --- Filtre sur population ---
    threshold = 20000  # exemple : villes de + de 20 000 habitants
    cities_filtered = cities[cities["population"].fillna(0) >= threshold]

    print(f"[INFO] Nombre de communes après filtre population >= {threshold}: {len(cities_filtered)}")
    return cities

# ----------------------
# Chargement des gares (gares-de-voyageurs.geojson)
# ----------------------
def load_gares(gares_fp):
    """
    Charge le GeoJSON des gares et adapte les colonnes au format attendu.
    gares_fp : chemin vers gares-de-voyageurs.geojson
    Retourne un DataFrame avec colonnes : codeinsee, gare_name, geometry
    """
    gares = gpd.read_file(gares_fp)
    # Vérification des colonnes typiques
    if "codeinsee" not in gares.columns:
        if "insee" in gares.columns:
            gares = gares.rename(columns={"insee": "codeinsee"})
        else:
            raise ValueError("Le fichier des gares doit contenir la colonne 'codeinsee' ou 'insee'.")
    if "gare_name" not in gares.columns:
        if "nom" in gares.columns:
            gares = gares.rename(columns={"nom": "gare_name"})
        elif "name" in gares.columns:
            gares = gares.rename(columns={"name": "gare_name"})
        else:
            raise ValueError("Le fichier des gares doit contenir la colonne 'gare_name', 'nom' ou 'name'.")
    # Vérification de la géométrie
    if gares.crs is None:
        gares.set_crs(epsg=4326, inplace=True)
    return gares

# ----------------------
# Associer villes ↔ gares
# ----------------------
def associate_stations_to_cities(cities_gdf, gares_df):
    """
    Associe chaque gare à la commune correspondante (via le code INSEE).
    cities_gdf : GeoDataFrame des communes avec colonne 'com_code'
    gares_df : DataFrame des gares avec colonne 'codeinsee'
    Retourne un DataFrame enrichi
    """
    # Vérif colonnes
    if 'com_code' not in cities_gdf.columns:
        raise ValueError("cities_gdf doit contenir la colonne 'com_code'")
    if 'codeinsee' not in gares_df.columns:
        raise ValueError("gares_df doit contenir la colonne 'codeinsee'")
    # Merge sur le code INSEE
    # Ok, en fait le code insee dans les gares est trop précis, alors que dans le code insee commune c'est plus pour la région de la commune globale.
    # Ce qu'il faut faire c'est prendre le code département dep_code dans communes et voir si ça correspond au préfixe du code insee de la gare.
    gares_df['dep_code'] = gares_df['codeinsee'].str[:2]
    cities_gdf['dep_code'] = cities_gdf['dep_code'].apply(lambda x: x[0] if isinstance(x, list) else x)
    
    # Bon en fait, check le fichier communes car il contient même pas les villes.

    gares_with_city = gares_df.merge(
        cities_gdf[['com_code', name_attr, 'dep_code', 'geometry']],
        left_on='dep_code',
        right_on='dep_code',
        how='left'
    )
    print(f"[DEBUG] Extrait des gares avec ville associée : {list(gares_with_city.head(5).to_dict('records'))}")
    print(f"[DEBUG] Nombre de valeurs na dans la colonne name_attr : {gares_with_city[name_attr].isna().sum()}")
    # Filtre les gares sans ville associée
    gares_with_city = gares_with_city[gares_with_city[name_attr].notna()]
    return gares_with_city

# ----------------------
# Vérification globale des fichiers
# ----------------------
def check_geojson_format(fp, expected_cols):
    """
    Vérifie qu'un fichier GeoJSON contient les colonnes attendues.
    fp : chemin du fichier
    expected_cols : liste de noms de colonnes
    """
    if not os.path.exists(fp):
        print(f"[ERREUR] Fichier introuvable : {fp}")
        return False
    try:
        gdf = gpd.read_file(fp)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire {fp} : {e}")
        return False
    missing = [col for col in expected_cols if col not in gdf.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes dans {fp} : {missing}")
        return False
    print(f"[OK] {fp} contient les colonnes requises : {expected_cols}")
    return True

# ----------------------
# Parse du fichier NeTEx
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

def load_netex_bis(netex_file):
    """
    Extraction exhaustive et propre des séquences de trajets Netex,
    fusionnées par trip_id avec horaires.
    """

    # Enlever critère géographique dans gares -> villes
    # Voir comment inclure des lignes avec villes intermédiaires manquantes ou prendre exhaustivement
    import datetime
    import xml.etree.ElementTree as ET
    import pandas as pd

    tree = ET.parse(netex_file)
    root = tree.getroot()

    # ---- Extraction arrêts physiques (StopPlace + Quay + ScheduledStopPoint) ----
    stops_data = []
    def add_stop(stop_id, stop_name, lat, lon):
        if stop_id and lat and lon:
            stops_data.append({
                "stop_id": stop_id,
                "stop_name": stop_name,
                "stop_lat": float(lat),
                "stop_lon": float(lon)
            })

    for sp in root.findall(".//{*}StopPlace"):
        stop_id = sp.attrib.get("id")
        name = sp.findtext("{*}Name")
        centroid = sp.find(".//{*}Centroid/{*}Location")
        lat = centroid.findtext("{*}Latitude") if centroid is not None else None
        lon = centroid.findtext("{*}Longitude") if centroid is not None else None
        add_stop(stop_id, name, lat, lon)
        for quay in sp.findall(".//{*}Quay"):
            quay_id = quay.attrib.get("id")
            quay_name = quay.findtext("{*}Name")
            centroid = quay.find(".//{*}Centroid/{*}Location")
            lat = centroid.findtext("{*}Latitude") if centroid is not None else None
            lon = centroid.findtext("{*}Longitude") if centroid is not None else None
            add_stop(quay_id, quay_name, lat, lon)

    for ssp in root.findall(".//{*}ScheduledStopPoint"):
        ssp_id = ssp.attrib.get("id")
        name = ssp.findtext("{*}Name")
        centroid = ssp.find(".//{*}Centroid/{*}Location")
        lat = centroid.findtext("{*}Latitude") if centroid is not None else None
        lon = centroid.findtext("{*}Longitude") if centroid is not None else None
        add_stop(ssp_id, name, lat, lon)

    stops = pd.DataFrame(stops_data).drop_duplicates("stop_id")

    # ---- Mapping ScheduledStopPoint -> arrêt physique ----
    ssp_map = build_ssp_map(root)

    # ---- Mapping PointInJourneyPatternRef -> ScheduledStopPointRef ----
    pijp_to_ssp = {}
    for sjp in root.findall(".//{*}ServiceJourneyPattern"):
        for spjp in sjp.findall(".//{*}pointsInSequence/{*}StopPointInJourneyPattern"):
            pijp_id = spjp.attrib.get("id")
            ssp_ref = spjp.find("{*}ScheduledStopPointRef")
            if pijp_id and ssp_ref is not None:
                pijp_to_ssp[pijp_id] = ssp_ref.attrib.get("ref")

    # ---- Mapping StopPointInServiceJourneyPattern -> ScheduledStopPointRef ----
    pijp_service_to_ssp = {}
    for sjs in root.findall(".//{*}StopPointInServiceJourneyPattern"):
        pijp_id = sjs.attrib.get("id")
        ssp_ref = sjs.find("{*}ScheduledStopPointRef")
        if pijp_id and ssp_ref is not None:
            pijp_service_to_ssp[pijp_id] = ssp_ref.attrib.get("ref")

    # ---- Extraction séquences par trip_id ----
    trip_sequences, trip_times = {}, {}

    def parse_time(t):
        if not t:
            return None
        try:
            return datetime.datetime.strptime(t, "%H:%M:%S")
        except ValueError:
            return None

    def add_trip_sequence(trip_id, seq, times=None):
        if len(seq) >= 2:
            trip_sequences[trip_id] = seq
            if times:
                trip_times[trip_id] = times

    for sj in root.findall(".//{*}ServiceJourney"):
        sj_id = sj.attrib.get("id")
        seq, times = [], []
        passing = sj.find("{*}passingTimes")
        if passing is not None:
            for tpt in passing.findall("{*}TimetabledPassingTime"):
                pijp_ref_el = tpt.find("{*}PointInJourneyPatternRef")
                if pijp_ref_el is None:
                    continue
                pijp_ref = pijp_ref_el.attrib.get("ref")

                # Prendre d'abord mapping ServiceJourneyPattern puis JourneyPattern
                ssp_id = pijp_service_to_ssp.get(pijp_ref) or pijp_to_ssp.get(pijp_ref)
                stop_id = ssp_map.get(ssp_id, ssp_id) if ssp_id else None
                if not stop_id:
                    continue

                arr = tpt.findtext("{*}ArrivalTime")
                dep = tpt.findtext("{*}DepartureTime")
                seq.append(stop_id)
                times.append((parse_time(arr), parse_time(dep)))

        add_trip_sequence(sj_id, seq, times)

    # ---- Nettoyage doublons consécutifs ----
    for trip_id in list(trip_sequences.keys()):
        seq = trip_sequences[trip_id]
        cleaned_seq, cleaned_times = [seq[0]], [trip_times[trip_id][0]]
        for idx in range(1, len(seq)):
            if seq[idx] != cleaned_seq[-1]:
                cleaned_seq.append(seq[idx])
                cleaned_times.append(trip_times[trip_id][idx])
        if len(cleaned_seq) < 2:
            del trip_sequences[trip_id]
            trip_times.pop(trip_id, None)
        else:
            trip_sequences[trip_id] = cleaned_seq
            trip_times[trip_id] = cleaned_times

    # ---- DataFrame stop_times final ----
    stop_times_data = []
    for trip_id, seq in trip_sequences.items():
        times = trip_times.get(trip_id, [(None, None)] * len(seq))
        for i, (stop_id, (arr, dep)) in enumerate(zip(seq, times), 1):
            stop_times_data.append({
                "trip_id": trip_id,
                "stop_id": stop_id,
                "stop_sequence": i,
                "arrival_time": arr.strftime("%H:%M:%S") if arr else None,
                "departure_time": dep.strftime("%H:%M:%S") if dep else None
            })

    stop_times = pd.DataFrame(stop_times_data, columns=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"])
    print(f"DEBUG: {len(stops)} arrêts extraits, {len(stop_times)} lignes stop_times construites")
    if stop_times.empty:
        print("WARNING: stop_times est vide. Vérifier le mapping PointInJourneyPattern -> SSP.")
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
    Associe chaque ville à la liste des arrêts dont le nom contient le nom de la ville (insensible à la casse)
    ET qui sont situés à moins de buffer_km km du centroïde de la ville.
    Si aucun n'est trouvé, fallback sur la méthode géographique (le plus proche).
    Retourne un mapping {ville: [stop_id1, stop_id2, ...]}.
    """
    city_to_stations = {}
    stops_gdf = stops_gdf.dropna(subset=["stop_lat", "stop_lon", "geometry", "stop_name"])
    stops_proj = stops_gdf.to_crs(epsg=2154)
    for _, row in cities.iterrows():
        city = row[name_attr]
        city_point = gpd.GeoSeries([row.geometry.centroid], crs=cities.crs).to_crs(epsg=2154).iloc[0]
        # 1. Recherche par nom ET distance
        mask = stops_gdf["stop_name"].str.contains(city, case=False, na=False)
        stops_match = stops_gdf[mask]
        if not stops_match.empty:
            stops_match_proj = stops_match.to_crs(epsg=2154)
            stops_match_proj["dist"] = stops_match_proj.geometry.distance(city_point)
            close_stops = stops_match_proj[stops_match_proj["dist"] <= buffer_km * 1000]
            stop_ids = list(close_stops["stop_id"].unique())
            if stop_ids:
                city_to_stations[city] = stop_ids
                continue
        # 2. Fallback géographique (le plus proche si dans le buffer)
        stops_proj["dist"] = stops_proj.geometry.distance(city_point)
        nearest = stops_proj.iloc[stops_proj["dist"].idxmin()]
        dist_km = nearest["dist"] / 1000
        if dist_km <= buffer_km:
            city_to_stations[city] = [nearest["stop_id"]]
        else:
            #print(f"[DEBUG] Aucun arrêt trouvé dans le buffer pour la ville '{city}' (fallback géographique).")
            city_to_stations[city] = []
    return city_to_stations

def expand_city_station_map(city_station_map, cities, stops_gdf, buffer_km=5):
    """
    Pour chaque ville, associe tous les arrêts géolocalisés dans un buffer autour de la ville à cette ville.
    city_station_map : {ville: [stop_id1, stop_id2, ...]}
    Retourne un mapping {stop_id: ville}
    """
    stop_to_city = {}
    stops_gdf = stops_gdf.dropna(subset=["stop_lat", "stop_lon", "geometry"])
    stops_proj = stops_gdf.to_crs(epsg=2154)
    for city, stop_ids in city_station_map.items():
        city_geom = cities.loc[cities[name_attr] == city].geometry.iloc[0].centroid
        city_point = gpd.GeoSeries([city_geom], crs=cities.crs).to_crs(epsg=2154).iloc[0]
        stops_proj["dist"] = stops_proj.geometry.distance(city_point)
        nearby = stops_proj[stops_proj["dist"] <= buffer_km * 1000]
        for stop_id in nearby["stop_id"]:
            stop_to_city[stop_id] = city
        # Ajoute aussi tous les arrêts explicitement associés par nom
        for stop_id in stop_ids:
            stop_to_city[stop_id] = city
    return stop_to_city

# ----------------------
# Construire le graphe
# ----------------------
def build_city_graph(city_station_map, stops, stop_times, stop_to_city):
    import datetime
    G_city = nx.Graph()
    for city in city_station_map:
        G_city.add_node(city)

    merged = stop_times.sort_values(["trip_id", "stop_sequence"])
    n_with_time = 0
    n_total = 0

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
            # Calcul du temps de trajet si possible
            duration = None
            n_total += 1
            if s1.departure_time and s2.arrival_time:
                try:
                    t1 = datetime.datetime.strptime(s1.departure_time, "%H:%M:%S")
                    t2 = datetime.datetime.strptime(s2.arrival_time, "%H:%M:%S")
                    duration = (t2 - t1).total_seconds() / 60
                    if duration < 0:
                        duration += 24 * 60  # gestion passage minuit
                    n_with_time += 1
                except Exception as e:
                    print(f"[DEBUG] Erreur parsing horaires: {e} | trip_id={trip_id} | s1.departure_time={s1.departure_time} | s2.arrival_time={s2.arrival_time}")
                    duration = None
            # Si l'arête existe déjà, on garde la distance/temps min
            if G_city.has_edge(c1, c2):
                if dist < G_city[c1][c2]["distance"]:
                    G_city[c1][c2]["distance"] = dist
                if duration is not None:
                    if (G_city[c1][c2].get("duration") is None) or (duration < G_city[c1][c2]["duration"]):
                        G_city[c1][c2]["duration"] = duration
            else:
                G_city.add_edge(c1, c2, distance=dist, duration=duration)
    # Supprimer les nodes sans arêtes
    nodes_to_remove = [n for n in G_city.nodes if G_city.degree(n) == 0]
    G_city.remove_nodes_from(nodes_to_remove)
    print(f"[DEBUG] {n_with_time} arêtes avec temps de trajet sur {n_total} arêtes possibles.")
    return G_city

def build_city_graph_from_netex(netex_fp, cities_fp, buffer_km_station=10, buffer_km_expand=5):
    """
    Construit un graphe de villes à partir d'un fichier Netex, où les noeuds sont les villes
    (matching via les fonctions existantes) et une arête existe s'il y a un train entre deux villes consécutives.
    """
    # Charger Netex
    stops, stop_times = load_netex_bis(netex_fp)
    # Charger les villes
    cities = load_cities_bis(cities_fp, 50000)
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326"
    )
    city_station_map = associate_cities_to_stations(cities, stops_gdf, buffer_km=buffer_km_station)
    stop_to_city = expand_city_station_map(city_station_map, cities, stops_gdf, buffer_km=buffer_km_expand)
    G_city = build_city_graph(city_station_map, stops, stop_times, stop_to_city)
    return G_city, cities

def match_gares_to_stops(gares_df, stops_df, max_dist_km=2):
    """
    Associe à chaque gare le(s) stop_id Netex dont le nom correspond (insensible à la casse)
    et qui sont situés à moins de max_dist_km km de la géométrie de la gare.
    Ajoute une colonne 'stop_id' (ou une liste de stop_id) à gares_df.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    # Créer GeoDataFrame pour les stops
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs="EPSG:4326"
    )
    gares_df = gares_df.copy()
    gares_df["stop_id"] = None
    stops_gdf_proj = stops_gdf.to_crs(epsg=2154)
    gares_df_proj = gares_df.to_crs(epsg=2154)
    for idx, gare in gares_df_proj.iterrows():
        nom_gare = gare["gare_name"]
        geom_gare = gare.geometry
        if geom_gare is None:
            continue
        # Filtrer les stops par nom (escape regex)
        mask = stops_gdf_proj["stop_name"].str.contains(re.escape(nom_gare), case=False, na=False)
        stops_match = stops_gdf_proj[mask]
        stops_match = stops_match.copy()  # Avoid SettingWithCopyWarning
        stops_match.loc[:, "dist"] = stops_match.geometry.distance(geom_gare)
        close_stops = stops_match[stops_match["dist"] <= max_dist_km * 1000]
        stop_ids = list(close_stops["stop_id"].unique())
        if stop_ids:
            gares_df.at[idx, "stop_id"] = stop_ids if len(stop_ids) > 1 else stop_ids[0]
    return gares_df

def build_city_graph_from_netex_with_station_merge(netex_fp, cities_fp, gares_fp, buffer_km_expand=5):
    """
    Construit un graphe de villes à partir d'un fichier Netex, où les noeuds sont les villes ayant une gare associée
    (via le code INSEE, merge entre villes et gares), et une arête existe s'il y a un train entre deux gares consécutives.
    """
    # Charger Netex
    stops, stop_times = load_netex_bis(netex_fp)
    # Charger les villes et gares
    cities_gdf = load_cities_bis(cities_fp, 50000)
    print(f"[DEBUG] {len(cities_gdf)} villes chargées depuis {cities_fp}")
    gares_df = load_gares(gares_fp)
    print(f"[DEBUG] {len(gares_df)} gares chargées depuis {gares_fp}")
    # Associer gares à stops Netex
    gares_df = match_gares_to_stops(gares_df, stops)
    print(f"[DEBUG] {gares_df['stop_id'].notnull().sum()} gares associées à des arrêts Netex")
    # Associer gares à villes
    gares_with_city = associate_stations_to_cities(cities_gdf, gares_df)
    print(f"[DEBUG] {len(gares_with_city)} gares associées à des villes")
    # Filtrer les villes ayant au moins une gare
    villes_avec_gare = gares_with_city[name_attr].dropna().unique()
    cities_gdf_filtered = cities_gdf[cities_gdf[name_attr].isin(villes_avec_gare)].copy()
    print(f"[DEBUG] {len(cities_gdf_filtered)} villes filtrées avec au moins une gare")
    # Mapping gare_id -> ville
    stop_to_city = {}
    for _, row in gares_with_city.iterrows():
        stop_ids = row['stop_id']
        city = row[name_attr]
        if city is None or city == "nan" or pd.isna(city):
            continue
        if isinstance(stop_ids, list):
            for sid in stop_ids:
                if pd.notnull(sid) and pd.notna(sid) and sid is not None:
                    stop_to_city[sid] = city
        elif pd.notnull(stop_ids) and pd.notna(stop_ids) and stop_ids is not None:
            stop_to_city[stop_ids] = city
    print(f"[DEBUG] {len(stop_to_city)} arrêts associés à des villes")
    print(f"[DEBUG] Exemple de mapping stop_to_city : {list(stop_to_city.items())[:5]}")
    # Mapping ville -> liste de gares
    city_station_map = gares_with_city.groupby(name_attr)['stop_id'].apply(list).to_dict()
    # Filtrer les villes sans gare
    city_station_map = {city: stops for city, stops in city_station_map.items() if stops or (isinstance(stops, list) and len(stops) > 0)}
    print(f"[DEBUG] {len(city_station_map)} villes avec gares associées")
    print(f"[DEBUG] Exemple de mapping city_station_map : {list(city_station_map.items())[:5]}")
    # Construire le graphe
    G_city = build_city_graph(city_station_map, stops, stop_times, stop_to_city)
    return G_city, cities_gdf_filtered, stop_to_city, city_station_map

# ----------------------
# Visualisation
# ----------------------
def plot_city_graph(G_city, cities):
    import pandas as pd
    # Only keep cities with valid names and centroids
    valid_cities = cities[cities[name_attr].notnull()]
    pos = {row[name_attr]: (row.geometry.centroid.x, row.geometry.centroid.y)
           for _, row in valid_cities.iterrows() if row[name_attr] in G_city.nodes}
    # Remove nodes with nan from the graph
    G_city.remove_nodes_from([n for n in G_city.nodes if pd.isnull(n)])
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(G_city, pos, alpha=0.3)
    nx.draw_networkx_nodes(G_city, pos, node_size=20, node_color="red")
    nx.draw_networkx_labels(G_city, pos, font_size=6)
    plt.show(block=False)

# ----------------------
# Vérifier ligne directe
# ----------------------
def has_direct_train_between(cities, stop_times, stop_to_city, ville1, ville2):
    """
    Retourne True s'il existe un trip où une station de ville1 est suivie immédiatement d'une station de ville2 (ou l'inverse).
    Affiche aussi les trips concernés si found=True.
    """
    # Récupérer tous les stop_id associés à chaque ville
    stops_ville1 = {stop_id for stop_id, city in stop_to_city.items() if city == ville1}
    stops_ville2 = {stop_id for stop_id, city in stop_to_city.items() if city == ville2}
    found = False
    trips_found = set()
    for trip_id, group in stop_times.groupby("trip_id"):
        seq = list(group.sort_values("stop_sequence")["stop_id"])
        for i in range(len(seq) - 1):
            if (seq[i] in stops_ville1 and seq[i+1] in stops_ville2) or (seq[i] in stops_ville2 and seq[i+1] in stops_ville1):
                found = True
                trips_found.add(trip_id)
    return found, trips_found

def prompt_train_between(cities, stop_times, stop_to_city, G_city, city_station_map, stops=None):
    """
    Demande à l'utilisateur deux villes, indique s'il existe une ligne directe, sinon affiche le plus court chemin et la gare associée à chaque ville.
    Affiche aussi la distance et le temps de trajet (si dispo) pour la liaison directe ou le plus court chemin.
    Retourne False si l'utilisateur entre 'x' pour quitter, True sinon.
    """
    ville1 = input("Première ville (ou 'x' pour quitter) : ").strip()
    if ville1.lower() == 'x':
        return False
    ville2 = input("Deuxième ville (ou 'x' pour quitter) : ").strip()
    if ville2.lower() == 'x':
        return False
    if ville1 not in G_city.nodes or ville2 not in G_city.nodes:
        print(f"{ville1} ou {ville2} n'est pas dans le graphe des villes.")
        return True
    found, trips = has_direct_train_between(cities, stop_times, stop_to_city, ville1, ville2)
    if found and G_city.has_edge(ville1, ville2):
        d = G_city[ville1][ville2].get("distance")
        t = G_city[ville1][ville2].get("duration")
        print(f"Il existe une ligne directe entre {ville1} et {ville2}.")
        if d is not None:
            print(f"  Distance : {d:.1f} km")
        if t is not None:
            print(f"  Temps de trajet : {t:.0f} min")
    else:
        print(f"Pas de ligne directe entre {ville1} et {ville2}.")
        try:
            path = nx.shortest_path(G_city, ville1, ville2, weight="distance")
            print(f"Plus court chemin ({len(path)-1} arêtes) : {' -> '.join(path)}")
            total_dist = 0
            total_time = 0
            has_time = True
            for i in range(len(path)-1):
                e = G_city[path[i]][path[i+1]]
                d = e.get("distance")
                t = e.get("duration")
                if d is not None:
                    total_dist += d
                if t is not None:
                    total_time += t
                else:
                    has_time = False
            print(f"  Distance totale : {total_dist:.1f} km")
            if has_time and total_time > 0:
                print(f"  Temps total estimé : {total_time:.0f} min")
            for v in path:
                gares = city_station_map.get(v, None)
                if gares and stops is not None:
                    if isinstance(gares, list):
                        noms = []
                        for gid in gares:
                            stop_row = stops.loc[stops['stop_id'] == gid]
                            if not stop_row.empty:
                                noms.append(stop_row.iloc[0]['stop_name'])
                            else:
                                noms.append(gid)
                        if noms:
                            print(f"  - {v} : gares principales {', '.join(noms)}")
                        else:
                            print(f"  - {v} : aucune gare associée")
                    else:
                        stop_row = stops.loc[stops['stop_id'] == gares]
                        if not stop_row.empty:
                            stop_name = stop_row.iloc[0]['stop_name']
                            print(f"  - {v} : gare principale {stop_name}")
                        else:
                            print(f"  - {v} : gare principale {gares}")
                elif gares:
                    if isinstance(gares, list):
                        print(f"  - {v} : gares principales {', '.join(gares)}")
                    else:
                        print(f"  - {v} : gare principale {gares}")
                else:
                    print(f"  - {v} : pas de gare associée")
        except nx.NetworkXNoPath:
            print(f"Aucun chemin entre {ville1} et {ville2} dans le graphe.")
    return True

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    netex_fp = "resources/netex/france/sncf-netex.xml"
    #cities_fp = "resources/villes_france.geojson"
    cities_fp = "resources/communes-france-datagouv-2025.csv" # Entièreté des communes de France

    # Charger Netex et afficher diagnostic
    stops, stop_times = load_netex_bis(netex_fp)
    #print("Exemples de stop_id dans stop_times:", stop_times['stop_id'].unique()[:10])
    #print("Exemples de stop_id dans stops:", stops['stop_id'].unique()[:10])
    n_match = sum(stop_times['stop_id'].isin(stops['stop_id']))
    print(f"Nombre de stop_id de stop_times présents dans stops : {n_match} / {len(stop_times)}")

    # Charger les villes
    cities = load_cities_bis(cities_fp, 50000)
    print(f"Nombre de villes chargées : {len(cities)}")
    print(f"Villes contenant Paris : {cities[cities[name_attr].str.contains('Paris', case=False, na=False)]}")
    #Villes contenant Paris :       com_code nom_standard dep_code  ...  latitude_centre  longitude_centre         geometry
                            #29244    75056        Paris       75  ...              NaN               NaN  POINT (NaN NaN)

    print("Exemples de villes :", list(cities['nom_standard'].unique())[:10])
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326"
    )
    #city_station_map = associate_cities_to_stations(cities, stops_gdf, buffer_km=10)
    #print("Exemples de correspondances ville-gare :", list(city_station_map.items())[:5])
    #stop_to_city = associate_stations_to_cities(cities, stops_gdf, buffer_km=10)
    #print("Exemples de correspondances gare-ville :", list(stop_to_city.items())[:5])
    #G_city = build_city_graph(city_station_map, stops, stop_times, stop_to_city)
    G_city, cities_gdf_filtered, stop_to_city, city_station_map = build_city_graph_from_netex_with_station_merge(
        netex_fp, cities_fp, gares_fp="resources/gares-de-voyageurs.geojson"
    )
    print(f"{G_city.number_of_nodes()} villes, {G_city.number_of_edges()} liaisons directes.")

    # Affichage du graphe en mode non bloquant (thread principal)
    plot_city_graph(G_city, cities_gdf_filtered)

    # --- Check Rennes ↔ Brest ---
    ville1 = "Paris"
    ville2 = "Amiens"
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

    # --- Vérification ligne directe ---
    found, trips = has_direct_train_between(cities, stop_times, stop_to_city, ville1, ville2)
    if found:
        print(f"Il existe une ligne directe entre {ville1} et {ville2} dans les séquences de stop_times.")
    else:
        print(f"Aucune ligne directe entre {ville1} et {ville2} dans les séquences de stop_times.")

    # s'il n'existe pas les villes intermédiaires dans une ligne de train alors on met quand même
    # la ligne et on garde bien la distance et temps de trajet correspondants
    # --- Interface utilisateur ---
    while True:
        if not prompt_train_between(cities, stop_times, stop_to_city, G_city, city_station_map, stops=stops):
            break