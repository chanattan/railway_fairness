from turtle import pos
from config import *
from graph import build_city_graph, build_city_graph_with_trips, plot_city_graph, plot_interactive_city_graph
import zipfile
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
from data_exporter import export_graph_for_web_dashboard

def load_gtfs(gtfs_zip):
    with zipfile.ZipFile(gtfs_zip, 'r') as z:
        stops = pd.read_csv(z.open('stops.txt'))
        stop_times = pd.read_csv(z.open('stop_times.txt'))
        trips = pd.read_csv(z.open('trips.txt'))
        routes = pd.read_csv(z.open('routes.txt'))
    return stops, stop_times, trips, routes

def load_cities(cities_fp):
    cities = gpd.read_file(cities_fp, dtype={"codeinsee": str})
    cities = cities.rename(columns={"cityLabel": "name"})
    if cities.crs is None:
        cities.set_crs(epsg=4326, inplace=True)
    return cities

import json
import geopandas as gpd
from shapely.geometry import Point

def load_cities_bis(csv_path):
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

    #df['com_code'] = df['code_insee'].apply(lambda x: x[0] if isinstance(x, list) else x)

    # replace the NaN latitude_centre and longitude_centre are NaN for a city, use latitude_mairie and longitude_mairie
    df['latitude_centre'].fillna(df['latitude_mairie'], inplace=True)
    df['longitude_centre'].fillna(df['longitude_mairie'], inplace=True)

    # si latitude_centre et longitude_centre sont toujours NaN, print un message de debug avec le nombre de villes concernées
    if df['latitude_centre'].isna().any() or df['longitude_centre'].isna().any():
        print(f"[DEBUG] {df['latitude_centre'].isna().sum()} villes ont latitude_centre NaN, {df['longitude_centre'].isna().sum()} villes ont longitude_centre NaN")

    # Oublier les colonnes inutiles
    df = df[[name_attr, 'code_unite_urbaine', 'nom_unite_urbaine', 'code_insee', 'population', 'latitude_centre', 'longitude_centre']]

    # Création géométrie
    geometry = [Point(xy) for xy in zip(df['longitude_centre'], df['latitude_centre'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Debug
    print(f"[DEBUG] {len(gdf)} communes chargées.")
    for city in ["Abancourt", "Toulouse"]:
        if city in gdf['nom_standard'].values:
            print(f"✅ {city} trouvée")
            # print the row
            print(gdf[gdf['nom_standard'] == city])
        else:
            print(f"❌ {city} manquante")

    return gdf

def resolve_stoparea_to_stoppoint(stop_id, stops_df, G_station):
    """If stop_id is a StopArea, pick a child stop that exists in G_station."""
    if stop_id in G_station.nodes:
        return stop_id
    if stop_id.startswith("StopArea"):
        children = stops_df[stops_df["parent_station"] == stop_id]["stop_id"].tolist()
        for child in children:
            if child in G_station.nodes:
                return child
    return None

def associate_stations_to_cities(cities_gdf, stops_to_gares_df, gares_df):
    """
    Associe chaque gare à la commune correspondante (via le code INSEE).
    cities_gdf : GeoDataFrame des communes avec colonne 'code_insee'
    gares_df : DataFrame des gares avec colonne 'codeinsee'
    Retourne un DataFrame enrichi
    """
    # Vérif colonnes
    if 'code_insee' not in cities_gdf.columns:
        raise ValueError("cities_gdf doit contenir la colonne 'code_insee'")
    if 'codeinsee' not in gares_df.columns:
        raise ValueError("gares_df doit contenir la colonne 'codeinsee'")
    
    # Merge sur le code INSEE les gares aux villes
    # Faire plutôt gares (qui ONT un STOP, à check avec stops_to_gares) qui n'ont pas de ville pour conserver gares et match second 
    gares_df['codeinsee'] = gares_df['codeinsee'].astype(str).str.zfill(5)
    cities_gdf['code_insee'] = cities_gdf['code_insee'].astype(str).str.zfill(5)

    # Réduire par le goulot d'étranglement des stops
    print(f"Nombre de gares préfiltrage stops : {len(gares_df)}")
    gares_df = gares_df[gares_df['code_uic'].isin(stops_to_gares_df['code_uic'])]
    print(f"Nombre de gares conservées : {len(gares_df)}")
    print(gares_df.head())

    # À la place de l'association par département, pour les entrées manquantes dans les communes,
    # on regarde au niveau des arrondissements pour rattacher à la ville.
    # On le fait en amont.
    df_communes = pd.read_csv("resources/v_commune_2025.csv")
    df_communes['COM'] = df_communes['COM'].apply(lambda x: str(int(x)) if isinstance(x, float) else str(x))
    df_communes['COMPARENT'] = df_communes['COMPARENT'].apply(lambda x: str(int(x)) if pd.notna(x) and isinstance(x, float) else str(x) if pd.notna(x) else None)

    # On supprime les communes centrales, i.e., où COMPARENT est vide
    mapping_equiv = df_communes.dropna(subset=['COMPARENT']).set_index('COM')['COMPARENT'].to_dict()

    gares_df['codeinsee_final'] = gares_df['codeinsee'].map(mapping_equiv).fillna(gares_df['codeinsee'])
    # Comptage des gares ayant trouvé une commune centrale
    nb_gares_centrale = (gares_df['codeinsee'] != gares_df['codeinsee_final']).sum()
    print(f"Nombre de gares ayant trouvé une commune centrale : {nb_gares_centrale}")
    print(f"[DEBUG] Extrait : {gares_df[gares_df['codeinsee'] != gares_df['codeinsee_final']].head()}")
    gares_df['codeinsee'] = gares_df['codeinsee_final']
    gares_df.drop(columns=['codeinsee_final'], inplace=True)

    # Distinction géométrie gare ville
    gares_df = gares_df.rename(columns={'geometry':'geometry_gare'})
    gares_with_cities = gares_df.merge(
        cities_gdf[['code_insee', name_attr, 'geometry']],
        left_on='codeinsee',
        right_on='code_insee',
        how='left'
    )
    print(f"[DEBUG] Paris Gare du Nord cities_with_gares:", "Paris Gare du Nord" in gares_with_cities['gare_name'].values)
    print(f"[DEBUG] Affichage pour Paris:", gares_with_cities[gares_with_cities['gare_name'].str.contains('Charles de Gaulle', case=False, na=False)][['gare_name', name_attr, 'codeinsee']])
    print(f"[DEBUG] Extrait cities_with_gares:", list(gares_with_cities.items())[:5])
    print(f"[DEBUG] {gares_with_cities[name_attr].isna().sum()}/{len(gares_with_cities)} gares sans ville associée.")
    
    cdg2 = gares_with_cities[gares_with_cities['gare_name'].str.contains('Charles de Gaulle', case=False, na=False)]
    tremblay = cities_gdf[cities_gdf['code_insee'] == '93073']
    # Afficher les premières valeurs pour voir le format exact
    print(cities_gdf['code_insee'].head(10).tolist())

    # Vérifier le type et la longueur de chaque code
    print(cities_gdf['code_insee'].apply(lambda x: (x, type(x), len(str(x)))).head(10).tolist())

    # Chercher Tremblay d'une autre façon
    print(cities_gdf[cities_gdf['nom_standard'] == 'Tremblay-en-France'])


    print("CDG2 codeinsee (gares_df) :", cdg2['codeinsee'].values, cdg2['codeinsee'].dtype)
    print("Tremblay code_insee (cities_gdf) :", tremblay['code_insee'].values, tremblay['code_insee'].dtype)

    # Vérifier si le merge serait possible
    print("Correspondance exacte ?", cdg2['codeinsee'].values[0] == tremblay['code_insee'].values[0])
    
    # Il se peut que certaines gares n'aient pas de ville, auquel cas on essaie de les associer par département
    gares_without_city = gares_with_cities[gares_with_cities[name_attr].isna()] # Toutes les gares ont un codeinsee, pas forcément une ville : name_attr
    
    print(f"[DEBUG] Paris Gare du Nord cities_without_gares:", "Paris Gare du Nord" in gares_without_city['gare_name'].values)
    if not gares_without_city.empty and False:
        pass
        """ Association par département, imprécis
        print("[DEBUG] Tentative de merge par département...")
        gares_with_cities['dep'] = gares_with_cities['codeinsee'].str[:2]
        cities_gdf['dep'] = cities_gdf['code_insee'].astype(str).str[:2]

        mask_missing = gares_with_cities[name_attr].isna()
        #mask_communes = cities_gdf['typecom'].str.
        gares_with_cities = gares_with_cities.loc[mask_missing].drop(columns=[name_attr, 'codeinsee', 'geometry', 'code_insee'])  # Important pour ne pas dupliquer les attributs dans le merge

        # On garde le code insee de la ville
        gares_with_cities = gares_with_cities.merge(
            cities_gdf[['code_insee', name_attr, 'dep', 'geometry']],
            on='dep',
            how='left'
        )

        print(f"[DEBUG] Paris Gare du Nord merged_dep :", "Paris Gare du Nord" in gares_with_cities['gare_name'].values)
        print(f"[DEBUG] Affichage :", gares_with_cities[gares_with_cities['gare_name'].str.contains('paris', case=False, na=False)][['gare_name', name_attr, 'dep', 'code_insee']])
        print("Colonnes gares_with_cities :", gares_with_cities.columns)

        gares_with_cities = gares_with_cities.to_crs(epsg=2154)
        gares_with_cities['distance'] = gares_with_cities.apply(
            lambda row: row['geometry_gare'].distance(row['geometry']),
            axis=1
        )

        # Pour chaque gare, garder la ville la plus proche
        idx_min_dist = gares_with_cities.groupby('code_uic')['distance'].idxmin()
        idx_min_dist = idx_min_dist.dropna().astype(int)
        gares_with_cities = gares_with_cities.loc[idx_min_dist].copy()
        """

        print(f"[DEBUG] {gares_with_cities[name_attr].notna().sum()} gares associées par département.")

    #print(f"[DEBUG] Extrait des villes avec gare associée : {list(cities_with_gares.head(5).to_dict('records'))}")
    
    # Filtre les gares sans ville associée
    print(f"[DEBUG2] {gares_with_cities[name_attr].isna().sum()}/{len(gares_with_cities)} gares sans ville associée.")
    gares_with_cities = gares_with_cities[gares_with_cities[name_attr].notna()]
    # Reconvertir geometry_x et geometry_y en une seule colonne geometry
    #cities_with_gares['geometry'] = cities_with_gares.apply(
    #    lambda row: row['geometry_x'] if row['geometry_x'] is not None else row['geometry_y'], axis=1
    #)
    print(f"[DEBUG] Extrait des gares avec ville associée 1: {list(gares_with_cities.head(5).to_dict('records'))}")
    gares_with_cities = gares_with_cities[['code_uic', name_attr, 'geometry', 'code_insee', 'gare_name']]
    print(f"[DEBUG] Extrait des gares avec ville associée 2: {list(gares_with_cities.head(5).to_dict('records'))}")
    print(f"[DEBUG] Affichage :", gares_with_cities[gares_with_cities['gare_name'].str.contains('Charles de Gaulle', case=False, na=False)][['gare_name', name_attr, 'code_insee']])
    # Il faut pas enlever les doublons, il faudrait plutôt avoir une paire d'index ou juste créer
    #cities_with_gares = cities_with_gares.drop_duplicates(subset='codeinsee', keep='first')
    #dic = (cities_with_gares
    #   .groupby('code_insee')[[name_attr, 'geometry']]
    #   .apply(lambda df: df.to_dict('records'))
    #   .to_dict())

    dic = gares_with_cities.set_index('code_uic').to_dict('index')
    print(f"[DEBUG] Dictionnaire gares -> villes : {list(dic.items())[:5]}")
    return dic, gares_with_cities

def expand_city_station_map(city_station_map, cities, stops_gdf, buffer_km=5):
    stop_to_city = {}
    stops_proj = stops_gdf.to_crs(epsg=2154)

    for city, main_stop_id in city_station_map.items():
        city_geom = cities.loc[cities["name"] == city].geometry.iloc[0].centroid
        city_point = gpd.GeoSeries([city_geom], crs=cities.crs).to_crs(epsg=2154).iloc[0]

        # Chercher tous les arrêts dans un rayon de buffer_km autour de la ville
        stops_proj["dist"] = stops_proj.geometry.distance(city_point)
        nearby = stops_proj[stops_proj["dist"] <= buffer_km * 1000]

        for stop_id in nearby["stop_id"]:
            stop_to_city[stop_id] = city

    return stop_to_city

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

import re

def extract_uic_from_stop_id(stop_id):
    """
    Extrait le code UIC (6 à 8 chiffres) situé en fin de stop_id.
    Retourne None si aucun code plausible trouvé.
    """
    if not isinstance(stop_id, str):
        return None
    match = re.search(r"(\d+)$", stop_id)
    if match:
        code = match.group(1)
        if 6 <= len(code) <= 8:
            return code
    return None

def match_stops_to_gares(stops, gares_df):
    """
    Associe les arrêts GTFS aux gares avec le nom et la géométrie.
    stops : DataFrame des arrêts GTFS
    gares : GeoDataFrame des gares
    Retourne un GeoDataFrame avec les arrêts associés aux gares.
    """
    print("[DEBUG] Extrait des stops GTFS:", stops.head(5).to_dict('records'))
    #print("[DEBUG] Extrait des gares:", gares_df.head(5).to_dict('records'))
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326"
    )
    stops_gdf["uic_code"] = stops_gdf["stop_id"].apply(extract_uic_from_stop_id)
    print("[DEBUG] Extrait des arrêts avec code UIC:", stops_gdf.head(5).to_dict('records'))

    print(f"[DEBUG] {stops_gdf['uic_code'].isna().sum()}/{len(stops_gdf)} arrêts sans code UIC.")

    merged = stops_gdf.merge( # Garder les stops qui ont une gare (il existe des stops en dehors des bordures)
        gares_df[["code_uic", "gare_name", "codeinsee", "geometry"]],
        left_on="uic_code",
        right_on="code_uic",
        how="inner",
        suffixes=("", "_gare")
    )
    merged = merged[merged['codeinsee'].notna()]
    print("[DEBUG] Extrait des arrêts associés aux gares:", merged.head(5).to_dict('records'))

    unmatched = merged["codeinsee"].isna().sum()
    print(f"[DEBUG] {unmatched} arrêts sans gare associée.")
    
    dic = merged.set_index('stop_id')[['uic_code', 'gare_name', 'codeinsee', 'geometry']].to_dict('index')
    print(f"[DEBUG] Dictionnaire stops -> gares : {list(dic.items())[:2]}")
    return dic, merged

key = 'code_unite_urbaine'

def aggregate_cities_to_agglomerations(cities_gdf):
    """
    Agrège les villes dans leur unité urbaine associée, avec le code d'unité urbaine.
    Retourne un dictionnaire {ville (code_insee): [nom, code_insee, code_unite_urbaine, nom_unite_urbaine, population, geometry]
    ainsi que les agrégations par unité urbaine.
    """
    unites_urbaines = {}
    agglomerations = {}
    for _, row in cities_gdf.iterrows():
        code_insee = row['code_insee']
        code_urbain = row['code_unite_urbaine']
        if unites_urbaines.get(code_urbain) is None:
            unites_urbaines[code_urbain] = {'nom': row['nom_unite_urbaine'], 'population': row['population']}
        else:
            unites_urbaines[code_urbain]['population'] += row['population']
        if agglomerations.get(code_insee) is None:
            agglomerations[code_insee] = {
                'code_unite_urbaine': row['code_unite_urbaine'],
                'nom_unite_urbaine': row['nom_unite_urbaine'],
                'population': row['population'],
                'geometry': row['geometry']
            }
    return agglomerations, unites_urbaines

def filter_agglomerations(agglomerations, unites_urbaines, cities_gdf, pop_threshold, by_agg=True):
    """
    Supprime toutes les villes ou agglomérations (by_agg = True) dont la population est inférieure à pop_threshold.
    Retourne les agglomerations avec les villes filtrées et cities_gdf filtré.
    """
    if not by_agg:
        to_remove = cities_gdf[cities_gdf['population'] < pop_threshold]['code_insee'].tolist()

        for code_insee in to_remove:
            if code_insee in agglomerations:
                del agglomerations[code_insee]

        cities_gdf = cities_gdf[~cities_gdf['code_insee'].isin(to_remove)]
    else:
        to_remove = set()
        for code_urbain, data in unites_urbaines.items():
            if data['population'] < pop_threshold:
                # marquer toutes les villes de cette unité pour suppression
                villes = [code_insee for code_insee, info in agglomerations.items()
                        if info['code_unite_urbaine'] == code_urbain]
                to_remove.update(villes)

        for code_insee in to_remove:
            if code_insee in agglomerations:
                del agglomerations[code_insee]

        cities_gdf = cities_gdf[~cities_gdf['code_insee'].isin(to_remove)]

    return agglomerations, cities_gdf

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

def prompt_train_between(cities, stop_times, stop_to_city, G_city, city_station_map, agglomerations, stops=None):
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
    e1 = cities[cities.nom_unite_urbaine == ville1]
    e2 = cities[cities.nom_unite_urbaine == ville2]
    try:
        ville1 = e1.code_unite_urbaine.iloc[0]
        ville2 = e2.code_unite_urbaine.iloc[0]
        ville1_nom = e1.nom_unite_urbaine.iloc[0]
        ville2_nom = e2.nom_unite_urbaine.iloc[0]
        if ville1 not in G_city.nodes or ville2 not in G_city.nodes:
            print(f"{ville1} ou {ville2} n'est pas dans le graphe des villes.")
            return True
    except IndexError:
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
        print(f"Pas de ligne directe entre {ville1_nom} et {ville2_nom}.")
        try:
            path = nx.shortest_path(G_city, ville1, ville2, weight="distance")
            path_names = list(map(lambda code: G_city.nodes[code]['name'], path))
            print(f"Plus court chemin ({len(path)-1} arêtes) : {' -> '.join(path_names)}")
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
                node = G_city.nodes[v]
                name = node['name']
                insee = node['insee'] # Faire avec les arêtes annotées par gares plutôt
                gares = city_station_map.get(insee, None)
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
                            print(f"  - {name} : gares principales {', '.join(noms)}")
                        else:
                            print(f"  - {name} : aucune gare associée")
                    else:
                        stop_row = stops.loc[stops['stop_id'] == gares]
                        if not stop_row.empty:
                            stop_name = stop_row.iloc[0]['stop_name']
                            print(f"  - {name} : gare principale {stop_name}")
                        else:
                            print(f"  - {name} : gare principale {gares}")
                elif gares:
                    if isinstance(gares, list):
                        print(f"  - {name} : gares principales {', '.join(gares)}")
                    else:
                        print(f"  - {name} : gare principale {gares}")
                else:
                    print(f"  - {name} : pas de gare associée")
        except nx.NetworkXNoPath:
            print(f"Aucun chemin entre {ville1} et {ville2} dans le graphe.")
    return True

if __name__ == "__main__":
    gtfs_zip = "resources/gtfs/france/opendata-sncf-transport.zip"
    cities_fp = "resources/communes-france-datagouv-2025.csv"

    # Charger les données GTFS
    stops, stop_times, trips, routes = load_gtfs(gtfs_zip)

    # Charger les villes
    cities = load_cities_bis(cities_fp)
    print(f"{len(cities)} villes chargées.")
    #print(f"Villes contenant Paris : {cities[cities[name_attr].str.contains('Paris', case=False, na=False)]}")
    print("Exemples de villes :", list(cities[name_attr].unique())[:10])

    # Aggréger les villes en agglomérations
    # /!\ À faire avant le filtrage sur population, plus lent mais nécessaire pour récupérer toutes les bonnes communes dans une agglomération.
    agglomerations, unites_urbaines = aggregate_cities_to_agglomerations(cities)
    print(f"{len(agglomerations)} agglomérations créées à partir des villes.") #unique?

    # Charger les gares
    gares_df = load_gares("resources/gares-de-voyageurs.geojson")
    print(f"{len(gares_df)} gares chargées.")
    # Déplier les codes UIC pour 1 par ligne
    gares_df = (
        gares_df.assign(code_uic=gares_df["codes_uic"].fillna("").str.split(";"))
             .explode("code_uic")
    )
    gares_df["code_uic"] = gares_df["code_uic"].str.strip()
    print("[DEBUG] Extrait des gares avec codes UIC dépliés:", gares_df.head(5).to_dict('records'))

    # Correspondance stops -> gares
    stops_to_gares, stops_to_gares_df = match_stops_to_gares(stops, gares_df) # Stops
    # TODO debug
    #nb_associes = stops_to_gares["gare_name"].notna().sum()
    #print(f"{nb_associes}/{len(stops)} arrêts associés à des gares.")

    # Associer les gares aux villes
    gares_to_cities, _ = associate_stations_to_cities(cities, stops_to_gares_df, gares_df) # Villes
    print(f"{len(gares_to_cities)} gares associées à des villes.")

    print("Correspondance villes -> agglomérations :", list(agglomerations.items())[5:10])

    by_agg = (key == 'code_unite_urbaine') # filtrage sur agglomérations et non villes

    for pop_threshold in range(0, 501, 10): # 10k - 500k, graphes préconstruits
        print("[DEBUG] Filtrage des", ("agglomérations" if by_agg else "villes"), "comme noeuds du graphe pour pop_threshold:", pop_threshold, "k")
        agglomerations_bis, cities_bis = filter_agglomerations(agglomerations, unites_urbaines, cities, pop_threshold * 1000, by_agg)
        G_city = build_city_graph_with_trips(cities_bis, stops, stop_times, trips, stops_to_gares, gares_to_cities, agglomerations_bis, unites_urbaines)

        print(f"{G_city.number_of_nodes()} villes, {G_city.number_of_edges()} liaisons directes.")

        print(f"[DEBUG] Export du graph dans output/france_railway_network_pop_threshold_{pop_threshold}k.json.")
        export_data = export_graph_for_web_dashboard(
            G_city, 
            cities_bis, 
            output_file="output/france_railway_network_pop_threshold_{}k.json".format(pop_threshold),
            country_code='france',
            key_column=key
        )

    # Visualisation
    fig = plot_interactive_city_graph(G_city, cities_bis)
    #fig.display_details("Contenu détaillé ici...", "Titre de la fenêtre")

        # s'il n'existe pas les villes intermédiaires dans une ligne de train alors on met quand même
    # la ligne et on garde bien la distance et temps de trajet correspondants
    # --- Interface utilisateur ---
    #while True:
    #    if not prompt_train_between(cities, stop_times, stops_to_gares, G_city, gares_to_cities, agglomerations, stops=stops):
    #        break

