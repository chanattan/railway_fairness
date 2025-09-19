from turtle import pos
import config
from graph import build_city_graph_with_trips, plot_interactive_city_graph
import zipfile
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
from data_exporter import export_graph_for_web_dashboard

"""
    Main file: load data, process, build graph, export results.
"""

def load_gtfs(gtfs_zip):
    # Print file names inside the zip
    with zipfile.ZipFile(gtfs_zip, 'r') as z:
        # if '/' in filename, add folder name to files
        folder_name = ''
        for filename in z.namelist():
            if '/' in filename:
                folder_name = (filename.split('/')[0]) + '/'
                break
        stops = pd.read_csv(z.open(folder_name + 'stops.txt'))
        stop_times = pd.read_csv(z.open(folder_name + 'stop_times.txt'))
        trips = pd.read_csv(z.open(folder_name + 'trips.txt'))
        routes = pd.read_csv(z.open(folder_name + 'routes.txt'))
        calendar = pd.read_csv(z.open(folder_name + 'calendar_dates.txt'))
    return stops, stop_times, trips, routes, calendar

def load_gtfs2(gtfs_zip):
    with zipfile.ZipFile(gtfs_zip, 'r') as z:
        stops = pd.read_csv(z.open('stops.txt'))
        stop_times = pd.read_csv(z.open('stop_times.txt'))
        trips = pd.read_csv(z.open('trips.txt'))
        routes = pd.read_csv(z.open('routes.txt'))
    return stops, stop_times, trips, routes

import geopandas as gpd
from shapely.geometry import Point

def load_cities(csv_path):
    """
        Charge les villes depuis un CSV et crée une GeoDataFrame.
        Tous les pays ne sont pas standardisés de la même façon, cela inclut les fichiers qui sont considérés.
        La fonction load_cities comme la fonction load_gares peut avoir un traitement spécifique au pays.
    """
    if config.country == 'france':
        # Lecture CSV
        dtype = {
            'code_insee': str,
            'nom_standard': str,
            'dep_code': str,
            'population': int,
            'latitude_centre': float,
            'longitude_centre': float,
            'latitude_mairie': float,
            'longitude_mairie': float,
            'canton_code': str,
            'epci_code': str
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
        df = df[[config.name_attr, 'code_insee', 'population', 'latitude_centre', 'longitude_centre']]

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
    elif config.country == 'switzerland':

        # Définition des types de données pour la Suisse
        dtype = {
            'xtf_id': str,
            'Name': str,
            'Nummer': str,
            'Gemeinde_Nummer': str,
            'Gemeinde_Name': str,
            'E': float,  # Coordonnée Est
            'N': float,  # Coordonnée Nord
            'H': float,  # Altitude
            'Transportunternehmen_Nummer': str,
            'Transportunternehmen_Abkuerzung': str,
            'Betriebspunkttyp_Code': str,
            'Verkehrsmittel_Code': str
        }
        # Lecture du CSV
        df = pd.read_csv(csv_path, dtype=dtype)

        # Nettoyage
        initial_count = len(df)
        df = df.dropna(subset=['E', 'N'])
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            print(f"[DEBUG] {dropped_count} lignes supprimées à cause de coordonnées manquantes.")
        
        # Agrégation par commune pour éviter les doublons (une ligne par commune)
        # On prend la première occurrence de chaque commune
        df_communes = df.groupby(['Gemeinde_Nummer', 'Gemeinde_Name']).agg({
            # mettre en liste les codes UIC (Nummer) associés à la commune
            'Nummer': lambda x: [n for n in x.unique() if pd.notna(n)],
            'E': 'mean',  # Moyenne des coordonnées si plusieurs points par commune
            'N': 'mean',
            'H': 'mean'
        }).reset_index()

        # Conversion des coordonnées du système suisse (CH1903+/LV95, EPSG:2056) vers WGS84 (EPSG:4326)
        #transformer = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

        #def convert_coordinates(row):
        #    try:
        #        lon, lat = transformer.transform(row['E'], row['N'])
        #        return pd.Series({'longitude': lon, 'latitude': lat})
        #    except:
        #        return pd.Series({'longitude': None, 'latitude': None})

        #coords_converted = df_communes.apply(convert_coordinates, axis=1)
        #df_communes = pd.concat([df_communes, coords_converted], axis=1)

        # Cleaning
        df_communes = df_communes.rename(columns={'E': 'longitude', 'N': 'latitude'})
        df_communes = df_communes.dropna(subset=['longitude', 'latitude'])
        if df_communes['latitude'].isna().any() or df_communes['longitude'].isna().any():
            print(f"[DEBUG] {df_communes['latitude'].isna().sum()} communes ont latitude NaN, {df_communes['longitude'].isna().sum()} communes ont longitude NaN")

        # Gemeinde_Nummer est l'équivalent de code_insee
        # Nummer est le code UIC
        df_final = df_communes[['Gemeinde_Name', 'Gemeinde_Nummer', 'Nummer', 'latitude', 'longitude']].copy()

        # Renommage pour standardiser
        df_final.rename(columns={
            'Gemeinde_Name': config.name_attr,
            'Gemeinde_Nummer': 'code_commune',
            'Nummer': 'codes_uic'
        }, inplace=True)

        # Création de la géométrie
        geometry = [Point(xy) for xy in zip(df_final['longitude'], df_final['latitude'])]
        gdf = gpd.GeoDataFrame(df_final, geometry=geometry, crs="EPSG:2056")

        # Identifier les gares étrangères et les filtrer
        def is_foreign_station(row):
            # Notamment, les gares passées les frontières (Allemagne, Italie...)
            return row[config.key] == 9998 or '(Ausland)' in row[config.name_attr]

        swiss_stations = gdf[~gdf.apply(is_foreign_station, axis=1)]
        foreign_stations = gdf[gdf.apply(is_foreign_station, axis=1)]
        print("[Filtrage] Nombre de villes suisses :", len(swiss_stations))
        print("[Filtrage] Nombre de gares étrangères :", len(foreign_stations))
        print("[Extrait] Gares étrangères :", foreign_stations.head().to_dict('records'))
        gdf = swiss_stations

        # On éclate les listes de codes UIC pour avoir une ligne par code UIC, après avoir clean et rename.
        # Cette version est utilisée pour les gares, en un temps par rapport au traitement de la France.
        gdf_gares = gdf.copy().explode('codes_uic')
        gdf_gares = gdf_gares.rename(columns={'codes_uic': 'code_uic'})

        # On ajoute les populations dans le gdf des villes
        # Ceci concerne en particulier la Suisse, qui n'a pas de population dans le fichier des communes.
        pop_df = pd.read_csv("resources/switzerland/swiss_populations_bfs.csv", skiprows=3)

        # Garder seulement les lignes où la première colonne contient "......" suivi du code commune puis de la population totale
        mask = pop_df.iloc[:, 0].fillna("").str.match(r"^\.+\d{4}\s")
        communes_df = pop_df[mask].copy()

        # Séparer "......0001 Aeugst am Albis" en code et nom et garder la pop totale (colonne 1)
        communes_df[["code_commune", "nom_commune"]] = communes_df.iloc[:, 0].str.extract(r"\.*(\d{4})\s+(.+)")
        communes_df = communes_df[["code_commune", "nom_commune", pop_df.columns[1]]]
        # Cleaning
        communes_df = communes_df.rename(columns={pop_df.columns[1]: "population"})
        communes_df["population"] = pd.to_numeric(communes_df["population"], errors="coerce").fillna(0).astype(int)

        # Merge avec le gdf des villes
        communes_df['code_commune'] = communes_df['code_commune'].astype(int).astype(str)
        communes_df = communes_df.rename(columns={'nom_commune': config.name_attr})
        gdf = gdf.merge(communes_df[['code_commune', 'population']], on='code_commune', how='left')
        print(f"[DEBUG] {gdf['population'].isna().sum()}/{len(gdf)} villes sans population associée (2024-2025 diff).")
        gdf = gdf[~gdf['population'].isna()]

        # Exemple
        print("[DEBUG] Extrait des villes populées :", gdf.head())

        # Debug
        print(f"[DEBUG] {len(gdf_gares)} gares suisses chargées.")
        print("Extrait des gares :", gdf_gares.head().to_dict('records'))
        print(f"[DEBUG] {len(gdf)} communes suisses chargées.")

        # Test avec quelques communes suisses connues
        debug = True
        if debug:
            test_cities = ["Zürich", "Genève", "Bern", "Lausanne", "Basel"]
            for city in test_cities:
                matches = gdf[gdf[config.name_attr].str.contains(city, case=False, na=False)]
                if not matches.empty:
                    print("Nombre de gares trouvées :", len(matches))
                    print(f"✅ {city} trouvée")
                    print(matches[[config.name_attr, 'code_commune', 'codes_uic', 'latitude', 'longitude']].head())
                else:
                    print(f"❌ {city} manquante")

            # Affichage d'un échantillon
            print("\n[DEBUG] Échantillon des données finales:")
            print(gdf.head())
            print(f"\n[DEBUG] Colonnes disponibles: {list(gdf.columns)}")
        return gdf, gdf_gares
    else:
        raise NotImplementedError("Le pays spécifié n'est pas supporté pour l'instant.")

    return None

def associate_stations_to_cities(cities_gdf, stops_to_gares_df, gares_df):
    """
    Associe chaque gare à la commune correspondante (via le code INSEE).
    cities_gdf : GeoDataFrame des communes avec colonne 'code_insee'
    gares_df : DataFrame des gares avec colonne 'codeinsee'
    Retourne un DataFrame enrichi
    """
    if config.country == 'france':
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
        df_communes = pd.read_csv("resources/france/v_commune_2025.csv")
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
            cities_gdf[['code_insee', config.name_attr, 'geometry']],
            left_on='codeinsee',
            right_on='code_insee',
            how='left'
        )
        print(f"[DEBUG] Paris Gare du Nord cities_with_gares:", "Paris Gare du Nord" in gares_with_cities['gare_name'].values)
        print(f"[DEBUG] Affichage pour Paris:", gares_with_cities[gares_with_cities['gare_name'].str.contains('Charles de Gaulle', case=False, na=False)][['gare_name', config.name_attr, 'codeinsee']])
        print(f"[DEBUG] Extrait cities_with_gares:", list(gares_with_cities.items())[:5])
        print(f"[DEBUG] {gares_with_cities[config.name_attr].isna().sum()}/{len(gares_with_cities)} gares sans ville associée.")
        
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
        gares_without_city = gares_with_cities[gares_with_cities[config.name_attr].isna()] # Toutes les gares ont un codeinsee, pas forcément une ville : name_attr
        
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
        print(f"[DEBUG2] {gares_with_cities[config.name_attr].isna().sum()}/{len(gares_with_cities)} gares sans ville associée.")
        gares_with_cities = gares_with_cities[gares_with_cities[config.name_attr].notna()]
        # Reconvertir geometry_x et geometry_y en une seule colonne geometry
        #cities_with_gares['geometry'] = cities_with_gares.apply(
        #    lambda row: row['geometry_x'] if row['geometry_x'] is not None else row['geometry_y'], axis=1
        #)
        print(f"[DEBUG] Extrait des gares avec ville associée 1: {list(gares_with_cities.head(5).to_dict('records'))}")
        gares_with_cities = gares_with_cities[['code_uic', config.name_attr, 'geometry', 'code_insee', 'gare_name']]
        print(f"[DEBUG] Extrait des gares avec ville associée 2: {list(gares_with_cities.head(5).to_dict('records'))}")
        print(f"[DEBUG] Affichage :", gares_with_cities[gares_with_cities['gare_name'].str.contains('Charles de Gaulle', case=False, na=False)][['gare_name', config.name_attr, 'code_insee']])
        # Il faut pas enlever les doublons, il faudrait plutôt avoir une paire d'index ou juste créer
        #cities_with_gares = cities_with_gares.drop_duplicates(subset='codeinsee', keep='first')
        #dic = (cities_with_gares
        #   .groupby('code_insee')[[name_attr, 'geometry']]
        #   .apply(lambda df: df.to_dict('records'))
        #   .to_dict())

        dic = gares_with_cities.set_index('code_uic').to_dict('index')
        print(f"[DEBUG] Dictionnaire gares -> villes : {list(dic.items())[:5]}")
    elif config.country == 'switzerland':
        # Cette étape concerne uniquement du debug et la création du dictionnaire.

        # Réduire par le goulot d'étranglement des stops
        print(f"Nombre de gares préfiltrage stops : {len(gares_df)}")
        gares_df = gares_df[gares_df['code_uic'].isin(stops_to_gares_df['code_uic'])]
        print(f"Nombre de gares conservées : {len(gares_df)}")
        print(gares_df.head())

        gares_with_cities = gares_df
        # Drop longitude et latitude pour éviter confusion
        gares_with_cities = gares_with_cities.drop(columns=['longitude', 'latitude'])
        dic = gares_with_cities.set_index('code_uic').to_dict('index')
        print(f"[DEBUG] Dictionnaire gares_to_cities : {list(dic.items())[:5]}")
    else:
        raise NotImplementedError("Le pays spécifié n'est pas supporté pour l'instant.")
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
    Retourne un DataFrame avec colonnes : codeinsee, gare_name, geometry
    La Suisse est déjà traitée dans load_cities.
    """
    gares = gpd.read_file(gares_fp)
    if config.country == 'france':
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
    else:
        raise NotImplementedError("Le pays spécifié n'est pas supporté pour l'instant.")
    return gares

import re

def extract_uic_from_stop_id(stop_id):
    """
    Extrait le code UIC.
    Retourne None si aucun code plausible trouvé.
    """
    if config.country == 'france':
        # Le code uic est contenu à la fin du stop_id.
        match = re.search(r"(\d+)$", stop_id)
        if match:
            code = match.group(1)
            if 6 <= len(code) <= 8:
                return code
    elif config.country == 'switzerland':
        # Pour le GTFS Suisse, les codes uic sont dans stop_id et il peut contenir les arrêts détaillés, e.g., étages stations.
        # Dans ce cas, le stop_id est de la forme code_uic:x:code_commune... On garde le stop_id dans son entierté pour mapper les stops
        # vers les gares, mais on extrait le code_uic pour faire le merge.
        # Le stop_id peut également être de la forme ParentCODEUIC.
        if stop_id.startswith('Parent'):
            return stop_id.replace('Parent', '')
        return stop_id.split(':')[0]
    else:
        raise NotImplementedError("Le pays spécifié n'est pas supporté pour l'instant.")
    return None

def match_stops_to_gares(stops, gares_df):
    """
    Associe les arrêts GTFS aux gares avec le nom et la géométrie.
    stops : DataFrame des arrêts GTFS
    gares : GeoDataFrame des gares
    Retourne un GeoDataFrame avec les arrêts associés aux gares.
    """

    if config.country == 'france':
        stops_gdf = gpd.GeoDataFrame(
            stops,
            geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
            crs="EPSG:4326"
        )
        stops_gdf["code_uic"] = stops_gdf["stop_id"].apply(extract_uic_from_stop_id)
        print("[DEBUG] Extrait des arrêts avec code UIC:", stops_gdf.head(5).to_dict('records'))

        print(f"[DEBUG] {stops_gdf['code_uic'].isna().sum()}/{len(stops_gdf)} arrêts sans code UIC.")

        merged = stops_gdf.merge( # Garder les stops qui ont une gare (il existe des stops en dehors des bordures)
            gares_df[["code_uic", "gare_name", "codeinsee", "geometry"]],
            on="code_uic",
            how="inner",
            suffixes=("", "_gare")
        )
        unmatched = merged["codeinsee"].isna().sum()
        print(f"[DEBUG] {unmatched} arrêts sans gare associée.")

        merged = merged[merged['codeinsee'].notna()]
        print("[DEBUG] Extrait des arrêts associés aux gares:", merged.head(5).to_dict('records'))
        
        dic = merged.set_index('stop_id')[['code_uic', 'gare_name', 'codeinsee', 'geometry']].to_dict('index')
        print(f"[DEBUG] Dictionnaire stops -> gares : {list(dic.items())[:2]}")
    elif config.country == 'switzerland':
        stops_gdf = gpd.GeoDataFrame(
            stops,
            geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
            crs="EPSG:4326"
        )

        # Reprojection en EPSG:2056 pour la Suisse
        stops_gdf = stops_gdf.to_crs(epsg=2056)

        stops_gdf["code_uic"] = stops_gdf["stop_id"].apply(extract_uic_from_stop_id)
        print("[DEBUG] Extrait des arrêts avec code UIC:", stops_gdf.head(5).to_dict('records'))
        print(f"[DEBUG] {stops_gdf['code_uic'].isna().sum()}/{len(stops_gdf)} arrêts sans code UIC.")

        # Les codes UIC sont les stop_id, à filtrer après avec les parent stations
        merged = stops_gdf.merge( # Garder les stops qui ont une gare (il existe des stops en dehors des bordures)
            gares_df[["code_uic", config.name_attr, "code_commune", "geometry"]],
            on="code_uic",
            how="inner",
            suffixes=("", "_gare")
        )
        unmatched = merged["code_uic"].isna().sum()
        print(f"[DEBUG] {unmatched} stops sans gare associée.")

        merged = merged[merged['code_uic'].notna()]
        print("[DEBUG] Extrait des arrêts associés aux gares:", merged.head(5).to_dict('records'))

        # On récolte également les données de la ville associée, on oublie la géométrie de la gare.
        dic = merged.set_index('stop_id')[['code_uic', 'stop_name', config.name_attr, 'code_commune', 'geometry']].to_dict('index')
        print(f"[DEBUG] Dictionnaire stops_to_gares : {list(dic.items())[:2]}")
    else:
        raise NotImplementedError("Le pays spécifié n'est pas supporté pour l'instant.")
    return dic, merged

def load_aires():
    """
    TODO: Relatif à la France seulement : charger les aires d'attraction pour agréger les communes nécessaires.
    """
    aires = pd.read_csv("resources/france/aires_attraction_insee_2020.csv", comment='#', delimiter=';')
    aires = aires[['CODGEO', 'AAV2020', 'LIBAAV2020', 'DEP', 'REG']].rename(
        columns={
            'CODGEO': 'code_insee',
            'AAV2020': 'code_aire',
            'LIBAAV2020': 'nom_aire'
        }
    )

    print(aires['nom_aire'].head())
    return aires

def find_main_city_coords(group_data):
    """
    Fonction d'agrégation pour trouver les coordonnées de la ville principale
    directement dans le groupby
    """
    nom_aire_lower = group_data['nom_aire'].iloc[0].lower()
            
    # Chercher la ville principale
    for idx, row in group_data.iterrows():
        nom_std = str(row['nom_standard']).lower()
        if nom_std in nom_aire_lower:
            lat = row['latitude'] if 'latitude' in row else row['latitude_centre']
            lon = row['longitude'] if 'longitude' in row else row['longitude_centre']
            if pd.notna(lat) and pd.notna(lon):
                return pd.Series({
                    'population': group_data['population'].sum(),
                    'latitude_centre': lat,
                    'longitude_centre': lon
                })
    
    # Fallback: coordonnées moyennes
    return pd.Series({
        'population': group_data['population'].sum(),
        'latitude_centre': group_data['latitude'].mean() if 'latitude' in group_data.columns else group_data['latitude_centre'].mean(),
        'longitude_centre': group_data['longitude'].mean() if 'longitude' in group_data.columns else group_data['longitude_centre'].mean()
    })
def aggregate_cities_to_aires(cities_gdf):
    """
    TODO: seulement pour la France, pour l'instant.
    Agrège les villes dans leur aire d'attraction, avec le code insee.
    Retourne un dictionnaire {ville (code_insee): [nom, code_insee, code_aire, nom_aire, population, geometry]
    ainsi que les agrégations par unité urbaine.
    """
    if config.country == 'france':
        aires = load_aires()
        
        # Merger cities_gdf avec aires
        # On garde toutes les communes de cities_gdf
        cities_with_aires = pd.merge(cities_gdf, aires[['code_insee', 'code_aire', 'nom_aire']], 
                                   on='code_insee', how='left')
        
        # Les fichiers des aires et des communes ne matchent pas exactement
        # en raison des différentes dates de collection. On affecte aux communes qui n'ont pas d'aire (au sens pas d'entrée dans aires.csv) le code 000.
        # (Toutes) Les communes ayant un code_aire de 000 n'ont pas d'aire d'attraction et on considère qu'elles sont à part entière.
        # Il faut assigner un code_aire unique à chaque commune hors aire, afin de les différencier dans les noeuds et pouvoir utiliser leur propre nom de commune.
        # /!\ Note importante : il peut y avoir des communes oubliées (notamment celles de petite taille).
        max_code_aire = aires['code_aire'].astype(str).apply(lambda x: int(x) if x.isdigit() else 0).max()
        print(f"Code aire maximum actuel: {max_code_aire}")
        
        # Et les communes sans aire trouvée et celles qui ne sont pas dans le fichier des aires
        mask_missing = cities_with_aires['code_aire'].isna() | (cities_with_aires['code_aire'] == '000')
        missing_count = mask_missing.sum()
        print(f"Nombre de communes sans code_aire: {missing_count}")
        
        if missing_count > 0:
            # Créer des nouveaux codes_aire uniques
            new_codes = range(max_code_aire + 1, max_code_aire + 1 + missing_count)
            new_codes_str = [str(code).zfill(3) for code in new_codes]
            print("Nouveaux codes_aire assignés:", new_codes_str[:5], "..." if len(new_codes_str) > 5 else "")
            
            # Assigner les nouveaux codes
            cities_with_aires.loc[mask_missing, 'code_aire'] = new_codes_str
            # nom_aire = nom_standard pour ces communes
            cities_with_aires.loc[mask_missing, 'nom_aire'] = cities_with_aires.loc[mask_missing, 'nom_standard']
            
            if 'latitude' in cities_with_aires.columns and 'longitude' in cities_with_aires.columns:
                cities_with_aires.loc[mask_missing, 'latitude_centre'] = cities_with_aires.loc[mask_missing, 'latitude']
                cities_with_aires.loc[mask_missing, 'longitude_centre'] = cities_with_aires.loc[mask_missing, 'longitude']
        
        # Grouper par by code_aire pour créer le dictionnaire associé
        aires_grouped = cities_with_aires.groupby(['code_aire', 'nom_aire']).apply(find_main_city_coords).reset_index()

        aires_grouped['population'] = aires_grouped['population'].astype(int)
        
        print(f"Aires grouped shape: {aires_grouped.shape}")
        
        # Premier dictionnaire mapping aires -> infos
        aires_dict = {}
        for _, row in aires_grouped.iterrows():
            geometry = Point(row['longitude_centre'], row['latitude_centre'])
            aires_dict[row['code_aire']] = {
                'nom_aire': row['nom_aire'],
                'population': row['population'],
                'geometry': geometry
            }
        
        # Second dictionnaire mapping ville -> infos
        insee_dict = {}
        for _, row in cities_with_aires.iterrows():
            insee_dict[row['code_insee']] = {
                'code_aire': row['code_aire'],
                'nom_aire': row['nom_aire'],
                'nom_standard': row['nom_standard'],
                'population': row['population'],
                'geometry': row['geometry']
            }
        
        # Màj cities_gdf
        cities_gdf['code_aire'] = cities_with_aires['code_aire']
        cities_gdf['nom_aire'] = cities_with_aires['nom_aire']
        
        # Vérification
        print(f"Communes avec code_aire null: {cities_gdf['code_aire'].isna().sum()}")
        print(f"Communes avec nom_aire null: {cities_gdf['nom_aire'].isna().sum()}")
        print(f"Communes avec nom_aire == 'Commune hors attraction des villes': {(cities_gdf['nom_aire'] == 'Commune hors attraction des villes').sum()}")
        print(f"Pour ces dernières communes, code_aire: {(cities_gdf[cities_gdf['nom_aire'] == 'Commune hors attraction des villes']['code_aire']).unique()}")
        
    return insee_dict, aires_dict
   
def filter_agglomerations(communes, aires, cities_gdf, pop_threshold, by_agg=True):
    """
    Supprime toutes les villes ou agglomérations (by_agg = True) dont la population est inférieure à pop_threshold.
    Pour la France :
        Retourne les communes (dictionnaire ville -> aire) filtrées et les villes (cities_gdf) filtrées.
    """
    
    if not by_agg:
        if config.country == 'france':
            to_remove = cities_gdf[cities_gdf['population'] < pop_threshold][config.key]

            for code_insee in to_remove:
                if code_insee in communes:
                    del communes[code_insee]

            cities_gdf = cities_gdf[~cities_gdf[config.key].isin(to_remove)]
        
            return communes, cities_gdf
        elif config.country == 'switzerland':
            to_remove = cities_gdf[cities_gdf['population'] < pop_threshold][config.key]

            cities_gdf = cities_gdf[~cities_gdf[config.key].isin(to_remove)]
        
            return cities_gdf
        else:
            raise NotImplementedError(f"Filter population is not supported yet for {config.country}")
    else:
        if config.country == 'france':
            to_remove = set()
            for code_aire, data in aires.items():
                if data['population'] < pop_threshold:
                    # marquer toutes les villes de cette unité pour suppression
                    villes = [code_insee for code_insee, info in communes.items()
                            if info['code_aire'] == code_aire]
                    to_remove.update(villes)

            for code_insee in to_remove:
                if code_insee in communes:
                    del communes[code_insee]

            cities_gdf = cities_gdf[~cities_gdf['code_insee'].isin(to_remove)]

            return communes, cities_gdf
        else:
            raise NotImplementedError(f"Aggregation for {config.country} is not supported.")

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

def __prompt_train_between(cities, stop_times, stop_to_city, G_city, city_station_map, agglomerations, stops=None):
    """
    DEPRECATED FUNCTION
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

def create_calendar(calendar, trips):
    """
        Retourne un dictionnaire trip_id -> première date de départ active.
    """
    # Garde uniquement les dates actives (1) où (2) : suppression.
    calendar_active = calendar[calendar['exception_type'] == 1][['service_id', 'date']]
    # Merge avec les trips pour lier chaque trip à ses dates
    trips_with_dates = trips[['trip_id', 'service_id']].merge(
        calendar_active, on='service_id', how='left'
    )
    # On garde que la première date, théoriquement c'est équivalent
    return trips_with_dates.groupby("trip_id")["date"].first().to_dict()

def generate_graph_france(gtfs_zip, cities_fp):
    # Charger les données GTFS
    stops, stop_times, trips, routes, calendar = load_gtfs(gtfs_zip)

    # Charger les villes
    cities = load_cities(cities_fp)
    print(f"{len(cities)} villes chargées.")
    #print(f"Villes contenant Paris : {cities[cities[name_attr].str.contains('Paris', case=False, na=False)]}")
    print("Exemples de villes :", list(cities[config.name_attr].unique())[:10])

    # Aggréger les villes en aires
    # /!\ À faire avant le filtrage sur population, plus lent mais nécessaire pour récupérer toutes les bonnes communes dans une agglomération.
    communes, aires = aggregate_cities_to_aires(cities)
    print(f"{len(aires)} aires créées à partir des villes.") #unique?

    # Charger les gares
    gares_df = load_gares("resources/france/gares-de-voyageurs.geojson")
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

    # Créer le mapping stop_id -> mode de transport
    # trip_id -> route_type
    trips_routes = trips.merge(routes[["route_id", "route_type"]], on="route_id", how="left")
    # stop_id -> route_type
    stop_route_types = stop_times.merge(trips_routes[["trip_id", "route_type"]], on="trip_id", how="left")
    # Grouper par stop_id et collecter les modes de transport
    stop_modes = stop_route_types.groupby("stop_id")["route_type"].apply(set).to_dict()

    # Associer les gares aux villes
    gares_to_cities, _ = associate_stations_to_cities(cities, stops_to_gares_df, gares_df) # Villes
    print(f"{len(gares_to_cities)} gares associées à des villes.")

    print("Communes :", list(communes.items())[5:10])
    print("Aires :", list(aires.items())[5:10])

    # Créer l'association trip_id -> date de départ (année, mois, jour) avec le calendrier.
    calendar = create_calendar(calendar, trips)

    for pop_threshold in range(0, 501, 10): # 10k - 500k, graphes préconstruits
        print("[DEBUG] Filtrage des", ("aires" if config.by_agg else "villes"), "comme noeuds du graphe pour pop_threshold:", pop_threshold, "k")
        communes_bis, cities_bis = filter_agglomerations(communes, aires, cities, pop_threshold * 1000, config.by_agg)
        G_city = build_city_graph_with_trips(cities_bis, stop_modes, stops, stop_times, trips, calendar, stops_to_gares, gares_to_cities, communes_bis, aires)

        print(f"{G_city.number_of_nodes()} villes, {G_city.number_of_edges()} liaisons directes.")

        print(f"[DEBUG] Export du graph dans output/france_new/france_railway_network_pop_threshold_{pop_threshold}k.json.")
        export_data = export_graph_for_web_dashboard(
            G_city, 
            cities_bis, 
            output_file="output/france_new/france_railway_network_pop_threshold_{}k.json".format(pop_threshold),
            country_code='france',
            key_column=config.key
        )

def generate_graph_switzerland(gtfs_zip, cities_fp):
    """
        Comparé au traitement pour la France, le fichier train_stations.csv match déjà gares et villes avec le code UIC.
        Toutefois, le fichier ne contient pas les populations, on les ajoute manuellement avec un autre fichier après.
    """
    # Charger les données GTFS
    stops, stop_times, trips, routes, calendar = load_gtfs(gtfs_zip)

    # Charger les villes et gares : elles sont déjà associées.
    # On disjoint villes gares pour garder la même logique que pour la France et pour construire le graphe également, avec les villes comme noeuds du graphe.
    cities, gares_df = load_cities(cities_fp)

    # Correspondance stops -> gares (et donc villes)
    stops_to_gares, stops_to_gares_df = match_stops_to_gares(stops, gares_df)

    # Créer le mapping stop_id -> mode de transport
    # trip_id -> route_type
    trips_routes = trips.merge(routes[["route_id", "route_type"]], on="route_id", how="left")
    # stop_id -> route_type
    stop_route_types = stop_times.merge(trips_routes[["trip_id", "route_type"]], on="trip_id", how="left")
    # Grouper par stop_id et collecter les modes de transport
    stop_modes = stop_route_types.groupby("stop_id")["route_type"].apply(set).to_dict()

    # Associer les gares aux villes (merge sur le code UIC)
    gares_to_cities, _ = associate_stations_to_cities(cities, stops_to_gares_df, gares_df) # Villes
    print(f"{len(gares_to_cities)} gares associées à des villes.")

    # Créer l'association trip_id -> date de départ (année, mois, jour) avec le calendrier.
    calendar = create_calendar(calendar, trips)

    for pop_threshold in range(0, 501, 10):
        print("[DEBUG] Filtrage des villes comme noeuds du graphe pour pop_threshold:", pop_threshold, "k")
        cities_bis = filter_agglomerations(None, None, cities, pop_threshold * 1000, by_agg=False)
        G_city = build_city_graph_with_trips(cities_bis, stop_modes, stops, stop_times, trips, calendar, stops_to_gares, gares_to_cities, None, None)

        print(f"{G_city.number_of_nodes()} villes, {G_city.number_of_edges()} liaisons directes.")

        print(f"[DEBUG] Export du graph dans output/switzerland/switzerland_railway_network_pop_threshold_{pop_threshold}k.json.")
        export_data = export_graph_for_web_dashboard(
            G_city, 
            cities, 
            output_file="output/switzerland/switzerland_railway_network_pop_threshold_{}k.json".format(pop_threshold),
            country_code='switzerland',
            key_column=config.key
        )

if __name__ == "__main__":
    config.setup_config()
    print("[DEBUG] Configuration initialisée.")
    if config.country == 'france':
        gtfs_zip = "resources/gtfs/france/opendata-sncf-transport.zip"
        cities_fp = "resources/france/communes-france-datagouv-2025.csv"
        print("Génération du graphe pour la France...")
        generate_graph_france(gtfs_zip, cities_fp)
    elif config.country == 'switzerland':
        gtfs_zip = "resources/gtfs/switzerland/gtfs_open_transport_data_2025.zip"
        cities_fp = "resources/switzerland/train_stations.csv"
        print("Génération du graphe pour la Suisse...")
        generate_graph_switzerland(gtfs_zip, cities_fp)
    else:
        raise ValueError("Pays non supporté.")

    # Visualisation
    #fig = plot_interactive_city_graph(G_city, cities_bis, aires)