import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import numpy as np
from geopy.distance import geodesic
import networkx as nx
from scipy.spatial.distance import cdist

class GTFSRailDistanceCalculator:
    def __init__(self, gtfs_path, rails_geojson_path):
        """
        Calculateur de distances sans shapes GTFS - utilise directement les rails
        
        Args:
            gtfs_path: chemin vers le dossier GTFS
            rails_geojson_path: chemin vers le fichier GeoJSON des rails
        """
        self.gtfs_path = gtfs_path
        self.rails_geojson_path = rails_geojson_path
        
        # Charger les données
        self.load_data()
        self.create_rail_network()
        
    def load_data(self):
        """Charge toutes les données nécessaires"""
        print("Chargement des données GTFS...")
        
        # Fichiers GTFS
        self.stops = pd.read_csv(f"{self.gtfs_path}/stops.txt")
        self.routes = pd.read_csv(f"{self.gtfs_path}/routes.txt")
        self.trips = pd.read_csv(f"{self.gtfs_path}/trips.txt")
        self.stop_times = pd.read_csv(f"{self.gtfs_path}/stop_times.txt")
        
        # Rails GeoJSON
        print("Chargement des rails GeoJSON...")
        self.rails = gpd.read_file(self.rails_geojson_path)
        
        # Ne garder que les rails exploités (optionnel)
        if 'libelle' in self.rails.columns:
            exploited_rails = self.rails[self.rails['libelle'].str.contains('Exploit', case=False, na=False)]
            print(f"Rails exploités: {len(exploited_rails)} sur {len(self.rails)}")
            # Décommentez la ligne suivante si vous voulez filtrer
            # self.rails = exploited_rails
        
        # Créer un GeoDataFrame des gares
        self.gares_gdf = gpd.GeoDataFrame(
            self.stops,
            geometry=gpd.points_from_xy(self.stops.stop_lon, self.stops.stop_lat),
            crs="EPSG:4326"
        )
        
        print(f"✓ {len(self.stops)} gares chargées")
        print(f"✓ {len(self.rails)} segments de rails chargés")
        
        # Analyser les stop_times pour voir quelles gares sont réellement utilisées
        used_stops = set(self.stop_times['stop_id'].unique())
        self.active_stops = self.stops[self.stops['stop_id'].isin(used_stops)]
        print(f"✓ {len(self.active_stops)} gares actives dans stop_times.txt")
        
    def create_rail_network(self):
        """Crée un réseau de rails pour les calculs de routage"""
        print("Création du réseau ferroviaire...")
        
        # Convertir en projection métrique
        self.rails_projected = self.rails.to_crs("EPSG:4326")  # Lambert 93 pour France
        
        # Créer un graphe NetworkX à partir des rails
        self.rail_graph = nx.Graph()
        
        # Ajouter les segments de rails comme edges
        for idx, rail in self.rails_projected.iterrows():
            if rail.geometry.geom_type == 'LineString':
                coords = list(rail.geometry.coords)
                
                # Ajouter les points comme nodes
                for i in range(len(coords) - 1):
                    start_point = coords[i]
                    end_point = coords[i + 1]
                    
                    # Distance du segment
                    distance = Point(start_point).distance(Point(end_point))
                    
                    # Ajouter l'edge avec la distance comme poids
                    self.rail_graph.add_edge(
                        start_point, end_point, 
                        weight=distance,
                        rail_id=idx
                    )
        
        print(f"✓ Graphe créé avec {len(self.rail_graph.nodes)} nodes et {len(self.rail_graph.edges)} edges")
        
    def find_nearest_rail_point(self, gare_point, max_distance_m=1000):
        """
        Trouve le point le plus proche sur le réseau ferré.
        Retourne les coordonnées en WGS84 et la distance en mètres.
        """
        # Convertir la gare en projection métrique
        gare_gdf = gpd.GeoDataFrame([1], geometry=[gare_point], crs="EPSG:4326")
        gare_projected = gare_gdf.to_crs("EPSG:2154").iloc[0].geometry

        min_distance = float("inf")
        nearest_point = None

        # Chercher le point le plus proche sur tous les rails projetés
        for idx, rail in self.rails_projected.iterrows():
            if rail.geometry.geom_type == 'LineString':
                nearest_on_line = rail.geometry.interpolate(rail.geometry.project(gare_projected))
                distance = gare_projected.distance(nearest_on_line)

                if distance < min_distance and distance <= max_distance_m:
                    min_distance = distance
                    nearest_point = nearest_on_line

        if nearest_point is not None:
            # Reprojeter le point en WGS84 pour cohérence avec les gares
            point_gdf = gpd.GeoDataFrame([1], geometry=[nearest_point], crs="EPSG:2154")
            point_wgs84 = point_gdf.to_crs("EPSG:4326").iloc[0].geometry
            print(f"Point le plus proche trouvé à {min_distance:.0f} m")
            return (point_wgs84.x, point_wgs84.y, min_distance)
        else:
            print(f"Aucun rail trouvé dans un rayon de {max_distance_m} m")
            return None
        
    def calculate_rail_distance_direct(self, origin_stop_id, destination_stop_id):
        """
        Calcule la distance directement via le réseau de rails (sans shapes GTFS)
        Utilise la fonction corrigée find_nearest_rail_point pour projeter les gares.
        """
        print(f"\n=== Calcul direct via réseau ferré ===")
        print(f"Trajet: {origin_stop_id} → {destination_stop_id}")

        # Vérifier que les gares existent
        origin_stop = self.stops[self.stops['stop_id'] == origin_stop_id].iloc[0]
        dest_stop = self.stops[self.stops['stop_id'] == destination_stop_id].iloc[0]

        origin_point = Point(origin_stop.stop_lon, origin_stop.stop_lat)
        dest_point = Point(dest_stop.stop_lon, dest_stop.stop_lat)

        # Trouver les points les plus proches sur les rails avec les distances
        nearest_origin = self.find_nearest_rail_point(origin_point)
        nearest_dest = self.find_nearest_rail_point(dest_point)

        if not nearest_origin or not nearest_dest:
            print("Erreur: Impossible de projeter les gares sur le réseau ferré")
            return None

        (ox, oy, d_origin) = nearest_origin
        (dx, dy, d_dest) = nearest_dest

        # Ajouter ces points comme nœuds au graphe avec les distances respectives
        origin_node = (ox, oy)
        dest_node = (dx, dy)

        # Si ces nœuds n'existent pas encore, les connecter au graphe ferroviaire
        if origin_node not in self.rail_graph:
            # Relier au nœud du rail le plus proche (dans le même CRS)
            self.rail_graph.add_node(origin_node)
        if dest_node not in self.rail_graph:
            self.rail_graph.add_node(dest_node)

        # Calculer la plus courte distance sur le graphe
        try:
            rail_distance = nx.shortest_path_length(
                self.rail_graph,
                source=origin_node,
                target=dest_node,
                weight='weight'
            )
            print(f"Distance projetée origine-rail: {d_origin:.0f} m, destination-rail: {d_dest:.0f} m")
            print(f"Distance ferroviaire calculée: {rail_distance:.0f} m")
            return {
                "origin_to_rail": d_origin,
                "dest_to_rail": d_dest,
                "rail_distance": rail_distance
            }
        except nx.NetworkXNoPath:
            print("Erreur: Pas de chemin ferroviaire trouvé entre les deux points")
            return None
            
    def find_connected_stations(self, max_distance_km=500):
        """
        Trouve des paires de gares connectées sur le réseau pour les tests
        
        Args:
            max_distance_km: distance maximale pour considérer une connexion
            
        Returns:
            Liste de paires de gares connectées
        """
        print("\nRecherche de gares connectées...")
        
        connected_pairs = []
        
        # Prendre un échantillon de gares actives françaises
        french_stops = self.active_stops[
            (self.active_stops['stop_lat'] >= 41) & 
            (self.active_stops['stop_lat'] <= 52) &
            (self.active_stops['stop_lon'] >= -5) & 
            (self.active_stops['stop_lon'] <= 10)
        ].head(20)  # Limiter pour les tests
        
        for i, stop1 in french_stops.iterrows():
            for j, stop2 in french_stops.iterrows():
                if i >= j:  # Éviter les doublons
                    continue
                    
                # Distance euclidienne rapide pour pré-filtrer
                euclidean_dist = geodesic(
                    (stop1['stop_lat'], stop1['stop_lon']),
                    (stop2['stop_lat'], stop2['stop_lon'])
                ).kilometers
                
                if 10 <= euclidean_dist <= max_distance_km:  # Au moins 10km, max 500km
                    connected_pairs.append((stop1['stop_id'], stop2['stop_id'], euclidean_dist))
        
        # Trier par distance croissante
        connected_pairs.sort(key=lambda x: x[2])
        
        return connected_pairs[:5]  # Retourner les 5 premiers
        
# Exemple d'utilisation adapté
def main():
    # Initialiser le calculateur
    calculator = GTFSRailDistanceCalculator(
        gtfs_path="resources/gtfs/france/opendata-sncf-transport/",
        rails_geojson_path="resources/lignes_rfn.geojson"
    )
    
    # Chercher des paires de gares connectées pour tester
    connected_pairs = calculator.find_connected_stations()
    
    if len(connected_pairs) > 0 and False:
        print(f"\n=== Test avec {len(connected_pairs)} paires de gares connectées ===")
        
        for i, (stop1, stop2, euclidean_dist) in enumerate(connected_pairs):
            print(f"\n--- Test {i+1}: Distance euclidienne ~{euclidean_dist:.1f}km ---")
            
            result = calculator.calculate_rail_distance_direct(stop1, stop2)
            
            if "error" in result:
                print(f"Erreur: {result['error']}")
            else:
                print(f"Trajet: {result['origin']['name']} → {result['destination']['name']}")
                print(f"Distance ferroviaire: {result.get('distance_km', 'N/A'):.1f} km")
                print(f"Rapport ferroviaire/euclidienne: {result.get('distance_km', 0)/euclidean_dist:.2f}")
                
            # Tester seulement les 2 premiers pour éviter de surcharger
            if i >= 1:
                break
    else:
        print("Aucune paire de gares connectées trouvée")
        
        # Fallback: tester avec gares françaises proches
        print("\n=== Test manuel avec gares françaises ===")
        french_stops = calculator.active_stops[
            (calculator.active_stops['stop_lat'] >= 41) & 
            (calculator.active_stops['stop_lat'] <= 52) &
            (calculator.active_stops['stop_lon'] >= -5) & 
            (calculator.active_stops['stop_lon'] <= 10)
        ]
        
        if len(french_stops) >= 2:
            stop1 = french_stops.iloc[0]['stop_id']
            stop2 = french_stops.iloc[1]['stop_id']
            
            result = calculator.calculate_rail_distance_direct(stop1, stop2)
            print(f"Test: {stop1} → {stop2}")
            print(f"Résultat: {result}")

if __name__ == "__main__":
    main()