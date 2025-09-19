import json
from config import *
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, Optional

class RailwayNetworkExporter:
    """Export railway networks from NetworkX graphs to JSON format for web dashboard"""
    
    def __init__(self):
        self.country_configs = { # Statistics to be filled with real data.
            'france': {
                'name': 'France',
                'major_city_threshold': 500000,
                'medium_city_threshold': 100000,
                'high_speed_trips': 30,
                'intercity_trips': 20,
                'coordinate_system': 'WGS84'
            },
            'italy': {
                'name': 'Italy', 
                'major_city_threshold': 400000,
                'medium_city_threshold': 80000,
                'high_speed_trips': 25,
                'intercity_trips': 15,
                'coordinate_system': 'WGS84'
            },
            'switzerland': {
                'name': 'Switzerland',
                'major_city_threshold': 200000,
                'medium_city_threshold': 50000,
                'high_speed_trips': 15,
                'intercity_trips': 10,
                'coordinate_system': 'WGS84'
            },
            'spain': {
                'name': 'Spain',
                'major_city_threshold': 300000,
                'medium_city_threshold': 100000,
                'high_speed_trips': 20,
                'intercity_trips': 12,
                'coordinate_system': 'WGS84'
            },
            'uk': {
                'name': 'United Kingdom',
                'major_city_threshold': 400000,
                'medium_city_threshold': 150000,
                'high_speed_trips': 35,
                'intercity_trips': 25,
                'coordinate_system': 'WGS84'
            },
            'germany': {
                'name': 'Germany',
                'major_city_threshold': 500000,
                'medium_city_threshold': 200000,
                'high_speed_trips': 40,
                'intercity_trips': 20,
                'coordinate_system': 'WGS84'
            }
        }
    
    def clean_nan_values(self, obj):
        """Recursively clean NaN and invalid values from data structures"""
        if isinstance(obj, dict):
            return {k: self.clean_nan_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_nan_values(v) for v in obj]
        elif pd.isna(obj) or (isinstance(obj, str) and str(obj).lower() == 'nan'):
            return None
        elif isinstance(obj, (float, np.float64, np.float32)) and np.isnan(obj):
            return None
        elif isinstance(obj, (int, np.int64, np.int32)) and pd.isna(obj):
            return 0
        else:
            return obj
    
    def classify_city_type(self, connections: int, population: int, country_config: Dict) -> str:
        """Classify city type based on connections and population"""
        if (connections >= 10 or 
            population >= country_config['major_city_threshold']):
            return 'major'
        elif (connections >= 5 or 
              population >= country_config['medium_city_threshold']):
            return 'medium'
        else:
            return 'small'
    
    def classify_route_type(self, distance: float, nb_trips: int, country_config: Dict) -> str:
        """Classify route type based on distance and frequency"""
        if (nb_trips >= country_config['high_speed_trips'] and distance >= 300):
            return 'high_speed'
        elif (nb_trips >= country_config['intercity_trips'] and distance >= 150):
            return 'intercity'
        else:
            return 'regional'
    
    def export_networkx_graph(self, 
                             G_city: nx.Graph, 
                             cities_gdf: pd.DataFrame, 
                             country_code: str = 'france',
                             output_dir: str = "output",
                             output_file: str = "railway_network.json",
                             key_column: str = 'code_unite_urbaine') -> Dict[str, Any]:
        """
        Export NetworkX graph for web dashboard
        
        Parameters:
        - G_city: NetworkX graph with railway connections
        - cities_gdf: GeoDataFrame with city information
        - country_code: Country code (france, italy, etc.)
        - output_dir: Output directory
        - key_column: Column name for city identifier (e.g., 'code_unite_urbaine')
        """
        
        if country_code not in self.country_configs:
            print(f"Warning: Unknown country code '{country_code}', using France config")
            country_code = 'france'
        
        config = self.country_configs[country_code]
        
        # Prepare nodes data
        nodes = []
        for node_id in G_city.nodes():
            node_data = G_city.nodes[node_id]
            
            # Get node properties and handle NaN values
            name = node_data.get('name', str(node_id))
            if pd.isna(name) or name == 'nan' or str(name).lower() == 'nan' or name is None:
                # Try to get name from cities_gdf if available
                if cities_gdf is not None:
                    try:
                        city_match = None
                        if key_column in cities_gdf.columns:
                            city_match = cities_gdf[cities_gdf[key_column] == node_id]
                        
                        if city_match is not None and not city_match.empty:
                            name_col = name_attr if name_attr in cities_gdf.columns else 'name'
                            if name_col in city_match.columns:
                                matched_name = city_match.iloc[0][name_col]
                                if not pd.isna(matched_name):
                                    name = str(matched_name)
                                else:
                                    name = f"City_{node_id}"
                            else:
                                name = f"City_{node_id}"
                        else:
                            name = f"City_{node_id}"
                    except:
                        name = f"City_{node_id}"
                else:
                    name = f"City_{node_id}"
            
            # Get coordinates - try different possible coordinate fields
            x = node_data.get('x', node_data.get('lon', node_data.get('longitude', 0)))
            y = node_data.get('y', node_data.get('lat', node_data.get('latitude', 0)))
            
            # Handle NaN coordinates
            if pd.isna(x) or str(x).lower() == 'nan':
                x = 0.0
            if pd.isna(y) or str(y).lower() == 'nan':
                y = 0.0
            
            # If coordinates are 0, try to get from cities_gdf
            if (x == 0 and y == 0) and cities_gdf is not None:
                try:
                    # Try to find matching city in GeoDataFrame
                    city_match = None
                    
                    # Try different matching strategies
                    if key_column in cities_gdf.columns:
                        city_match = cities_gdf[cities_gdf[key_column] == node_id]
                    
                    if city_match is None or city_match.empty:
                        # Try matching by name
                        name_col = 'nom_standard' if 'nom_standard' in cities_gdf.columns else 'name'
                        if name_col in cities_gdf.columns:
                            city_match = cities_gdf[cities_gdf[name_col] == name]
                    
                    if city_match is not None and not city_match.empty:
                        city_info = city_match.iloc[0]
                        if hasattr(city_info, 'geometry') and city_info.geometry:
                            x, y = city_info.geometry.x, city_info.geometry.y
                        elif 'longitude_centre' in city_match.columns and 'latitude_centre' in city_match.columns:
                            x = city_info['longitude_centre']
                            y = city_info['latitude_centre']
                        
                except Exception as e:
                    print(f"Warning: Could not get coordinates for {name}: {e}")
            
            population = node_data.get('population', 50000)  # default
            connections = G_city.degree[node_id]
            city_type = self.classify_city_type(connections, population, config)
            
            node = {
                'id': str(node_id),
                'name': name,
                'x': float(x),
                'y': float(y),
                'connections': connections,
                'population': int(population),
                'type': city_type,
                # Additional node attributes (exclude coordinate fields to avoid duplication)
                **{k: v for k, v in node_data.items() 
                   if k not in ['name', 'x', 'y', 'lon', 'lat', 'longitude', 'latitude', 'population']}
            }
            nodes.append(node)
        
        # Prepare edges data
        edges = []
        for edge in G_city.edges(data=True):
            source, target, edge_data = edge
            
            # Skip edges where nodes are not in our node list
            source_exists = any(n['id'] == str(source) for n in nodes)
            target_exists = any(n['id'] == str(target) for n in nodes)
            if not (source_exists and target_exists):
                continue
            
            # Get edge properties with fallback values
            distance = edge_data.get('distance', edge_data.get('weight', 100))
            nb_trips = edge_data.get('nb_trips', edge_data.get('trips', 5))
            duration = edge_data.get('duration', edge_data.get('time', 60))
            
            # Ensure numeric values
            try:
                distance = float(distance)
                nb_trips = int(nb_trips)
                duration = float(duration)
            except (ValueError, TypeError):
                print(f"Warning: Invalid edge data for {source}-{target}, using defaults")
                distance, nb_trips, duration = 100.0, 5, 60.0
            
            # Classify route type
            route_type = self.classify_route_type(distance, nb_trips, config)
            
            edge_info = {
                'source': str(source),
                'target': str(target),
                'distance': distance,
                'duration': duration,
                'trips': nb_trips,
                'route_type': route_type,
                # Additional edge attributes
                **{k: v for k, v in edge_data.items() 
                   if k not in ['distance', 'duration', 'nb_trips', 'trips', 'weight', 'time']}
            }
            edges.append(edge_info)
        
        # Create export data structure
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'country_code': country_code,
                'country_name': config['name'],
                'graph_type': 'railway_network',
                'node_count': len(nodes),
                'edge_count': len(edges),
                'coordinate_system': config['coordinate_system'],
                'description': f'{config["name"]} Railway Network Data',
                'key_column': key_column,
                'exporter_version': '1.0'
            },
            'nodes': nodes,
            'edges': edges,
            'statistics': {
                'total_cities': len(nodes),
                'total_connections': len(edges),
                'major_cities': len([n for n in nodes if n['type'] == 'major']),
                'medium_cities': len([n for n in nodes if n['type'] == 'medium']),
                'small_cities': len([n for n in nodes if n['type'] == 'small']),
                'total_trips': sum(e['trips'] for e in edges),
                'avg_distance': sum(e['distance'] for e in edges) / len(edges) if edges else 0,
                'avg_duration': sum(e['duration'] for e in edges) / len(edges) if edges else 0,
                'high_speed_routes': len([e for e in edges if e['dominant_train_type'] == 'TGV']),
                'intercity_routes': len([e for e in edges if e['dominant_train_type'] == 'INTERCITES']),
                'regional_routes': len([e for e in edges if e['dominant_train_type'] == 'TER']),
                'max_connections': max([n['connections'] for n in nodes]) if nodes else 0,
                'total_population': sum([n['population'] for n in nodes]) if nodes else 0
            }
        }
        
        # Clean all data before JSON serialization
        export_data = self.clean_nan_values(export_data)
        
        # Save to JSON file
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {config['name']} network exported to {output_file}")
        print(f"   - {len(nodes)} nodes ({export_data['statistics']['major_cities']} major, "
              f"{export_data['statistics']['medium_cities']} medium, "
              f"{export_data['statistics']['small_cities']} small)")
        print(f"   - {len(edges)} edges ({export_data['statistics']['high_speed_routes']} high-speed, "
              f"{export_data['statistics']['intercity_routes']} intercity, "
              f"{export_data['statistics']['regional_routes']} regional)")
        
        return export_data

# Function to be called from data_test.py
def export_graph_for_web_dashboard(G_city: nx.Graph, 
                                  cities_gdf: pd.DataFrame = None, 
                                  output_file: str = "output/france_railway_network.json",
                                  country_code: str = 'france',
                                  key_column: str = 'code_aire'):
    """
    Convenience function to export graph from data_test.py
    
    Parameters:
    - G_city: NetworkX graph
    - cities_gdf: Cities GeoDataFrame (optional)
    - output_file: Output file path
    - country_code: Country identifier
    - key_column: Key column for city matching
    """
    
    exporter = RailwayNetworkExporter()
    
    # Extract directory and filename
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "output"
    
    return exporter.export_networkx_graph(
        G_city, 
        cities_gdf, 
        country_code=country_code,
        output_dir=output_dir,
        output_file=output_file,
        key_column=key_column
    )

# Example usage for your data_test.py:
"""
from data_exporter import export_graph_for_web_dashboard

# At the end of your data_test.py, add:
export_data = export_graph_for_web_dashboard(
    G_city, 
    cities, 
    output_file="output/france_railway_network.json",
    country_code='france',
    key_column='code_unite_urbaine'  # or whatever key you're using
)
"""