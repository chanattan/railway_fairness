from config import *
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

# haversine distance en km
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    dlon, dlat = radians(lon2 - lon1), radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def build_city_graph(cities, stops, stop_times, trips, stops_to_gares, gares_to_cities, agglomerations):
    import networkx as nx
    import pandas as pd

    G_city = nx.Graph()
    # Ensemble des agglomérations valides
    valid_agglos = set(cities[key])
    print(f"[DEBUG] Nombre de communes valides :", len(valid_agglos))

    # Ajouter les noeuds
    for _, row in cities.iterrows():
        city = row[key]
        name = row['nom_aire' if key == 'code_aire' else name_attr]
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
            gare = stops_to_gares.get(s.stop_id, {}).get('uic_code', None)
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

            # Récupérer stops originaux pour distance/duration
            st1 = stops.loc[stops.stop_id == s1.stop_id].iloc[0]
            st2 = stops.loc[stops.stop_id == s2.stop_id].iloc[0]
            dist = haversine(st1.stop_lon, st1.stop_lat, st2.stop_lon, st2.stop_lat)

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

def plot_city_graph(G_city, cities):
    pos = {str(row[key]): (row.geometry.x, row.geometry.y)
           for _, row in cities.iterrows() if row[key] in G_city.nodes}

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(G_city, pos, alpha=0.3)
    nx.draw_networkx_nodes(G_city, pos, node_size=20, node_color="red")
    labels = nx.get_node_attributes(G_city, "name")
    nx.draw_networkx_labels(G_city, pos, labels, font_size=6)
    plt.show(block=False)

def extract_train_type(stop_id):
    """
    Extrait le type de train depuis le stop_id
    Retourne: 'TER', 'TGV', 'INTERCITES', 'ICE', 'UNKNOWN'
    """
    if 'OCECar' in stop_id:
        return 'CAR'  # À filtrer
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

def build_city_graph_with_trips(cities, stops, stop_times, trips, stops_to_gares, gares_to_cities, communes, aires):
    """
    Version modifiée qui stocke les trip_ids et autres infos dans les arêtes
    """
    import networkx as nx
    import pandas as pd

    G_city = nx.Graph()
    # Ensemble des agglomérations valides
    valid_agglos = set(cities[key])
    print(f"[DEBUG] Nombre de communes valides :", len(valid_agglos))

    # Ajouter les noeuds (unités urbaines ou villes)
    for _, row in cities.iterrows():
        city = row[key]
        name = row['nom_aire' if key == 'code_aire' else name_attr]
        #if name is not None and not pd.isna(name) and "Toulouse" in name:
        #    print(f"[DEBUG] POPULATION {name} : {unites_urbaines[row['code_aire']]['population'] if key == 'code_aire' else city['population']}")
        #print("Aires :::::", aires.get(row['code_aire'], row) if key == 'code_aire' else row)
        #print("DEBUG city:", city, name, row['code_insee'], aires[row['code_aire']]['population'] if key == 'code_aire' else row['population'])
        pos = (aires[row['code_aire']]['geometry'].x, aires[row['code_aire']]['geometry'].y) if key == 'code_aire' else (row.geometry.x, row.geometry.y)
        G_city.add_node(city,
                        name=name,
                        insee=row['code_insee'],
                        population=aires.get(row['code_aire'], {}).get('population', row['population']) if key == 'code_aire' else row['population'],
                        x=pos[0],
                        y=pos[1])

    merged = stop_times.merge(trips, on="trip_id").sort_values(["trip_id", "stop_sequence"])

    stops_without_gare = 0
    gares_without_city = 0
    filtered_cars = 0

    # Dictionnaire pour stocker les trip_ids par arête
    edge_trips = {}

    for trip_id, group in merged.groupby("trip_id"):
        group = group.sort_values("stop_sequence")

        # Construire la séquence d'agglomérations avec les infos distances/durations
        agglos_sequence = []
        stops_sequence = []
        uic_sequence = []
        train_types = []

        for _, s in group.iterrows():
            train_type = extract_train_type(s.stop_id)
            if train_type == 'CAR':
                filtered_cars += 1
                continue

            gare = stops_to_gares.get(s.stop_id, {}).get('uic_code', None)
            if gare is None:
                stops_without_gare += 1
                continue

            city_codeinsee = gares_to_cities.get(gare, {}).get('code_insee', None)
            if city_codeinsee is None:
                gares_without_city += 1
                continue

            if key == 'code_aire':
                aggl = communes.get(city_codeinsee, {}).get('code_aire', None)
                if aggl is None:
                    continue

            agglos_sequence.append(aggl if key == 'code_aire' else city_codeinsee)
            stops_sequence.append(s)
            uic_sequence.append(gare)
            train_types.append(train_type)

        # Filtrer seulement les agglos présentes dans cities_df et supprimer doublons consécutifs
        reduced_seq = []
        reduced_stops = []
        reduced_types = []
        for idx, aggl in enumerate(agglos_sequence):
            if aggl in valid_agglos:
                if not reduced_seq or aggl != reduced_seq[-1]:
                    reduced_seq.append(aggl)
                    reduced_stops.append(stops_sequence[idx])
                    reduced_types.append(train_types[idx])

        # Créer les arêtes entre agglomérations consécutives
        for i in range(len(reduced_seq) - 1):
            c1, c2 = reduced_seq[i], reduced_seq[i + 1]
            s1, s2 = reduced_stops[i], reduced_stops[i + 1]
            type1, type2 = reduced_types[i], reduced_types[i + 1]

            # Prioriser le type le plus spécifique (non UNKNOWN)
            segment_type = type1 if type1 != 'UNKNOWN' else type2

            # Récupérer stops originaux pour distance/duration
            st1 = stops.loc[stops.stop_id == s1.stop_id].iloc[0]
            st2 = stops.loc[stops.stop_id == s2.stop_id].iloc[0]
            dist = haversine(st1.stop_lon, st1.stop_lat, st2.stop_lon, st2.stop_lat)

            t1 = pd.to_datetime(s1.departure_time, errors="coerce")
            t2 = pd.to_datetime(s2.arrival_time, errors="coerce")
            if pd.isna(t1) or pd.isna(t2):
                continue
            duration = (t2 - t1).seconds / 60

            # Stocker les trip_ids pour cette arête
            edge_key = tuple(sorted([c1, c2]))
            if edge_key not in edge_trips:
                edge_trips[edge_key] = []
            edge_trips[edge_key].append({
                'trip_id': trip_id,
                'route_id': s1.route_id,
                'stop1': st1.stop_name,
                'stop2': st2.stop_name,
                'departure': s1.departure_time,
                'arrival': s2.arrival_time,
                'train_type': segment_type
            })

            if G_city.has_edge(c1, c2):
                #if dist < G_city[c1][c2]["distance"]:
                if duration < G_city[c1][c2]["duration"]:
                    G_city[c1][c2]["duration"] = duration
                    G_city[c1][c2]["distance"] = dist
                    G_city[c1][c2]["trip_id"] = trip_id  # trip le plus rapide
            else:
                G_city.add_edge(c1, c2, distance=dist, duration=duration, trip_id=trip_id)

    # Ajouter les trip_ids aux arêtes
    for edge in G_city.edges():
        edge_key = tuple(sorted(edge))
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
            
            G_city[edge[0]][edge[1]]['trips'] = trips_data
            G_city[edge[0]][edge[1]]['nb_trips'] = len(trips_data)
            G_city[edge[0]][edge[1]]['train_types'] = train_type_counts
            
            # Type dominant pour cette arête
            dominant_type = max(train_type_counts, key=train_type_counts.get)
            G_city[edge[0]][edge[1]]['dominant_train_type'] = dominant_type

    print(f"[DEBUG] Isolated nodes: {len(list(nx.isolates(G_city)))}")
    G_city.remove_nodes_from(list(nx.isolates(G_city)))
    print(f"[DEBUG] {stops_without_gare} arrêts sans gare associée, {gares_without_city} gares sans ville associée.")
    return G_city

def plot_interactive_city_graph(G_city, cities, aires):
    """
    Crée un graphe interactif avec Plotly - Version avec points virtuels pour hover des arêtes
    Basée sur la solution du forum Plotly
    """
    # Position des nœuds
    pos = {str(row[key]): (aires[row['code_aire']]['geometry'].x, aires[row['code_aire']]['geometry'].y) if key == 'code_aire' else (row.geometry.x, row.geometry.y)
           for _, row in cities.iterrows() if row[key] in G_city.nodes}
    
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