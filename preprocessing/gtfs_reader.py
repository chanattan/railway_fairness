import pandas as pd

resources = "resources/"
stops = pd.read_csv(resources + "stops.txt")
routes = pd.read_csv(resources + "routes.txt")
trips = pd.read_csv(resources + "trips.txt")
stoptimes = pd.read_csv(resources + "stop_times.txt")

tgv_routes = routes[routes['agency_id'].str.contains('TGV', na=False)]

# Pour chaque trip, extraire stop sequence minimal : créer arêtes entre arrêts consécutifs
print(tgv_routes)
