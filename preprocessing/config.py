from datetime import datetime, timedelta
import pytz

name_attr = "nom_standard"
key = 'code_commune' # [code_aire, code_commune, code_insee]
POPULATION_THRESHOLD = 0
country = 'france' # Supported: 'france', 'switzerland'
city_codes = None
by_agg = True # True => key = code_aire / agglo_id, False => key = code_insee / code_commune
COUNTRY_TIMEZONES = {
    "france": "Europe/Paris",
    "switzerland": "Europe/Zurich",
    "united_kingdom": "Europe/London"
}

def setup_config():
    """
        On pourrait également standardiser directement dans le code.
    """
    global key, POPULATION_THRESHOLD, city_codes, country
    if country == 'france':
        if not by_agg:
            key = 'code_insee'
        else:
            key = 'code_aire'
    elif country == 'switzerland':
        if not by_agg:
            key = 'code_commune'
        else:
            key = 'agglo_id'
    else:
        raise ValueError(f"Unknown country: {country}")
    city_codes = { # On aurait pu standardiser, j'ai souhaité garder les noms d'origine.
        'france': 'code_insee',
        'switzerland': 'code_commune'
    }

def get_departure_utc(trip_id, departure_time, calendar_dict, country='france'):
    """
    Convertit la date + heure locale d'un stop en RFC3339 UTC pour Google Routes API.
    
    Parameters
    ----------
    trip_id : str
        ID du trip
    departure_time : str
        départ du train au stop associé, format HH:MM:SS
    calendar_dict : dict
        Dictionnaire trip_id -> date (format YYYYMMDD)
    country : str
        Pays pour déterminer le fuseau horaire
    
        ex: '2025-09-12T08:37:00Z'
    """
    date_str = calendar_dict.get(trip_id)
    if date_str is None:
        raise ValueError(f"/!\ Trip_id {trip_id} absent du calendrier")

    hours, minutes, seconds = map(int, departure_time.split(":"))
    # Gestion heures >= 24 (c'est bien fait...)
    extra_days, hour = divmod(hours, 24)

    # datetime local
    date_dt = datetime.strptime(str(date_str), "%Y%m%d")
    dt_local = datetime(
        year=date_dt.year,
        month=date_dt.month,
        day=date_dt.day,
        hour=hour,
        minute=minutes,
        second=seconds
    ) + timedelta(days=extra_days)

    tz = pytz.timezone(COUNTRY_TIMEZONES[country.lower()])
    dt_localized = tz.localize(dt_local)

    # to UTC
    dt_utc = dt_localized.astimezone(pytz.UTC)

    # format RFC3339
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")