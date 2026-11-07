# smhi_station_finder.py
import requests
import json

SMHI_BASE_URL = "https://opendata-download-metobs.smhi.se/api/version/latest"

CITIES_OF_INTEREST = [
    "stockholm",
    "göteborg",
    "malmö",
    "uppsala",
    "umeå",
    "kiruna",
    "åre",
]


def find_stations():
    """
    Query SMHI API for all stations with temperature (parameter 1) data.
    Print stations matching our cities of interest.
    """
    url = f"{SMHI_BASE_URL}/parameter/1.json"
    print(f"Fetching station list from: {url}")

    response = requests.get(url, timeout=30)
    print(f"Status: {response.status_code}")

    if response.status_code != 200:
        print("ERROR: Could not fetch station list")
        return

    data = response.json()
    stations = data.get("station", [])
    print(f"\nTotal stations with temperature data: {len(stations)}")
    print("\n--- Stations matching cities of interest ---")

    for station in stations:
        name = station.get("name", "").lower()
        station_id = station.get("id")
        active = station.get("active", False)
        for city in CITIES_OF_INTEREST:
            if city in name:
                print(
                    f"City match: {city.upper()} | Station: {station.get('name')} | ID: {station_id} | Active: {active}"
                )

    print("\n--- Full station list (all active) ---")
    for station in stations:
        if station.get("active"):
            print(f"ID: {station.get('id')} | Name: {station.get('name')}")


if __name__ == "__main__":
    find_stations()
