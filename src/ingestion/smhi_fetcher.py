import os
import requests
import pandas as pd
import time

SMHI_BASE_URL = "https://opendata-download-metobs.smhi.se/api/version/latest"

STATIONS = {
    "stockholm": {"id": 98230, "name": "Stockholm", "wind_station_id": 97200},
    "gothenburg": {"id": 71420, "name": "Gothenburg"},
    "malmo": {"id": 52350, "name": "Malmö"},
    "uppsala": {"id": 97510, "name": "Uppsala"},
    "umea": {"id": 140480, "name": "Umeå"},
    "kiruna": {"id": 180940, "name": "Kiruna"},
    "are": {"id": 134110, "name": "Åre-Östersund"},
}

PARAMETERS = {
    "temperature": 1,
    "wind_speed": 4,
    "humidity": 6,
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def fetch_station_parameter(
    station_id: int, parameter: int, period: str = "corrected-archive"
) -> pd.DataFrame:
    # corrected-archive only supports CSV, not JSON
    url = f"{SMHI_BASE_URL}/parameter/{parameter}/station/{station_id}/period/{period}/data.csv"
    print(f"  Fetching parameter {parameter} from station {station_id}...")
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        print(
            f"  WARNING: Status {response.status_code} for station {station_id}, parameter {parameter}"
        )
        return pd.DataFrame()

    # SMHI CSV has metadata rows at the top before the actual data
    # We skip rows until we find the header line containing "Datum"
    lines = response.text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Datum") or line.startswith('"Datum'):
            header_idx = i
            break

    if header_idx is None:
        print(
            f"  WARNING: Could not find data header for station {station_id}, parameter {parameter}"
        )
        return pd.DataFrame()

    from io import StringIO

    data_str = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(data_str), sep=";", on_bad_lines="skip", low_memory=False)

    # SMHI CSV columns: Datum, Tid (UTC), Lufttemperatur, Kvalitet
    # Combine Datum + Tid into timestamp
    try:
        # Find date and time columns regardless of exact name
        date_col = [c for c in df.columns if "Datum" in c][0]
        time_col = [c for c in df.columns if "Tid" in c][0]
        df["timestamp"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            utc=True,
            errors="coerce",
        )
        value_col = [
            c
            for c in df.columns
            if c not in (date_col, time_col, "Kvalitet", "timestamp")
        ][0]
        df = df[["timestamp", value_col]].rename(columns={value_col: "value"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value", "timestamp"])
        df = df.set_index("timestamp").sort_index()
    except Exception as e:
        print(
            f"  WARNING: Could not parse CSV for station {station_id}, parameter {parameter}: {e}"
        )
        return pd.DataFrame()

    return df


def fetch_all_stations(start_year: int = 2018, end_year: int = 2024) -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    all_frames = []

    for city_key, station_info in STATIONS.items():
        station_id = station_info["id"]
        city_name = station_info["name"]
        print(f"\nFetching data for {city_name} (station {station_id})...")

        city_data = {}

        for param_name, param_id in PARAMETERS.items():
            # Use fallback wind station if defined and this is wind_speed
            fetch_id = station_id
            if param_name == "wind_speed" and "wind_station_id" in station_info:
                fetch_id = station_info["wind_station_id"]
                print(f"  Using fallback wind station {fetch_id} for {city_name}")
            df = fetch_station_parameter(fetch_id, param_id)
            time.sleep(0.5)

            if df.empty:
                print(f"  No data for {param_name} in {city_name}")
                continue

            df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

            if df.empty:
                print(
                    f"  No data in range {start_year}-{end_year} for {param_name} in {city_name}"
                )
                continue

            city_data[param_name] = df["value"]

        if not city_data:
            print(f"  Skipping {city_name} — no data retrieved")
            continue

        city_df = pd.DataFrame(city_data)
        city_df.index.name = "timestamp"
        city_df = city_df.resample("h").mean(numeric_only=True)
        city_df["city"] = city_name
        city_df = city_df.reset_index()
        city_df = city_df.dropna(subset=["temperature"])

        print(
            f"  {city_name}: {len(city_df)} hourly records ({city_df['timestamp'].min()} to {city_df['timestamp'].max()})"
        )
        all_frames.append(city_df)

    if not all_frames:
        print("ERROR: No data retrieved from SMHI.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["city", "timestamp"]).reset_index(drop=True)

    output_path = os.path.join(RAW_DIR, "smhi_weather_raw.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Total records: {len(combined)}")
    print(f"Cities: {combined['city'].unique()}")
    return combined


if __name__ == "__main__":
    print("=== SMHI Weather Data Fetcher ===")
    print("Fetching 2018-2024 hourly weather for 7 Swedish cities...")
    df = fetch_all_stations(start_year=2018, end_year=2024)
    if not df.empty:
        print("\nSample:")
        print(df.head(5).to_string())
