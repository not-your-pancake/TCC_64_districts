#!/usr/bin/env python3
"""
Generate tcc_pipeline/assets/zones.csv and name_map.csv.

Zone membership, assignment status and exemplar choices are carried over
verbatim from the project's original zone definition script.
"""
from __future__ import annotations

import csv
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "tcc_pipeline", "assets")
HIST_DIR = os.path.join(HERE, "1980-2024-dataset")

zones = {
    "1. Northwest Extreme Heat Zone (Dry Hot Plains)": {
        "confirmed": ["Bogra", "Pabna", "Rajshahi"],
        "provisional": ["Natore", "Naogaon", "Nawabganj", "Joypurhat", "Sirajganj",
                        "Rangpur", "Dinajpur", "Thakurgaon", "Panchagarh", "Nilphamari",
                        "Lalmonirhat", "Kurigram", "Gaibandha"],
    },
    "2. Central Urban Heat Zone (Urban Heat-Island Region)": {
        "confirmed": ["Dhaka", "Gazipur", "Narayanganj"],
        "provisional": ["Narsingdi", "Manikganj", "Munshiganj", "Tangail", "Kishoreganj",
                        "Mymensingh", "Netrakona", "Jamalpur", "Sherpur", "Faridpur",
                        "Madaripur", "Shariatpur", "Gopalganj", "Rajbari"],
    },
    "3. Southwest Hot-Dry Zone (Drought or Saline)": {
        "confirmed": ["Jessore", "Khulna", "Kushtia"],
        "provisional": ["Chuadanga", "Meherpur", "Jhenaidah", "Magura", "Narail",
                        "Satkhira", "Bagerhat"],
    },
    "4. Coastal Humid Heat Zone (Coastal Belt)": {
        "confirmed": ["Chittagong", "Feni", "Noakhali"],
        "provisional": ["Lakshmipur", "Chandpur", "Comilla", "Brahmanbaria", "Cox_s Bazar",
                        "Barisal", "Bhola", "Patuakhali", "Barguna", "Pirojpur", "Jhalokati"],
    },
    "5. Haor or Wetland Humid Zone (Haor Basin)": {
        "confirmed": ["Habiganj", "Moulvibazar", "Sylhet"],
        "provisional": ["Sunamganj"],
    },
    "6. Hill Tract Zone (Elevated Terrain)": {
        "confirmed": ["Khagrachhari", "Bandarban", "Rangamati"],
        "provisional": [],
    },
}

# Representative (densest) district per zone, used for main-text figures.
exemplars = {"Rajshahi", "Dhaka", "Kushtia", "Chittagong", "Sylhet", "Khagrachhari"}

# Dataset spelling -> geojson ADM2_EN spelling, for the choropleth builder.
known = {
    "Bogra": "Bogura", "Jessore": "Jashore", "Comilla": "Cumilla",
    "Chittagong": "Chattogram", "Barisal": "Barishal", "Cox_s Bazar": "Cox's Bazar",
    "Nawabganj": "Chapai Nawabganj", "Brahmanbaria": "Brahamanbaria",
    "Moulvibazar": "Maulvibazar",
}


def main():
    os.makedirs(ASSET_DIR, exist_ok=True)

    rows = []
    for zone_id, (zname, groups) in enumerate(zones.items(), 1):
        for status in ("confirmed", "provisional"):
            for d in groups[status]:
                rows.append([d, zone_id, zname, status, d in exemplars])
    rows.sort(key=lambda r: r[0])

    with open(os.path.join(ASSET_DIR, "zones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["district", "zone_id", "zone_name", "assignment_status", "is_exemplar"])
        w.writerows(rows)

    districts = [r[0] for r in rows]
    with open(os.path.join(ASSET_DIR, "name_map.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset_name", "geojson_adm2_en", "verified"])
        for d in districts:
            w.writerow([d, known.get(d, d), "yes" if d in known else "assumed"])

    print(f"zones.csv    : {len(rows)} districts")
    c = Counter(r[3] for r in rows)
    print(f"  confirmed  : {c['confirmed']}   provisional: {c['provisional']}")
    zc = Counter(str(r[1]) for r in rows)
    print(f"  per-zone   : {dict(sorted(zc.items()))}")
    print(f"  exemplars  : {sorted(exemplars)}")
    print(f"name_map.csv : {len(districts)} rows ({len(known)} explicit corrections)")

    # Cross-check every district against the actual dataset filenames.
    missing = [d for d in districts
               if not os.path.exists(os.path.join(HIST_DIR, f"{d}_historical_weather_1980_2024.csv"))]
    on_disk = {f.replace("_historical_weather_1980_2024.csv", "")
               for f in os.listdir(HIST_DIR) if f.endswith("_1980_2024.csv")}
    unmapped = sorted(on_disk - set(districts))

    print("\n--- dataset cross-check ---")
    print(f"  zones.csv districts without a history CSV : {missing or 'none'}")
    print(f"  history CSVs not present in zones.csv     : {unmapped or 'none'}")
    if not missing and not unmapped:
        print("  OK: 1:1 match between zones.csv and the dataset directory.")


if __name__ == "__main__":
    main()
