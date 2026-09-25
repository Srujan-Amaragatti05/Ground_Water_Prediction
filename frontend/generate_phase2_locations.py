"""
frontend/generate_phase2_locations.py
======================================
Utility to export curated, valid Phase-2 station locations with valid
5-consecutive-year sequences to frontend/phase2_locations.json for
the interactive All-India Leaflet map.
"""

import sys
import os
import json

DL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "dl")
if DL_DIR not in sys.path:
    sys.path.insert(0, DL_DIR)

import phase2_location_service as ls

def generate():
    ls._ensure_index()

    test_loc_id = 'West Bengal_Purulia_Santuri_Leadson_23.51992_86.82893'
    stations = []
    seen_districts = {}

    def add_station(loc_id, grp, is_featured=False):
        years = grp['YEAR'].values
        idx = ls._find_latest_consecutive_window(years)
        if idx is None:
            return False
        parts = loc_id.split('_')
        sp = ls._spatial_index[loc_id]
        lat = float(sp['LATITUDE'])
        lng = float(sp['LONGITUDE'])
        if not (6.0 <= lat <= 38.0 and 68.0 <= lng <= 98.0):
            return False
        avail_years = ls.get_available_prediction_years(loc_id)
        stations.append({
            'id': loc_id,
            'state': parts[0],
            'district': parts[1],
            'block': parts[2],
            'village': parts[3],
            'lat': round(lat, 5),
            'lng': round(lng, 5),
            'start': int(years[idx]),
            'end': int(years[idx + 4]),
            'pred': int(years[idx + 4] + 1),
            'available_years': avail_years,
            'recent_wl': round(float(grp.iloc[idx + 4]['WL(mbgl)']), 2),
            'featured': is_featured
        })
        return True

    if test_loc_id in ls._annual_index:
        add_station(test_loc_id, ls._annual_index[test_loc_id], is_featured=True)

    for loc_id, grp in ls._annual_index.items():
        if loc_id == test_loc_id:
            continue
        parts = loc_id.split('_')
        state = parts[0]
        district = parts[1]
        dist_key = f'{state}_{district}'
        if seen_districts.get(dist_key, 0) >= 5:
            continue
        if add_station(loc_id, grp):
            seen_districts[dist_key] = seen_districts.get(dist_key, 0) + 1

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase2_locations.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stations, f, separators=(',', ':'))

    print(f'Successfully wrote {len(stations)} stations to {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)')

if __name__ == '__main__':
    generate()
