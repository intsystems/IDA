import numpy as np
import os
from math import radians, cos, sin

def latlon_to_xyz(lat, lon):
    # Convert degrees to radians
    lat_rad = radians(lat)
    lon_rad = radians(lon)
    x = cos(lat_rad) * cos(lon_rad)
    y = cos(lat_rad) * sin(lon_rad)
    z = sin(lat_rad)
    return x, y, z

def parse_hurdat2_to_npz(hurdat_path, npz_path):
    subj = []
    seq = []
    ids = []
    info = np.array([
        ['Cyclone Nr', 'Name', 'Number of Entries'],
        ['Date', 'Time', 'Status Type', 'Latitude', 'Longitude', 'Max Wind', 'Min Pressure', 'x', 'y', 'z']
    ], dtype=object)

    with open(hurdat_path, 'r', encoding='utf8') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]

    i = 0
    idx = 0
    while i < len(lines):
        header = lines[i]
        parts = [p.strip() for p in header.split(',')]
        if len(parts) < 3:
            i += 1
            continue
        try:
            nrecs = int(parts[2])
        except Exception:
            i += 1
            continue
        storm_id = parts[0]
        name = parts[1]
        subj.append([storm_id, name, nrecs])
        ids.append(idx)
        for j in range(nrecs):
            rec = lines[i + 1 + j]
            rec_parts = [p.strip() for p in rec.split(',')]
            # Defensive: fill missing fields with empty string
            while len(rec_parts) < 7:
                rec_parts.append('')
            date = rec_parts[0]
            time = rec_parts[1]
            rec_id = rec_parts[2] if len(rec_parts) > 2 else ''
            status = rec_parts[3] if len(rec_parts) > 2 else ''
            lat_str = rec_parts[4] if len(rec_parts) > 3 else ''
            lon_str = rec_parts[5] if len(rec_parts) > 4 else ''
            wind = rec_parts[6] if len(rec_parts) > 5 else ''
            pres = rec_parts[7] if len(rec_parts) > 6 else ''
            # Parse lat/lon
            lat = float(lat_str[:-1]) * (1 if lat_str.endswith('N') else -1 if lat_str.endswith('S') else 1)

            lon = float(lon_str[:-1]) * (1 if lon_str.endswith('E') else -1 if lon_str.endswith('W') else 1)
            # Convert to x, y, z
            x, y, z = latlon_to_xyz(lat, lon)
            # Store all fields
            seq.append([date, time, status, lat, lon, wind, pres, x, y, z])
            idx += 1
        i += 1 + nrecs
    subj = np.array(subj, dtype=object)
    seq = np.array(seq, dtype=object)
    ids = np.array(ids, dtype=int)
    np.savez(npz_path, subj=subj, seq=seq, ids=ids, info=info)
    print(f"Saved NPZ to {npz_path} with subj.shape={subj.shape}, seq.shape={seq.shape}, ids.shape={ids.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert HURDAT2 raw file to NPZ format for hurricane analysis.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw HURDAT2 file")
    parser.add_argument("--output", type=str, required=True, help="Path to output NPZ file")
    args = parser.parse_args()
    parse_hurdat2_to_npz(args.input, args.output)
