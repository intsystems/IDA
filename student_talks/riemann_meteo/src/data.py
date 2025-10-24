import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional


def synthetic_dataset(n_trials=100, n_channels=5, n_times=50, random_state=0):
    """Generate synthetic multivariate timeseries and target lat/lon.

    Returns:
        X: ndarray (n_trials, n_channels, n_times)
        y: ndarray (n_trials, 2) -- lat, lon in degrees
    """
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_trials, n_channels, n_times)
    # Make a simple relation: target depends on average of channel 0 and 1
    lat = (X[:, 0, :].mean(axis=1) * 5.0) + 25.0
    lon = (X[:, 1, :].mean(axis=1) * 5.0) - 60.0
    y = np.vstack([lat, lon]).T
    return X, y


def _parse_hurdat2_block(lines: List[str]) -> Dict:
    """Parse a single storm block from HURDAT2 lines (block starting with header).

    Returns a dict with keys: 'name', 'id', 'records' where records is a list of dicts
    containing 'time', 'lat', 'lon', 'wind', 'pressure', and 'status'.
    """
    header = lines[0].strip().split(',')
    storm_id = header[0].strip()
    storm_name = header[1].strip()
    nrecs = int(header[2].strip())
    records = []
    for r in lines[1:1 + nrecs]:
        parts = [p.strip() for p in r.split(',')]
        # Format per HURDAT2: date(YYYYMMDD), time(HHMM), record_id, status, lat, lon, max_wind, pres, ...
        try:
            dt = datetime.strptime(parts[0] + parts[1], '%Y%m%d%H%M')
        except Exception:
            dt = None
        status = parts[3]
        # lat like 25.0N, lon like 75.0W
        def _parse_coord(c):
            if c.endswith('N') or c.endswith('S'):
                val = float(c[:-1]) * (1 if c.endswith('N') else -1)
            elif c.endswith('E') or c.endswith('W'):
                # sometimes lat/long fields may be swapped; handle generically
                val = float(c[:-1]) * (1 if c.endswith('E') else -1)
            else:
                val = float(c)
            return val

        lat = _parse_coord(parts[4])
        lon = _parse_coord(parts[5])
        wind = None
        pres = None
        try:
            wind = float(parts[6]) if parts[6] != '' else None
        except Exception:
            wind = None
        try:
            pres = float(parts[7]) if parts[7] != '' else None
        except Exception:
            pres = None
        records.append({'time': dt, 'lat': lat, 'lon': lon, 'wind': wind, 'pressure': pres, 'status': status})
    return {'id': storm_id, 'name': storm_name, 'records': records}


def load_hurdat2_year(file_path: str, year: int = 2021, min_points: int = 6, n_times: Optional[int] = 50) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load HURDAT2 file and extract tracks for storms that occurred in the given year.

    Args:
        file_path: path to HURDAT2 text file.
        year: year to filter storms (e.g., 2021).
        min_points: minimum number of track points to keep a storm.
        n_times: if provided, resample/pad each track to this many time steps.

    Returns:
        X: ndarray (n_storms, n_channels, n_times) with channels=[lat, lon, wind, pressure]
        y: ndarray (n_storms, 2) -- final lat/lon of the storm (target)
        meta: list of dicts with storm metadata (id, name, original_length)

    Notes:
        - HURDAT2 file format is plain text with storm header lines followed by record lines.
        - This loader is conservative and aims to be robust to small format variants.
    """
    with open(file_path, 'r', encoding='utf8') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]

    # split into blocks where header lines have 3 comma-separated fields (id,name,nrecs)
    blocks = []
    i = 0
    while i < len(lines):
        header = lines[i]
        parts = [p.strip() for p in header.split(',')]
        if len(parts) < 3:
            i += 1
            continue
        try:
            nrecs = int(parts[2])
        except Exception:
            # not a header, skip
            i += 1
            continue
        block = lines[i:i + 1 + nrecs]
        blocks.append(block)
        i += 1 + nrecs

    tracks = []
    for b in blocks:
        parsed = _parse_hurdat2_block(b)
        # check if any record has the requested year
        years = [r['time'].year for r in parsed['records'] if r['time'] is not None]
        if any(y == year for y in years):
            tracks.append(parsed)

    # prepare arrays
    prepared = []
    meta = []
    for t in tracks:
        recs = t['records']
        # filter out records with missing time
        recs = [r for r in recs if r['time'] is not None]
        if len(recs) < min_points:
            continue
        lats = np.array([r['lat'] for r in recs], dtype=float)
        lons = np.array([r['lon'] for r in recs], dtype=float)
        winds = np.array([r['wind'] if r['wind'] is not None else np.nan for r in recs], dtype=float)
        press = np.array([r['pressure'] if r['pressure'] is not None else np.nan for r in recs], dtype=float)

        # stack channels: lat, lon, wind, pressure
        channels = np.vstack([lats, lons, winds, press])

        # resample or pad/truncate to n_times along time axis
        if n_times is not None:
            # simple linear interpolation along time dimension
            orig_t = np.linspace(0, 1, channels.shape[1])
            new_t = np.linspace(0, 1, n_times)
            new_ch = np.empty((channels.shape[0], n_times), dtype=float)
            for ci in range(channels.shape[0]):
                x = channels[ci]
                # handle NaNs by linear interpolation over valid points
                valid = ~np.isnan(x)
                if valid.sum() == 0:
                    new_ch[ci, :] = 0.0
                elif valid.sum() == 1:
                    new_ch[ci, :] = x[valid][0]
                else:
                    new_ch[ci, :] = np.interp(new_t, orig_t[valid], x[valid])
            channels = new_ch

        prepared.append(channels)
        meta.append({'id': t['id'], 'name': t['name'], 'length': len(recs)})

    if not prepared:
        return np.empty((0, 4, n_times if n_times is not None else 0)), np.empty((0, 2)), meta

    X = np.stack(prepared, axis=0)
    # target: final lat/lon of each storm (last observed point)
    y = np.array([[p[0, -1], p[1, -1]] for p in prepared], dtype=float)
    return X, y, meta

