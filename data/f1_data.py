#!/usr/bin/env python3
"""
build_dataset.py

This script builds a SINDy-ready dataset from Formula 1 telemetry
using the FastF1 library.

Main features:
- Downloads F1 telemetry (FIA Live Timing via FastF1)
- Extracts telemetry of a selected driver
- Tries to use full telemetry (including X, Y, Z) when available
- Falls back to car data only when position data is not available
- Interpolates all channels to a constant sampling time dt
- Normalizes the data
- Saves X, t and metadata to disk

Dependencies:
    pip install fastf1 pandas numpy scipy scikit-learn

Example usage:

    python build_dataset.py \
        --year 2023 \
        --gp Monza \
        --session R \
        --driver VER \
        --dt 0.02 \
        --output-dir dataset_monza_ver
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import fastf1
from fastf1.core import Laps

from scipy.interpolate import interp1d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a SINDy-ready dataset from F1 telemetry using FastF1."
    )
    parser.add_argument("--year", type=int, default=2023,
                        help="Championship year (e.g., 2023)")
    parser.add_argument("--gp", type=str, default="Monza",
                        help="Grand Prix name (e.g., 'Monza', 'Silverstone')")
    parser.add_argument("--session", type=str, default="R",
                        help="Session type: R (Race), Q (Qualifying), FP1, FP2, FP3")
    parser.add_argument("--driver", type=str, default="VER",
                        help="Driver 3-letter code (e.g., VER, HAM, LEC)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Time step (in seconds) for interpolation (default: 0.02 = 50 Hz)")
    parser.add_argument("--output-dir", type=str, default="dataset_output",
                        help="Output directory where dataset files will be stored")
    parser.add_argument("--all-laps", action="store_true",
                        help="If set, use ALL laps instead of only the fastest lap")
    parser.add_argument("--min-lap-time", type=float, default=0.0,
                        help="Ignore laps shorter than this time (in seconds). 0 = no filtering")
    return parser.parse_args()


def enable_fastf1_cache(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def load_session(year: int, gp: str, session_type: str):
    print(f"[INFO] Loading session: {year} {gp} {session_type}")
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def select_laps(session, driver_code: str, use_all_laps: bool, min_lap_time: float) -> Laps:
    # NOTE: pick_driver is deprecated in FastF1 3.7+, but still works for now.
    # Future replacement: session.laps.pick_drivers([driver_code])
    laps_driver = session.laps.pick_driver(driver_code)
    if len(laps_driver) == 0:
        raise ValueError(f"No laps found for driver {driver_code}")

    # filter out in/out laps or laps shorter than min_lap_time
    if min_lap_time > 0.0:
        laps_driver = laps_driver[laps_driver["LapTime"].dt.total_seconds() >= min_lap_time]

    if len(laps_driver) == 0:
        raise ValueError(
            f"No valid laps for {driver_code} after applying min_lap_time={min_lap_time}"
        )

    if use_all_laps:
        print(f"[INFO] Using ALL laps for driver {driver_code}: {len(laps_driver)} laps")
        return laps_driver
    else:
        fastest_lap = laps_driver.pick_fastest()
        if fastest_lap is None:
            raise ValueError(f"No fastest lap found for driver {driver_code}")

        # IMPORTANT: slice laps_driver so we keep the original Session reference
        laps_fast = laps_driver[laps_driver["LapNumber"] == fastest_lap["LapNumber"]]

        print(
            f"[INFO] Using only the FASTEST lap for driver {driver_code}: "
            f"LapNumber={fastest_lap['LapNumber']}, LapTime={fastest_lap['LapTime']}"
        )
        return laps_fast


def telemetry_to_uniform_timeseries(tel: pd.DataFrame, dt: float, state_cols: list) -> tuple:
    """
    Converts telemetry of a single lap into:
        - t_new: uniformly spaced time axis (seconds)
        - X: interpolated state matrix (N, n_state)
    """
    if "Time" not in tel.columns:
        raise ValueError("Column 'Time' not found in telemetry.")

    t = (tel["Time"] - tel["Time"].iloc[0]).dt.total_seconds().to_numpy()
    t_min, t_max = t[0], t[-1]

    if t_max <= t_min:
        raise ValueError("Invalid time axis (t_max <= t_min).")

    t_new = np.arange(t_min, t_max, dt)

    X_cols = []
    for col in state_cols:
        if col not in tel.columns:
            raise ValueError(f"Column '{col}' not found in telemetry.")

        values = tel[col].to_numpy()

        # Convert boolean to float if needed
        if values.dtype == bool:
            values = values.astype(float)

        f = interp1d(
            t,
            values,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        X_cols.append(f(t_new))

    X = np.stack(X_cols, axis=1)  # shape: (N, n_state)

    return t_new, X


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("fastf1_cache")
    enable_fastf1_cache(cache_dir)

    # Load session
    session = load_session(args.year, args.gp, args.session)

    # Select laps
    laps = select_laps(
        session,
        driver_code=args.driver,
        use_all_laps=args.all_laps,
        min_lap_time=args.min_lap_time,
    )

    # Base state variables (always from car data)
    base_state_cols = [
        "Distance",  # distance along the lap
        "Speed",     # car speed
        "Throttle",  # throttle input
        "Brake",     # brake input
        "nGear",     # gear (mapped from Gear if needed)
        "RPM",       # engine RPM
    ]
    # Extra state variables if full telemetry with position is available
    xyz_cols = ["X", "Y", "Z"]

    state_cols = None  # will be determined on first successful lap

    all_t = []
    all_X = []
    lap_indices = []

    t_offset = 0.0
    n_laps_used = 0

    for idx, lap in laps.iterlaps():
        print(
            f"[INFO] Processing lap index={idx}, LapNumber={lap['LapNumber']}, "
            f"LapTime={lap['LapTime']}"
        )

        tel = None
        used_full_telemetry = False

        # Try full telemetry (car data + position X,Y,Z)
        try:
            full_tel = lap.get_telemetry().add_distance()
            # Some seasons use 'Gear' instead of 'nGear'
            if "nGear" not in full_tel.columns and "Gear" in full_tel.columns:
                full_tel["nGear"] = full_tel["Gear"]

            if all(c in full_tel.columns for c in xyz_cols):
                tel = full_tel
                used_full_telemetry = True
                print("[INFO] Using full telemetry (car data + X,Y,Z) for this lap.")
            else:
                print("[INFO] Full telemetry available but X,Y,Z missing; will fall back to car data.")
        except Exception as e:
            print(f"[INFO] Full telemetry not available for this lap: {e}")

        # Fallback: use car data only
        if tel is None:
            try:
                car_data = lap.get_car_data().add_distance()
                if "nGear" not in car_data.columns and "Gear" in car_data.columns:
                    car_data["nGear"] = car_data["Gear"]
                tel = car_data
                print("[INFO] Using car data only for this lap.")
            except Exception as e:
                print(f"[WARN] Unable to load car data for this lap: {e}")
                continue

        # Decide state_cols on first successful lap
        if state_cols is None:
            state_cols = list(base_state_cols)
            if used_full_telemetry and all(c in tel.columns for c in xyz_cols):
                state_cols.extend(xyz_cols)
                print(f"[INFO] State columns set to (with X,Y,Z): {state_cols}")
            else:
                print(f"[INFO] State columns set to (no X,Y,Z): {state_cols}")

        # Now build uniform time series
        try:
            t_new, X = telemetry_to_uniform_timeseries(
                tel=tel,
                dt=args.dt,
                state_cols=state_cols,
            )
        except Exception as e:
            print(f"[WARN] Error processing lap: {e}")
            continue

        # Offset the time axis so laps can be concatenated
        t_new = t_new + t_offset
        t_offset = t_new[-1] + args.dt

        all_t.append(t_new)
        all_X.append(X)
        lap_indices.extend([n_laps_used] * len(t_new))
        n_laps_used += 1

    if len(all_X) == 0:
        raise RuntimeError("No laps processed successfully. Dataset is empty.")

    t_all = np.concatenate(all_t, axis=0)
    X_all = np.concatenate(all_X, axis=0)
    lap_indices = np.array(lap_indices, dtype=int)

    print(f"[INFO] Total samples: {X_all.shape[0]}, state dimension: {X_all.shape[1]}")
    print(f"[INFO] Laps used: {n_laps_used}")
    print(f"[INFO] Final state columns: {state_cols}")

    # Normalization
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_all)

    # Save dataset
    np.save(output_dir / "X.npy", X_scaled)
    np.save(output_dir / "t.npy", t_all)
    np.save(output_dir / "lap_indices.npy", lap_indices)

    # Metadata
    metadata = {
        "year": args.year,
        "gp": args.gp,
        "session": args.session,
        "driver": args.driver,
        "dt": args.dt,
        "all_laps": args.all_laps,
        "min_lap_time": args.min_lap_time,
        "n_samples": int(X_all.shape[0]),
        "n_state": int(X_all.shape[1]),
        "n_laps_used": int(n_laps_used),
        "state_cols": state_cols,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Save scaler parameters for inverse-transform later
    scaler_info = {
        "X_mean": scaler_X.mean_.tolist(),
        "X_scale": scaler_X.scale_.tolist(),
    }
    with open(output_dir / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, indent=4)

    print(f"[INFO] Dataset saved in: {output_dir.resolve()}")
    print("[INFO] Files saved: X.npy, t.npy, lap_indices.npy, metadata.json, scaler.json")


if __name__ == "__main__":
    main()
