"""
getExamples.py — TESS Exoplanet Candidate Manifest Generator

Queries the ExoFOP TESS TOI catalog, filters targets by TFOPWG disposition label,
and writes a manifest CSV (classified_targets.csv) listing the targets to be downloaded.
The manifest includes ephemeris data (period, epoch, duration) required for phase-folding.
"""

import pandas as pd
import requests
import io

# Optional magnitude filter to ensure 2-minute cadence data availability
APPLY_MAG_FILTER = False 

def create_tess_csv():
    # The direct CSV export link for the TESS Objects of Interest catalog
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"

    # Headers to prevent the server from blocking the request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("Connecting to NASA/ExoFOP servers...")

    try:
        # Fetch the data
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Load into pandas
        df = pd.read_csv(io.StringIO(response.text))

        # Clean column names (lowercase and strip whitespace)
        df.columns = df.columns.str.strip().str.lower()

        # Fix 5: Apply magnitude filter only if enabled (ensures 2-minute cadence availability for bright stars)
        if APPLY_MAG_FILTER:
            df = df[df['tess mag'] < 13]

        # Ask the user how many examples they want
        false_num = int(input("How many false positives do you want to download? - "))
        true_num = int(input("How many true positives do you want to download? - "))

        # Fix 2: Expand the label pools
        # Positives: KP (Confirmed Planet), CP (Candidate Planet), PC (Planet Candidate)
        # Negatives: FP (False Positive), EB (Eclipsing Binary), NEB (Non-Eclipsing Binary)
        # Exclusions: FA, APC are dropped entirely
        positives_mask = df['tfopwg disposition'].isin(['KP', 'CP', 'PC'])
        negatives_mask = df['tfopwg disposition'].isin(['FP', 'EB', 'NEB'])

        positives_pool = df[positives_mask]
        negatives_pool = df[negatives_mask]

        # Fix 4: Drop rows with null or zero ephemeris values before sampling
        # These cannot be phase-folded and would cause failures downstream
        def drop_null_ephemeris(pool, device_name):
            initial_len = len(pool)
            pool = pool.dropna(subset=['period (days)', 'epoch (bjd)', 'duration (hours)'])
            pool = pool[(pool['period (days)'] != 0) &
                       (pool['epoch (bjd)'] != 0) &
                       (pool['duration (hours)'] != 0)]
            dropped = initial_len - len(pool)
            if dropped > 0:
                print(f"  {device_name}: Dropped {dropped} rows with null/zero ephemeris values")
            return pool

        positives_pool = drop_null_ephemeris(positives_pool, "Positives")
        negatives_pool = drop_null_ephemeris(negatives_pool, "Negatives")

        # Fix 3: Use .sample() instead of .head() to avoid systematic bias toward early rows
        # If either pool is too small, reduce both to match and warn the user
        true_available = len(positives_pool)
        false_available = len(negatives_pool)

        if true_available < true_num or false_available < false_num:
            actual_true_num = min(true_num, true_available)
            actual_false_num = min(false_num, false_available)
            # Reduce both to the minimum to maintain balance
            actual_true_num = min(actual_true_num, actual_false_num)
            actual_false_num = actual_true_num
            print(f"\nWARNING: Requested {true_num} positives and {false_num} negatives, but only "
                  f"{true_available} positives and {false_available} negatives available after filtering.")
            print(f"Sampling {actual_true_num} of each class for balance.")
        else:
            actual_true_num = true_num
            actual_false_num = false_num

        # Sample randomly but reproducibly with random_state=42
        if actual_true_num > 0 and len(positives_pool) > 0:
            positives = positives_pool.sample(n=actual_true_num, random_state=42)
        else:
            positives = pd.DataFrame()

        if actual_false_num > 0 and len(negatives_pool) > 0:
            negatives = negatives_pool.sample(n=actual_false_num, random_state=42)
        else:
            negatives = pd.DataFrame()

        # Fix 1: Include ephemeris columns in the manifest
        relevant_columns = [
            'tic id', 'toi', 'tfopwg disposition',
            'period (days)', 'epoch (bjd)', 'duration (hours)', 'sectors'
        ]

        # Combine positives and negatives
        final_data = pd.concat([positives, negatives])[relevant_columns]

        # Save to CSV
        final_data.to_csv("classified_targets.csv", index=False)

        print(f"\nSuccess! Created 'classified_targets.csv' with {len(final_data)} examples in total.")
        print(f"Contains: {len(positives)} Positive Targets (KP/CP/PC), {len(negatives)} Negative Targets (FP/EB/NEB)")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_tess_csv()