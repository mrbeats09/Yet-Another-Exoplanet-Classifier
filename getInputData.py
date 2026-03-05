"""
getInputData.py — TESS Light Curve Processing Pipeline (Async Version)

Reads the manifest of TESS exoplanet candidates (classified_targets.csv) and
executes a three-phase pipeline:

Phase 1 (Sequential): Query MAST via lightkurve to discover available FITS files
for each target. Lightweight metadata queries that share internal astroquery state.

Phase 2 (Async): Download all discovered FITS files concurrently using aiohttp
with a semaphore cap of 15 simultaneous connections. This provides ~15x speedup
over sequential downloading. Files already in cache are skipped automatically,
allowing safe resumption if interrupted.

Phase 3 (Sequential): For each target, load its locally-cached FITS files,
replicate lightkurve's sector-stitching normalization, and apply the full
processing pipeline (quality masking, sigma-clipping, out-of-transit detection,
flux and centroid normalization, phase-folding, binning). Write each result
incrementally to the output CSV (tess_training_data.csv) formatted for input
to a 1D CNN. The output has 3,006 columns: metadata + 1000 flux bins +
1000 m1 centroid bins + 1000 m2 centroid bins.

Dependencies: pandas, lightkurve, numpy, scipy.stats, aiohttp, asyncio, astropy, tqdm
"""

import pandas as pd
import lightkurve as lk
import numpy as np
import os
import time
import random
import warnings
import asyncio
import aiohttp
from scipy.stats import binned_statistic
from astropy.io import fits
from tqdm import tqdm


# ============================================================================
# Module-level constants and helper functions
# ============================================================================

# Module-level constant — number of phase bins for the CNN input
NUM_BINS = 1000


def interpolate_nans(arr):
    """Fill any NaN values in arr using linear interpolation from neighbouring
    populated bins. This handles sparse phase coverage without introducing
    large discontinuities in the input tensor.

    Args:
        arr: array of length NUM_BINS, possibly containing NaN values

    Returns:
        arr with all NaN values filled by interpolation
    """
    nans = np.isnan(arr)
    if nans.any():
        x = np.arange(NUM_BINS)
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


# ============================================================================
# Phase 1: Sequential MAST search collection
# ============================================================================

def collect_search_results(manifest, cache_dir):
    """
    Phase 1: Query MAST for each target's available FITS files.
    Returns a list of (row, file_list) tuples where file_list contains
    the download URL, filename, and intended local path for each FITS file.
    This phase is sequential because MAST search queries share internal
    astroquery cache state that is not safe to access concurrently.
    """
    search_results = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc="Phase 1/3: Querying MAST", unit="target"):

        # Rate limiting: space out search queries to avoid hitting MAST limits
        time.sleep(random.uniform(0.1, 0.5))

        try:
            tic_id = f"TIC {int(row['tic id'])}"

            # Two-stage cadence fallback: prefer 2-minute, fall back to 10-minute
            search = lk.search_lightcurve(tic_id, mission="TESS",
                                           author="SPOC", cadence="short")
            if len(search) == 0:
                search = lk.search_lightcurve(tic_id, mission="TESS",
                                               author="SPOC", cadence="long")
            if len(search) == 0:
                with open("failed_targets.log", "a") as f:
                    f.write(f"TIC {row['tic id']}: ValueError: No SPOC light curves found\n")
                continue

            # Cap at 5 sectors — diminishing returns beyond this for phase-folded CNNs
            MAX_SECTORS = 5
            if len(search) > MAX_SECTORS:
                search = search[-MAX_SECTORS:]

            # Extract download URLs and filenames from the search result table.
            # lightkurve's SearchResult stores metadata in search.table, an astropy Table.
            # 'dataURI' contains the MAST URI (e.g. mast:TESS/product/filename.fits).
            # We construct the full HTTPS download URL from it.
            file_list = []
            for i in range(len(search)):
                try:
                    # Get the data URI for this result row
                    uri = search.table['dataURI'][i]
                    filename = search.table['productFilename'][i]
                    # Construct the full MAST download URL from the URI
                    url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={uri}"
                    local_path = os.path.join(cache_dir, filename)
                    file_list.append({
                        'url': url,
                        'filename': filename,
                        'local_path': local_path
                    })
                except (KeyError, IndexError):
                    # If URI extraction fails for one sector, skip that sector only
                    continue

            if len(file_list) == 0:
                with open("failed_targets.log", "a") as f:
                    f.write(f"TIC {row['tic id']}: ValueError: Could not extract any download URLs\n")
                continue

            search_results.append((row, file_list))

        except Exception as e:
            with open("failed_targets.log", "a") as f:
                f.write(f"TIC {row['tic id']}: {type(e).__name__}: {str(e)}\n")
            continue

    tqdm.write(f"Phase 1 complete: {len(search_results)} targets with valid search results.")
    return search_results


# ============================================================================
# Phase 2: Async FITS file downloading
# ============================================================================

async def async_download_all(search_results):
    """
    Phase 2: Download all FITS files concurrently using aiohttp.
    A semaphore caps simultaneous connections at 15 to respect MAST rate limits
    while still providing ~15x speedup over sequential downloading.
    Files already present in the cache are skipped automatically.
    """
    # Flatten search_results into a single list of files to download,
    # filtering out any that already exist in the cache.
    all_files = []
    for row, file_list in search_results:
        for f in file_list:
            if not os.path.exists(f['local_path']):
                all_files.append(f)

    if len(all_files) == 0:
        tqdm.write("Phase 2: All files already cached — skipping download.")
        return

    tqdm.write(f"Phase 2/3: Downloading {len(all_files)} FITS files...")

    # Semaphore caps concurrent connections to 15 — enough for significant
    # speedup without overwhelming MAST's servers.
    semaphore = asyncio.Semaphore(15)

    # Configure aiohttp with a generous timeout (5 minutes per file) and
    # limit the connector to 20 total connections.
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=20)

    async def download_one(session, file_info, pbar):
        """Download a single FITS file, skipping if already cached."""
        async with semaphore:
            try:
                async with session.get(file_info['url']) as response:
                    if response.status == 200:
                        content = await response.read()
                        # Write atomically: write to a temp file then rename,
                        # so a crashed download never leaves a partial FITS file
                        # in the cache that would be mistaken for a complete one.
                        tmp_path = file_info['local_path'] + '.tmp'
                        with open(tmp_path, 'wb') as f:
                            f.write(content)
                        os.rename(tmp_path, file_info['local_path'])
                    else:
                        with open("failed_targets.log", "a") as f:
                            f.write(f"HTTP {response.status}: {file_info['filename']}\n")
            except Exception as e:
                with open("failed_targets.log", "a") as f:
                    f.write(f"Download error {file_info['filename']}: "
                            f"{type(e).__name__}: {str(e)}\n")
            finally:
                pbar.update(1)

    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=connector) as session:
        with tqdm(total=len(all_files),
                  desc="Phase 2/3: Downloading FITS", unit="file") as pbar:
            tasks = [download_one(session, f, pbar) for f in all_files]
            await asyncio.gather(*tasks)

    tqdm.write("Phase 2 complete.")


# ============================================================================
# Phase 3: Process cached FITS files and apply pipeline
# ============================================================================

def process_single_target(row, file_list):
    """
    Phase 3: Load locally-cached FITS files for one target, replicate
    lightkurve's stitch() normalization, and apply the full processing
    pipeline. Returns the completed entry dict or None on failure.
    All processing logic from quality masking onwards is unchanged.
    """
    try:
        tic_id = f"TIC {int(row['tic id'])}"

        # Verify that at least one FITS file exists for this target.
        # Files missing from cache mean the Phase 2 download failed for them.
        available = [f for f in file_list if os.path.exists(f['local_path'])]
        if len(available) == 0:
            raise ValueError("No cached FITS files found — all sector downloads failed")

        # Read each cached FITS file and extract the four required columns.
        # Replicate lightkurve's stitch() behaviour: normalise each sector's
        # PDCSAP_FLUX by its own median before concatenating, preventing
        # inter-sector baseline jumps from dominating the time series.
        time_parts, flux_parts, m1_parts, m2_parts, quality_parts = [], [], [], [], []

        for file_info in available:
            try:
                with fits.open(file_info['local_path'], memmap=False) as hdul:
                    lc_data = hdul['LIGHTCURVE'].data

                    t    = lc_data['TIME'].astype(np.float64)
                    f    = lc_data['PDCSAP_FLUX'].astype(np.float64)
                    q    = lc_data['QUALITY'].astype(np.int32)
                    m1   = lc_data['MOM_CENTR1'].astype(np.float64)
                    m2   = lc_data['MOM_CENTR2'].astype(np.float64)

                    # Drop cadences with NaN time (sometimes present at sector edges)
                    valid = np.isfinite(t)
                    t, f, q, m1, m2 = t[valid], f[valid], q[valid], m1[valid], m2[valid]

                    # Replicate stitch() normalization: divide this sector's flux
                    # by its median so all sectors share a common baseline of ~1.0
                    sector_median = np.nanmedian(f)
                    if sector_median == 0 or not np.isfinite(sector_median):
                        continue  # Skip sectors with degenerate flux
                    f = f / sector_median

                    time_parts.append(t)
                    flux_parts.append(f)
                    quality_parts.append(q)
                    m1_parts.append(m1)
                    m2_parts.append(m2)

            except Exception as e:
                # One bad sector should not abort the whole target — skip it
                with open("failed_targets.log", "a") as log:
                    log.write(f"TIC {row['tic id']} sector {file_info['filename']}: "
                              f"{type(e).__name__}: {str(e)}\n")
                continue

        if len(time_parts) == 0:
            raise ValueError("No valid sectors could be read from cache")

        # Concatenate all sectors into single arrays, then sort by time
        time_arr    = np.concatenate(time_parts)
        flux_arr    = np.concatenate(flux_parts)
        quality_arr = np.concatenate(quality_parts)
        m1_arr      = np.concatenate(m1_parts)
        m2_arr      = np.concatenate(m2_parts)

        sort_idx  = np.argsort(time_arr)
        time_arr  = time_arr[sort_idx]
        flux_arr  = flux_arr[sort_idx]
        quality_arr = quality_arr[sort_idx]
        m1_arr    = m1_arr[sort_idx]
        m2_arr    = m2_arr[sort_idx]

        # Apply quality mask using the concatenated quality array.
        # This replicates what lightkurve does when lc.quality.value == 0.
        good = quality_arr == 0
        time_arr = time_arr[good]
        flux_arr = flux_arr[good]
        m1_arr   = m1_arr[good]
        m2_arr   = m2_arr[good]

        if len(time_arr) == 0:
            raise ValueError("No good cadences after quality masking")

        # ====================================================================
        # All code from this point onwards is completely unchanged
        # ====================================================================

        # Apply sigma-clipping to remove residual outliers
        flux_median = np.nanmedian(flux_arr)
        flux_std = np.nanstd(flux_arr)
        good_mask = np.abs(flux_arr - flux_median) <= 5 * flux_std
        time_arr = time_arr[good_mask]
        flux_arr = flux_arr[good_mask]
        m1_arr = m1_arr[good_mask]
        m2_arr = m2_arr[good_mask]

        if len(time_arr) == 0:
            raise ValueError("No cadences survive sigma-clipping")

        # Compute phase early and identify out-of-transit cadences
        duration_days = row['duration (hours)'] / 24.0
        period = row['period (days)']
        epoch = row['epoch (bjd)'] - 2457000.0  # Convert BJD to BTJD

        # Compute phase so it can be reused for both the OOT mask and later binning
        phase = ((time_arr - epoch) / period) % 1.0
        phase[phase > 0.5] -= 1.0  # Centre transit at phase 0

        # Out-of-transit mask: a cadence is out-of-transit if its phase falls
        # outside the transit window of ±1.5 × (duration / period).
        half_window = 1.5 * (duration_days / period)
        oot_mask = np.abs(phase) > half_window

        # Normalise the flux channel
        oot_flux = flux_arr[oot_mask]
        if len(oot_flux) < 10:
            raise ValueError("Insufficient out-of-transit cadences for flux normalisation")
        flux_arr = flux_arr / np.nanmedian(oot_flux)

        # Normalise the centroid channels
        # Baseline subtraction (already present)
        m1_arr = m1_arr - np.nanmedian(m1_arr[oot_mask])
        m2_arr = m2_arr - np.nanmedian(m2_arr[oot_mask])

        # Standardise by OOT scatter so values are comparable across targets
        m1_std = np.nanstd(m1_arr[oot_mask])
        m2_std = np.nanstd(m2_arr[oot_mask])
        if m1_std > 1e-10:
            m1_arr = m1_arr / m1_std
        if m2_std > 1e-10:
            m2_arr = m2_arr / m2_std

        # Phase-fold and sort all four arrays by phase
        sort_idx = np.argsort(phase)
        phase = phase[sort_idx]
        flux_arr = flux_arr[sort_idx]
        m1_arr = m1_arr[sort_idx]
        m2_arr = m2_arr[sort_idx]

        # Bin to exactly 1,000 phase bins using scipy.stats.binned_statistic
        flux_binned, _, _ = binned_statistic(
            phase, flux_arr, statistic='median', bins=NUM_BINS, range=(-0.5, 0.5)
        )
        m1_binned, _, _ = binned_statistic(
            phase, m1_arr, statistic='median', bins=NUM_BINS, range=(-0.5, 0.5)
        )
        m2_binned, _, _ = binned_statistic(
            phase, m2_arr, statistic='median', bins=NUM_BINS, range=(-0.5, 0.5)
        )

        # Recompute bin_centres for the final standardisation step below.
        bin_edges = np.linspace(-0.5, 0.5, NUM_BINS + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Check sparsity before interpolating
        empty_fraction = np.sum(np.isnan(flux_binned)) / NUM_BINS
        if empty_fraction > 0.15:
            with open("sparse_targets.log", "a") as log_file:
                log_file.write(f"TIC {row['tic id']}: {empty_fraction:.1%} empty bins — discarded\n")
            raise ValueError(f"Too many empty bins ({empty_fraction:.1%})")

        # Interpolate NaN bins using linear interpolation from adjacent populated bins
        flux_binned = interpolate_nans(flux_binned)
        m1_binned = interpolate_nans(m1_binned)
        m2_binned = interpolate_nans(m2_binned)

        # Apply final flux standardisation
        oot_bin_mask = np.abs(bin_centres) > 2 * (duration_days / period)
        oot_bins = flux_binned[oot_bin_mask]

        if len(oot_bins) < 10:
            raise ValueError("Insufficient out-of-transit bins for final standardisation")

        oot_median = np.nanmedian(oot_bins)
        oot_std = np.nanstd(oot_bins)

        if oot_std < 1e-10:
            raise ValueError("Near-zero out-of-transit standard deviation — flat or corrupt light curve")

        flux_binned = (flux_binned - oot_median) / oot_std

        # Build the output row in the exact order specified
        entry = {
            'tic_id': row['tic id'],
            'label': row['label'],
            'tfopwg_disp': row['tfopwg disposition'],
            'period_days': row['period (days)'],
            'epoch_btjd': epoch,
            'duration_hours': row['duration (hours)'],
        }

        # Add 1000 flux bins
        for i in range(NUM_BINS):
            entry[f'f_{i}'] = flux_binned[i]

        # Add 1000 m1 centroid bins
        for i in range(NUM_BINS):
            entry[f'm1_{i}'] = m1_binned[i]

        # Add 1000 m2 centroid bins
        for i in range(NUM_BINS):
            entry[f'm2_{i}'] = m2_binned[i]

        return entry

    except Exception as e:
        with open("failed_targets.log", "a") as log_file:
            log_file.write(f"TIC {row['tic id']}: {type(e).__name__}: {str(e)}\n")
        return None


def process_targets(manifest_path="classified_targets.csv", output_path="tess_training_data.csv"):
    """
    Main pipeline orchestrator: execute three phases in order.
    Phase 1: Query MAST for available FITS files (sequential).
    Phase 2: Download all FITS files concurrently via aiohttp (async).
    Phase 3: Process each target from its cached FITS files (sequential).
    """
    print("="*70)
    print("TESS Light Curve Processing Pipeline (Async Download)")
    print("="*70)

    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Ensure that you have run getExamples.py to create the file classified_targets.csv first.")
        return

    manifest = pd.read_csv(manifest_path)

    # Add label mapping
    label_map = {'CP': 1, 'KP': 1, 'PC': 1, 'FP': 0, 'EB': 0, 'NEB': 0}
    manifest['label'] = manifest['tfopwg disposition'].map(label_map)
    manifest = manifest.dropna(subset=['label'])
    manifest['label'] = manifest['label'].astype(int)

    # User prompt: how many examples to download
    total_available = len(manifest)
    print(f"\nManifest loaded: {total_available} valid examples available after label filtering.")
    print(f"  Label 1 (planets): {(manifest['label'] == 1).sum()}")
    print(f"  Label 0 (FP/EB):   {(manifest['label'] == 0).sum()}")
    print()

    while True:
        raw = input("How many examples do you want to download? (enter a number, or 'a' for all): ").strip().lower()
        if raw == 'a':
            n_requested = total_available
            print(f"Downloading all {total_available} available examples.")
            break
        elif raw.isdigit() and int(raw) > 0:
            n_requested = int(raw)
            if n_requested > total_available:
                print(f"Warning: you requested {n_requested} examples but only {total_available} are available.")
                print(f"Proceeding with all {total_available} available examples instead.")
                n_requested = total_available
            else:
                print(f"Downloading {n_requested} examples.")
            break
        else:
            print("Invalid input. Please enter a positive integer or 'a' for all examples.")

    # Slice the manifest to the requested number
    manifest = (
        manifest
        .groupby('label', group_keys=False)          # Group by class
        .apply(lambda g: g.sample(                   # Sample from each group
            n=min(len(g), n_requested // 2),         # Half from each class
            random_state=42
        ))
        .sample(frac=1, random_state=42)             # Shuffle the combined result
        .reset_index(drop=True)
)

    cache_dir = "./tpf_temp"
    os.makedirs(cache_dir, exist_ok=True)

    output_path = "tess_training_data.csv"
    print("(This can take a while depending on network speed and FITS file sizes)")

    # Clear or create the log files
    for log_file in ["failed_targets.log", "sparse_targets.log"]:
        with open(log_file, "w") as f:
            f.write("")

    # Redirect all warnings to avoid disrupting progress bars
    warnings.filterwarnings("ignore")

    # =========================================================================
    # Phase 1: Query MAST for all targets' available FITS files (sequential)
    # =========================================================================
    search_results = collect_search_results(manifest, cache_dir)

    if len(search_results) == 0:
        print("ERROR: No valid search results found for any target.")
        return

    # =========================================================================
    # Phase 2: Download all FITS files concurrently via aiohttp
    # =========================================================================
    asyncio.run(async_download_all(search_results))

    # =========================================================================
    # Phase 3: Process each target from its locally-cached FITS files
    # =========================================================================
    header_written = False
    success_count  = 0
    label_0_count  = 0
    label_1_count  = 0

    for row, file_list in tqdm(search_results, desc="Phase 3/3: Processing",
                               unit="target"):
        result = process_single_target(row, file_list)

        if result is not None:
            result_df = pd.DataFrame([result])
            result_df.to_csv(output_path, mode='a',
                             header=not header_written, index=False)
            header_written = True
            success_count += 1
            if result['label'] == 0:
                label_0_count += 1
            else:
                label_1_count += 1
            if success_count % 100 == 0:
                tqdm.write(f"[{success_count}] examples saved so far")

    # Print final summary
    failed_count = len(search_results) - success_count
    print(f"\n{'='*70}")
    print(f"Processing complete! Results saved to {output_path}")
    print(f"Total examples saved:          {success_count}")
    print(f"Label 0 (negatives):           {label_0_count}")
    print(f"Label 1 (positives):           {label_1_count}")
    print(f"Targets failed/discarded:      {failed_count}")
    print(f"{'='*70}")

    # Preserve cached FITS files to allow safe restarts without re-downloading
    print(f"Cache preserved at {cache_dir} for future runs.")


if __name__ == "__main__":
    process_targets()
