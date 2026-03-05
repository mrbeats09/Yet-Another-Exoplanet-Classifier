"""
getInputData.py — TESS Light Curve Processing Pipeline

Reads the manifest of TESS exoplanet candidates (classified_targets.csv), downloads
raw SPOC light curves, processes them through a multi-stage pipeline (quality masking,
sigma-clipping, out-of-transit detection, normalization, phase-folding, binning),
and writes all examples to a training dataset CSV (tess_training_data.csv) formatted
for input to a 1D CNN. The output has 3,006 columns: metadata + 1000 flux bins +
1000 m1 centroid bins + 1000 m2 centroid bins.
"""

import pandas as pd
import lightkurve as lk
import numpy as np
import os
import time
import random
from tqdm import tqdm


def process_targets(manifest_path="classified_targets.csv", output_path="tess_training_data.csv"):
    """
    Main pipeline: read manifest, download TESS light curves, process them,
    and write normalized, phase-folded examples to the output CSV.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Ensure that you have run getExamples.py to create the file classified_targets.csv first.")
        return

    manifest = pd.read_csv(manifest_path)

    # ============================================================================
    # Fix 1: Add label mapping at the top
    # Map string dispositions to binary integers before any further processing.
    # Rows with unmapped dispositions are silently dropped.
    # ============================================================================
    label_map = {'CP': 1, 'KP': 1, 'PC': 1, 'FP': 0, 'EB': 0, 'NEB': 0}
    manifest['label'] = manifest['tfopwg disposition'].map(label_map)
    manifest = manifest.dropna(subset=['label'])
    manifest['label'] = manifest['label'].astype(int)

    final_data = []
    cache_dir = "./tpf_temp"
    os.makedirs(cache_dir, exist_ok=True)

    output_path = "tess_training_data.csv"
    print("Initializing TESS Data Extraction Pipeline...")
    print("(This can take a REALLY long time)")

    # Clear or create the log files
    for log_file in ["failed_targets.log", "sparse_targets.log"]:
        with open(log_file, "w") as f:
            f.write("")

    # tqdm will automatically calculate the time remaining based on the length of the manifest
    for index, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing Targets", unit="star"):
        tic_id = f"TIC {row['tic id']}"

        try:
            # Rate limiting: be polite to ExoFOP/MAST servers to avoid hitting rate limits
            time.sleep(random.uniform(0.5, 2.0))

            # Search for SPOC short-cadence (2-min) light curves
            search = lk.search_lightcurve(tic_id, mission="TESS", author="SPOC", cadence="short")

            if len(search) == 0:
                raise ValueError("No SPOC short-cadence light curves found")

            # ====================================================================
            # Fix 2: Download all available sectors and stitch them
            # This concatenates all sectors into one continuous light curve,
            # with each sector independently normalized to prevent baseline jumps
            # between sectors from dominating the time series.
            # ====================================================================
            lc_collection = search.download_all(download_dir=cache_dir)
            if lc_collection is None or len(lc_collection) == 0:
                raise ValueError("Failed to download light curve collection")

            lc = lc_collection.stitch()
            if lc is None:
                raise ValueError("Failed to stitch light curve sectors")

            # ====================================================================
            # Fix 3: Extract only the four required channels from the light curve
            # Removed: sap_bkg and all derived background flux columns (fb_orig, flux_bg, fb_{i})
            # Kept:    flux (calibrated photometry), mom_centr1, mom_centr2, and time
            # ====================================================================
            time_arr = lc.time.value
            flux_arr = lc.flux.value
            m1_arr = lc.mom_centr1.value
            m2_arr = lc.mom_centr2.value

            # ====================================================================
            # Fix 4: Apply the quality mask
            # Remove cadences flagged by SPOC as having quality issues (momentum dumps,
            # scattered light, cosmic rays, etc.). Synchronise all four arrays.
            # ====================================================================
            if hasattr(lc, 'quality') and lc.quality is not None:
                good = lc.quality.value == 0
                time_arr = time_arr[good]
                flux_arr = flux_arr[good]
                m1_arr = m1_arr[good]
                m2_arr = m2_arr[good]

            if len(time_arr) == 0:
                raise ValueError("No good cadences after quality masking")

            # ====================================================================
            # Fix 5: Apply sigma-clipping to remove residual outliers
            # A cadence is removed if its flux deviates from the median by more than
            # 5 standard deviations. This removes transient cosmic rays and hot pixels
            # that escaped the SPOC quality flags. Keep all arrays synchronised.
            # ====================================================================
            flux_median = np.nanmedian(flux_arr)
            flux_std = np.nanstd(flux_arr)
            good_mask = np.abs(flux_arr - flux_median) <= 5 * flux_std
            time_arr = time_arr[good_mask]
            flux_arr = flux_arr[good_mask]
            m1_arr = m1_arr[good_mask]
            m2_arr = m2_arr[good_mask]

            if len(time_arr) == 0:
                raise ValueError("No cadences survive sigma-clipping")

            # ====================================================================
            # Fix 6: Identify out-of-transit cadences for later normalisation
            # A cadence is out-of-transit if it is more than 1.5 × duration_days away
            # from the nearest transit centre. Transit centres are computed as
            # epoch + n × period for all integer n placing them within the time range.
            # ====================================================================
            duration_days = row['duration (hours)'] / 24.0
            period = row['period (days)']
            epoch = row['epoch (bjd)'] - 2457000.0  # Convert BJD to BTJD (lightkurve's time reference)

            # Generate all transit centres within the observed time range
            n_min = int(np.floor((time_arr.min() - epoch) / period))
            n_max = int(np.ceil((time_arr.max() - epoch) / period))
            transit_centres = epoch + np.arange(n_min, n_max + 1) * period

            # For each cadence, find the distance to the nearest transit centre
            dist_to_transit = np.min(np.abs(time_arr[:, None] - transit_centres[None, :]), axis=1)
            oot_mask = dist_to_transit > 1.5 * duration_days  # True = out-of-transit

            # ====================================================================
            # Fix 7: Normalise the flux channel
            # Compute the median of out-of-transit flux and divide the entire
            # flux array by it. This sets the out-of-transit baseline to ~1.0
            # and in-transit to slightly or significantly below 1.0.
            # ====================================================================
            oot_flux = flux_arr[oot_mask]
            if len(oot_flux) < 10:
                raise ValueError("Insufficient out-of-transit cadences for flux normalisation")
            flux_arr = flux_arr / np.nanmedian(oot_flux)

            # ====================================================================
            # Fix 8: Normalise the centroid channels
            # Subtract the median out-of-transit centroid from each channel.
            # This converts from absolute pixel positions to displacements from
            # the quiescent centroid, which is the physically meaningful quantity.
            # ====================================================================
            m1_arr = m1_arr - np.nanmedian(m1_arr[oot_mask])
            m2_arr = m2_arr - np.nanmedian(m2_arr[oot_mask])

            # ====================================================================
            # Fix 9: Phase-fold all four arrays
            # Compute phase as (time - epoch) / period, mapped to [−0.5, 0.5]
            # with the primary transit centred at phase 0. Sort by ascending phase.
            # ====================================================================
            phase = ((time_arr - epoch) / period) % 1.0
            phase[phase > 0.5] -= 1.0  # Centre transit at phase 0

            # Sort everything by phase
            sort_idx = np.argsort(phase)
            phase = phase[sort_idx]
            flux_arr = flux_arr[sort_idx]
            m1_arr = m1_arr[sort_idx]
            m2_arr = m2_arr[sort_idx]

            # ====================================================================
            # Fix 10: Bin to exactly 1,000 phase bins
            # Divide [−0.5, 0.5] into 1,000 evenly spaced bins. For each bin,
            # compute the median of all cadences within that bin. Initially mark
            # empty bins as NaN. If >15% of bins are empty, discard the target.
            # Fill remaining NaNs by linear interpolation from adjacent bins.
            # ====================================================================
            NUM_BINS = 1000
            bin_edges = np.linspace(-0.5, 0.5, NUM_BINS + 1)
            bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_indices = np.digitize(phase, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, NUM_BINS - 1)

            def bin_median(values):
                """Compute the median of values in each bin."""
                binned = np.full(NUM_BINS, np.nan)
                for b in range(NUM_BINS):
                    pts = values[bin_indices == b]
                    if len(pts) > 0:
                        binned[b] = np.nanmedian(pts)
                return binned

            flux_binned = bin_median(flux_arr)
            m1_binned = bin_median(m1_arr)
            m2_binned = bin_median(m2_arr)

            # Check sparsity before interpolating
            empty_fraction = np.sum(np.isnan(flux_binned)) / NUM_BINS
            if empty_fraction > 0.15:
                with open("sparse_targets.log", "a") as log:
                    log.write(f"TIC {row['tic id']}: {empty_fraction:.1%} empty bins — discarded\n")
                raise ValueError(f"Too many empty bins ({empty_fraction:.1%})")

            # Interpolate NaN bins using linear interpolation from adjacent populated bins
            def interpolate_nans(arr):
                """Fill NaN values by linear interpolation."""
                nans = np.isnan(arr)
                if nans.any():
                    x = np.arange(NUM_BINS)
                    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
                return arr

            flux_binned = interpolate_nans(flux_binned)
            m1_binned = interpolate_nans(m1_binned)
            m2_binned = interpolate_nans(m2_binned)

            # ====================================================================
            # Fix 11: Apply final flux standardisation
            # Identify out-of-transit bins (phase outside ±2 × duration_days / period).
            # Subtract the median of those bins from the entire flux array, then divide
            # by their standard deviation. This produces a zero-centred, unit-variance
            # representation suitable for a neural network. Do NOT apply to centroids.
            # ====================================================================
            oot_bin_mask = np.abs(bin_centres) > 2 * (duration_days / period)
            oot_bins = flux_binned[oot_bin_mask]

            if len(oot_bins) < 10:
                raise ValueError("Insufficient out-of-transit bins for final standardisation")

            oot_median = np.nanmedian(oot_bins)
            oot_std = np.nanstd(oot_bins)

            if oot_std < 1e-10:
                raise ValueError("Near-zero out-of-transit standard deviation — flat or corrupt light curve")

            flux_binned = (flux_binned - oot_median) / oot_std

            # ====================================================================
            # Build the output row in the exact order specified
            # ====================================================================

            # Fix 14: Update output CSV column structure
            # Columns: tic_id, label, tfopwg_disp, period_days, epoch_btjd,
            # duration_hours, then f_0 to f_999, m1_0 to m1_999, m2_0 to m2_999
            entry = {
                'tic_id': row['tic id'],
                'label': row['label'],
                'tfopwg_disp': row['tfopwg disposition'],
                'period_days': row['period (days)'],
                'epoch_btjd': epoch,  # Already converted to BTJD
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

            final_data.append(entry)

        # ========================================================================
        # Fix 12: Improved error handling and logging
        # Log every failure to failed_targets.log with TIC ID and full error message
        # so patterns can be diagnosed. Never crash on a single target failure.
        # ========================================================================
        except Exception as e:
            with open("failed_targets.log", "a") as log:
                log.write(f"TIC {row['tic id']}: {type(e).__name__}: {str(e)}\n")
            continue

    # Convert to CSV and save
    if final_data:
        pd.DataFrame(final_data).to_csv(output_path, index=False)
        # Fix 15: Print summary at the end
        label_0_count = sum(1 for d in final_data if d['label'] == 0)
        label_1_count = sum(1 for d in final_data if d['label'] == 1)
        failed_count = len(manifest) - len(final_data)
        print(f"\n{'='*70}")
        print(f"Processing complete! Results saved to {output_path}")
        print(f"Total examples saved:          {len(final_data)}")
        print(f"Label 0 (negatives):           {label_0_count}")
        print(f"Label 1 (positives):           {label_1_count}")
        print(f"Targets failed/discarded:      {failed_count}")
        print(f"{'='*70}")
    else:
        print("ERROR: No targets were successfully processed.")

    # Fix 13: Do NOT delete the cache directory
    # Preserve cached FITS files to allow safe restarts without re-downloading
    print(f"Cache preserved at {cache_dir} for future runs.")


if __name__ == "__main__":
    process_targets()