#!/usr/bin/env python3
"""PFS synthetic magnitude minus Pan-STARRS 1 for FLUXSTD objects.

The difference is computed separately for the coadd spectrum and for up to 10
shared pfsCalibrated visits, so the two can be compared with each other.
Calibrated magnitudes are the median of the visits an object actually has.
Visit files are read with one worker process each.

The PDF has five pages: band differences, the g−y colour residual, then band
differences against exposure time, median S/N, and mean PFI offset.
"""

import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lsst.daf.butler import Butler
from matplotlib.backends.backend_pdf import PdfPages
from pfs.datamodel import TargetType

repo = "/lfs_pfs/Subaru/PFS/data/datastore_20260226"
collections = "PFS/science/run29/coadd.20260909"
combination = "brn_run29"

SYNTHMAG_PATH = Path("/home/sali/synthmag")
WAV_NM_TO_A = 10.0
FNU = True
BANDS = ("g", "r", "i", "z", "y")
MIN_MEDIAN_SNR = 20
N_CAL_VISITS = 10  # median this many calibrated visits; fewer if the object has fewer
N_WORKERS = 10     # one worker process per calibrated visit file
SIGMA_CLIP_NMAD = 10  # dotted ±1σ uses the std inside this many NMADs of the median
COADD_MIDRES = "m" in combination.split("_")[0]

BAND_MAP = {
    "g": ("PFS_g_mag", "PS1_g_mag"),
    "r": ("PFS_r_mag", "PS1_r_mag"),
    "i": ("PFS_i_mag", "PS1_i_mag"),
    "z": ("PFS_z_mag", "PS1_z_mag"),
    "y": ("PFS_y_mag", "PS1_y_mag"),
}

BAND_COLORS = {
    "g": "royalblue",
    "r": "forestgreen",
    "i": "darkorange",
    "z": "crimson",
    "y": "purple",
}

_path = str(SYNTHMAG_PATH)
if _path not in sys.path:
    sys.path.insert(0, _path)
import synthmag


def discover_cat_ids(butler, combination):
    """Sorted FLUXSTD cat_ids (numeric id starts with 3) for this combination."""
    refs = butler.registry.queryDatasets("pfsCoadd", dataId={"combination": combination})
    all_cat_ids = {ref.dataId["cat_id"] for ref in refs}
    return sorted(cat_id for cat_id in all_cat_ids if str(cat_id).startswith("3"))


def discover_obj_groups(butler, combination, cat_id):
    """Sorted obj_group values for one (combination, cat_id)."""
    refs = butler.registry.queryDatasets(
        "pfsCoadd",
        dataId={"combination": combination, "cat_id": cat_id},
    )
    return sorted({ref.dataId["obj_group"] for ref in refs})


def compute_magnitudes(sci):
    """AB magnitudes from pfsConfig catalog fluxes, as in the magnitude table."""
    total_flux = np.array([list(f) for f in sci.totalFlux], dtype=float)
    psf_flux = np.array([list(f) for f in sci.psfFlux], dtype=float)
    flux = np.where((total_flux > 0) & np.isfinite(total_flux), total_flux, psf_flux)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, -2.5 * np.log10(flux) + 31.4, np.nan)


def _ps1_band(filter_name):
    return filter_name.lower().replace("ps1_", "").replace("_ps1", "")


def mag_columns_for_object(sci, idx, mag):
    obj_filters = list(sci.filterNames[idx])
    seen, obj_cols = {}, []
    for fn in obj_filters:
        band = _ps1_band(fn)
        count = seen.get(fn, 0)
        if count > 0 or obj_filters.count(fn) > 1:
            obj_cols.append(f"PS1_{band}_mag_{count}")
        else:
            obj_cols.append(f"PS1_{band}_mag")
        seen[fn] = count + 1
    return {col: mag[idx, k] for k, col in enumerate(obj_cols)}


def ps1_mags_from_config(pfs_config, objid):
    """PS1_*_mag for one objId from an already-open pfsConfig."""
    sci = pfs_config.select(targetType=TargetType.FLUXSTD, fiberStatus=1)
    if len(sci) == 0:
        return None
    mag = compute_magnitudes(sci)
    for idx in range(len(sci)):
        if int(sci.objId[idx]) != int(objid):
            continue
        info = {"spectrograph": int(sci.spectrograph[idx])}
        info.update(mag_columns_for_object(sci, idx, mag))
        return info
    return None


def visits_and_arms(observations):
    """Unique visits in observation order, with the arms recorded on each."""
    arms_by_visit = {}
    for visit, arm in zip(observations.visit, observations.arm):
        if isinstance(arm, bytes):
            arm = arm.decode()
        arm = str(arm).strip().lower()
        arms_by_visit.setdefault(int(visit), set()).add(arm)
    return arms_by_visit


def assign_shared_visits(object_visits, n_visits):
    """Assign each object up to n_visits shared visits.

    object_visits is a list of (key, set of visit numbers). Each round takes
    the visit that appears on the most objects still short of n_visits and
    gives that visit to every one of them that has it. An object stops when
    it has n_visits or no unused visits remain. Ties go to the smaller visit
    number.
    """
    remaining = [[key, set(visits), []] for key, visits in object_visits if visits]
    assignment = {}
    while remaining:
        counts = defaultdict(int)
        for _, visits, _ in remaining:
            for visit in visits:
                counts[visit] += 1
        if not counts:
            for key, _, assigned in remaining:
                assignment[key] = assigned
            break
        visit = max(counts, key=lambda v: (counts[v], -v))
        still_waiting = []
        for key, visits, assigned in remaining:
            if visit in visits:
                visits.remove(visit)
                assigned.append(visit)
            if len(assigned) < n_visits and visits:
                still_waiting.append([key, visits, assigned])
            else:
                assignment[key] = assigned
        remaining = still_waiting
    return assignment


def mean_pfi_offset_mm(observations):
    """Mean fiber offset between measured and nominal PFI position."""
    delta = np.asarray(observations.pfiCenter, dtype=float) - np.asarray(observations.pfiNominal, dtype=float)
    if delta.ndim != 2 or delta.shape[0] == 0:
        return np.nan
    return float(np.mean(np.sqrt((delta ** 2).sum(axis=1))))


def visit_exptime(observations):
    """Exposure time of one visit. The arms are simultaneous, so use their median."""
    exp = np.asarray(observations.expTime, dtype=float).ravel()
    exp = exp[np.isfinite(exp)]
    if exp.size == 0:
        return np.nan
    return float(np.median(exp))


def coadd_metadata(pfsobject):
    """Columns the original magnitude table stores from the coadd spectrum."""
    obs = pfsobject.observations
    return {
        "ra": float(pfsobject.target.ra),
        "dec": float(pfsobject.target.dec),
        "nVisit": int(pfsobject.nVisit),
        "total_exptime": float(np.asarray(obs.expTime).sum()),
        "pfi_offset_mean_mm": mean_pfi_offset_mm(obs),
    }


def median_snr(pfsobject):
    """Median per-pixel S/N, as in the magnitude table."""
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.asarray(pfsobject.flux, dtype=float) / np.sqrt(np.asarray(pfsobject.variance, dtype=float))
    return float(np.nanmedian(snr))


def calibrated_metadata(pfsobject):
    """Exposure time, median S/N, and mean PFI offset for one calibrated visit."""
    obs = pfsobject.observations
    return {
        "total_exptime": visit_exptime(obs),
        "median_snr": median_snr(pfsobject),
        "pfi_offset_mean_mm": mean_pfi_offset_mm(obs),
    }


def _finite_median(values):
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.median(vals))


def median_calibrated(mag_list, param_list):
    """Per-band median magnitude and median visit parameters."""
    mags = {
        f"PFS_{band}_mag": _finite_median(item.get(f"PFS_{band}_mag", np.nan) for item in mag_list)
        for band in BANDS
    }
    params = {
        "total_exptime": _finite_median(item.get("total_exptime", np.nan) for item in param_list),
        "median_snr": _finite_median(item.get("median_snr", np.nan) for item in param_list),
        "pfi_offset_mean_mm": _finite_median(item.get("pfi_offset_mean_mm", np.nan) for item in param_list),
    }
    return mags, params


def calibrated_csv_fields(samples, median_params, median_mags):
    """Median calibrated columns, then one block per visit slot up to N_CAL_VISITS."""
    fields = {
        "n_cal_visits": len(samples),
        "cal_visits": ",".join(str(sample["visit"]) for sample in samples),
        "cal_total_exptime": median_params.get("total_exptime", np.nan),
        "cal_median_snr": median_params.get("median_snr", np.nan),
        "cal_pfi_offset_mean_mm": median_params.get("pfi_offset_mean_mm", np.nan),
    }
    for band in BANDS:
        fields[f"cal_PFS_{band}_mag"] = median_mags.get(f"PFS_{band}_mag", np.nan)
    for n in range(1, N_CAL_VISITS + 1):
        if n <= len(samples):
            sample = samples[n - 1]
            params = sample["params"]
            mags = sample["mags"]
            fields[f"cal_visit_{n}"] = sample["visit"]
            fields[f"cal_total_exptime_{n}"] = params.get("total_exptime", np.nan)
            fields[f"cal_median_snr_{n}"] = params.get("median_snr", np.nan)
            fields[f"cal_pfi_offset_mean_mm_{n}"] = params.get("pfi_offset_mean_mm", np.nan)
            for band in BANDS:
                fields[f"cal_PFS_{band}_mag_{n}"] = mags.get(f"PFS_{band}_mag", np.nan)
        else:
            fields[f"cal_visit_{n}"] = np.nan
            fields[f"cal_total_exptime_{n}"] = np.nan
            fields[f"cal_median_snr_{n}"] = np.nan
            fields[f"cal_pfi_offset_mean_mm_{n}"] = np.nan
            for band in BANDS:
                fields[f"cal_PFS_{band}_mag_{n}"] = np.nan
    return fields


def calibrated_csv_columns():
    """Column order matching calibrated_csv_fields."""
    median_cols = [
        "n_cal_visits", "cal_visits",
        "cal_total_exptime", "cal_median_snr", "cal_pfi_offset_mean_mm",
    ] + [f"cal_PFS_{band}_mag" for band in BANDS]
    visit_cols = []
    for n in range(1, N_CAL_VISITS + 1):
        visit_cols.extend([
            f"cal_visit_{n}",
            f"cal_total_exptime_{n}",
            f"cal_median_snr_{n}",
            f"cal_pfi_offset_mean_mm_{n}",
        ])
        visit_cols.extend(f"cal_PFS_{band}_mag_{n}" for band in BANDS)
    return median_cols + visit_cols


def _robust_ylim(ax, values, plo=1, phi=99, pad=0.50):
    """Set y-limits from plo–phi percentile range with fractional padding."""
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    lo, hi = np.percentile(finite, [plo, phi])
    margin = (hi - lo) * pad
    ax.set_ylim(lo - margin, hi + margin)


def _robust_xlim(ax, values, plo=0.1, phi=99.9, pad=0.05):
    """Set x-limits from the outer 0.1% of the plotted x values, with a 5% margin.

    One extreme x value stays outside the frame. When every plotted x value is
    non-negative, the lower limit stays at zero.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    lo, hi = np.percentile(finite, [plo, phi])
    span = hi - lo
    if span == 0:
        span = abs(lo) if lo != 0 else 1.0
    margin = span * pad
    left, right = lo - margin, hi + margin
    if np.min(finite) >= 0:
        left = max(left, 0.0)
    if right > left:
        ax.set_xlim(left, right)


def _sigma_within_nmad(values, n_nmad=SIGMA_CLIP_NMAD):
    """Sample standard deviation of points within n_nmad times NMAD of the median.

    NMAD is 1.4826 times the median absolute deviation. Points outside the cut
    stay on the plot; they are omitted only from this standard deviation.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return None
    med = float(np.median(finite))
    nmad = 1.4826 * float(np.median(np.abs(finite - med)))
    if np.isfinite(nmad) and nmad > 0:
        kept = finite[np.abs(finite - med) <= n_nmad * nmad]
        if kept.size >= 2:
            finite = kept
    sigma = float(np.std(finite, ddof=1))
    if not np.isfinite(sigma) or sigma == 0:
        return None
    return sigma


def _axhline_pm_sigma(ax, values, label="±1σ"):
    """Dotted horizontal lines at ±1 standard deviation inside the NMAD cut."""
    sigma = _sigma_within_nmad(values)
    if sigma is None:
        return
    text = f"{label}  ±{sigma:.3f} mag"
    ax.axhline(sigma, color="black", linewidth=1.0, linestyle=":", zorder=4, label=text)
    ax.axhline(-sigma, color="black", linewidth=1.0, linestyle=":", zorder=4)


def _format_title(subtitle, n):
    return (
        f"{subtitle}  —  N = {n}\n"
        f"combination={combination}  |  collections={collections}"
    )


def plot_all_band_diffs(
    ax, df,
    subtitle="Synthetic magnitude differences by band",
    x_col=None,
    xlabel="PS1 $g$-band magnitude [AB mag]",
):
    """Scatter all bands on one axes: y = PFS − PS1. x defaults to PS1 g."""
    if x_col is None:
        x_col = BAND_MAP["g"][1]
    n_total = 0
    all_diffs = []
    all_x = []
    g_diff = None
    stats_lines = [f"{'Band':>4}  {'Median':>7}  {'10th%':>7}  {'90th%':>8}"]
    stats_lines.append("─" * 36)
    for band, (pfs_col, ps1_col) in BAND_MAP.items():
        diff_col = f"diff_{band}"
        cols_needed = list(dict.fromkeys([pfs_col, ps1_col, diff_col, x_col]))
        if x_col not in df.columns:
            stats_lines.append(f"{band:>4}  {'—':>7}  {'—':>7}  {'—':>8}")
            continue
        valid = df[cols_needed].dropna()
        if valid.empty:
            stats_lines.append(f"{band:>4}  {'—':>7}  {'—':>7}  {'—':>8}")
            continue
        ax.scatter(
            valid[x_col], valid[diff_col],
            s=10, alpha=0.45, edgecolors="none",
            color=BAND_COLORS[band],
            label=f"{band}   PFS_{band}_mag − PS1_{band}_mag",
            rasterized=True,
        )
        n_total = max(n_total, len(valid))
        diff = valid[diff_col]
        all_diffs.append(diff.values)
        all_x.append(np.asarray(valid[x_col], dtype=float))
        if band == "g":
            g_diff = diff.values
        med = np.median(diff)
        p10, p90 = np.percentile(diff, [10, 90])
        stats_lines.append(f"{band:>4}  {med:>+7.3f}  {p10:>+7.3f}  {p90:>+8.3f}")

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=3)
    if g_diff is not None:
        _axhline_pm_sigma(ax, g_diff, label="±1σ (g)")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r"$\Delta m$ = PFS $-$ PS1 [mag]", fontsize=11)
    ax.set_title(_format_title(subtitle, n_total), fontsize=10)
    ax.legend(loc="upper right", markerscale=2, fontsize=9, framealpha=0.35)
    ax.text(
        0.02, 0.98, "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.65", alpha=0.35),
    )
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.minorticks_on()
    if all_diffs:
        _robust_ylim(ax, np.concatenate(all_diffs))
    if all_x:
        _robust_xlim(ax, np.concatenate(all_x))


def plot_gy_color_diff(ax, df, subtitle="$g - y$ colour difference"):
    """Scatter: x = PS1 g mag, y = (g−y)_PFS − (g−y)_PS1."""
    pfs_g, ps1_g = BAND_MAP["g"]
    pfs_y, ps1_y = BAND_MAP["y"]
    cols = [pfs_g, pfs_y, ps1_g, ps1_y]
    valid = df[cols].dropna().copy()

    if valid.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title(_format_title(subtitle, 0), fontsize=10)
        return

    valid["gy_diff"] = (valid[pfs_g] - valid[pfs_y]) - (valid[ps1_g] - valid[ps1_y])
    gy = valid["gy_diff"]
    med = np.median(gy)
    p10, p90 = np.percentile(gy, [10, 90])

    ax.scatter(valid[ps1_g], gy, s=10, alpha=0.45, color="steelblue",
               edgecolors="none", rasterized=True)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=3)
    _axhline_pm_sigma(ax, gy.values)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.35)
    _robust_ylim(ax, gy.values)
    ax.set_xlabel("PS1 $g$-band magnitude [AB mag]", fontsize=11)
    ax.set_ylabel(r"$\Delta(g - y)$ = PFS $-$ PS1 [mag]", fontsize=11)
    ax.set_title(_format_title(subtitle, len(valid)), fontsize=10)

    stats_lines = [
        f"{'':>4}  {'Median':>7}  {'10th%':>7}  {'90th%':>8}",
        "─" * 36,
        f"{'g-y':>4}  {med:>+7.3f}  {p10:>+7.3f}  {p90:>+8.3f}",
    ]
    ax.text(
        0.02, 0.98, "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.65", alpha=0.35),
    )
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.minorticks_on()
    _robust_xlim(ax, np.asarray(valid[ps1_g], dtype=float))


def band_frame(rows, pfs_by_key, ps1_by_key, param_by_key):
    """One row per object with PFS_*, PS1_*, diff_*, and the parameter columns."""
    records = []
    for row in rows:
        key = (row["cat_id"], row["obj_group"], row["objId"])
        pfs = pfs_by_key.get(key) or {}
        ps1 = ps1_by_key.get(key) or {}
        params = param_by_key.get(key) or {}
        rec = {}
        for band in BANDS:
            pfs_mag = pfs.get(f"PFS_{band}_mag", np.nan)
            ps1_mag = ps1.get(f"PS1_{band}_mag", np.nan)
            rec[f"PFS_{band}_mag"] = pfs_mag
            rec[f"PS1_{band}_mag"] = ps1_mag
            if np.isfinite(pfs_mag) and np.isfinite(ps1_mag):
                rec[f"diff_{band}"] = float(pfs_mag) - float(ps1_mag)
            else:
                rec[f"diff_{band}"] = np.nan
        for name in ("total_exptime", "median_snr", "pfi_offset_mean_mm"):
            rec[name] = params.get(name, np.nan)
        records.append(rec)
    return pd.DataFrame(records)


def _save_side_by_side(pdf, draw_left, draw_right):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 6))
    draw_left(ax_left)
    draw_right(ax_right)
    ax_right.set_ylim(ax_left.get_ylim())
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_comparison_plot(coadd_df, cal_df, plot_pdf):
    """Five pages, coadd on the left and pfsCalibrated on the right."""
    param_pages = [
        ("total_exptime", "Total exposure time [s]", "Median visit exposure time [s]",
         "mag difference vs total exposure time", "mag difference vs median visit exposure time"),
        ("median_snr", "Median S/N per pixel", "Median visit S/N per pixel",
         "mag difference vs median S/N", "mag difference vs median visit S/N"),
        ("pfi_offset_mean_mm", "Mean PFI positioning offset [mm]", "Median visit PFI offset [mm]",
         "mag difference vs mean PFI offset", "mag difference vs median visit PFI offset"),
    ]
    with PdfPages(plot_pdf) as pdf:
        _save_side_by_side(
            pdf,
            lambda ax: plot_all_band_diffs(ax, coadd_df),
            lambda ax: plot_all_band_diffs(
                ax, cal_df, subtitle="Median pfsCalibrated magnitude differences by band",
            ),
        )
        _save_side_by_side(
            pdf,
            lambda ax: plot_gy_color_diff(ax, coadd_df, subtitle="Coadd $g - y$ colour difference"),
            lambda ax: plot_gy_color_diff(
                ax, cal_df, subtitle="Median pfsCalibrated $g - y$ colour difference",
            ),
        )
        for x_col, coadd_xlabel, cal_xlabel, coadd_subtitle, cal_subtitle in param_pages:
            _save_side_by_side(
                pdf,
                lambda ax, x_col=x_col, xlabel=coadd_xlabel, subtitle=coadd_subtitle: plot_all_band_diffs(
                    ax, coadd_df, subtitle=f"Coadd {subtitle}", x_col=x_col, xlabel=xlabel,
                ),
                lambda ax, x_col=x_col, xlabel=cal_xlabel, subtitle=cal_subtitle: plot_all_band_diffs(
                    ax, cal_df, subtitle=f"Median pfsCalibrated {subtitle}", x_col=x_col, xlabel=xlabel,
                ),
            )
    print(f"Wrote 5 page(s) to {plot_pdf}")


def good_pixels(pfsobject):
    pixels_bad = (pfsobject.mask & pfsobject.flags.get("BAD", "CR", "SAT", "NO_DATA")) != 0
    return ~pixels_bad


def pfs_synthetic_mags(pfsobject, synth, use_midres=False):
    """PFS_{band}_mag synthetic AB mags, as in the magnitude table."""
    good = good_pixels(pfsobject)
    empty = {f"PFS_{band}_mag": np.nan for band in BANDS}
    if not np.any(good):
        return empty

    wav = np.asarray(pfsobject.wavelength[good], dtype=np.float64) * WAV_NM_TO_A
    flux = np.asarray(pfsobject.flux[good], dtype=np.float64)
    ok = np.isfinite(wav) & np.isfinite(flux) & (flux > 0)
    wav, flux = wav[ok], flux[ok]
    if wav.size == 0:
        return empty

    order = np.argsort(wav)
    wav, flux = wav[order], flux[order]

    mags = {}
    for filt in synth.filters:
        band = filt.split("_")[1]
        mags[f"PFS_{band}_mag"] = synth.getMag(
            wav, flux,
            filterName=filt,
            verbose=False,
            fnu=FNU,
            midres=use_midres,
        )
    return mags


_butler = None
_synth = None


def _init_worker():
    """One Butler and one synthmag per worker process."""
    global _butler, _synth
    _butler = Butler(repo, collections=collections)
    _synth = synthmag.synthmag(path=str(SYNTHMAG_PATH))


def _calibrated_worker(args):
    """Synthetic mags and visit parameters for every assigned object on one visit."""
    visit, use_midres, objects = args
    try:
        pfs_config = _butler.get("pfsConfig", dict(visit=visit))
        pfs_calibrated = _butler.get("pfsCalibrated", visit=visit)
    except Exception as exc:
        return {
            "visit": visit,
            "n_objects": len(objects),
            "n_missing": len(objects),
            "results": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    results = []
    n_missing = 0
    for cat_id, obj_group, objid in objects:
        mask = pfs_config.objId == objid
        if not np.any(mask):
            n_missing += 1
            continue
        catid = int(pfs_config.catId[mask][0])
        try:
            pfs_single = pfs_calibrated[catid, int(objid)]
        except (KeyError, IndexError):
            n_missing += 1
            continue
        results.append({
            "key": (cat_id, obj_group, objid),
            "visit": int(visit),
            "mags": pfs_synthetic_mags(pfs_single, _synth, use_midres),
            "params": calibrated_metadata(pfs_single),
            "ps1": ps1_mags_from_config(pfs_config, objid),
        })
    return {
        "visit": visit,
        "n_objects": len(objects),
        "n_missing": n_missing,
        "results": results,
        "error": None,
    }


def main():
    synth = synthmag.synthmag(path=str(SYNTHMAG_PATH))
    output_csv = f"coadd_mags_{combination}_FLUXSTD.csv"
    plot_pdf = f"coadd_calibrated_mag_diff_{combination}_FLUXSTD_snr={MIN_MEDIAN_SNR}.pdf"
    print(f"coadd synthmag midres={COADD_MIDRES}")

    butler = Butler(repo, collections=collections)
    cat_ids = discover_cat_ids(butler, combination)
    if not cat_ids:
        raise RuntimeError(f"No FLUXSTD cat_ids for combination={combination!r}")

    pairs = [
        (cat_id, obj_group)
        for cat_id in cat_ids
        for obj_group in discover_obj_groups(butler, combination, cat_id)
    ]
    print(f"Total (cat_id, obj_group) pairs to process: {len(pairs)}")

    # --- all visits on every coadd object; note which visits use the medium red arm ---
    chosen = []
    object_visits = []
    visit_midres = {}
    coadd_pfs_by_key = {}
    median_snr_by_key = {}
    for cat_id, obj_group in pairs:
        ogm = butler.get("objectGroupMap", combination=combination, cat_id=cat_id)
        objids = sorted(int(o) for o in ogm.objId if ogm[int(o)] == obj_group)
        if not objids:
            print(f"  No objects for cat_id={cat_id} obj_group={obj_group}, skipping")
            continue

        pfscoadd = butler.get(
            "pfsCoadd",
            combination=combination,
            cat_id=cat_id,
            instrument="PFS",
            obj_group=obj_group,
        )
        print(f"  Opened pfsCoadd: cat_id={cat_id} obj_group={obj_group}  ({len(objids)} objects)")

        for objid in objids:
            pfsobject = pfscoadd[int(objid)]
            arms_by_visit = visits_and_arms(pfsobject.observations)
            for visit, arms in arms_by_visit.items():
                visit_midres[visit] = visit_midres.get(visit, False) or ("m" in arms)
            key = (cat_id, obj_group, objid)
            coadd_pfs_by_key[key] = pfs_synthetic_mags(pfsobject, synth, COADD_MIDRES)
            median_snr_by_key[key] = median_snr(pfsobject)
            chosen.append({
                "objId": objid,
                "cat_id": cat_id,
                "obj_group": obj_group,
                "combination": combination,
                **coadd_metadata(pfsobject),
            })
            object_visits.append((key, set(arms_by_visit)))

    assignment = assign_shared_visits(object_visits, N_CAL_VISITS)
    by_visit = defaultdict(list)
    n_assigned = []
    for row, (key, _) in zip(chosen, object_visits):
        visits = assignment.get(key, [])
        n_assigned.append(len(visits))
        for visit in visits:
            by_visit[visit].append((row["cat_id"], row["obj_group"], row["objId"]))

    n_midres = sum(visit_midres[visit] for visit in by_visit)
    print(
        f"chosen visits: {len(by_visit)}  "
        f"medium-res red: {n_midres}  low-res red: {len(by_visit) - n_midres}"
    )
    print(f"visits assigned per object: {dict(sorted(Counter(n_assigned).items()))}")

    # --- one pfsCalibrated per chosen visit, read in parallel ---
    tasks = [
        (visit, bool(visit_midres[visit]), objects)
        for visit, objects in sorted(by_visit.items())
    ]
    cal_samples = defaultdict(list)
    n_missing_object = 0
    n_failed_visits = 0
    n_workers = min(N_WORKERS, len(tasks)) if tasks else 0
    print(f"Reading {len(tasks)} calibrated visit(s) with {n_workers} worker(s)")
    if tasks:
        with Pool(processes=n_workers, initializer=_init_worker) as pool:
            finished = 0
            for payload in pool.imap_unordered(_calibrated_worker, tasks):
                finished += 1
                if payload["error"]:
                    n_failed_visits += 1
                    print(
                        f"  Warning: visit={payload['visit']} failed "
                        f"({finished}/{len(tasks)}): {payload['error']}"
                    )
                    continue
                n_missing_object += payload["n_missing"]
                for item in payload["results"]:
                    cal_samples[item["key"]].append(item)
                print(
                    f"  Opened pfsCalibrated visit={payload['visit']}  "
                    f"({payload['n_objects']} objects, {finished}/{len(tasks)})"
                )

    cal_pfs_by_key = {}
    cal_param_by_key = {}
    cal_samples_by_key = {}
    ps1_by_key = {}
    for key, samples in cal_samples.items():
        samples = sorted(samples, key=lambda sample: sample["visit"])
        cal_samples_by_key[key] = samples
        cal_pfs_by_key[key], cal_param_by_key[key] = median_calibrated(
            [sample["mags"] for sample in samples],
            [sample["params"] for sample in samples],
        )
        for sample in samples:
            if sample["ps1"]:
                ps1_by_key[key] = sample["ps1"]
                break

    kept = [
        row for row in chosen
        if median_snr_by_key[(row["cat_id"], row["obj_group"], row["objId"])] >= MIN_MEDIAN_SNR
    ]
    print(
        f"S/N cut (>= {MIN_MEDIAN_SNR}): {len(chosen)} -> {len(kept)} objects "
        f"({len(chosen) - len(kept)} removed)"
    )
    coadd_param_by_key = {
        (row["cat_id"], row["obj_group"], row["objId"]): {
            "total_exptime": row["total_exptime"],
            "median_snr": median_snr_by_key.get((row["cat_id"], row["obj_group"], row["objId"]), np.nan),
            "pfi_offset_mean_mm": row["pfi_offset_mean_mm"],
        }
        for row in kept
    }
    coadd_df = band_frame(kept, coadd_pfs_by_key, ps1_by_key, coadd_param_by_key)
    cal_df = band_frame(kept, cal_pfs_by_key, ps1_by_key, cal_param_by_key)
    write_comparison_plot(coadd_df, cal_df, plot_pdf)

    records = []
    n_used = []
    for row in chosen:
        key = (row["cat_id"], row["obj_group"], row["objId"])
        samples = cal_samples_by_key.get(key, [])
        n_used.append(len(samples))
        ps1 = ps1_by_key.get(key) or {}
        coadd_pfs = coadd_pfs_by_key.get(key) or {}
        rec = {
            "objId": row["objId"],
            "cat_id": row["cat_id"],
            "obj_group": row["obj_group"],
            "combination": row["combination"],
            "spectrograph": ps1.get("spectrograph", np.nan),
            "ra": row["ra"],
            "dec": row["dec"],
            "nVisit": row["nVisit"],
            "total_exptime": row["total_exptime"],
            "median_snr": median_snr_by_key.get(key, np.nan),
            "pfi_offset_mean_mm": row["pfi_offset_mean_mm"],
        }
        rec.update({name: value for name, value in ps1.items() if name.startswith("PS1_")})
        for band in BANDS:
            rec[f"PFS_{band}_mag"] = coadd_pfs.get(f"PFS_{band}_mag", np.nan)
        rec.update(calibrated_csv_fields(
            samples,
            cal_param_by_key.get(key) or {},
            cal_pfs_by_key.get(key) or {},
        ))
        records.append(rec)

    table = pd.DataFrame(records)
    ps1_cols_all = [col for col in table.columns if col.startswith("PS1_")]
    empty_ps1 = [col for col in ps1_cols_all if table[col].isna().all()]
    table = table.drop(columns=empty_ps1)
    ps1_cols = sorted(col for col in ps1_cols_all if col not in empty_ps1)
    meta_cols = [
        "objId", "cat_id", "obj_group", "combination", "spectrograph",
        "ra", "dec", "nVisit", "total_exptime", "median_snr", "pfi_offset_mean_mm",
    ]
    pfs_cols = [f"PFS_{band}_mag" for band in BANDS]
    table = table[meta_cols + ps1_cols + pfs_cols + calibrated_csv_columns()]
    table.to_csv(output_csv, index=False)

    print(f"\nWrote {len(table)} row(s) to {output_csv}")
    print(f"visits opened: {len(by_visit)}")
    print(f"calibrated visits used per object: {dict(sorted(Counter(n_used).items()))}")
    if n_failed_visits:
        print(f"Warning: {n_failed_visits} calibrated visit(s) failed to open")
    if n_missing_object:
        print(f"Warning: {n_missing_object} object-visit lookup(s) were missing from pfsCalibrated")


if __name__ == "__main__":
    main()
