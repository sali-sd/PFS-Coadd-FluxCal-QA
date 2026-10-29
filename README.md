# PFS Coadd Flux Calibration QA

Quality-assurance tool for PFS/Subaru flux calibration. For every FLUXSTD object it measures PFS synthetic magnitude minus Pan-STARRS 1 (PS1) catalog magnitude.

That difference is computed separately for the coadd spectrum and for the calibrated spectra. The calibrated value for each object is the median of up to 10 shared `pfsCalibrated` visits, using every visit the object has when fewer than 10 are available. Because both sides are PFS − PS1, the coadd and the calibrated results can be compared with each other as well as with PS1. Visit files are read with one worker process each.

## Outputs

| File | Contents |
| --- | --- |
| `coadd_mags_{combination}_FLUXSTD.csv` | Coadd magnitudes, the calibrated median, and each visit |
| `coadd_calibrated_mag_diff_{combination}_FLUXSTD_snr={cut}.pdf` | Five-page PFS − PS1 comparison |

The PDF pages are PFS − PS1 band differences, the g−y colour residual, then those band differences against exposure time, median signal-to-noise, and mean PFI offset. Each page has the coadd result on the left and the calibrated result on the right.

## Requirements

- The LSST / PFS 2D DRP stack (`lsst.daf.butler`, `pfs.datamodel`)
- `numpy`, `pandas`, `matplotlib`
- `synthmag`, with the PS1 filter curves

## Usage

Edit `repo`, `collections`, `combination`, and `SYNTHMAG_PATH` at the top of [`fluxcalQA_coadd.py`](fluxcalQA_coadd.py), then run:

```bash
python fluxcalQA_coadd.py
```

The PFS data store and the DRP stack are available on the Hilo, Mitaka, and Science Platform servers.
