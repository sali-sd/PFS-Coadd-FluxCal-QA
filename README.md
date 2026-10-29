# PFS Coadd Flux Calibration QA

Quality-assurance tool for PFS/Subaru flux calibration. For every FLUXSTD object in a coadd combination it compares synthetic PS1 magnitudes from the coadd spectrum with synthetic magnitudes from up to 10 shared `pfsCalibrated` visits.

Coadd magnitudes use the coadd spectrum. Calibrated magnitudes are the median across those visits, using every visit an object has when fewer than 10 are available. Visit files are read with one worker process each.

## Outputs

| File | Contents |
| --- | --- |
| `coadd_mags_{combination}_FLUXSTD.csv` | Coadd magnitudes, the calibrated median, and each visit |
| `coadd_calibrated_mag_diff_{combination}_FLUXSTD_snr={cut}.pdf` | Five-page PFS − PS1 comparison |

The PDF pages are band differences, the g−y colour residual, then band differences against exposure time, median signal-to-noise, and mean PFI offset. Each page has the coadd on the left and the calibrated median on the right.

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
