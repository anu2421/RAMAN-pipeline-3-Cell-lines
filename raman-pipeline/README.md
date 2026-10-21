# Single-cell Raman analysis of bacteria-exposed cancer cells

Code used to compare single-cell Raman spectra of A549, K562 and KB cells after exposure to
*E. coli* or *S. aureus* (heat-inactivated, sonicated or secretome) against untreated controls.

## Steps

1. **Loading** (`loading.py`): finds the Control / E. coli / S. aureus folders under each cell-line
   folder and sorts the CSV files into replicates (R1, R2, R3).
2. **Preprocessing** (`preprocessing.py`): cosmic-spike removal, 600–1800 cm⁻¹ range, Savitzky–Golay
   smoothing (11, 3), ALS baseline (λ = 1e5, p = 0.01), subtraction of the quartz band near 800 cm⁻¹,
   area normalisation.
3. **Alignment** (`alignment.py`): each replicate is shifted (0.5 cm⁻¹ steps, up to ±15 cm⁻¹) to the
   pooled control mean. The shared axis is then anchored using the median offset of several strong bands.
4. **Bands** (`bands.py`): peaks are detected in every group mean and matched across groups. Each band
   is given one fixed position, so the same band has the same label in every cell line and run.
5. **Statistics** (`stats.py`, `anova.py`): one-way ANOVA at each wavenumber, cluster-based permutation
   test, η² ≥ 0.06 effect-size filter and Games–Howell post-hoc comparisons.
6. **Peak shifts** (`peak_shifts.py`): bootstrap confidence intervals for band position changes.
7. **Classification** (`classification.py`): PCA, LDA, 5-fold cross-validation and
   leave-one-replicate-out validation.

The quartz/substrate regions (600–625, 792–845 and 915–995 cm⁻¹) stay in the plotted spectra. They are
left out of normalisation, alignment, ANOVA, band tables, PCA and LDA.

## Installation

```
pip install -r requirements.txt
```

## Running

Every data set has a settings file in `experiments/`. Edit the folder paths in that file, then run:

```
python run.py experiments/heat_inactivated.py
python run.py experiments/secretome.py
python run.py experiments/sonicated_A549.py
python run.py experiments/sonicated_K562.py
python run.py experiments/sonicated_KB.py
```

`--output <folder>` changes where the results are written.

Any value in `ramanpipe/config.py` can be overridden in a settings file. The ones that change between
data sets are:

| setting | meaning |
|---|---|
| `DATASETS` | cell line → folder with the spectra |
| `CONTROL_ROOT` | cell line → folder to take the control from, if it is not in the same folder |
| `EXPOSURE` | label used in titles and for choosing folders (`heat-inactivated`, `sonicated`, `secretome`) |
| `LAYOUT` | `folders` (Control / E.coli / S.aureus folders) or `sets` (`SET-n/C-n, E-n, S-n`) |
| `RUN_TAG` | text added to the output folder names |
| `OUTPUT_DIR` | where results are written |

## Input format

One CSV file per cell, containing a header line with `RamanShift` and `Intensity` and two numeric columns.
Replicates are read from folder or file names starting with `R1`, `S1`, `Rep1` and similar.

## Test data

```
python tests/make_mock_data.py
python run.py experiments/mock.py --permutations 100
```

This writes a small synthetic data set to `data/mock` and runs the full analysis on it.

## Output

For each cell line:
- stacked and overlaid spectra
- ANOVA spectrum with the significant clusters
- band tables (`*_all_consensus_bands.csv`, `*_significant_peaks.csv`)
- bar chart of the significant bands
- peak-position shifts
- PCA, LDA and confusion matrices

The output folder also holds QC files (applied shifts, phenylalanine position per replicate) and a
summary table covering all cell lines.
