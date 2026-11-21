# Edge-Robust SHM Classification — Reproduction Code

Companion code for the paper *"Resource-Efficient Self-Supervised Classification for
Edge-Based Structural Health Monitoring Under Corrupted Sensor Data."* Runs end-to-end on
a laptop CPU (no GPU required) in a few minutes, or on Google Colab's free tier.

## What this does

1. `simulate_structure.py` — numerically simulates a 4-DOF shear-building structure
   (healthy vs. story-1 stiffness loss at 20/35/50%) and generates 80 runs of noisy
   4-channel acceleration time series. **This stands in for a real recorded SHM dataset**
   (see "Using real data" below) because the LANL three-story-structure archive and the
   KW51 bridge-monitoring record on Zenodo were not reachable from the sandboxed
   environment this project was originally built in.
2. `corrupt.py` — injects MCAR, MAR (burst dropout), and MNAR (amplitude clipping)
   missingness at controllable rates.
3. `models.py` — four PyTorch models: a full-signal CNN, a GRU reconstruction baseline
   (Recon-GRU), and a compact self-supervised-pretrained classifier (the proposed model)
   plus its no-pretraining ablation.
4. `run_experiment.py` — the main pipeline: builds the dataset, trains all models,
   evaluates every model × corruption-type × corruption-rate combination, profiles
   model size/latency before and after int8 dynamic quantization, and writes all
   results to `outputs/*.csv`.
5. `make_figures.py` — turns the CSVs into the four figures and summary tables used in
   the paper.

## Quick start (laptop)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

mkdir -p outputs
python simulate_structure.py    # ~10-20s, writes outputs/raw_runs.npz
python run_experiment.py        # ~2-5 min on CPU, writes outputs/*.csv
python make_figures.py          # writes outputs/fig1..4.png + summary tables
```

All outputs land in `outputs/`. Total runtime on a typical laptop CPU is under five
minutes; no GPU is used or required anywhere in the pipeline.

## Quick start (Google Colab)

1. Upload all five `.py` files to a Colab session (or `git clone` if you host this
   folder in a repo).
2. In a cell:
   ```python
   !pip install -q torch scipy pandas matplotlib scikit-learn
   !mkdir -p outputs
   !python simulate_structure.py
   !python run_experiment.py
   !python make_figures.py
   ```
3. Download `outputs/` or view the PNGs inline with `IPython.display.Image`.

Colab's free CPU tier is sufficient — none of the models exceed 37k parameters, so a GPU
gives no meaningful speed-up.

## Using real data instead of the simulation

The pipeline is deliberately factored so the simulation is the *only* data-generating
step. To substitute real recorded data:

- Replace `simulate_structure.py`'s output with your own `outputs/raw_runs.npz`
  containing `runs` (N runs × T samples × C channels, float32), `labels` (0=healthy,
  1=damaged), and `meta` (optional, any string array) — matching arrays are all
  `run_experiment.py` expects.
- Two real, open, widely used candidates: the **LANL three-story structure** benchmark
  (distributed via OSTI / the Los Alamos National Laboratory SHM data repository) and
  the **KW51 railway bridge** continuous monitoring dataset (Zenodo, CC-BY-NC-SA). Both
  are standard reference structures in the SHM machine-learning literature; substituting
  either is the natural next step recommended in the paper's Limitations section.
- Everything downstream (corruption injection, spectral feature extraction, all four
  models, training, evaluation, quantization, figure generation) requires no changes.

## Files

| File | Purpose |
|---|---|
| `simulate_structure.py` | Physics simulation → `outputs/raw_runs.npz` |
| `corrupt.py` | MCAR / MAR / MNAR corruption functions |
| `models.py` | BaselineCNN, ReconGRU, CompactEncoder/Decoder/Classifier |
| `run_experiment.py` | Full training + evaluation + quantization pipeline |
| `make_figures.py` | Figures 1-4 and summary tables from the CSV outputs |
| `build_paper.js` | Generates the IEEE-format `.docx` from the results (Node.js, `docx` package) |
| `requirements.txt` | Python dependencies |

## Reproducibility note

All results in the paper reflect a single simulation seed (42) and a single training run
per model — there is no repeated-seed variance estimate. If you re-run this pipeline,
expect numbers close to but not bit-identical to the paper (PyTorch's CPU RNG is not
fully deterministic across versions/platforms even with a fixed seed). To get a variance
estimate, wrap `run_experiment.py`'s `main()` in a loop over several seeds and aggregate.
