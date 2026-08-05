# Hybrid CNN-Based Models for Breast Cancer Prediction

A reproducible ML framework comparing **CNN-SVM**, **CNN-LR**, and **CNN-KNN**
hybrid models — each with and without PCA — on the Wisconsin Breast Cancer
Dataset (WBCD/WDBC), using Stratified K-Fold cross-validation and
GridSearchCV, exactly as specified in the project brief.

**No TensorFlow, PyTorch, or any deep-learning framework is required.** The
1D CNN is implemented from scratch in plain NumPy (`src/numpy_nn.py`) —
Conv1D, MaxPool1D, Dense, ReLU, Dropout, backpropagation, and an Adam
optimizer, all hand-written. This was a deliberate choice after TensorFlow
proved too fragile to install reliably (missing wheels for newer Python
releases, 32-bit/64-bit mismatches, large downloads failing on restricted
networks). Every remaining dependency — numpy, pandas, scikit-learn,
matplotlib, seaborn — has small, universal prebuilt wheels for every Python
version, OS, and architecture, so this class of install failure should no
longer be possible. It also means the full pipeline runs in well under a
minute on an ordinary laptop CPU, with nothing to download beyond a few
megabytes.

This package has been **built and test-run end-to-end** against your
uploaded `data.csv` (569 samples, 30 features) before being handed to you.

---

## 1. Required Libraries

Everything is pinned in `requirements.txt`:

| Library | Purpose |
|---|---|
| `numpy` | numerical arrays, and the hand-rolled CNN itself |
| `pandas` | data loading/cleaning |
| `scikit-learn` | SVM, LR, KNN, PCA, pipelines, CV, GridSearchCV, metrics |
| `matplotlib`, `seaborn` | all plots |
| `scipy` | scikit-learn dependency |

That's the complete list — five small, well-established packages, all with
prebuilt wheels.

---

## 2. Project Structure

```
project/
├── data/
│   └── data.csv                      # your WBCD/WDBC dataset (already included)
├── src/
│   ├── config.py                     # all settings: seeds, paths, CV folds, grids
│   ├── data_preprocessing.py         # load, clean, encode, split, scale (Section 2)
│   ├── numpy_nn.py                   # hand-rolled Conv1D/Dense/ReLU/Dropout/Adam layers
│   ├── cnn_feature_extractor.py      # 1D CNN + tuning + feature extraction (Sections 3, 7)
│   ├── hybrid_models.py              # CNN-SVM/LR/KNN pipelines + param grids (Sections 4, 5)
│   ├── cross_validation.py           # Stratified K-Fold CV (Section 6)
│   ├── hyperparameter_tuning.py      # GridSearchCV (Section 7)
│   ├── evaluation.py                 # test-set metrics, confusion matrix (Section 8)
│   └── visualization.py              # all plots (Section 9)
├── outputs/
│   ├── figures/                      # PNGs generated on each run
│   ├── results/                      # CSVs generated on each run (Section 10)
│   └── models/                       # saved CNN weights (.npz)
├── main.py                           # orchestrates the entire pipeline
├── requirements.txt
└── README.md
```

Every numbered brief-section is implemented in a specific file, so you can
cite exact line locations in your report if needed.

---

## 3. Step-by-Step VS Code Setup

### Step 1 — Install Python
Install **Python 3.10, 3.11, or 3.12** (64-bit) from
[python.org](https://www.python.org/downloads/) — make sure you pick the
**64-bit** installer. During installation on Windows, tick **"Add Python to
PATH."** (Any of these versions works fine now that TensorFlow isn't
involved; you no longer need to hunt for a specific version.)

### Step 2 — Install VS Code + the Python extension
1. Install [VS Code](https://code.visualstudio.com/).
2. Open VS Code → Extensions (`Ctrl+Shift+X`) → install **"Python"** (by Microsoft).

### Step 3 — Open the project folder
`File → Open Folder…` → select the unzipped `project` folder.

### Step 4 — Create a virtual environment
Open the integrated terminal (`` Ctrl+` ``) and run:

```bash
python -m venv venv
```

### Step 5 — Activate the virtual environment
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
  (If you get an execution-policy error, first run:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` — this only
  applies to that one terminal session, so you'll need to repeat it if you
  open a fresh terminal later.)
- **Windows (cmd.exe):** `venv\Scripts\activate.bat`
- **macOS / Linux:** `source venv/bin/activate`

You should see `(venv)` appear at the start of your terminal prompt.

### Step 6 — Select the interpreter in VS Code
`Ctrl+Shift+P` → type **"Python: Select Interpreter"** → choose the one
inside `venv` (e.g. `.\venv\Scripts\python.exe` or `./venv/bin/python`).

### Step 7 — Install the libraries
With `(venv)` active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This should finish in well under a minute — everything here is small.
Watch for `.whl` files downloading; if you ever see pip trying to build
something from a `.tar.gz` source archive with a compiler, something is
off (wrong Python version/architecture) — but that scenario is far less
likely now than it was with TensorFlow in the mix.

### Step 8 — Confirm your data file is in place
`data/data.csv` is already included (your uploaded WDBC file). To use a
different export of the same dataset, just replace that file — the loader
expects a `diagnosis` column (`M`/`B`) plus the 30 numeric feature columns.

### Step 9 — Run it
In the integrated terminal (with `(venv)` active):

```bash
python main.py --quick
```

Run this fast smoke-test version first (well under a minute) to confirm
everything works on your machine. Then run the full experiment:

```bash
python main.py
```

Typically finishes in under a minute on an ordinary laptop CPU.

### Step 10 — Find your outputs
- `outputs/figures/` — every plot (Section 9) as PNG
- `outputs/results/` — every CSV table (cross-validation results,
  GridSearchCV summary, the final comparison table from Section 10, and a
  text file of full classification reports)
- `outputs/models/` — the trained CNN feature extractor's weights (`.npz`)

---

## 4. Command-Line Options

```bash
python main.py                 # full run (default 5-fold CV, full grids)
python main.py --quick         # fast smoke-test (reduced CNN grid/epochs, 3-fold CV)
python main.py --folds 10      # switch to 10-fold Stratified CV
python main.py --data path\to\other_export.csv   # use a different CSV
```

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` when running `main.py` | Your venv isn't active, or VS Code is using the wrong interpreter — redo Steps 5–6. Confirm with `where python` (Windows) / `which python` (Mac/Linux) that it points inside `venv`. |
| VS Code terminal doesn't show `(venv)` | Close and reopen the terminal panel after creating the venv, or reactivate manually. |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, in that same terminal, then activate again. |
| `pip install` seems to hang or times out | You may be on a restricted network (e.g. campus/institutional Wi-Fi) that throttles PyPI. Try a personal hotspot as a test — but this should be far less likely now, since nothing in `requirements.txt` is large. |
| Numbers differ slightly from a previous run | All seeds are fixed (`RANDOM_SEED = 42` in `src/config.py`), so results are reproducible **on the same machine**. Tiny differences can appear across different OS/CPU architectures due to floating-point rounding order — this is normal, not a bug. |

---

## 6. Methodology Notes (for your report)

**How data leakage is avoided (Section 6's "critical requirement"):**
- The train/validation/test split happens once, at the very start, before
  anything is fit (`data_preprocessing.py`).
- `StandardScaler` for the *raw tabular* features is fit on the training
  split only.
- The CNN is trained using only the training split, with the validation
  split used purely for early stopping / model selection — the test split
  is never seen during CNN training.
- CNN features are then extracted for train, validation, and test
  separately. Train + validation features are pooled into one "training
  feature pool" that cross-validation and GridSearchCV operate on; the test
  features are held out until the very last step.
- Inside every pipeline (`hybrid_models.py`), `StandardScaler → [PCA] →
  Classifier` is a single `sklearn.pipeline.Pipeline`. Because
  `cross_validate` and `GridSearchCV` refit the *entire pipeline* on each
  fold's training portion, PCA and the second scaler are always refit from
  scratch inside each fold — they never see a fold's held-out data, and
  never see the outer test set at all until final evaluation.

**One honest simplification worth naming in your limitations section:**
the CNN feature extractor itself is trained **once** (on the train/val
split), not retrained inside every outer CV fold. Retraining a CNN inside
every fold of every model's cross-validation would multiply runtime
substantially and is impractical for a coursework timeline. This is
standard practice in comparable hybrid CNN+classical-ML studies, but it
does mean the *classifier-stage* cross-validation is fully leakage-safe,
while the CNN's own feature learning sees the full training pool once.
State this explicitly in your methodology/limitations section — it is a
real and defensible limitation, not an error to hide.

**Why a hand-rolled NumPy CNN instead of TensorFlow/PyTorch:** this is
worth a sentence in your methodology section too — it's a legitimate
engineering decision (portability and zero fragile dependencies), not a
shortcut. `src/numpy_nn.py` implements the same architecture
(Conv1D → ReLU → MaxPool1D → Conv1D → ReLU → MaxPool1D → Flatten → Dense →
ReLU → Dropout → Dense(feature layer) → ReLU → Dropout → Dense(1) →
Sigmoid) with manually derived forward and backward passes and an Adam
optimizer — mathematically the same model TensorFlow would have built, just
without the framework underneath it.

---

## 7. What Each Brief Section Maps To

| Brief Section | Where it's implemented |
|---|---|
| 1. Libraries | `requirements.txt` |
| 2. Preprocessing | `src/data_preprocessing.py` |
| 3. CNN feature extraction | `src/numpy_nn.py`, `src/cnn_feature_extractor.py` |
| 4. Hybrid model construction | `src/hybrid_models.py` |
| 5. PCA | `src/hybrid_models.py` (`_base_steps`), `src/visualization.py` (variance curve) |
| 6. Cross-validation | `src/cross_validation.py` |
| 7. Hyperparameter tuning | `src/hyperparameter_tuning.py`, `cfe.tune_cnn` in `cnn_feature_extractor.py` |
| 8. Evaluation metrics | `src/evaluation.py` |
| 9. Visualization | `src/visualization.py` |
| 10. Comparison table | `outputs/results/final_comparison_table.csv` |
| 11–13. Discussion, limitations, improvements | see `REPORT_NOTES.md` — a template grounded in this package's actual test-run results |
