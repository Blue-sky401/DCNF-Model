# DCNF Geological Modeling – Minimal Reproducible Example

This repository provides a simplified demonstration of the Deep Clustering Neural Field (DCNF) method for 3D coal‑bearing strata modeling using borehole data.

---

## Important Note on Data
[train_dcnf.py](src/train_dcnf.py)
The original field data used in the paper are proprietary and subject to confidentiality agreements, and therefore **cannot be released**.  
To ensure full reproducibility of the code workflow, this repository includes a synthetic data file `synthetic_borehole_data.csv`. All results reported here are based on this synthetic dataset.

---

## Requirements

- Python 3.8 or higher
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `torch` (PyTorch)
- `xgboost`
- `matplotlib`

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

## Quick Start (End-to-End)

```bash
# 1. Clone the repository
git clone https://github.com/Blue-sky401/DCNF-Model.git
cd DCNF-Model/DCNF_demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full DCNF training
python src/train_dcnf.py
```

---

## Detailed Step-by-Step Instructions

### Step 1: Obtain the Synthetic Data

Place `synthetic_borehole_data.csv` in the `data/` directory.  
(This file is already included in this repository; no separate generation is required.)

**Data Description:**
- Contains 220 virtual boreholes, each with 10 layers, totaling 2,200 sample points.
- Fields: `x` (coordinate), `y` (coordinate), `z_true` (true elevation), `layer` (layer index).

---

### Step 2: Run the Full DCNF Training

```bash
python src/train_dcnf.py
```

**What this script does:**
1. Loads `data/synthetic_borehole_data.csv`.
2. Generates soft partition probabilities from layer information as additional input to DCNF.
3. Splits the data into training and validation sets at an 80%/20% ratio.
4. Trains the full DCNF network.
5. Evaluates DCNF on the validation set and compares it against multiple baseline methods:
   - IDW (Inverse Distance Weighting)
   - Random Forest
   - RBF Interpolation (thin‑plate spline kernel)
   - Polynomial Regression (degree 2)
   - XGBoost

**Training output (displayed every 200 epochs):**
```
Epoch 200/1000, Train Loss: 0.002345, Val Loss: 0.003876
Epoch 400/1000, Train Loss: 0.001234, Val Loss: 0.002567
```

**Final comparison results (example):**
```
============================================================
Comparison on Validation Set (Best to Worst)
============================================================
Method               RMSE (m)     R²        
------------------------------------------------------------
DCNF (Ours)          5.135        0.9866
XGBoost              6.238        0.9802
Polynomial (deg=2)   22.960       0.7318
Random Forest        39.153       0.2200
IDW                  41.768       0.1123
RBF (thin_plate)     73.359       -1.7384
============================================================
```

**Expected results:**
- DCNF should achieve an RMSE in the range of **5–8 m** with R² > 0.98.
- XGBoost typically ranks second, with RMSE slightly higher than DCNF.
- Traditional interpolation methods (IDW, RBF) perform significantly worse.

---

### Step 3: (Optional) Run the Simplified DCNF Training

```bash
python src/train_simple_model.py
```

This script uses a **lightweight DCNF network** with fewer parameters and faster training, suitable for quick testing. The output format is similar to `train_dcnf.py`, with a runtime of approximately 2–5 minutes.

---

## Repository Structure

```
DCNF_demo/
├── data/
│   └── synthetic_borehole_data.csv   # Synthetic borehole data
├── src/
│   ├── autoencoder.py                # Autoencoder module
│   ├── dec_clustering.py             # Deep clustering components
│   ├── inr_network.py                # Implicit neural representation
│   ├── residual_block.py             # Residual block definition
│   ├── train_dcnf.py                 # Full DCNF training (main script)
│   └── train_simple_model.py         # Simplified DCNF training (quick test)
├── docs/                             # Additional documentation
├── train_dcnf.py                 # Full DCNF training (main script)
├── requirements.txt                  # Python dependencies
├── LICENSE.txt                       # MIT License
└── README.md                         # This file
```

---

## Baseline Methods Included

The script `train_dcnf.py` compares DCNF against the following methods:

- **IDW** (Inverse Distance Weighting)
- **Random Forest**
- **RBF Interpolation** (thin‑plate spline kernel)
- **Polynomial Regression** (degree 2)
- **XGBoost**

All baselines use the same training/validation split and the same input features (coordinates + soft partition probabilities).

---

## Frequently Asked Questions

**Q: I get `ModuleNotFoundError: No module named 'xgboost'` when running the script.**  
A: Run `pip install xgboost`. XGBoost is only used for baseline comparison and does not affect the DCNF training itself.

**Q: Where is the synthetic data file located?**  
A: It is located at `data/synthetic_borehole_data.csv` by default.

**Q: Results vary slightly between runs?**  
A: This is expected due to random number generation (e.g., weight initialization). However, RMSE fluctuations are typically within ±0.5 m and do not affect the conclusions.

**Q: How long does training take?**  
A: `train_dcnf.py` takes approximately 15–30 minutes, while `train_simple_model.py` takes about 2–5 minutes (both on a standard CPU).

---

## License

This code is distributed under the **MIT License**. See `LICENSE.txt` for details.

---

## Acknowledgments

We thank the journal *Computers & Geosciences* for its open‑science policy, and for providing templates and guidelines that facilitated this reproducible research.