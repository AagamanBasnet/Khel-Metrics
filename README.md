Khel Metrics — System Architecture & Pipeline

Khel Metrics is an end-to-end football analytics system that:

1. Synthesizes realistic match-level data**
2. Predicts goals using XGBoost Poisson models**
3. Converts goal expectations into probabilities**
4. Runs large-scale Monte Carlo simulations**
5. Visualizes results via an interactive Streamlit dashboard**

 File Responsibilities (What each file does)

dataformat.ipynb — Intelligent Data Generation & Feature Engineering

Purpose: Prepare a production-ready dataset for ML.

 Core Responsibilities:

* Cleans raw match data (`data1.csv`)
* Fills Future fixtures using:

  * Recent form (rolling averages)
  * Head-to-head history
  * Seasonal team statistics
  * Dynamic **ELO ratings**
* Applies:

  * ELO-based scaling
  * Controlled randomness (deterministic RNG)
  * Stat capping (realism constraints)
* Generates **derived features**:

  * xG difference
  * Shot ratios
  * Possession ratios
  * Rolling last-3 match stats
  * Days since last match
  * Match count per team

Output:
`data2_filled_safe.csv` → fully ML-ready dataset

 `finalmodel.py` — Predictive Modeling & Simulation Engine

Purpose: Learn goal distributions and simulate the league.

 Model Layer

* Two independent **XGBoost Poisson models**:

  * Home goals λₕ
  * Away goals λₐ
* Time-aware validation using **TimeSeriesSplit**
* Metrics tracked:

  * RMSE
  * R²
  * Match outcome accuracy

   Probability Layer

* Converts λₕ and λₐ into:

  * Full scoreline distributions
  * Win / Draw / Loss probabilities
* Optional **logistic calibration** using out-of-fold predictions

   Match Simulation

* For each fixture:

  * 5,000 Poisson goal simulations
  * Expected goals
  * Outcome probabilities

  Output:
`simulated_matches_with_probs.csv`

  Season Simulation (GPU-Accelerated)

* 10,000 full-season simulations using **PyTorch**
* Each simulation:

  * Samples match outcomes
  * Allocates points
  * Ranks teams by:

    1. Points
    2. Goal difference
    3. Goals scored
* Aggregates:

  * Average points
  * Title probability
  * Top-4 probability
  * Relegation probability
  * Average league position

  Output:
`season_simulation_distributions.csv`

---

   `st.py` — Interactive Visualization Layer

Purpose: Turn model outputs into an explainable UI.

   Features:

* Match predictor:

  * Expected goals
  * Win/draw/loss probabilities
  * Visual probability bars
* Team form:

  * Last 5 matches
  * Head-to-head history
* Season simulation:

  * League table with probability badges
  * Title & relegation races
  * Interactive charts (Plotly)
* Polished UI:

  * Custom CSS
  * Team logos
  * Responsive layout



  End-to-End Pipeline

Raw Match Data
      ↓
dataformat.py
(Feature Engineering + ELO + Forecasting)
      ↓
data2_filled_safe.csv
      ↓
finalmodel.py
(XGBoost Poisson Models)
      ↓
Match Simulations
      ↓
Monte Carlo Season Simulation (GPU)
      ↓
CSV Outputs
      ↓
st.py (Streamlit Dashboard)

## 🧪 Why This Approach Is Strong

* Uses **probabilistic modeling**, not hard predictions
* Time-aware validation (no data leakage)
* Combines:

  * ML (XGBoost)
  * Statistics (Poisson)
  * Simulation (Monte Carlo)
  * Systems design (GPU acceleration)
* Scales easily to other leagues
