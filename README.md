# World Cup 2026 Stochastic Predictor

An AI predictive engine developed in Python to simulate and forecast the outcomes of the 2026 FIFA World Cup. This model abandons traditional heuristics in favor of a rigorous stochastic architecture based on Bayesian statistics, bivariate distributions, and time decay factors.

## Features

* **Bivariate Poisson Matrix:** Calculates Expected Goals (xG) probabilities by isolating the neutral attacking and defensive strengths of each national team against the FIFA historical global average.
* **Dixon-Coles Correction Factor:** Applies a critical algorithmic adjustment (ρ = -0.13) to correct the underestimation of low-scoring draws (e.g., 0-0, 1-1) inherent in pure Poisson models for international football.
* **Time Decay Weighting:** Mathematically prioritizes recent match results to capture the true momentum of the teams, heavily penalizing older data.
* **Home Field Advantage (HFA):** Injects a dynamic field advantage multiplier specifically for the host nations (Mexico, United States, Canada).
* **Value Betting Analysis:** Compares model probabilities against bookmaker odds to detect positive Expected Value (+EV) and recommends bankroll management via the Kelly Criterion.
* **Custom Live Tracking:** Reads real-world results dynamically and simulates the remainder of the tournament in real-time, including a Backtracking algorithm to handle FIFA's best third-placed teams logic.

## Setup

1.  Download or clone this repository:
    ```bash
    git clone [https://github.com/Fernandohj/world-cup-stochastic-predictor.git](https://github.com/Fernandohj/world-cup-stochastic-predictor.git)
    cd world-cup-stochastic-predictor
    ```

2.  (Recommended) Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the predictor, execute any of the simulation modules from your terminal depending on the analysis stage you want to perform. For example, to run the live tournament tracker:

```bash
python tournament_simulator.py
```

The script will print the simulation outputs, group stage standings, advancement probabilities, and the Top 3 exact scorelines directly to the console:

```text
[ ✓ Completado ] Grupo A: Mexico 2 - 1 South Africa -> Avanzó: Mexico

[ SIMULACIÓN PENDIENTE: Match 73 | Canada vs South Korea ]
90 Min: [1] 45.2% | [X] 28.5% | [2] 26.3%
Clasificar: Canada (58.1%) vs South Korea (41.9%)
Avanza a la siguiente ronda: Canada
Goles: >1.5 (68.4%) | >2.5 (41.2%) | A/A (52.1%)
Top 3 Marcadores Exactos:
   1. 1 - 1 (Prob: 13.5%)
   2. 1 - 0 (Prob: 12.1%)
   3. 0 - 0 (Prob: 9.8%)
```

## File Structure

### Core Modules
* `tournament_simulator.py`: **Live Tracker.** Dynamically reads real-world results from the dataset and simulates the remainder of the tournament in real-time.
* `official_simulator.py`: **Pure Simulator.** Executes the entire tournament from scratch using a Backtracking algorithm for the perfect allocation of the best third-placed teams, strictly complying with the FIFA rulebook.
* `fc26_live_tracker.py`: **Specialized Interceptor.** Overrides official matchmaking rules to simulate alternative Knockout Phase scenarios (Career Mode logic) with a fast-forward feature.
* `knockout_scanner.py`: **CLI Tactical Audit.** Analyzes individual knockout matches, calculating extra-time probabilities, penalty shootout effectiveness, and Top 10 exact scores.
* `group_stage_scanner.py`: **Financial Analyzer.** Scans group stage matchups to detect mathematical Edge and recommends bets using the Kelly Criterion.

### Data Sources (Pipeline)
The core datasets for this project are sourced from the data science community on Kaggle:
* `results.csv` & `shootouts.csv`: Historical dataset of official international football matches from 1872 to present, originally compiled by [Mart Jürisoo on Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017).
* `ranking_fifa.csv`: Global scoring dataset used as a base strength multiplier.

### Audits & Backtesting
* `audits/`: Directory containing Excel performance reports (`Auditoria_Jornada_1.xlsx`, etc.). These files document the model's backtesting, tracking its real-world predictive accuracy, hit rates, and Kelly Criterion profitability across the group stage matchdays.
