# 🏆 World Cup 2026 Stochastic Predictor

An AI predictive engine developed in Python to simulate and forecast the outcomes of the 2026 FIFA World Cup. This model abandons traditional heuristics in favor of a rigorous stochastic architecture based on Bayesian statistics, bivariate distributions, and time decay factors.

## 🧠 Mathematical Architecture

The core of this ecosystem is built upon the following statistical principles:
* **Bivariate Poisson Matrix:** Expected Goals (xG) probability calculation that isolates the neutral attacking and defensive strengths of each national team against the FIFA historical global average.
* **Dixon-Coles Correction Factor (ρ = -0.13):** A critical algorithmic adjustment to correct the underestimation of low-scoring draws (e.g., 0-0, 1-1) inherent in pure Poisson models for international football.
* **Time Decay (Exponential Weighting):** A dynamic weighting system that mathematically prioritizes recent match results to capture the true *momentum* of the teams, heavily penalizing older data.
* **Home Field Advantage (HFA):** A dynamic field advantage multiplier specifically injected for the host nations (Mexico, United States, Canada).

## 📂 Module Ecosystem

The project is modularized into 5 executable scripts, engineered for different analysis stages and data consumption:

1. **`tournament_simulator.py`** *(Live Tracker)*: Dynamically reads real-world results from the dataset and simulates the remainder of the tournament in real-time, adapting to predefined brackets.
2. **`official_simulator.py`** *(Pure Simulator)*: Executes the entire tournament from scratch. It features a custom **Backtracking algorithm** for the perfect allocation of the best third-placed teams, strictly complying with the official FIFA rulebook.
3. **`knockout_scanner.py`** *(CLI Tactical Audit)*: Analyzes individual knockout stage matches. It calculates extra-time probabilities, integrates historical penalty shootout effectiveness, and outputs the Top 10 most probable exact scores.
4. **`group_stage_scanner.py`** *(Financial Analyzer)*: Scans group stage matchups and compares the model's probabilities against bookmaker odds to detect mathematical *Edge*, recommending bankroll management via the **Kelly Criterion**.
5. **`match_scanner.py`** *(Refactored Core)*: Acts as a pure function returning JSON/Dictionary objects. It is specifically designed to be consumed by a web Frontend or a REST API in future iterations.

## 📊 Data Pipeline
* `results.csv`: Historical dataset of official international matches (filtered to exclude friendlies).
* `shootouts.csv`: Historical log of penalty shootouts to resolve ties in knockout stages.
* `ranking_fifa.csv`: Global scoring dataset used as a base strength multiplier.

## 🎯 Accuracy and Validation (Edge)
The objective of this model is not deterministic clairvoyance, but rather the detection of inefficiencies in sports forecasting markets (*Value Betting*). By isolating neutral strength and applying the Dixon-Coles factor, the model consistently seeks a **Positive Expected Value (+EV)**. Success metrics are evaluated by iterating the Kelly Criterion over the differential between the model's Fair Odd and the commercial market odds.

## 🚀 Tech Stack
* **Language:** Python 3.x
* **Core Libraries:** `pandas` (Data manipulation), `numpy` (Matrix operations), `scipy.stats` (Stochastic probability mass functions).
