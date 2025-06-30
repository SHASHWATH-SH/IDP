<<<<<<< HEAD
# IDP
=======
# CPG Supply Chain Digital Twin Dashboard

A modern digital twin for CPG supply chain simulation, disruption prediction, and optimization using 4 ML models.

## Features
- Inventory, logistics, and product condition simulation
- Unified dashboard with map visualization (Leaflet.js)
- Real-time and simulated data inputs
- ML-powered risk prediction (stockout, delay, anomaly, condition)
- Rule-based optimization and decision support
- Weather and news API integration

## Setup
1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your 4 PKL models and sample CSVs in the project root.
4. Run the Flask app:
   ```bash
   python app.py
   ```
5. Open your browser at http://localhost:5000/

## Methodology
1. **Scope**: Simulates inventory, logistics, product condition, and lead time anomalies.
2. **Data Layer**: Integrates real-time/simulated data, weather/news APIs, and logs events.
3. **Intelligence Layer**: Uses 4 ML models for risk prediction and optimization logic.
4. **Visualization**: Interactive dashboard and map for live and simulated scenarios.
5. **Optimization**: Rule-based suggestions for route, vehicle, replenishment, etc.
6. **Feedback Loop**: Compares predicted vs. actual, supports scenario testing.
7. **Validation**: Simulate historic data, run what-if scenarios, validate decisions.

---

For more, see comments in `app.py` and the dashboard UI. 
>>>>>>> f91f404 (Initial commit with project files)
