# Energy Tracker Project

This project simulates and tracks energy usage across different dorms at Bates College, providing insights into energy consumption patterns and enabling predictive analysis using machine learning.

## Features
- Simulated Energy Data**: Generates synthetic energy usage data for dormitories and appliances, considering seasonal variations, peak usage hours, and weather impacts.
- Data Visualization**: Provides graphs and charts for energy usage trends by dorm and over time.
- Machine Learning Integration**: Includes regression models for predicting energy consumption.
- Web Dashboard (Future)**: Plans to develop a user-friendly dashboard using Flask or Django for visualization and insights.

## Usage
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the simulation script:
    ```bash
    python simulation.py
    ```
3. Visualize the generated data or train machine learning models using the provided scripts.

## Files Included
- `simulation.py`**: Script for generating and analyzing energy data.
-`energy_data_full.csv`**: Simulated dataset for five years of energy usage.
- `ml_ready_energy_data.csv`**: Preprocessed data for machine learning.
- `energy_usage_by_dorm.png`**: Visualization of energy usage by dorm.
- `energy_usage_over_time.png`**: Visualization of energy usage trends over time.

## Technologies Used
- Python**
  - pandas
  - NumPy
  - Matplotlib
  - Scikit-learn
  - Seaborn
- Machine Learning
  - Linear Regression
- **Future Technologies**
  - Flask/Django (for web dashboard)
  - Hosting on AWS/Google Cloud/Heroku

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
