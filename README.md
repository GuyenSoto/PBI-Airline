# Airline Passenger Satisfaction Analysis

## Project Overview
This Streamlit application analyzes airline passenger satisfaction data through exploratory data analysis and predictive modeling. It helps business leaders understand factors affecting passenger satisfaction and enables targeted customer service interventions by identifying potentially unsatisfied customers.

The application provides:
- Data loading and preprocessing capabilities
- Exploratory data visualization 
- Machine learning model for predicting passenger satisfaction
- Threshold optimization for focusing customer service resources
- Model evaluation and export functionality

## Key Visualizations

### Flight Distance Distribution
![Flight Distance Distribution](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/flight_distance_views_part1.png)

### Feature Importance
![Feature Importance](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/feature_importance_top15_part1.png)

### Precision-Recall Curve
![Precision-Recall Curve](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/pr_curve_explained.png)

## Business Problem
Airlines need to efficiently identify unsatisfied passengers and understand key factors affecting satisfaction. With limited customer service resources, it's crucial to prioritize outreach to potentially unsatisfied customers to address their concerns promptly.

## Features
- **Interactive Data Loading**: Upload your data or use the provided example dataset
- **Variable Visualization**: Explore distributions with boxplots, histograms, and percentile plots
- **Feature Engineering**: Create derived features and handle outliers
- **Model Training**: Build a Gradient Boosting classifier with customizable parameters
- **Model Evaluation**: View comprehensive performance metrics and visualizations
- **Threshold Optimization**: Tune decision thresholds to balance precision and recall
- **Export Functionality**: Save models, thresholds, and processed data for deployment

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Clone the Repository
```bash
git clone https://github.com/GuyenSoto/PBI-Airline.git
cd PBI-Airline
```

### Set up a Virtual Environment (recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
```bash
streamlit run airline_2025.py
```

This will start the Streamlit server and open the application in your default web browser. If it doesn't open automatically, navigate to the URL displayed in your terminal (typically http://localhost:8501).

### Application Workflow

1. **Load Data**: 
   - Upload your CSV file or use the provided example dataset
   - Review basic statistics and configure your target column

2. **Variable Visualization**:
   - Select variables to visualize their distributions
   - Create new features if needed
   - Remove outliers to improve model performance

3. **Model Training and Evaluation**:
   - Configure model parameters
   - Train the model
   - Review performance metrics, confusion matrices, and feature importance

4. **Threshold Optimization**:
   - Adjust decision thresholds to balance precision and recall
   - Understand the trade-offs between different threshold values

5. **Export Results**:
   - Save your trained model, optimized threshold, and processed data
   - Generate a summary report of findings

## Dataset

The sample dataset (`satisfaction.csv`) contains:
- Passenger demographic information
- Flight and trip details
- Ratings for various service aspects
- Verified satisfaction labels

Key features include:
- Flight distance
- Departure and arrival delays
- Service ratings (food, entertainment, etc.)
- Customer demographics (age, gender, etc.)

### Delay Distributions

#### Departure Delay
![Departure Delay Distribution](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/departure_delay_in_minutes_views_part1.png)

#### Arrival Delay
![Arrival Delay Distribution](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/arrival_delay_in_minutes_views_part1.png)

## Model Details

The application uses a Gradient Boosting Classifier with:
- Customizable hyperparameters (n_estimators, learning_rate, max_depth)
- Preprocessing pipelines for numerical and categorical features
- Feature importance analysis
- Optimized decision thresholds for focusing on unsatisfied passengers

### Model Evaluation

#### Confusion Matrix (Default Threshold)
![Confusion Matrix](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/confusion_matrix_part1.png)

#### Confusion Matrix (Optimized Threshold)
![Optimized Confusion Matrix](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/confusion_matrix_thr95.png)

### Threshold Analysis

#### Precision vs Threshold
![Precision vs Threshold](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/precision_vs_threshold.png)

#### Recall vs Threshold
![Recall vs Threshold](https://github.com/GuyenSoto/PBI-Airline/raw/main/output/recall_vs_threshold.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data based on airline passenger satisfaction surveys
- Built with Streamlit, Pandas, Scikit-learn, and Matplotlib
