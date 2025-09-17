# CatBoost Binary Classification Project

This project implements a supervised binary classification model using the CatBoost library. The model is trained on a synthetic dataset (`syndata`) and utilizes Z-standardization for numerical features. The project includes data preprocessing, model training, and evaluation through 10-fold cross-validation with Hinge Loss.

## Project Structure

```
catboost-binary-classification
├── src
│   ├── data_preprocessing.py      # Functions for loading and preprocessing data
│   ├── train_model.py              # Functions for training the CatBoost model
│   ├── cross_validation.py          # Implements 10-fold cross-validation
│   ├── utils
│   │   └── __init__.py             # Initialization file for utils module
│   └── types
│       └── index.py                 # Custom types and data structures
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd catboost-binary-classification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing**: Use the `data_preprocessing.py` module to load and preprocess the dataset. This includes Z-standardization of numerical variables.

2. **Model Training**: Train the CatBoost model using the `train_model.py` module. This module provides functions to initialize the model, fit it to the training data, and save the trained model.

3. **Cross-Validation**: Evaluate the model's performance using 10-fold cross-validation with Hinge Loss through the `cross_validation.py` module.

## Example

```python
from src.data_preprocessing import load_data, standardize_numerical, prepare_data
from src.train_model import train_catboost_model, save_model
from src.cross_validation import cross_validate_model

# Load and preprocess data
data = load_data('path/to/syndata.csv')
data = standardize_numerical(data)
X, y = prepare_data(data)

# Train model
model = train_catboost_model(X, y)

# Save model
save_model(model, 'catboost_model.cbm')

# Cross-validation
cv_results = cross_validate_model(model, X, y)
print(cv_results)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.