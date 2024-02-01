!pip install scikit-optimize

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

# Load your data
train_features = pd.read_csv('train_features.csv')
train_label = pd.read_csv('train_label.csv')
train_data = pd.merge(train_features, train_label, on='Id')

# Extract features (X) and target variable (y)
X = train_data.drop(['Id', 'feature_2', 'feature_20', 'feature_10', 'label'], axis=1)
y = train_data['label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize an XGBoost classifier with early stopping
xgb_model = xgb.XGBClassifier(random_state=20)

# Define the hyperparameter search space for BayesianOptimization for XGBoost
param_dist_xgb = {
    'n_estimators': (100, 2000),
    'max_depth': (3, 9),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.8, 1.0),
    'colsample_bytree': (0.8, 1.0),
}

# Define the scoring metric (F1 macro)
scorer = make_scorer(f1_score, average='macro')

# Use StratifiedKFold for cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform BayesianOptimization for XGBoost
bayesian_search_xgb = BayesSearchCV(xgb_model, param_dist_xgb, n_iter=50, cv=cv_strategy, scoring=scorer, random_state=42)
bayesian_search_xgb.fit(X_scaled, y)

# Get the best model from BayesianOptimization for XGBoost
best_xgb_model = bayesian_search_xgb.best_estimator_

# Print the best hyperparameters for XGBoost
print("Best Hyperparameters for XGBoost:")
print(bayesian_search_xgb.best_params_)

# Perform cross-validation with the best model
cv_scores_xgb = cross_val_score(best_xgb_model, X_scaled, y, cv=cv_strategy, scoring=scorer)

# Display the cross-validated F1 macro scores for XGBoost
print('\nCross-Validated F1 Macro Scores for XGBoost:', cv_scores_xgb)
print(f'Mean F1 Macro Score for XGBoost: {cv_scores_xgb.mean():.4f}')

# Load test features
test_features = pd.read_csv('test_features.csv')

# Feature Engineering for test set (Add your feature engineering code here)

# Standardize test features
X_test_scaled = scaler.transform(test_features.drop(['Id', 'feature_2', 'feature_20', 'feature_10'], axis=1))

# Make predictions on the test set
test_predictions = best_xgb_model.predict(X_test_scaled)

# Create a DataFrame with Id and predicted labels for submission
submission_df = pd.DataFrame({'Id': test_features['Id'], 'label': test_predictions})

# Save the DataFrame to a CSV file for submission
submission_df.to_csv('submission.csv', index=False)
