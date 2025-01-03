# To choose the best model using criteria like AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion),
# fit multiple models with different combinations of predictors and compare their AIC or BIC values.
# Note: Lower values of AIC or BIC indicate a better-fitting model, balancing goodness of fit and model complexity.

# Steps to Select the Best Model:
# 1.Generate Candidate Models:
#     Start with a simple model (e.g., only Specialty as a predictor).
#     Incrementally add predictors to the model.
#
# 2.Compute AIC and BIC:
#     Use statsmodels to fit each model and extract AIC/BIC values.
#
# 3.Select the Best Model:
#     Compare the AIC or BIC values of all models.
#     Choose the model with the lowest AIC or BIC.

import statsmodels.api as sm
import pandas as pd
from itertools import combinations

# Load the data
data = pd.read_excel("C:/Users/RXY149/OneDrive - MedStar Health/Project 3/Lachman exam_survey results_cleaned.xlsx")

# Encode categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Dominant hand', 'Specialty', 'Preferred exam for ACL',
                                     'Familiarity w/ lachman', 'How often you perform lachman',
                                     'Confidence in  lachman skill'], drop_first=True)

# Define the outcome variable
y = data['Avg Score']

# Define all potential predictors dynamically
predictors = [col for col in data.columns if col != 'Avg Score']

# Store results
results = []

# Iterate through all possible combinations of predictors
for k in range(1, len(predictors) + 1):
    for combo in combinations(predictors, k):
        X = data[list(combo)]
        X = sm.add_constant(X)  # Add intercept

        # Fit the model
        model = sm.OLS(y, X).fit()

        # Store model summary, AIC, and BIC
        results.append({
            'Predictors': combo,
            'AIC': model.aic,
            'BIC': model.bic
        })

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results)

# Find the best model by AIC
best_model_aic = results_df.loc[results_df['AIC'].idxmin()]
print("Best Model by AIC:")
print(best_model_aic)

# Find the best model by BIC
best_model_bic = results_df.loc[results_df['BIC'].idxmin()]
print("Best Model by BIC:")
print(best_model_bic)
















