# outcome variable is continuous
# all predictor variables are categorical
# use ANOVA (which allows one to compare the mean of the continuous outcome across
# different levels of the categorical predictor(s).

# in this case, we have multiple categorical predictors, use a factorial ANOVA to examine the combined effects of these predictors on the continuous outcome.

# To interpret the results:
# 1. F-statistic:
# The ANOVA test provides an F-statistic, which indicates whether there is a significant overall difference between the group means.
#
# 2. Post-hoc tests:
# If the ANOVA finds a significant effect, use post-hoc tests like Tukey's HSD or Bonferroni correction to identify which specific categories differ significantly from each other.

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the data
data = pd.read_excel("C:/Users/RXY149/OneDrive - MedStar Health/Project 3/Lachman exam_survey results_cleaned.xlsx")

# Simplify column names
data.columns = data.columns.str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')

# Ensure categorical variables are treated as such
categorical_vars = ["Gender", "Dominant_hand", "Specialty", "Years__of_experience",
                    "Preferred_exam_for_ACL", "Familiarity_w_lachman",
                    "How_often_you_perform_lachman", "Confidence_in__lachman_skill"]
for var in categorical_vars:
    data[var] = data[var].astype("category")

# Fit a linear model with Specialty and all other predictors
# Categorical variables are included using C(variable_name) in the formula to treat them as factors with discrete levels.
model = smf.ols(
    formula="""Avg_Score ~ C(Specialty) + C(Gender) + C(Dominant_hand) + C(Years__of_experience) + 
               C(Preferred_exam_for_ACL) + C(Familiarity_w_lachman) + 
               C(How_often_you_perform_lachman) + C(Confidence_in__lachman_skill)""",
    data=data
).fit()

# Perform ANOVA
# The ANOVA test uses Type-II sum of squares (typ=2) to calculate F-statistics and p-values,
# accounting for the hierarchical contribution of predictors after controlling for others.
anova_results = sm.stats.anova_lm(model, typ=2)

# Display the ANOVA results
print(anova_results)


# Post-hoc test: Tukey's HSD for "Confidence in lachman skill"
tukey = pairwise_tukeyhsd(
    endog=data['Avg_Score'],
    groups=data['Confidence_in__lachman_skill'],
    alpha=0.05
)

# Display the Tukey HSD results
print(tukey)