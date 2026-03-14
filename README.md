# Insurance Charges Prediction Project

## Project Overview
This project aims to analyze and preprocess a dataset containing insurance charges to prepare it for a machine learning model. The goal is to understand the factors influencing insurance costs and to build a clean, feature-engineered dataset for subsequent predictive modeling.

## Data Source
The dataset used in this project is `insurance.csv`.

## Analysis Steps Performed

### 1. Data Loading and Initial Exploration
- Loaded the `insurance.csv` file into a pandas DataFrame.
- Examined the dataset dimensions (`df.shape`).
- Displayed the first few rows (`df.head()`).
- Checked data types and non-null counts (`df.info()`).
- Generated descriptive statistics for numerical columns (`df.describe()`).
- Verified for missing values (`df.isnull().sum()`).

### 2. Exploratory Data Analysis (EDA)
- Visualized the distribution of numerical features (`age`, `bmi`, `children`, `charges`) using histograms.
- Count plots for categorical features (`children`, `sex`, `smoker`, `region`).
- Identified potential outliers in numerical features using box plots.
- Explored correlations between numerical features using a heatmap.

### 3. Data Cleaning
- Created a copy of the original DataFrame (`df_cleaned`).
- Removed duplicate rows from the dataset.
- Confirmed no remaining null values.

### 4. Feature Engineering and Encoding
- **Categorical Encoding:**
    - Converted `sex` ('male', 'female') to numerical (`is_female`: 0 for male, 1 for female).
    - Converted `smoker` ('yes', 'no') to numerical (`is_smoker`: 0 for no, 1 for yes).
    - Renamed columns `sex` and `smoker` to `is_female` and `is_smoker` for clarity.
    - Applied one-hot encoding to the `region` column, dropping the first category to avoid multicollinearity.
- **BMI Categorization:**
    - Created a new categorical feature `bmi_category` based on `bmi` values (Underweight, Normal, Overweight, Obese).
    - Applied one-hot encoding to `bmi_category`, dropping the first category.
- **Data Type Conversion:**
    - Converted all relevant columns in `df_cleaned` to integer types where appropriate.

### 5. Feature Scaling
- Applied StandardScaler to numerical features (`age`, `bmi`, `children`) to normalize their ranges.

### 6. Feature Selection
- **Pearson Correlation:**
    - Calculated Pearson correlation coefficients between `charges` and selected numerical and binary categorical features.
    - `is_smoker` showed the highest positive correlation with `charges`.
- **Chi-squared Test:**
    - Binned `charges` into quartiles (`charges_bin`).
    - Performed Chi-squared tests between `charges_bin` and other categorical features to assess their independence.
    - Features with p-value < 0.05 (`is_smoker`, `region_southeast`, `is_female`, `bmi_category_Obese`) were selected.

### 7. Final Dataset for Modeling
- A final DataFrame `final_df` was created containing the selected features (`age`, `is_female`, `bmi`, `children`, `is_smoker`, `charges`, `region_southeast`, `bmi_category_Obese`), ready for model training.

## Next Steps
- Build and train various machine learning models (e.g., Linear Regression, Random Forest, Gradient Boosting) for predicting insurance charges.
- Evaluate model performance using appropriate metrics (e.g., R-squared, MAE, RMSE).
- Perform hyperparameter tuning to optimize model performance.
- Deploy the best-performing model.
