# Lasso Regression — Complete Experimental Summary (E1–E5)

## 1. Research Objective

The objective of this experiment is to predict the total number of wins (`W`) achieved by Major League Baseball (MLB) teams during a season using historical team-season statistics.

Lasso Regression was applied to evaluate whether L1 regularization can improve prediction accuracy while performing automatic feature selection.

Five experiments were conducted:

- **E1:** Baseline Lasso Regression using 47 features.
- **E2:** Hyperparameter tuning to identify the best alpha.
- **E3:** Tuned Lasso Regression using 47 features.
- **E4:** Permutation Feature Importance for the E3 model.
- **E5:** Tuned Lasso Regression using only the original 44 features, followed by permutation importance analysis.

The experiments also assess whether feature engineering improves predictive performance.

---

## 2. Dataset and Model Preparation

### 2.1 Dataset

The original training dataset contains 1,412 MLB team-season observations.

| Dataset | Observations | Input Features |
|---|---:|---:|
| Training | 1,129 | 44 original features |
| Testing | 283 | 44 original features |
| Feature-engineered training | 1,129 | 47 features |
| Feature-engineered testing | 283 | 47 features |

The same training and testing observations were maintained across the experiments to support a consistent comparison.

**Target variable:** `W` — total season wins.

**Primary evaluation metric:** Mean Absolute Error (MAE).

A lower MAE indicates better predictive accuracy.

### 2.2 Feature Engineering

Three additional features were created:

| Engineered Feature | Formula | Purpose |
|---|---|---|
| R_per_game | R / G | Average runs scored per game |
| RA_per_game | RA / G | Average runs allowed per game |
| Run_Differential | R − RA | Difference between runs scored and runs allowed |

These features increased the number of model inputs from 44 to 47.

### 2.3 Data Preprocessing

Numerical features were standardized using `StandardScaler`, while the binary era and decade indicator features were retained without scaling.

For cross-validation, preprocessing was fitted separately within each training fold to reduce the risk of data leakage.

The held-out testing dataset was not used to select the best alpha.

---

## 3. Understanding Lasso Regression

Lasso stands for **Least Absolute Shrinkage and Selection Operator**.

It is a linear regression method that applies L1 regularization to reduce the magnitude of regression coefficients.

The model minimizes two components:

1. Prediction error, measured through squared residuals during model fitting.
2. A penalty based on the absolute values of the regression coefficients.

The strength of the penalty is controlled by the hyperparameter **alpha (α)**.

### How Alpha Works

| Alpha | Regularization | Expected Effect |
|---|---|---|
| Large | Strong | More coefficients shrink toward zero |
| Small | Weak | More coefficients remain active |
| Zero | No L1 penalty | Approaches ordinary least-squares regression |

A larger alpha does not automatically produce a better model. Excessive regularization can remove useful predictive information and cause underfitting.

A smaller alpha may preserve more information but may also reduce the benefits of regularization.

**The best alpha is selected using cross-validation performance, not simply by choosing the largest or smallest value.**

---

## 4. E1 — Baseline Lasso Regression

### Objective

Establish an initial Lasso Regression benchmark using 47 features and an alpha of 1.0.

### Configuration

- Model: Lasso Regression
- Features: 47
- Alpha: 1.0
- Maximum iterations: 100,000

### Results

| Metric | Training | Testing |
|---|---:|---:|
| MAE | 3.2763 | 3.1609 |
| RMSE | 4.1409 | 3.8578 |
| R² | 0.9051 | 0.9069 |

### Feature Selection

- Total features: 47
- Selected features: 3
- Zero coefficients: 44

### Interpretation

The baseline Lasso model demonstrated strong regularization.

With alpha set to 1.0, Lasso retained only three active features, eliminating the direct contributions of the remaining 44 features.

Although the model achieved reasonable predictive accuracy, its testing MAE of 3.1609 was higher than the instructor's original Linear Regression baseline MAE of 2.7025.

This suggests that alpha = 1.0 imposed excessive regularization for this dataset.

**E1 Conclusion:** Hyperparameter tuning was required to improve prediction accuracy and retain useful information.

---

## 5. E2 — Hyperparameter Tuning: Finding the Best Alpha

### 5.1 Objective

Identify the alpha value that achieves the lowest cross-validation MAE.

Rather than selecting alpha arbitrarily, six candidate values were evaluated systematically.

### 5.2 Methodology

**Cross-validation:** 5-fold KFold

**Settings:**

- Number of folds: 5
- Shuffle: True
- Random state: 42
- Evaluation metric: MAE
- Selection criterion: Lowest average cross-validation MAE

### 5.3 How Five-Fold Cross-Validation Works

The 1,129 training observations are divided into five approximately equal groups.

The model is trained and validated five times:

| Round | Model Training | Validation |
|---|---|---|
| Fold 1 | Groups 2–5 | Group 1 |
| Fold 2 | Groups 1, 3–5 | Group 2 |
| Fold 3 | Groups 1–2, 4–5 | Group 3 |
| Fold 4 | Groups 1–3, 5 | Group 4 |
| Fold 5 | Groups 1–4 | Group 5 |

For each alpha:

1. Split the training data into five folds.
2. Fit preprocessing using the current training folds.
3. Train Lasso using that alpha.
4. Predict the validation fold.
5. Calculate validation MAE.
6. Repeat for all five folds.
7. Average the five validation MAE values.
8. Compare the average MAE across candidate alpha values.

The alpha with the lowest average validation MAE is selected.

### 5.4 E2 Cross-Validation Results

| Alpha | CV MAE ↓ | Selected Features | Zero Coefficients |
|---|---:|---:|---:|
| 1.000 | 3.3000 | 3 | 44 |
| 0.500 | 3.1373 | 6 | 41 |
| 0.100 | 2.9599 | 13 | 34 |
| 0.050 | 2.8752 | 18 | 29 |
| **0.010** | **2.8195** | **32** | **15** |
| 0.001 | 2.8244 | 43 | 4 |

### 5.5 Why Was Alpha = 0.01 Selected?

The cross-validation results showed that reducing alpha from 1.0 to 0.01 improved predictive performance.

At alpha = 1.0, the model retained only three features and achieved CV MAE of approximately 3.3000.

At alpha = 0.01, the model retained 32 active features and achieved the lowest CV MAE of approximately 2.8195.

Reducing alpha further to 0.001 increased the number of active features to 43 but slightly worsened CV MAE to 2.8244.

Therefore:

**Best tested alpha = 0.01**

**Best cross-validation MAE = 2.8195**

This is the best alpha among the six tested candidates; other untested values may also perform well.

### 5.6 Why Not Choose Alpha Using Testing MAE?

The testing dataset should be reserved for evaluating the final selected model.

If alpha is repeatedly chosen based on testing performance, the testing dataset effectively becomes part of model selection, potentially producing an overly optimistic evaluation.

Cross-validation allows alpha to be selected using only the training data.

**E2 Conclusion:** Alpha = 0.01 provided the best balance of predictive accuracy and regularization among the tested values.

---

## 6. E3 — Tuned Lasso Regression (47 Features)

### Objective

Train and evaluate the Lasso model using the optimal alpha identified in E2.

### Configuration

- Input features: 47
- Alpha: 0.01
- Training observations: 1,129
- Testing observations: 283

### Results

| Metric | Training | Testing |
|---|---:|---:|
| MAE | 2.6970 | 2.6883 |
| RMSE | 3.4097 | 3.3242 |
| R² | 0.9357 | 0.9308 |

### Feature Selection

- Total features: 47
- Active features: 32
- Zero coefficients: 15

### Interpretation

Compared with E1, reducing alpha from 1.0 to 0.01 significantly improved predictive accuracy.

Testing MAE decreased from 3.1609 to 2.6883 wins.

The model retained 32 active features rather than only three, indicating that weaker regularization preserved more useful predictive information.

Training and testing MAE were also very similar, suggesting no substantial generalization gap in this particular data split.

### Feature Engineering Results

Among the three engineered features:

- `Run_Differential` retained a non-zero coefficient.
- `R_per_game` received a zero coefficient.
- `RA_per_game` received a zero coefficient.

The model therefore used Run Differential but did not directly rely on the two per-game features.

**E3 Conclusion:** Alpha tuning significantly improved Lasso performance, producing a testing MAE of 2.6883 wins.

---

## 7. E4 — Permutation Feature Importance (47 Features)

### Objective

Investigate which features the tuned E3 Lasso model relied on most strongly.

### Methodology

Permutation importance was calculated by randomly shuffling one feature at a time while leaving the other features unchanged.

The increase in testing MAE after shuffling indicates how strongly the model depends on that feature under the permutation test.

- Model: E3 Lasso
- Features: 47
- Repetitions: 10
- Metric: Increase in MAE

### E4 Results — Top Features

| Rank | Feature | Permutation Importance |
|---|---|---:|
| 1 | Run_Differential | 6.3306 |
| 2 | IPouts | 3.7905 |
| 3 | AB | 3.5255 |
| 4 | SV | 2.4091 |
| 5 | CG | 1.8690 |
| 6 | H | 1.2335 |
| 7 | R | 0.8639 |

### Interpretation

Run Differential recorded the highest permutation importance.

Shuffling this feature increased testing MAE by approximately 6.3306 wins on average.

This indicates that the E3 model relied heavily on Run Differential when generating predictions.

However, Run Differential is mathematically related to Runs Scored and Runs Allowed. Shuffling it independently can create unrealistic combinations of baseball statistics.

Therefore, its importance should be interpreted as model reliance rather than an independent causal effect.

**E4 Conclusion:** Run Differential was the most influential feature under E3's individual-feature permutation analysis.

---

## 8. E5 — Lasso Regression Using 44 Original Features

### Objective

Determine whether the original 44 features can achieve comparable predictive performance without the three engineered features.

### Configuration

- Model: Lasso Regression
- Features: 44
- Alpha tuning: GridSearchCV
- Cross-validation: 5-fold
- Best alpha: 0.01
- Best CV MAE: 2.8190

### Model Results

| Metric | Training | Testing |
|---|---:|---:|
| MAE | 2.6956 | 2.6898 |
| RMSE | — | 3.3248 |
| R² | — | 0.9308 |

### Feature Selection

- Total features: 44
- Active features: 33
- Zero coefficients: 11

### Interpretation

The E5 model achieved nearly identical accuracy to E3 despite using only the original 44 features.

The optimal alpha remained 0.01.

This suggests that the original features already provided much of the information needed for this prediction task.

**E5 Conclusion:** The 44-feature Lasso model achieved comparable prediction accuracy without additional feature engineering.

---

## 9. E5 — Permutation Feature Importance (44 Features)

### Objective

Identify which original features the E5 Lasso model relied on most strongly.

### Results

| Rank | Feature | Permutation Importance |
|---|---|---:|
| 1 | R — Runs Scored | 6.9199 |
| 2 | RA — Runs Allowed | 4.7082 |
| 3 | IPouts | 3.8395 |
| 4 | AB | 3.5519 |
| 5 | SV | 2.4201 |
| 6 | CG | 1.8815 |
| 7 | H | 1.2422 |

### Interpretation

Runs Scored (`R`) and Runs Allowed (`RA`) were the two most influential features in E5.

This contrasts with E4, where Run Differential was the most influential feature.

The result suggests that Lasso distributed model reliance differently when the engineered Run Differential feature was introduced.

### E4 vs E5 Permutation Comparison

| Feature | E4 — 47 Features | E5 — 44 Features |
|---|---:|---:|
| Run_Differential | 6.3306 | Not included |
| R | 0.8639 | 6.9199 |
| RA | 0.0000 | 4.7082 |
| IPouts | 3.7905 | 3.8395 |
| AB | 3.5255 | 3.5519 |
| SV | 2.4091 | 2.4201 |
| CG | 1.8690 | 1.8815 |

Because Run Differential equals `R − RA`, these variables contain overlapping information.

Introducing Run Differential changed the model coefficients and permutation rankings without meaningfully changing predictive performance.

This illustrates why feature importance must be interpreted carefully when predictors are correlated.

---

## 10. Overall Model Comparison — E1 to E5

E2 is the hyperparameter tuning stage, while E4 is an interpretability analysis rather than a separately trained prediction model.

| Experiment | Features | Alpha | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Instructor Linear Regression | 44 | N/A | — | 2.7025 | 3.3310 | 0.9306 |
| E1 — Baseline Lasso | 47 | 1.0 | — | 3.1609 | 3.8578 | 0.9069 |
| E2 — Alpha Tuning | 47 | **0.01** | **2.8195** | — | — | — |
| E3 — Tuned Lasso | 47 | 0.01 | 2.8195 | **2.6883** | **3.3242** | **0.9308** |
| E4 — Permutation Importance | 47 | 0.01 | — | — | — | — |
| E5 — Original-Feature Lasso | 44 | 0.01 | **2.8190** | 2.6898 | 3.3248 | 0.9308 |

### 10.1 E1 vs E3 — Effect of Alpha Tuning

E1 testing MAE: 3.1609

E3 testing MAE: 2.6883

**MAE reduction: 0.4726 wins**

This demonstrates that tuning alpha substantially improved the baseline Lasso model.

### 10.2 E3 vs E5 — Effect of Feature Engineering

E3 testing MAE: 2.6883

E5 testing MAE: 2.6898

**MAE difference: 0.0015 wins**

E3 performed marginally better on the testing dataset, while E5 achieved a marginally better cross-validation MAE.

These differences are too small to establish a meaningful predictive advantage for either feature set.

### 10.3 Linear Regression vs Lasso

The instructor's 44-feature Linear Regression achieved a testing MAE of 2.7025.

E5 Lasso, using the same 44 original features, achieved a testing MAE of 2.6898.

The observed improvement was 0.0127 wins.

Although E5 performed slightly better on this testing split, the difference was very small and does not establish that Lasso will consistently outperform ordinary Linear Regression.

---

## 11. Key Research Findings

### Finding 1 — Hyperparameter Tuning Was Important

The initial alpha of 1.0 imposed excessive regularization, leaving only three active features.

Five-fold cross-validation identified alpha = 0.01 as the best candidate among the tested values, substantially reducing prediction error.

### Finding 2 — Lasso Performed Automatic Feature Selection

E3 retained 32 of 47 features, while E5 retained 33 of 44 features.

This demonstrates the ability of L1 regularization to shrink some regression coefficients to zero.

### Finding 3 — Feature Engineering Did Not Meaningfully Improve Accuracy

The three engineered features produced only a negligible difference in testing MAE.

This suggests that the original features already contained substantial predictive information.

### Finding 4 — Feature Importance Changed When Correlated Features Were Introduced

E3 relied strongly on Run Differential, whereas E5 relied strongly on Runs Scored and Runs Allowed.

The difference reflects how Lasso handles overlapping predictive information.

### Finding 5 — A Simpler Model Can Perform Equally Well

E5 achieved virtually the same predictive accuracy as E3 while requiring fewer input features and no additional feature engineering.

This supports selecting the simpler model when predictive differences are negligible.

---

## 12. Model Selection and Limitations

### Recommended Model

**E5 — Lasso Regression with 44 original features and alpha = 0.01**

Reasons:

1. Achieved a testing MAE of 2.6898 wins.
2. Produced nearly identical accuracy to E3.
3. Achieved slightly better cross-validation MAE.
4. Did not require the three engineered features.
5. Retained automatic feature selection through L1 regularization.

### Limitations

- The experiments used one training/testing split, so performance may vary with different samples.
- The alpha search evaluated six candidate values and does not guarantee a globally optimal alpha.
- Correlated features can affect Lasso coefficient selection and permutation importance.
- Small differences in MAE do not establish statistically meaningful superiority.
- Historical full-season statistics were used to predict historical season wins. These results should not be interpreted as a validated preseason forecasting model because many of the input statistics are only known after a season has progressed or finished.

### Final Conclusion

The E1–E5 experiments demonstrate that hyperparameter tuning has a greater observed impact on this Lasso model's predictive accuracy than the three additional engineered features.

Five-fold cross-validation identified alpha = 0.01 as the best tested regularization parameter.

The tuned models achieved testing MAE values of approximately 2.69 wins, with E3 and E5 showing nearly identical predictive performance.

Permutation importance analysis showed that the engineered Run Differential feature changed how the model relied on Runs Scored and Runs Allowed but did not meaningfully improve prediction accuracy.

**The 44-feature E5 Lasso model is selected as the preferred simpler model, while E3 and E4 provide useful evidence about feature engineering and model interpretability.**

Overall, these experiments illustrate the importance of systematic hyperparameter tuning, cross-validation, feature selection, and evaluating whether additional features genuinely improve predictive performance.
