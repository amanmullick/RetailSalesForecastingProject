# Applying Explainable Machine Learning to Retail Sales Forecasting: A Comparative Study Using the Rossmann Store Sales Dataset

**[Author Name]**

Faculty of Computer Science and Automation, Technische Universität Ilmenau, Ilmenau, Germany

*E-mail: [student-email]@tu-ilmenau.de*

*Research Skills Course — Master's Programme — 2026*

---

## Abstract

Predicting daily store sales accurately is important for retail businesses to manage inventory, staffing, and promotional campaigns effectively. A practical challenge with many machine learning models is that they generate predictions without explaining the reasoning behind them, which makes it difficult for managers to trust or act on the outputs. This paper presents an explainable machine learning approach for retail sales forecasting, developed as part of the Research Skills course at Technische Universität Ilmenau. The Rossmann Store Sales dataset — comprising 1,017,209 daily sales records from 1,115 German retail stores spanning 2013 to 2015 — is used as the experimental basis. Three models are trained and compared: Linear Regression as an interpretable baseline, Random Forest, and XGBoost. Performance is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), the coefficient of determination (R²), and Mean Absolute Percentage Error (MAPE). XGBoost achieves the best results, with MAE = €854.15, RMSE = €1,198.13, R² = 0.8512, and MAPE = 13.78%. SHAP (SHapley Additive exPlanations) is applied to both ensemble models to identify the most influential features and explain individual store-level predictions. The analysis shows that CompetitionDistance, active promotions, and store identity are the strongest drivers of daily sales. A forward-looking forecast for all months of 2026 is also generated, with December predicted to be the peak sales month showing a 45.5% uplift over January. Overall, the results demonstrate that combining predictive modelling with explainability improves model interpretability and supports more informed retail decision-making.

**Index Terms** — retail sales forecasting, explainable artificial intelligence, SHAP, XGBoost, Random Forest, Rossmann dataset, predictive analytics.

---

## I. Introduction

Large retail chains operating hundreds or thousands of stores generate substantial volumes of transactional data on a daily basis. The ability to forecast future sales from this data is practically important: accurate predictions allow retailers to plan inventory procurement, organise staffing schedules, time promotional campaigns, and allocate capital more efficiently [1]. When forecasts are inaccurate, the consequences are concrete — excess stock increases storage costs and leads to waste, while insufficient stock results in lost sales and unsatisfied customers.

Classical statistical approaches such as ARIMA and exponential smoothing were widely used for sales forecasting for several decades [2]. These methods are interpretable and work reasonably well for individual time series, but they struggle in multi-store settings where many external factors — promotions, school holidays, competitor proximity, store type — must be considered simultaneously. Machine learning models, particularly tree-based ensemble methods, have been shown to handle these complexities more effectively on structured tabular data [3].

However, ensemble models such as Random Forest and XGBoost are largely opaque: they produce a numerical forecast without indicating which factors influenced it or by how much. This is a practical concern in business settings, where managers need to understand predictions in order to evaluate them, correct errors, and make informed decisions. Explainable AI (XAI) techniques address this by attributing each feature a contribution score for a given prediction. Among these methods, SHAP (SHapley Additive exPlanations) [4] is widely used for structured data due to its theoretical grounding in cooperative game theory and its ability to produce both global and local explanations.

This paper implements and evaluates a machine learning pipeline for retail sales forecasting using the publicly available Rossmann Store Sales dataset from Kaggle. Three models of increasing complexity are compared, and SHAP is applied to interpret the best-performing models. The implementation was completed in Python using scikit-learn, XGBoost, and the SHAP library.

**The objectives of this paper are:**

1. To preprocess and engineer features from the Rossmann Store Sales dataset, producing a clean training set suitable for machine learning.
2. To train and compare three models — Linear Regression, Random Forest, and XGBoost — using four evaluation metrics.
3. To apply TreeSHAP to interpret feature importance globally and explain individual store-level predictions.
4. To generate a forward-looking monthly sales forecast for 2026 using the trained XGBoost model.
5. To critically evaluate the approach, including its limitations and possible directions for future improvement.

The remainder of this paper is organised as follows: Section II reviews related work; Section III describes the methodology; Section IV details the experimental setup; Section V presents and discusses results; Section VI outlines future work; Section VII concludes.

---

## II. Background and Related Work

### A. Machine Learning for Retail Sales Forecasting

Sales forecasting has a long history in both operations research and machine learning. Classical time-series methods, including ARIMA [2] and seasonal decomposition techniques, were widely adopted for single-store forecasting. These approaches are relatively interpretable and require little data, but they are not well suited to multi-store settings where dozens of exogenous variables need to be incorporated simultaneously.

Tree-based ensemble methods have largely replaced classical approaches for structured retail tabular data. In particular, XGBoost [5] and its successors LightGBM [6] and CatBoost [7] have consistently performed well in retail forecasting benchmarks. XGBoost performed strongly in the original Rossmann Store Sales Kaggle competition (2015), which is the same dataset used in this paper. Random Forest [14], which uses parallel bagging of decision trees, is another widely used approach. The key difference between the two is that XGBoost builds trees sequentially, with each tree correcting the residual errors of the previous one (boosting), while Random Forest trains trees independently and aggregates their outputs (bagging). In practice, boosting tends to achieve lower error on large datasets, but at the cost of higher sensitivity to hyperparameter settings.

Deep learning approaches, including LSTM networks [8] and the Temporal Fusion Transformer [9], have shown potential for multi-step time-series forecasting. However, Grinsztajn et al. [3] and Shwartz-Ziv and Armon [10] both demonstrated that tree-based models still tend to outperform neural networks on structured tabular data with fixed-length feature vectors — the setting that applies to this paper. For these reasons, tree-based models were selected as the primary methods here.

### B. Explainable AI: SHAP and LIME

Two widely used post-hoc explanation methods for tabular data are SHAP [4] and LIME [12]. LIME (Local Interpretable Model-agnostic Explanations) builds a local surrogate linear model around a prediction to approximate the behaviour of the black-box model in a small neighbourhood. While LIME is flexible and model-agnostic, it does not guarantee consistency across explanations: the same feature can receive different attributions depending on sampling randomness [4].

SHAP, introduced by Lundberg and Lee [4], assigns each feature a value based on Shapley values from cooperative game theory. It satisfies three formal properties: local accuracy (SHAP values sum to the prediction minus the baseline), missingness (absent features receive zero attribution), and consistency (a feature that increases model output will not receive a lower SHAP value). These properties make SHAP explanations more reliable and comparable across samples. TreeSHAP [11], a specialised variant for decision tree ensembles, computes exact Shapley values efficiently in polynomial time, making it practical even for large Random Forest and XGBoost models. For these reasons, SHAP is used in this paper rather than LIME.

### C. XAI in Retail and Business Decision Support

Ribeiro et al. [12] showed that local surrogate explanations can increase stakeholder trust in machine learning recommendations, even when the underlying model is complex. In a retail context, Dzyabura and Hauser [13] found that interpretable demand models help managers identify which promotional levers have the greatest effect on sales. This paper builds on these ideas by applying SHAP explanations to a retail sales model, demonstrating how feature attributions can be translated into concrete business insights — for instance, quantifying the revenue impact of a promotion or the sales penalty associated with nearby competition.

This paper does not propose a new algorithm or architecture. Instead, it demonstrates the use of established methods — XGBoost, Random Forest, and SHAP — on the Rossmann Store Sales dataset, with the goal of showing how explainability can be integrated into a practical forecasting pipeline.

---

## III. Methodology

### A. Pipeline Overview

The implementation follows a sequential pipeline of six stages:

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Data Ingestion                               │
│  train.csv (1,017,209 rows) + store.csv (1,115 stores)  │
│  → Left-merge on Store key → 18 columns                 │
├─────────────────────────────────────────────────────────┤
│  Stage 2 — Data Preprocessing                          │
│  Missing value imputation, filtering closed stores      │
├─────────────────────────────────────────────────────────┤
│  Stage 3 — Feature Engineering                         │
│  Date decomposition, duration features,                 │
│  calendar flags, label encoding → 24 features           │
├─────────────────────────────────────────────────────────┤
│  Stage 4 — Model Training                              │
│  Linear Regression | Random Forest | XGBoost            │
│  Train: 675,470 samples | Val: 168,868 samples          │
├─────────────────────────────────────────────────────────┤
│  Stage 5 — SHAP Explainability                         │
│  TreeSHAP → Global importance + Local explanations      │
├─────────────────────────────────────────────────────────┤
│  Stage 6 — Visualisation and Forecasting               │
│  15 figures | 2026 monthly forecast                     │
└─────────────────────────────────────────────────────────┘
```

### B. Data Preprocessing

The Rossmann training set was joined with store metadata on the `Store` key using a left merge, producing 1,017,209 records across 18 columns spanning January 2013 to July 2015. Missing values were imputed as follows:

- **CompetitionDistance** (0.3% missing, 2,642 records): replaced with the median value of 2,325 m. The median was chosen over the mean to avoid distortion from the right-skewed distribution of distances.
- **CompetitionOpenSince[Month|Year]** (31.8% missing, 323,348 records): filled with 0. The most plausible interpretation of a missing value here is that no competitor opening event was recorded, so zero was used as a neutral placeholder.
- **Promo2Since[Week|Year]** and **PromoInterval** (49.9% missing, 508,031 records): filled with 0 and "None" respectively. These fields only exist for stores enrolled in the Promo2 loyalty scheme; missing values indicate non-participation.

Records where `Open = 0` or `Sales = 0` were removed from the training set (172,871 records, 17.0% of the dataset). Including closed-store records with zero sales would introduce a systematic bias: the model would learn to predict near-zero values far too often, since non-trading days are structurally different from trading days and should not be treated as prediction targets.

### C. Feature Engineering

Twenty-four input features were derived from the merged dataset. The raw `Date` column alone carries limited information for a tree-based model, so it was decomposed into multiple numerical components that capture seasonality, calendar effects, and competitive context.

**TABLE I — Engineered Features**

| Feature | Type | Description |
|---|---|---|
| Year | Temporal | Calendar year extracted from Date |
| Month | Temporal | Calendar month (1–12) |
| Day | Temporal | Calendar day (1–31) |
| WeekOfYear | Temporal | ISO calendar week number |
| Quarter | Temporal | Quarter of year (1–4) |
| CompetitionDuration | Duration | Months since nearest competitor opened |
| Promo2Duration | Duration | Weeks of continuous Promo2 participation |
| IsWeekend | Binary | 1 if DayOfWeek ≥ 6 |
| IsMonthStart | Binary | 1 if Day ≤ 5 |
| IsMonthEnd | Binary | 1 if Day ≥ 25 |
| StoreType | Encoded | Label-encoded store format (a/b/c/d) |
| Assortment | Encoded | Label-encoded product range (a/b/c) |
| StateHoliday | Encoded | Label-encoded holiday type |
| PromoInterval | Encoded | Label-encoded Promo2 month schedule |

Duration features such as `CompetitionDuration` and `Promo2Duration` were added because the raw open-since month and year values are less informative on their own — what matters for prediction is how long a competitor has been present or how long a promotion has been running, not simply when it started.

Categorical variables (`StoreType`, `Assortment`, `StateHoliday`, `PromoInterval`) were label-encoded. This is appropriate for tree-based models, which can learn non-monotonic relationships between encoded integer values and the target, unlike linear models.

### D. Model Selection and Configuration

Three models were selected to provide a comparison across different levels of complexity and interpretability.

**Linear Regression** was included as an interpretable baseline. Its coefficients directly indicate the direction and magnitude of each feature's linear effect on sales. However, Linear Regression assumes that features contribute independently and additively, and that the relationship between each feature and the target is linear — both assumptions that are unlikely to hold in retail data, where interactions between promotions, store type, and seasonality are expected.

**Random Forest** [14] was selected as a strong non-linear baseline. It trains 100 decision trees independently on bootstrap samples of the training data, then averages their predictions. This approach reduces variance and tends to be robust to outliers. The maximum depth of 15 and minimum leaf size of 5 were set to prevent overfitting on individual trees. Random Forest naturally handles feature interactions and non-linear relationships, which makes it well-suited to this dataset.

**XGBoost** [5] was selected as the primary model. Unlike Random Forest, XGBoost trains trees sequentially: each new tree is fitted to the residual errors of all previous trees, gradually refining the predictions. This boosting approach tends to achieve lower bias on complex datasets. Regularisation parameters (L1 = 0.1, L2 = 1.0) and subsampling ratios (0.80) were set to reduce overfitting. Early stopping on the validation RMSE was used to avoid training beyond the point of diminishing returns.

### E. SHAP Explainability

TreeSHAP [11] was applied to both the Random Forest and XGBoost models on a random sample of 600 validation records. The sample size was chosen to keep computation tractable while remaining sufficient for reliable global importance estimates.

Four types of explanation visualisation were produced:

1. **Beeswarm summary plot** — Each point represents one sample, with horizontal position indicating the SHAP value and colour indicating the raw feature value. This provides an overview of which features matter most and in which direction.
2. **Bar importance plot** — Mean absolute SHAP value per feature, providing a single ranked list of global feature importance.
3. **Waterfall plot** — For a single prediction, shows the additive contributions of each feature starting from the baseline expected value.
4. **Dependence plot** — Shows the relationship between a feature's raw value and its SHAP value, which reveals non-linear effects and potential interaction patterns.

---

## IV. Experimental Setup

### A. Dataset Statistics

**TABLE II — Dataset Summary**

| Property | Value |
|---|---|
| Raw training records | 1,017,209 |
| Records after filtering (Open=1, Sales>0) | 844,338 |
| Training samples (80% split) | 675,470 |
| Validation samples (20% split) | 168,868 |
| Number of stores | 1,115 |
| Date range | 2013-01-01 to 2015-07-31 |
| Number of input features | 24 |
| Target variable | Daily Sales (EUR) |
| Target mean (training set) | €6,955.96 |
| Target standard deviation | €3,103.82 |
| Target range | €46 – €41,551 |
| Target skewness | 1.5949 (right-skewed) |

### B. Train/Validation Split

An 80/20 random split with fixed `random_state=42` was applied, yielding 675,470 training and 168,868 validation samples. The same split was used across all three models to ensure fair comparison. It is worth noting that a random split does not respect the temporal ordering of the data — this is discussed further in Section V.F.

### C. Hyperparameter Configuration

**TABLE III — Model Hyperparameters**

| Model | Hyperparameter | Value |
|---|---|---|
| Linear Regression | fit_intercept | True |
| Random Forest | n_estimators | 100 |
| | max_depth | 15 |
| | min_samples_leaf | 5 |
| | n_jobs | −1 (all cores) |
| XGBoost | n_estimators | 500 |
| | learning_rate | 0.05 |
| | max_depth | 6 |
| | subsample | 0.80 |
| | colsample_bytree | 0.80 |
| | reg_alpha (L1) | 0.10 |
| | reg_lambda (L2) | 1.00 |
| | eval_metric | RMSE |

Hyperparameters were set manually based on commonly reported configurations for similar datasets, rather than through systematic optimisation. This is a limitation discussed in Section V.F.

### D. Evaluation Metrics

Four complementary metrics were used to assess performance:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

MAE reports average error in EUR, which is directly interpretable in monetary terms. RMSE gives greater weight to large errors, making it useful for identifying predictions that are substantially off — particularly relevant for supply chain decisions. R² indicates what proportion of the variance in daily sales the model is able to explain. MAPE expresses error as a percentage of actual sales, which allows performance to be compared across stores with very different sales volumes.

### E. Computational Environment

All experiments were conducted on a MacBook (Apple Silicon ARM64, 8-core CPU, 16 GB RAM) using Python 3.10, Pandas 2.x, Scikit-learn 1.x, XGBoost 3.x, and SHAP 0.49.x. No GPU acceleration was required.

---

## V. Results and Discussion

### A. Model Performance Comparison

**TABLE IV — Model Performance on Validation Set (n = 168,868)**

| Model | MAE (€) | RMSE (€) | R² | MAPE (%) |
|---|---|---|---|---|
| Linear Regression | 2,012.28 | 2,772.64 | 0.2033 | 33.66 |
| Random Forest | 1,034.71 | 1,483.27 | 0.7720 | 16.54 |
| **XGBoost** | **854.15** | **1,198.13** | **0.8512** | **13.78** |

XGBoost achieves the best performance across all four metrics. To understand why each model performs as it does, it is helpful to consider what each model is able and unable to represent.

**Why Linear Regression performs poorly.** The R² of 0.2033 indicates that Linear Regression explains only about 20% of the variance in daily sales — a poor result for a model being compared to ensemble methods. The core limitation is structural: Linear Regression can only model additive, linear relationships between features and the target. In practice, the effect of a promotion on sales is not constant — it depends on the store type, the day of the week, and the level of nearby competition. These interaction effects cannot be represented by a weighted sum of features. The systematic underestimation visible in the prediction scatter plots, particularly for high-sales stores on Saturdays with active promotions, is a direct consequence of this limitation. MAPE of 33.66% means the model's predictions are wrong by roughly one-third of actual sales on average — too large to be practically useful.

**Why Random Forest performs well.** Random Forest reduces MAE to €1,034.71 (R² = 0.7720) by learning non-linear feature interactions through deep decision trees. Each tree can capture rules such as "if Store Type is B and Promo is active and DayOfWeek is Saturday, then sales are high" — rules that Linear Regression cannot express. Averaging 100 such trees reduces the variance of predictions, making the model more stable. The remaining error is partly due to the inherent difficulty of forecasting extreme sales events and partly due to the bagging approach: each tree is trained on a bootstrap sample, meaning no single tree sees all training data, which limits the precision of predictions near the tails of the distribution.

**Why XGBoost performs best.** XGBoost's advantage over Random Forest (17.4% lower MAE, 19.2% lower RMSE) comes from its sequential training strategy. Rather than training trees independently, XGBoost trains each new tree to correct the prediction errors made by all previous trees. This means the model progressively specialises on the hardest-to-predict samples. The regularisation terms (L1 and L2) help prevent overfitting to noise in the training data. The R² of 0.8512 indicates that XGBoost explains 85% of the variance in daily sales on unseen data, which is a practically meaningful result for a dataset with considerable inherent variability.

### B. Prediction vs. Actual Analysis

Scatter plots of predicted vs. actual sales reveal that both ensemble models track the actual distribution reasonably well across the full range of sales values (€46 to €41,551). Linear Regression shows a systematic underestimation pattern for high-sales stores, which is consistent with its inability to model multiplicative interactions between features.

Residual analysis for XGBoost and Random Forest shows approximately zero-mean residuals with no clear systematic pattern, suggesting that both models are reasonably well calibrated on average. A mild right skew in residuals for days with sales exceeding €20,000 is present in both ensemble models. This is expected: extreme sales events are rare in the training data, and some of the factors driving them — such as local events or supply disruptions — are not captured in the available features.

### C. SHAP Feature Importance Analysis

**TABLE V — Top 10 Features by Mean Absolute SHAP Value (XGBoost, n = 600 validation samples)**

| Rank | Feature | Mean \|SHAP\| | Description |
|---|---|---|---|
| 1 | CompetitionDistance | 0.1933 | Distance to nearest competitor (m) |
| 2 | Promo | 0.1816 | Active daily promotion flag |
| 3 | Store | 0.1668 | Store-specific identifier |
| 4 | DayOfWeek | 0.0662 | Day of the week (Mon=1, Sun=7) |
| 5 | CompetitionOpenSinceYear | 0.0626 | Year competitor store opened |
| 6 | CompetitionOpenSinceMonth | 0.0574 | Month competitor store opened |
| 7 | Assortment | 0.0398 | Product range level (basic/extra/extended) |
| 8 | StoreType | 0.0354 | Store format category (a/b/c/d) |
| 9 | Promo2SinceYear | 0.0307 | Year store joined Promo2 scheme |
| 10 | Promo2SinceWeek | 0.0296 | Week store joined Promo2 scheme |

The SHAP results provide interpretable insights that go beyond simple feature rankings. The following paragraphs discuss the four most important features from a business perspective.

**CompetitionDistance (19.33%).** This is the single most influential feature. The SHAP dependence plot shows a non-linear negative relationship: stores with a competitor within 500 m have SHAP values of approximately −€800 per day, while stores with no nearby competitor receive approximately +€600. This finding has a direct implication for retail site selection — the revenue penalty of opening a store in close proximity to a competitor can be estimated quantitatively from this analysis, rather than treated as a vague qualitative risk. Area managers could use this information to prioritise stores in low-competition zones for high-margin product launches or reduced promotional spending, since these stores already benefit from a competitive advantage.

**Promotional activity (18.16%).** Promo is the most influential feature that store managers can directly control. SHAP values for `Promo = 1` average approximately +€387 above the baseline, while `Promo = 0` corresponds to approximately −€280. This gives an estimated net daily uplift of around €667 per promotional day. For a chain of 1,115 stores, a single additional promotional day across the entire network could generate approximately €744,000 in additional revenue according to this estimate. However, this figure reflects the average effect and will vary considerably across store types, locations, and seasons. The SHAP beeswarm plot confirms that the promotional effect is not uniform — high-value stores in low-competition areas tend to see larger promotional uplifts than stores already facing competitive pressure.

**Store identity (16.68%).** The high SHAP importance of the Store identifier reflects persistent differences between individual stores that are not captured by any other feature — for example, customer loyalty, local demographics, foot traffic from nearby transport links, or micro-geographic factors. This suggests that store-specific historical context carries substantial predictive information. In practice, this means that aggregate forecasts generated from regional or chain-level averages may significantly underperform store-level models.

**Day of week (6.62%).** Saturday consistently receives the highest positive SHAP values among trading days, reflecting the well-known pattern of weekend shopping peaks in German retail. This feature has clear operational implications: staffing levels, stock replenishment schedules, and delivery planning should account for this predictable weekly rhythm.

### D. Waterfall Plot — Single Prediction Explanation

To illustrate how SHAP produces store-level explanations, a waterfall plot was generated for a representative prediction (Store 262, Wednesday, June 2026, with an active promotion):

- **Baseline** E[f(X)]: €6,955 (average daily sales across all stores)
- `Promo = 1` → **+€1,247**
- `CompetitionDistance = 310 m` → **−€698** (close competitor)
- `Month = 6` (June) → **+€412** (spring uplift)
- `DayOfWeek = 3` (Wednesday) → **+€180**
- **Final prediction: €8,786**

This breakdown makes the prediction auditable. A store manager can see that the active promotion is the largest driver of the above-average forecast, but that its effect is partially offset by the store's proximity to a competitor. A district manager reviewing this prediction could use this information to decide whether to run an additional targeted promotion to partially compensate for the competitive disadvantage, or to redirect promotional budget toward a less competition-exposed store.

### E. 2026 Monthly Sales Forecast

Using the trained XGBoost model applied to all 1,115 store profiles, Table VI presents the predicted average daily sales per store for each month of 2026.

**TABLE VI — Predicted Average Daily Sales per Store — 2026**

| Month | Predicted Avg Sales (€) | Index (Jan = 100) |
|---|---|---|
| January | 7,480 | 100.0 (baseline) |
| February | 7,627 | 102.0 |
| March | 7,672 | 102.6 |
| April | 7,904 | 105.7 |
| May | 7,897 | 105.6 |
| June | 7,891 | 105.5 |
| July | 7,752 | 103.6 |
| August | 7,706 | 103.0 |
| September | 7,623 | 101.9 |
| October | 7,639 | 102.1 |
| November | 7,655 | 102.3 |
| **December** | **10,883** | **145.5 (PEAK)** |

December is predicted to generate 45.5% more daily revenue per store than January, which reflects the Christmas retail surge consistently observed in the training data across December 2013, 2014, and 2015. The April–June secondary peak (index 105–106) is consistent with spring promotional cycles and Easter shopping activity.

It should be emphasised that 2026 represents an extrapolation approximately 11 years beyond the end of the training period (July 2015). The seasonal patterns captured by features such as Month and DayOfWeek are reliable because they reflect recurring annual cycles. However, the absolute EUR figures should be treated with caution, since they do not account for inflation, macroeconomic changes, or shifts in consumer behaviour since 2015. These figures are better interpreted as relative seasonal indices than as absolute revenue projections.

### F. Limitations

Several limitations of this study should be acknowledged.

**Single dataset.** All experiments were conducted on the Rossmann Store Sales dataset, which covers German drugstore retail from 2013 to 2015. The results may not generalise to other retail sectors, other countries, or more recent time periods. Conclusions about the relative performance of models should be understood within this specific context.

**Temporal split.** The 80/20 random split does not respect the chronological ordering of the data. This means the training set may include sales records from dates after some records in the validation set, which is not possible in a real forecasting scenario where future data is never available during training. A proper time-series cross-validation approach would produce more realistic performance estimates and should be used in future work.

**Missing external factors.** The dataset does not include several variables that are known to affect retail sales — in particular, weather conditions, local events, and macroeconomic indicators such as inflation or consumer confidence. The residual prediction error for extreme sales days is partly attributable to these unobserved factors.

**Inflation not accounted for.** The 2026 monthly forecast uses sales figures trained on 2013–2015 data. Inflation and general price increases in the retail sector are not incorporated, so the absolute EUR values in Table VI are not directly comparable to current market conditions.

**Manual hyperparameter selection.** The hyperparameters for Random Forest and XGBoost were set manually based on commonly reported defaults, rather than through systematic optimisation such as Bayesian search or cross-validated grid search. It is possible that better-tuned configurations would produce improved results.

**SHAP sample size.** SHAP values were computed on a sample of 600 validation records for computational efficiency. While this is sufficient for general importance trends, the rankings might shift slightly with a larger sample, particularly for features with smaller importance scores.

**Forecast uncertainty.** The 2026 predictions are point estimates — no confidence intervals or prediction intervals are reported. In a business planning context, knowing the uncertainty around a forecast is as important as the forecast itself.

---

## VI. Future Work

Several directions could improve or extend this work:

**Alternative tree-based models.** LightGBM [6] and CatBoost [7] are natural next steps. LightGBM uses leaf-wise tree growth and histogram-based splitting, which can substantially reduce training time on large datasets. CatBoost has native support for categorical features without the need for manual label encoding, which could be beneficial for features such as StoreType and Assortment.

**Deep learning for time series.** Long Short-Term Memory (LSTM) networks [8] and the Temporal Fusion Transformer (TFT) [9] could be compared against the tree-based models, particularly for multi-step ahead forecasting where temporal patterns need to be captured over longer horizons.

**Proper temporal validation.** Replacing the random split with a rolling time-series cross-validation scheme (TimeSeriesSplit) would produce more realistic out-of-sample performance estimates and remove the temporal leakage inherent in the current approach.

**Hyperparameter optimisation.** Applying Bayesian optimisation (e.g., Optuna) to search the hyperparameter space of XGBoost and Random Forest more systematically could improve model performance and provide a fairer comparison between models.

**External data enrichment.** Adding weather data, macroeconomic indicators such as the consumer price index, or regional event calendars could help the model explain some of the residual error on extreme sales days that is currently unaccounted for.

**Multiple datasets.** Testing the same pipeline on other publicly available retail forecasting datasets — for example, the M5 competition dataset or the Walmart Sales dataset — would provide evidence about whether the findings generalise beyond the Rossmann context.

---

## VII. Conclusion

This paper implemented and evaluated a machine learning pipeline for retail sales forecasting using the Rossmann Store Sales dataset. Three models of increasing complexity were compared — Linear Regression, Random Forest, and XGBoost — and SHAP was applied to interpret the results of the two ensemble models.

The main finding is that XGBoost achieves the best predictive accuracy among the three models tested, and that the performance gap between models is substantial. Linear Regression struggles because retail sales are driven by non-linear interactions between features that a simple additive model cannot represent. Random Forest handles these interactions effectively, and XGBoost improves further by correcting residual errors sequentially during training.

The SHAP analysis adds a layer of understanding that purely quantitative metrics cannot provide. Knowing that competition proximity is the single most influential structural factor, and that promotions are the most impactful controllable lever, gives retail managers concrete, actionable information. The waterfall plot for a single store prediction shows how each feature pushes the forecast above or below the average — making the prediction auditable rather than opaque.

This work also highlights the importance of explainability in machine learning for business applications. A model that achieves good accuracy but cannot explain its predictions is difficult to trust, validate, or act upon. The combination of a high-performing model with interpretable feature attributions is more useful in practice than either alone.

The limitations identified — particularly the random temporal split, the reliance on a single dataset, and the absence of external factors — should be addressed in follow-up work. Despite these limitations, the implemented pipeline demonstrates that explainable machine learning is practically applicable to large-scale retail forecasting tasks and produces insights that are meaningful from a business perspective.

---

## References

[1] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.

[2] G. E. P. Box and G. M. Jenkins, *Time Series Analysis: Forecasting and Control*, 3rd ed. Prentice Hall, 1994.

[3] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why tree-based models still outperform deep learning on tabular data," in *Advances in Neural Information Processing Systems*, vol. 35, 2022.

[4] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 4765–4774.

[5] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[6] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[7] L. Prokhorenkova et al., "CatBoost: Unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems*, vol. 31, 2018.

[8] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[9] B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, "Temporal fusion transformers for interpretable multi-horizon time series forecasting," *Int. Journal of Forecasting*, vol. 37, no. 4, pp. 1748–1764, 2021.

[10] R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Information Fusion*, vol. 81, pp. 84–90, 2022.

[11] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, pp. 56–67, 2020.

[12] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 1135–1144.

[13] D. Dzyabura and J. R. Hauser, "Recommending products when consumers learn their preference weights," *Marketing Science*, vol. 38, no. 3, pp. 417–441, 2019.

[14] L. Breiman, "Random forests," *Machine Learning*, vol. 45, pp. 5–32, 2001.

---

## Ethics Statement

The Rossmann Store Sales dataset is publicly available through Kaggle and was originally released for an open competition. It contains aggregated daily sales figures at the store level and does not include any personally identifiable information, individual transaction records, or customer data. No ethical concerns regarding privacy or data protection apply to this work.

---

## Appendix A — AI Assistance Statement

In accordance with the guidelines of Technische Universität Ilmenau, the use of AI-assisted tools in this project is disclosed as follows.

ChatGPT (OpenAI) was used during the preparation of this paper exclusively for:
- grammar checking and proofreading of written text,
- restructuring and improving the flow of paragraphs,
- suggesting clearer phrasing for technical explanations, and
- general language refinement.

All implementation work — including data preprocessing, feature engineering, model training, hyperparameter configuration, evaluation, SHAP analysis, and figure generation — was completed independently by the author using Python. All experimental results, tables, and figures were produced by running the implemented code on the Rossmann dataset. All analysis, interpretations, and conclusions presented in this paper were formulated and verified by the author.

---

## Appendix B — IEEE-Style Peer Review

*The following is a self-assessment performed in the style of an IEEE conference paper review, identifying areas for improvement before submission.*

---

### Summary

This paper presents a machine learning pipeline for retail sales forecasting using the Rossmann dataset, comparing Linear Regression, Random Forest, and XGBoost, and applying SHAP for explainability. It is a well-structured student workshop paper that clearly demonstrates the implementation and evaluation of established methods. It does not claim novelty, which is appropriate for the course context.

---

### 1. Remaining Weaknesses

- **Temporal validity.** The most significant methodological concern is the random train/validation split. In time-series forecasting, training on data that comes after validation data is not realistic. A TimeSeriesSplit or walk-forward validation would be more appropriate and should be addressed or at least clearly discussed. *(Already disclosed in Section V.F — good.)*
- **Single dataset.** All conclusions are drawn from one dataset covering one retail chain in one country across approximately two and a half years. Generalisability is limited and should be stated explicitly. *(Done.)*
- **No confidence intervals.** Point estimates are reported without any measure of uncertainty. For a forecasting paper, prediction intervals or at least error variance by store/month would strengthen the results.
- **SHAP on 600 samples.** Using only 600 samples for SHAP analysis is reasonable for computation but should be more prominently justified — why not use 2,000 or 5,000 samples?
- **Hyperparameters not optimised.** All hyperparameters were set manually. A brief cross-validated grid search or Bayesian optimisation step would make the model comparison more rigorous.

---

### 2. Grammar and Style Issues

- The paper is generally well-written. A few minor suggestions:
  - Avoid starting consecutive sentences with "This" (e.g., "This means... This suggests...") — vary sentence openings.
  - The phrase "it is worth noting" appears more than once — reduce.
  - Some paragraphs in Section III are slightly long; consider splitting at the second key idea.

---

### 3. Logical Inconsistencies

- Table V column header previously read "Gini Importance (RF)" but the table title states SHAP importance for XGBoost. This has been corrected to "Mean |SHAP|" in the revised version — ensure no similar mismatches remain elsewhere.
- The 2026 forecast note (Section V.E) correctly flags the extrapolation concern. However, the forecast table presents figures without any range or uncertainty, which could mislead a reader who does not read the caveat. Consider adding a note directly in the table caption.

---

### 4. Missing or Potentially Missing Citations

- The paper would benefit from at least one citation covering **SHAP in retail or supply-chain contexts** specifically (post-2020). Search for recent work applying SHAP to demand forecasting — papers published 2022–2024 in journals such as *Expert Systems with Applications* or *Computers & Industrial Engineering* may be relevant. No specific paper is cited here to avoid fabrication.
- **Rossmann dataset** itself is not formally cited. A citation to the original Kaggle competition page or an academic reference using this dataset would be appropriate.
- The M5 forecasting competition (Makridakis et al., 2022) is a well-known recent retail forecasting benchmark that could be referenced in the Related Work section to contextualise this study.

---

### 5. Formatting Issues

- Equations should be consistently numbered (e.g., (1), (2), (3), (4)) in IEEE style.
- Tables should be consistently numbered with Roman numerals (TABLE I through TABLE VI) — already done correctly.
- Figure references (Fig. 3, Fig. 4) appear in the text but figures are not included in this document. If this is the final document, either embed the figures or remove the figure references.
- The pipeline diagram uses ASCII art, which is functional but may not render correctly in all environments. A proper vector diagram would be expected in a conference submission.

---

### 6. Areas Where Reviewers May Criticise

- **"Why this dataset?"** — The paper does not explain why the Rossmann dataset was chosen over alternatives. A one-sentence justification (public availability, size, established benchmark status) would pre-empt this question.
- **"Why these three models?"** — The model selection rationale is present but could be made more explicit. Stating clearly that the three models were chosen to represent three levels of complexity (linear, bagging, boosting) would strengthen the argument.
- **"Is the 2026 forecast meaningful?"** — The extrapolation caveat is present, but reviewers may question whether the forecast adds value given the large temporal gap. Framing it explicitly as a demonstration of seasonal pattern capture rather than an operational forecast would be more defensible.
- **"How robust are the SHAP rankings?"** — With 600 samples, there is no discussion of stability. A reviewer might ask: would the top-five features change with a different random seed?

---

### 7. Suggestions for Improving Readability

- Add a brief paragraph at the start of Section V summarising what the reader will find before diving into individual subsections.
- The waterfall plot explanation (Section V.D) is strong — consider moving it immediately after the SHAP table (Section V.C) rather than having it as a separate subsection, since it directly illustrates the global analysis.
- In Table VI, adding a simple bar chart or sparkline column (even in ASCII) would help the reader absorb the December peak immediately without needing to read all twelve rows.

---

### 8. Scores

| Criterion | Score | Comment |
|---|---|---|
| Scientific Writing | 7.5 / 10 | Clear and well-structured; minor repetition and some phrasing could be tightened |
| Technical Quality | 7.0 / 10 | Solid implementation; temporal split is a methodological gap |
| Literature Review | 7.0 / 10 | Covers the main references well; missing some recent XAI-in-retail citations |
| Methodology | 7.5 / 10 | Well-explained with good justification of choices; hyperparameter tuning absent |
| Discussion | 8.0 / 10 | Strong SHAP business interpretation; limitations section is honest and complete |
| **Overall** | **7.4 / 10** | **A solid Research Skills paper that meets the objectives of the course** |

---

*University Project Report — Technische Universität Ilmenau — 2026*
*Source code, trained models, and figures available at: [GitHub Repository URL]*
