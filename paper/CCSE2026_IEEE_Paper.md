# Explainable Machine Learning for Retail Sales Forecasting: A Comparative Study on the Rossmann Store Sales Dataset

**Aman Ismail Mullick**

Faculty of Computer Science and Automation, Technische Universität Ilmenau, Ilmenau, Germany

aman-ismail.mullick@tu-ilmenau.de

---

## Abstract

Retail sales forecasting is a central operational challenge in the retail sector, yet models that achieve strong predictive performance often offer no explanation for what drives a given forecast. This paper evaluates three models — Linear Regression, Random Forest, and XGBoost — applied to the Rossmann Store Sales dataset, which contains 1,017,209 daily sales records from 1,115 German drugstores spanning 2013 to 2015. Following preprocessing and feature engineering, 24 input features were derived and models were evaluated on a held-out set of 168,868 samples. XGBoost produced the best results across all four metrics: MAE = €854.15, RMSE = €1,198.13, R² = 0.8512, and MAPE = 13.78%, reducing mean absolute error by 57.6% relative to Linear Regression and by 17.4% relative to Random Forest. TreeSHAP was then applied to both ensemble models to explain predictions at the global and individual levels. Competition proximity (mean |SHAP| = 0.193), promotional activity (0.182), and store identity (0.167) emerged as the three strongest predictors of daily sales. A store-level waterfall plot confirms that these attributions can be read and interpreted directly by business managers without statistical background. Applying the trained model to 2026 store profiles reveals a clear seasonal pattern, with December forecast as the peak month at 45.5% above the January baseline. The study demonstrates that combining ensemble forecasting with post-hoc explainability produces outputs that are both accurate and auditable.

**Index Terms** — retail sales forecasting, explainable AI, SHAP, XGBoost, Random Forest, Rossmann dataset.

---

## I. Introduction

Managing inventory, staffing, and promotional spend across a large retail network requires reliable daily sales forecasts at the individual store level [1]. Aggregate or region-level projections are insufficient: a store near a new competitor behaves differently from one in an established location, and the effect of a promotion varies substantially by store type and assortment. Classical time-series approaches such as ARIMA [2] are well understood and interpretable, but they are designed for single univariate sequences and do not readily incorporate the mix of categorical and continuous features that define retail store behaviour. Tree-based ensemble methods have become the standard for structured tabular data in forecasting applications [3], and the original Rossmann Store Sales Kaggle competition (2015) confirmed their practical dominance in this domain.

Despite their accuracy, ensemble models present a deployment problem: store managers and operations teams cannot verify or act on predictions they cannot understand. A forecast that says a store will sell €8,800 on a given day is useful; one that explains why — because the promotional calendar adds €1,200, partially offset by a €700 penalty from a nearby competitor — is operationally actionable. SHAP (SHapley Additive exPlanations) [4], particularly the TreeSHAP variant [11], provides a principled and computationally efficient way to generate such explanations for tree-based models. Its grounding in cooperative game theory ensures that attributions are consistent and faithful to the model, unlike approximation-based alternatives.

This paper reports on a study that trains and compares Linear Regression, Random Forest, and XGBoost on the Rossmann dataset, then applies TreeSHAP to the two ensemble models to produce both global feature rankings and local, store-specific prediction explanations. The goals were to identify which factors most influence daily sales, quantify the performance gap between model classes, and assess whether SHAP attributions are practically interpretable in a retail management context. Section II situates this work relative to related forecasting and explainability research. Section III describes the data, preprocessing decisions, model configurations, and evaluation approach. Section IV presents results and discusses their practical interpretation. Section V concludes.

---

## II. Related Work

### A. Sales Forecasting Methods

ARIMA and Holt-Winters smoothing [2] remain standard benchmarks for univariate time-series forecasting, but their performance degrades when the target depends on multiple interacting features — promotions, competitor activity, store format — that do not fit cleanly into a single time series structure. Random Forest [14] addressed this by learning non-linear feature interactions through bootstrap-aggregated decision trees, which also produce built-in importance rankings via Gini impurity. XGBoost [5] extended this further with gradient boosting: rather than building trees independently, it trains each tree on the residual errors of the ensemble so far, achieving lower bias on high-dimensional tabular data. LightGBM [6] and CatBoost [7] offer further refinements — histogram-based splitting and native categorical handling, respectively — but the core gradient boosting approach is shared.

Deep learning methods, including LSTMs [8] and the Temporal Fusion Transformer [9], are increasingly applied to multi-horizon retail forecasting and handle long temporal dependencies well. However, Grinsztajn et al. [3] showed that on fixed-length tabular datasets, tree-based models consistently outperform neural approaches, and Shwartz-Ziv and Armon [10] reached a similar conclusion across multiple benchmarks. Given that the Rossmann dataset is a row-per-day tabular structure rather than a continuous sequence, tree-based models are the appropriate focus here.

### B. Explainability: SHAP and LIME

The two most widely used post-hoc local explanation methods are LIME [12] and SHAP [4]. LIME approximates model behaviour in the neighbourhood of a single input by fitting a linear surrogate model to perturbed samples, but this approach is sensitive to the perturbation distribution: the same input can produce meaningfully different explanations across runs, which undermines trust in high-stakes decisions. SHAP takes a different foundation, distributing each prediction's departure from the dataset mean across input features according to Shapley values from cooperative game theory. This guarantees that contributions are consistent across models and that they sum exactly to the difference between the prediction and the baseline — a property LIME does not satisfy. For tree ensembles specifically, TreeSHAP [11] computes exact Shapley values in O(TLD²) time rather than by approximation, making it practical on models with hundreds of trees and tens of features.

### C. Interpretability in Retail Applications

Ribeiro et al. [12] showed that local explanations measurably increase user trust in machine learning recommendations even when users cannot inspect the model directly. In a demand forecasting context, Dzyabura and Hauser [13] found that interpretable models help managers make better promotional decisions by making the relative impact of different levers explicit. The present study builds on this line of work by applying SHAP explanations not just as a research diagnostic but as a practical output — the waterfall plot for a specific store and date is designed to be readable by a non-technical manager and to support concrete decisions about promotions, stocking, and staffing.

---

## III. Methodology and Experimental Setup

### A. Dataset

The Rossmann Store Sales dataset contains 1,017,209 daily sales records from 1,115 German drugstores between January 2013 and July 2015. The sales file was joined with a store metadata file on the store identifier, adding structural attributes such as store type (A, B, C, D), product assortment level, and information about nearby competitors and loyalty promotion schemes. Records where the store was closed (Open = 0) or sales were zero were excluded, as including them would bias every model toward zero for inputs that are structurally different from normal operating days. After filtering, 844,338 usable records remained.

The dataset was split 80/20 using a fixed random seed (42), yielding 675,470 training samples and 168,868 validation samples. Daily sales, the prediction target, ranged from €0 to €41,551 with a mean of €6,955.96, a standard deviation of €3,103.82, and a right skew of 1.59, reflecting the presence of a minority of very high-sales days.

### B. Preprocessing and Feature Engineering

Three types of missing data required different treatment. CompetitionDistance was missing for 0.3% of records and was filled with the column median (2,325 m), which is appropriate for a nearly complete continuous variable with no strong pattern to the missingness. The CompetitionOpenSince fields (31.8% missing) and Promo2-related fields (49.9% missing) were filled with zero, meaning "no recorded event," which is the correct domain interpretation: a missing entry indicates that no competitor opening or loyalty scheme enrolment was recorded for that store.

From the merged dataset, 24 features were constructed. The date was decomposed into year, month, day, week of year, and quarter to expose seasonal periodicity. Two duration features were derived: CompetitionDuration (months elapsed since the nearest competitor opened) and Promo2Duration (weeks of continuous Promo2 participation). These capture the cumulative effect of competitor presence and promotional momentum, which the raw calendar dates alone do not express. Three binary flags — IsWeekend, IsMonthStart, and IsMonthEnd — were added based on known retail demand patterns around weekends and pay cycles. Categorical variables (StoreType, Assortment, StateHoliday, PromoInterval) were label-encoded; this is sufficient for tree-based models, which can discover the appropriate non-monotonic mapping between encoded values and the target without requiring one-hot representation.

### C. Models

Three models were selected to span a range of complexity and interpretability. Linear Regression serves as the reference baseline: it is fast to train, fully interpretable by coefficient inspection, but assumes that all feature effects are additive and linear, which is unlikely to hold in a retail environment driven by interaction effects between promotions, competition, and store format. Random Forest trains 100 decision trees independently on bootstrap samples of the training data (max depth 15, minimum samples per leaf 5), then averages their outputs; this reduces variance substantially relative to a single tree while preserving the ability to model non-linear relationships. XGBoost trains 500 trees sequentially, where each tree fits the residuals of the current ensemble (learning rate 0.05, max depth 6, L1 regularisation 0.1, L2 regularisation 1.0, subsample fraction 0.80 per tree). Early stopping on validation RMSE was applied to prevent overfitting.

### D. Evaluation and SHAP Configuration

All models were evaluated on the held-out 168,868-sample validation set using four metrics: MAE (mean absolute error in EUR, the primary measure of forecast usefulness), RMSE (root mean squared error, which penalises large errors more heavily), R² (proportion of variance explained), and MAPE (mean absolute percentage error, which is scale-independent and useful for comparing stores of different sizes).

TreeSHAP was applied to both ensemble models using 600 randomly sampled validation records — sufficient to produce stable global importance rankings while keeping computation tractable. Four explanation artefacts were produced: a beeswarm summary plot showing global importance with directional colour-coding, a ranked bar chart of mean absolute SHAP values, a waterfall plot for a specific store-date pair, and a dependence plot for the CompetitionDistance feature.

---

## IV. Results and Discussion

### A. Model Performance

Table I summarises validation performance across all three models.

**TABLE I — Validation Set Performance (n = 168,868)**

| Model | MAE (€) | RMSE (€) | R² | MAPE (%) |
|---|---|---|---|---|
| Linear Regression | 2,012.28 | 2,772.64 | 0.2033 | 33.66 |
| Random Forest | 1,034.71 | 1,483.27 | 0.7720 | 16.54 |
| **XGBoost** | **854.15** | **1,198.13** | **0.8512** | **13.78** |

Linear Regression's R² of 0.20 establishes the ceiling achievable by purely additive, linear assumptions. The model captures broad trends — weekday and seasonal effects — but fails wherever predictors interact. Stores close to a competitor during an active promotion behave differently from stores without those conditions simultaneously, and a linear model has no mechanism to represent this. The MAPE of 33.66% is substantial: on an average day, the prediction is off by roughly one-third of actual sales.

Random Forest raises R² to 0.77 by learning non-linear thresholds and interaction effects through its ensemble of deep trees. The improvement over Linear Regression is large and reflects how much of the variance in daily sales is driven by exactly these kinds of interacting factors. Residuals from the Random Forest are approximately zero-centred with no obvious systematic pattern, suggesting the model is reasonably well calibrated across the full range of store types and competitive conditions.

XGBoost achieves R² = 0.8512 and MAPE = 13.78%, the best result on every metric. The MAE of €854.15 represents a 17.4% improvement over Random Forest and a 57.6% improvement over Linear Regression. The sequential boosting mechanism is the likely explanation: each successive tree is fitted specifically to the hardest-to-predict records in the current ensemble, progressively reducing the large errors that remain after the first few hundred trees. L1 and L2 regularisation control the complexity of individual trees, preventing the model from overfitting to noise in the training data while still fitting complex patterns. A mild right skew in the residuals above €20,000 reflects a small number of extreme sales events — likely driven by unrecorded local factors such as community events or supply anomalies — that no model trained on the available features can fully anticipate.

### B. SHAP Feature Importance

Table II reports mean absolute SHAP values for the top eight features from the XGBoost model, computed over 600 validation records. Fig. 1 shows the beeswarm summary plot.

**TABLE II — Feature Importance by Mean |SHAP| (XGBoost, n = 600)**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | CompetitionDistance | 0.1933 |
| 2 | Promo | 0.1816 |
| 3 | Store | 0.1668 |
| 4 | DayOfWeek | 0.0662 |
| 5 | CompetitionOpenSinceYear | 0.0626 |
| 6 | CompetitionOpenSinceMonth | 0.0574 |
| 7 | Assortment | 0.0398 |
| 8 | StoreType | 0.0354 |

CompetitionDistance is the most influential feature. The dependence plot shows a non-linear negative relationship: stores within approximately 500 m of a competitor carry a SHAP penalty of around −€800 per day on average, while stores with no recorded nearby competitor receive a positive contribution of roughly +€600. The relationship is steep at close distances and flattens beyond 3 km, which is consistent with how localised foot-traffic competition operates in urban retail.

Active promotions (Promo) rank second. On promotional days, the SHAP contribution averages around +€387; on non-promotional days, it averages −€280. The net difference — approximately €667 per store per day — is a rough estimate of the incremental sales lift from a single promotional activation, under the model's average-effect assumption. Scaled across all 1,115 stores, an additional chain-wide promotional day corresponds to an estimated €744,000 in incremental revenue, though this will vary considerably by store type and competitive context and should be validated against actual promotional records before informing budget decisions.

Store identity (0.167) accounts for persistent, store-specific effects that are not captured by the structural features available in the dataset — local demographics, historical customer loyalty, foot traffic patterns specific to the store's location. The size of this contribution suggests that aggregate chain-level forecasts will systematically miss store-level variation, and that store-specific calibration adds meaningful predictive value.

DayOfWeek (0.066) reflects the weekly demand cycle. Saturdays consistently receive the highest SHAP values, which is consistent with German retail patterns where Saturday is a peak shopping day. This finding directly supports weekly staffing and replenishment decisions.

### C. Store-Level Prediction Explanation

Fig. 2 shows the SHAP waterfall plot for Store 262 on a Wednesday in June 2026, with an active promotion in effect. The baseline — the mean prediction across all training records — is €6,955. The active promotion contributes +€1,247 above this baseline. The store's proximity to a competitor (approximately 310 m) reduces the forecast by €698. A seasonal spring effect adds €412, and a mid-week weekday pattern adds €180. The sum of all feature contributions lands at a final forecast of €8,786.

This decomposition is directly readable. A store manager can see that the promotional effect is real and large, that the competitive penalty is specific and quantifiable, and that the net result is a particular EUR figure with identifiable contributing factors. If the promotion were cancelled, the prediction would drop by approximately €1,247. If a new competitor were to open within 200 m, the competitive penalty would increase. These are not abstract model properties; they are directly actionable pieces of information.

### D. Seasonal Forecast for 2026

Applying the trained XGBoost model to representative store profiles for each month of 2026 reveals the seasonal structure that the model learned from the 2013–2015 training data. January is the annual low point at an estimated €7,480 per store per day. Sales rise moderately through spring, reaching a secondary peak in April and May (roughly 6% above the January baseline), before dipping slightly through summer. December reaches the annual peak at €10,883 per store per day — a 45.5% uplift over January — consistent with the Christmas shopping surge visible across all three training years.

These seasonal indices are reliable in the sense that the underlying pattern is stable and consistent in the training data. The absolute EUR figures are indicative rather than operational: they do not account for inflation, structural changes in the store network since 2015, or macroeconomic conditions specific to 2026. They are best used as relative benchmarks — for example, to anticipate proportional uplift in procurement or staffing requirements for December relative to January.

### E. Limitations

Several limitations should be noted. The 80/20 validation split was performed randomly rather than on a time boundary, which means some later records in the training set could inform predictions for earlier-dated records in the validation set. While the effect is likely small given the dataset size, time-series cross-validation would produce a more conservative and realistic performance estimate. The dataset covers German drugstore retail over a specific two-year window; the findings may not transfer directly to other retail sectors, geographies, or more recent market conditions. External factors known to influence daily sales — local weather, economic indicators, regional events — are absent from the dataset and contribute to residual prediction error, particularly on extreme-sales days. Finally, the hyperparameters for all three models were set manually; systematic optimisation using Bayesian search or cross-validated grid search could yield further performance gains.

---

## V. Conclusion

This study compared Linear Regression, Random Forest, and XGBoost for store-level retail sales forecasting on the Rossmann dataset, and applied TreeSHAP to explain ensemble model predictions at both the global and individual levels. XGBoost achieved the strongest performance across all four evaluation metrics, with R² = 0.8512 and MAPE = 13.78%, substantially outperforming both the linear baseline and the Random Forest. SHAP analysis identified competition proximity, promotional activity, and store-specific characteristics as the three dominant drivers of daily sales variance, and the waterfall explanation for Store 262 showed that these attributions are legible to non-technical users in a management context. The 2026 seasonal forecast confirmed that the model captures recurring annual demand patterns, with December projected as the peak month.

The main limitation to address in future work is the random validation split; replacing it with temporal cross-validation would give a fairer picture of out-of-sample performance. Other natural extensions include evaluating LightGBM [6] and CatBoost [7] against XGBoost, testing LSTM [8] and Temporal Fusion Transformer [9] baselines on the sequential structure of the data, incorporating external data sources such as weather or economic indicators, and applying Bayesian hyperparameter optimisation. Validating the approach on a second public benchmark — such as the M5 competition dataset or the Walmart Sales dataset — would assess how well the pipeline generalises beyond the Rossmann context. The combination of strong predictive accuracy and store-level SHAP explanations demonstrated here offers a practical direction for deploying machine learning in retail forecasting settings where transparency is required alongside performance.

---

## References

[1] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.

[2] G. E. P. Box and G. M. Jenkins, *Time Series Analysis: Forecasting and Control*, 3rd ed. Prentice Hall, 1994.

[3] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why tree-based models still outperform deep learning on tabular data," in *Adv. Neural Inf. Process. Syst.*, vol. 35, 2022.

[4] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst.*, vol. 30, 2017, pp. 4765–4774.

[5] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794.

[6] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Adv. Neural Inf. Process. Syst.*, vol. 30, 2017.

[7] L. Prokhorenkova et al., "CatBoost: Unbiased boosting with categorical features," in *Adv. Neural Inf. Process. Syst.*, vol. 31, 2018.

[8] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[9] B. Lim et al., "Temporal fusion transformers for interpretable multi-horizon time series forecasting," *Int. J. Forecasting*, vol. 37, no. 4, pp. 1748–1764, 2021.

[10] R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Information Fusion*, vol. 81, pp. 84–90, 2022.

[11] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, pp. 56–67, 2020.

[12] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 1135–1144.

[13] D. Dzyabura and J. R. Hauser, "Recommending products when consumers learn their preference weights," *Marketing Science*, vol. 38, no. 3, pp. 417–441, 2019.

[14] L. Breiman, "Random forests," *Machine Learning*, vol. 45, pp. 5–32, 2001.
