Dataset Construction
The Olist raw data comes as eight relational CSVs. A daily order-count time series was built by aggregating delivered orders per purchase date. Only 'delivered' orders were counted 
because cancelled, processing, and unavailable orders do not represent confirmed sales. Missing dates
(days with zero delivered orders) were filled with zero after reindexing to a continuous daily range.
The analysis window was restricted to January 2017 – August 2018. The pre-2017 period is too sparse 
(a few weeks of ramp-up data) to provide useful seasonal information, and post-August 2018 data is incomplete. This gave 602 daily records.
•	Outliers: A single-day spike in November 2017 to 1,147 orders (vs ~160 typical). This is a holiday event, not a data error.
outliers were ceiled to the upperbpound defined by q3+ 1.5 IQR

### 1. E-Commerce Stationarity & Patterns
- **Stationarity:** The ADF test on daily orders reveals a p-value < 0.05, strongly indicating the series is stationary.
- **Patterns:** While statistically stationary overall, the rolling mean plot displays slight shifting trends, and rolling standard deviation spikes heavily around November (holiday sales).
- **Seasonality & Trend Plot:** The seasonal decomposition reveals a strong weekly recurring pattern (day-of-week seasonality, typically lower on weekends) and a gradual overall positive trend leading into late 2017 before tapering off in 2018.
- **ACF/PACF Conclusions:** The ACF plot shows significant periodic correlations every 7 lags, confirming the weekly seasonality. The PACF shows a sharp drop-off after the first few lags with significant spikes at multiples of 7, which informs our choice of utilizing models that support seasonal parameters (like SARIMA or Exponential Smoothing) configured with a 7-day period.

### 2. Sensor Data Quality & Treatment
The `sensor.csv` data comes with significant NaNs across continuous multi-minute sequences.
- **Issue:** The core issue is missing values (NaNs), which breaks sequence-based forecasting models (such as ARIMA, LSTMs, etc.).
- **Fix:** We used Forward Fill (`.ffill()`) as the primary imputation method, assuming sensors carry forward the last known state until renewed, supplemented by Backward Fill (`.bfill()`) to cover any early missing sequences.

### 3. Baseline Forecasting Model Choice & Strategy
- **Model:** Holt-Winters Exponential Smoothing.
- **Justification:** Exponential smoothing natively captures level, trend, and seasonality without requiring stationary conversions or heavy parameter fitting.
- **Hold-out:** We employed an 80/20 chronological train-test split, respecting temporal ordering (no random shuffling).
- **Metric:** Mean Absolute Error (MAE) was selected because it is interpretable to inventory teams—an MAE of "X" implies forecasts stray by exactly X actual orders on average.

### 4. Advanced Forecasting Model
- **Model:** A Random Forest Regressor using simple lag configurations, day-of-week, and month variables.
- **Justification:** Standard baseline models can fail to capture volatile local spikes. A Random Forest naturally isolates step-function patterns (holidays/Black Friday) given lag features.
- **Outcome:** The evaluation against the common hold-out set determines whether the feature overhead produces a lower MAE worth deploying over Holt-Winters.

### 5. Sensor Machine Failure Prediction
- **Strategy:** We shifted the `machine_status == "BROKEN"` label backwards by 24 hours (1440 minutes) on a rolling maximum to establish a `target_24h_fail` label.
- **Model:** A Random Forest Classifier. Given the asymmetric cost between missing an equipment failure (devastating repair costs) versus triggering a false alarm (cheaper inspection), evaluation heavily relies on Recall (catching the failure events) over simple accuracy.

### 6. Single Sensor Rule vs ML Model
- **Identified Signal:** Based on absolute correlation with the failure target, the pipeline dynamically grabs the most predictive single sensor signal, and fits an optimal numerical threshold over the training distribution.
- **Cost Matrix Simulation:** We simulated a real-world asymmetry where Missed Failures cost $10,000 (repairing a failure offline) and False Alarms cost $100 (sending a tech for unneeded spot-check).
- **Conditions of Outperformance/Failure:** A single rule strategy outperforms the ML model only under conditions where the risk profile heavily penalizes false negatives, and the single sensor reliably steps universally into a deterministic range just prior to failure. However, it fails drastically in multi-collinear varying load environments, where the sensor might natively trigger the threshold simply from heavy utilization, incurring massive false alarm overhead.
- **Recommendation:** Deploy the **ML Model**. A Random Forest captures compounding intersections of multiple sensors effectively limiting false-alarms significantly while maintaining identical or superior Recall. The generalized cost at scale relies heavily on an optimized False-Positive rate, pushing the final cost of the ML Model decisively lower than a rigid Single Rule.
