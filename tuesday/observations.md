### 1. Data Quality Audit & Cleaning
- **Age Issue:** The Patient `Age` column was binned into subjective string labels (`"51-60"`, `"11-20"`). Sequence/classification models rely on absolute variance thresholds and ordinal distributions instead of categorical strings. 
- **Treatment Strategy:** We converted the binned distributions into numeric float representations by extracting the median value of each bin (e.g. `55.0` for `"51-60"`). 
- **Target Extraction:** Rather than performing Multi-class continuous prediction on raw strings, we extracted readmission risk from the `Stay` label (where `Stay` $\geq 31$ implies a 1 high-readmission). Missing categorical constraints across the healthcare dataframe were backfilled utilizing Simple Modes implicitly over Label Encoders. 

### 2. Neural Network Construction
- A custom 2-Layer Neural Network (Multilayer Perceptron) was written entirely through absolute NumPy matrix multiplication natively capturing back-propagation utilizing a Cross-Entropy log-loss sequence. Weight parameters are safely initialized utilizing He Initializations ensuring convergence. 

### 3. Neural Embeddings Feature Extractor
- We bypassed the traditional NN classification architecture directly by cutting off the model explicitly across the final Hidden Dense Layer ($A_1$). These layer values were successfully extracted as continuous $N$ dimensional data clusters.
- **Improvement Conclusion:** By explicitly feeding the extracted Neural Layer embedded coordinates (abstractions) into a downstream baseline model (Logistic Regression), the $F_1$ metric demonstrated a measurable delta logic performance. Intermediate representations explicitly warp categorical correlations inside latent layers, offering the regressor a drastically cleaner boundary to separate `high-risk` verses `low-risk` readmission candidates over the Raw Feature distribution (confirmed visually across the executed PCA components plot).
