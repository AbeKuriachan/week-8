### 1. Social Media Data Audit
- The `social_media` constraint encountered class imbalances where 'Neutral/Positive' posts vastly overshadow Explicit 'Negative/Hate' configurations. 
- Over 70% of standard baseline models push metrics heavily toward default majority classes, dropping our Recall rates. Thus, an explicit threshold matrix measuring exact 'Missed Harmful Posts' forces us into the Stage-2 Similarity framework avoiding simple binary inaccuracies. 

### 2 & 3. MNIST Distribution and CNN Filter Properties
- Raw MNIST digit constraints naturally distribute across $28\times 28$ scaled dimension matrices inside normalized float parameters. The ConvNet trained effectively across these images. 
- **CNN Interpretation:** The First layer ConvNet Kernels cleanly adapted into discrete edge-detectors mapping explicit textural variations (such as distinct horizontal thresholds vs diagonal highlights). The neural network algorithm constructs spatial embedding patterns which bypasses flattened dimensional vectors seamlessly.

### 4 & 5. Semantic System and Moderation Deployments
- A baseline standard Linear configuration easily categorized the obvious textual correlations relying strictly on TF-IDF word presences, but heavily failed on disjoint syntaxes and paraphrased contexts.
- **2-Stage Evaluation:** The Semantic pipeline drastically improved Recall. By checking previously 'Cleared/Safe' datasets utilizing dense text similarity (Cosine Cosine scores checking back against known Harmful representations inside Latent space), we discovered multiple implicitly dangerous contexts hidden from stage 1 models. 
- **Cost Scaling Analysis:** We deployed estimations simulating manual intervention cost logic—discovering that implementing Semantic checks scales precisely without generating overwhelmingly expensive unverified manual flags relative to 100,000 threshold processing pipelines.
