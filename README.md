### scikit-learn v0.24.1

| Section | Title | Contents |
| ------- | ----- | -------- |
| 00      | **Getting Started** | Estimators, Transformers, Preprocessors, Pipelines, Model Evaluation, Parameter Searches, Next Steps |
| 01      | **Linear Models** | OLS, LS <br> Ridge <br> Lasso <br> Elastic-Net <br> Least Angle Regression (LARS) <br> LARS Lasso <br> OMP <br. Bayes <br> Generalized Linear Models (GLM) <br> Tweedie Models <br> Stochastic Gradient Descent (SGD) <br> Perceptrons <br> Passive-Aggressive Algos <br> RANSAC, Huber, Thiel-Sen <br> Polynomial Regression |
| 01a     | **Logistic Regression** | Basics, Examples |
| 02      | **Discriminant Analysis** | LDA <br> QDA <br> Math Foundations, Shrinkage, Estimators |
| 03 | **Kernel Ridge Regression** | KRR vs SVR |
| 04 | **Support Vector Machines (SVMs)** | Classifiers (SVC, NuSVC, LinearSVC), <br> Regressors (SVR, NuSVR, LinearSVR),<br> Scoring, Weights,  Complexity, Kernels |
| 05 | **Stochastic Gradient Descent (SGD)** | Classifier <br> Classifier (Multiclass) <br> Classifier (Weighted) <br> Solvers <br> Regressors <br> Sparse Data; Complexity; Stopping/Convergence; Tips |
| 06 | **K Nearest Neighbors (KNN)** | Algos (Ball Tree, KD Tree, Brute Force) <br> Radius-based Classifiers <br> Radius-based Regressors <br> Nearest Centroid Classifiers <br> Caching <br> Neighborhood Components Analysis (NCA) |
| 07 | **Gaussian Processes (GPs)** | GP Regressors |
| 08 | **Cross Decomposition** | Partial Least Squares (PLS) <br> Canonical PLS <br> SVD PLS <br> PLS Regression <br> Canonical Correlation Analysis (CCA) |
| 09 | **Naive Bayes (NB)** | Gaussian NB <br> Multinomial NB <br> Complement NB <br> Bernoulli NB <br> Categorical NB <br> Out-of-core fitting |
| 10 | **Decision Trees (DTs)** | Classifiers <br> Graphviz <br> Regressions <br> Multiple Outputs <br> Extra Trees <br> Complexity, Algorithms <br> Gini, Entropy, Misclassification <br> Minimal cost-complexity Pruning |
| 11a | **Ensembles/Bagging** | Methods <br> Random Forests, Extra Trees <br> Parameters, Parallel Execution, Feature Importance <br> Random Tree Embedding
| 11b | **Ensembles/Boosting** | AdaBoost <br> Gradient Boosting (GBs) <br> GB Classifiers <br> GB Regressions <br> Tree Sizes, Math (TODO), Loss Functions, Shrinkage, Subsampling, Feature Importance <br> Histogram Gradient Boosting (HGB) <br> HGB - Monotonic Constraints <br> Stacked Generalization |
| 11c | **Ensembles/Voting** | Hard Voting, Soft Voting, Voting Regressor |
| 11d | **Ensembles/General Stacking** | Summary |
| 12 | **Multiclass/Multioutput Problems** | Label Binarization <br> One vs Rest (OvR), One vs One (OvO) Classification <br> Output Codes <br> Multilabel, Multioutput Classification <br> Classifier Chains <br> Multioutput Regressions <br> Regression Chains |
| 13 | **Feature Selection (FS)** | Removing Low-Variance Features <br> Univariate FS <br> | Recursive FS | Model-based FS | Impurity-based FS | Sequential FS | Pipeline Usage |
| 14 | **Semi-Supervised/Unsupervised Learning** | Self-Training Classifier <br> Label Propagation, Label Spreading |
| 15 | **Isotonic Regression** | Example |
| 16 | **Calibration Curves** | Intro/Example, Cross-Validation, Metrics <br> Regressors |
| 17 | **Perceptrons** | Intro, Classification, Regression, Regularization, Training, Complexity, Tips |
| 21 | **Gaussian Mixtures (GMs)** | Expectation Maximization <br> Variational Bayes GM |
| 22 | **Manifolds** | Isomap, Locally Linear Embedding (LLE), Modified LLE, Hessian LLE, Local Tangent Space Alignment (LTSA), Multidimensional Scaling (MDS)<br> Random Trees Embedding, Spectral Embedding, t-SNE, Neighborhood Components Analysis (NCA) |
| 23 | **Clustering** | K-Means, Voronoi Diagrams <br> Affinity Propagation <br> Mean Shift <br> Spectral Clustering <br> Agglomerative Clustering, Dendrograms, Connectivity Constraints, Distance Metrics <br> DBSCAN, Optics, Birch |
| 23a | **Clustering Metrics** | Rand Index, Mutual Info Score, Homogeneity, Completeness, V-Measure, Fowlkes-Mallows, Silhouette Coefficient, Calinski-Harabasz, Davies-Bouldin <br> Contingency Matrix <br> Pair Confusion Matrix |
| 24 | **Biclustering** | Spectral Co-Clustering, Spectral Bi-Clustering <br> Metrics |
| 25 | **Component Analysis / Matrix Factorization** | PCA, Incremental PCA, PCA w/ Random SVD, PCA w/ Sparse Data, Kernel PCA <br> Dimension Reduction Comparison <br> Truncated SVD / LSA <br> Dictionary Learning <br> Factor Analysis <br> Independent Component Analysis <br> Non-Negative Matrix Factorization (NNMF) <br> Latent Dirichlet Allocation (LDA) |
| 26 | **Covariance** | Empirical CV, Shrunk CV, Max Likelihood Estimation (MLE) <br> Ledoit-Wolf Shrinkage, Oracle Approximating Shrinkage <br> Sparse Inverse CV, aka Precision Matrix <br> Mahalanobis Distance |
| 27 | **Novelties & Outliers** | One-Class SVMs, Elliptic Envelope, Isolation Forest, Local Outlier Factor |
| 28 | **Density Estimation (DE)** | Histograms, Kernel DE |
| 29 | **Restricted Boltzmann Machines (RBMs)** | Intro, Training |
| 31 | **Cross Validation (CV)** | Intro, Metrics <br> Parameter Estimation, Pipelines, Prediction Plots, Nesting <br> K-Fold, Stratified K-Fold <br> Leave One Out, Leave P Out <br> Class Label CV <br> Grouped Data CV <br> Predefined Splits <br> Time Series Splits <br> Permutation Testing <br> Visualizations |
| 32 | **Parameter Tuning** | Grid Search, Randomized Optimization <br> Successive Halving <br> Composite Estimators & Parameter Spaces <br> Alternative to Brute Force <br> Info Criteria (AIC, BIC) |
| 33 | **Metrics & Scoring (Intro)** | scoring, make_scorer | 
| 33a | **Classification Metrics** | Accuracy, Top-K Accuracy, Balanced Accuracy <br> Cohen's Kappa <br> Confusion Matrix <br> Classification Report <br> Hamming Loss <br> Precision, Recall, F-Measure, Precision-Recall Curve, Average Precision <br> Jaccard Similarity, Hinge Loss, Log Loss, Matthews Correlation Coefficient <br> Receiver Operating Characteristic (ROC) Curves, ROC-AUC <br> Detection Error Tradeoff (DET), Zero One Loss, Brier Score |
| 33b | **Multilabel Ranking Metrics** | Coverage Error, Label Ranking Avg Precision (LRAP), Label Ranking Loss <br> Discounted Cumulative Gain (DCG), Normalized DCG |
| 33c | **Regression Metrics** | Explained Variance, Max Error, Mean Absolute Error (MAE), Mean Squared Error (MSE), Mean Squared Log Error (MSLE), Mean Absolute Pct Error (MAPE) <br> R^2 score, aka Coefficient of Determination  <br> Tweedie Deviances |
| 33d | **Dummy Metrics** | Dummy Classifiers, Dummy Regressors | 
| 34 | **Validation Curves** | Example, Validation Curve, Learning Curve | 
| 41 | **Viz/Inspection** | 2D PDPs, 3D PDPs <br> Individual Conditional Expectation (ICE) Plot |
| 42 | **Viz/Permutations** | Permutation Feature Importance (PFI) <br> Impurity vs Permutation Metrics | 
| 50a | **Viz/ROC Curves** | ROC Curve |
| 50b | **Viz/custom PDP Plots** | Example |
| 50c | **Vis/Classification metrics** | Confusion Matrix, ROC Curve, Precision-Recall Curve | 
| 61 | **Composite Transformers** | Pipelines, Caching, Examples <br> Regression Target xforms <br> Feature Unions <br> Column Transformers |
| 62a | **Text Feature Extraction** | Bag of Words (BoW) <br> Sparsity, Count Vectorizer, Stop Words, Tf-Idf <br> Binary Markers, Text file decoding, Hashing Trick <br> Out-of-core Scaling, Custom Vectorizers |
| 62b | **Image Patch Extraction** | Extract from Patches, Reconstruct from Patches <br> Connectivity Graphs |
| 63 | **Data Preprocessing** | Scaling, Quantile Transforms, Power Maps (Box-Cox, Yeo-Johnson) <br> Category Coding, One-Hot Coding <br> Quantization aka Binning <br> Feature Binarization |
| 64 | **Missing Value Imputation** | Univariate, Multivariate, Multiple-vs-Single, Nearest-Neighbors <br> Marking Imputed Values | 
| 66 | **Dimensionality Reduction/Random Projections** | Johnson-Lindenstrauss lemma <br> Gaussian RP, Sparse RP, Empirical Validation |
| 67 | **Kernel Approximations** | Nystroem <br> RBF Sampler, Additive Chi-Squared Sampler, Skewed Chi-Squared Sampler  <br> Polynomial Sampling - Tensor Sketch |
| 68 | **Pairwise Ops** | Distances vs Kernels <br> Cosine Similarity, Linear / Polynomial / Sigmoid / RBF / Laplacian / Chi-Squared kernels |
| 69 | **Transforming Prediction Targets** | Label Binarization, Multilabel Binarization, Label Encoding |
| 71 | **Example Datasets** | Boston, Iris, Diabetes, Digits, Linnerud, Wine, Breast Cancer <br> Olivetti faces, 20 newsgroups, Labeled faces, Forest covertypes, Reuters corpus, KDD, Cal housing |
| 73 | **Artificial Data** | random-nclass-data, Gaussian blobs, Gaussian quantiles, Circles, Moons, Multilabel class data, Hastie data <br> BiClusters, Checkerboards <br> Regression, Friedman1/2/3 <br> S-Curve, Swiss Roll <br> Low-Rank Matrix, Sparse Coded Signal, Sparse Symmetric Positive Definite (SPD) Matrix | 
| 74 | **Other Data** | Sample images, SVMlight/LibSVM formats, OpenML <br> pandas.io, scipy.io, numpy.routines.io, scikit-image, imageio, scipy.io.wavfile |
| 81 | **Scaling** | Out-of-core ops (**BUG = TODO**) | 
| 82 | **Latency** | Bulk-vs-atomic ops, Latency vs Validation, Latency vs #Features, Latency vs Datatype, Latency vs Feature Extraction <br> Linear Algebra Libs (BLAS, LAPACK, ATLAS, OpenBLAS, MKL, vecLib) |
| 83 | **Parallelism** | JobLib, OpenMP, NumPy <br> Oversubscription <br> config switches |
| 90 | **Persistence** | Pickle, Joblib |
