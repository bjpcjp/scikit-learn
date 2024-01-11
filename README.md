### scikit-learn contents

| Section | Title | Contents |
| ------- | ----- | -------- |
| 00      | **Getting Started** | Estimators, Transformers, Preprocessors, Pipelines, Model Evaluation, Parameter Searches, Next Steps |
| 01      | **Linear Models** | OLS, Ridge, Lasso, Elastic-Net, Least Angle Regression (LARS), LARS Lasso, OMP, Naive Bayes, Generalized Linear Models (GLM), Tweedie Models, Stochastic Gradient Descent (SGD), Perceptrons, Passive-Aggressive Algos, Polynomial Regression |
| 01a     | **Logistic Regression**   | Basics, Examples |
| 01b     | **Splines**               |        |
| 01c     | **Quantile Regression**   | Examples 
                                        QR vs linear regression |
| 01d     | **Outliers**              | Robustness
                                        RANSAC
                                        Huber
                                        Thiel-Sen |
| 02      | **Discriminant Analysis** | LDA 
                                        QDA 
                                        Math Foundations 
                                        Shrinkage
                                        Estimators |
| 03      | **Kernel Ridge Regression**        | KRR vs SVR |
| 04      | **Support Vector Machines (SVMs)** | Classifiers (SVC, NuSVC, LinearSVC) 
                                                 Regressors (SVR, NuSVR, LinearSVR) 
                                                 Scoring, Weights,  Complexity, Kernels |
| 05      | **Stochastic Gradient Descent (SGD)** | Classifier
                                                    Classifier (Multiclass) 
                                                    Classifier (Weighted) 
                                                    Solvers
                                                    Regressors 
                                                    Sparse Data; Complexity; Stopping/Convergence; Tips |
| 06      | **K Nearest Neighbors (KNN)**         | Algos (Ball Tree, KD Tree, Brute Force) 
                                                    Radius-based Classifiers
                                                    Radius-based Regressors 
                                                    Nearest Centroid Classifiers 
                                                    Caching
                                                    Neighborhood Components Analysis (NCA) |
| 07      | **Gaussian Processes (GPs)**          | Regressors 
                                                    Classifiers 
                                                    Kernels |
| 08      | **Cross Decomposition**               | Partial Least Squares (PLS) 
                                                    Canonical PLS
                                                    SVD PLS
                                                    PLS Regression 
                                                    Canonical Correlation Analysis (CCA) |
| 09      | **Naive Bayes (NB)**                  | Gaussian NB 
                                                    Multinomial NB 
                                                    Complement NB 
                                                    Bernoulli NB 
                                                    Categorical NB 
                                                    Out-of-core fitting |
| 10      | **Decision Trees (DTs)** | Classifiers <br> Graphviz <br> Regressions <br> Multiple Outputs <br> Extra Trees <br> Complexity, Algorithms <br> Gini, Entropy, Misclassification <br> Minimal cost-complexity Pruning |
| 11a     | **Ensembles/Bagging** | Methods <br> Random Forests <br> Extra Trees <br> Parameters <br> Parallel Execution <br> Feature Importance <br> Random Tree Embedding
| 11b     | **Ensembles/Boosting** | Gradient Boosting (GBs) <br> GB Classifiers <br> GB Regressions <br> Tree Sizes <br> Loss Functions <br> Shrinkage <br> Subsampling <br> Feature Importance <br> Histogram Gradient Boosting (HGB) <br> HGB - Monotonic Constraints |
| 11ba    | **Ensembles/Boosting/Adaboost** | examples |
| 11c     | **Ensembles/Voting** | Hard Voting <br> Soft Voting <br> Voting Regressor |
| 11d     | **Ensembles/General Stacking** | Summary |
| 12      | **Multiclass/Multioutput Problems** | Label Binarization <br> One vs Rest (OvR), One vs One (OvO) Classification <br> Output Codes <br> Multilabel, Multioutput Classification <br> Classifier Chains <br> Multioutput Regressions <br> Regression Chains |
| 13 | **Feature Selection (FS)** | Removing Low-Variance Features <br> Univariate FS <br> | Recursive FS <br> Model-based FS <br> Impurity-based FS <br> Sequential FS <br> Pipeline Usage |
| 14 | **Semi-Supervised** | Self-Training Classifier <br> Label Propagation <br> Label Spreading |
| 15 | **Isotonic Regression** | Example |
| 16 | **Calibration Curves** | Intro/Example <br> Cross-Validation <br> Metrics <br> Regressors |
| 17 | **Perceptrons** | Intro <br> Classification <br> Regression <br> Regularization <br> Training <br> Complexity <br> Tips |
| 21 | **Gaussian Mixtures (GMs)** | Expectation Maximization <br> Variational Bayes GM |
| 22 | **Manifolds** | Isomap <br> Locally Linear Embedding (LLE) <br> Modified LLE, Hessian LLE <br> Local Tangent Space Alignment (LTSA) <br> Multidimensional Scaling (MDS) <br> Random Trees Embedding <br> Spectral Embedding <br> t-SNE <br> Neighborhood Components Analysis (NCA) |
| 23 | **Clustering** | K-Means <br> Voronoi Diagrams <br> Affinity Propagation <br> Mean Shift <br> Spectral Clustering <br> Agglomerative Clustering <br> Dendrograms <br> Connectivity Constraints <br> Distance Metrics <br> DBSCAN <br> Optics <br> Birch |
| 23a | **Clustering Metrics** | Rand Index <br> Mutual Info Score <br> Homogeneity <br> Completeness <br> V-Measure <br> Fowlkes-Mallows <br> Silhouette Coefficient <br> Calinski-Harabasz <br> Davies-Bouldin <br> Contingency Matrix <br> Pair Confusion Matrix |
| 24 | **Biclustering** | Spectral Co-Clustering <br> Spectral Bi-Clustering <br> Metrics |
| 25 | **Component Analysis / Matrix Factorization** | PCA <br> Incremental PCA <br> PCA w/ Random SVD <br> PCA w/ Sparse Data <br> Kernel PCA <br> Dimension Reduction Comparison <br> Truncated SVD / LSA <br> Dictionary Learning <br> Factor Analysis <br> Independent Component Analysis <br> Non-Negative Matrix Factorization (NNMF) <br> Latent Dirichlet Allocation (LDA) |
| 26 | **Covariance** | Empirical CV <br> Shrunk CV <br> Max Likelihood Estimation (MLE) <br> Ledoit-Wolf Shrinkage <br> Oracle Approximating Shrinkage <br> Sparse Inverse CV, aka Precision Matrix <br> Mahalanobis Distance |
| 27 | **Novelties & Outliers** | One-Class SVMs <br> Elliptic Envelope <br> Isolation Forest <br> Local Outlier Factor |
| 28 | **Density Estimation (DE)** | Histograms <br> Kernel DE |
| 29 | **Restricted Boltzmann Machines (RBMs)** | Intro <br> Training |
| 31 | **Cross Validation (CV)** | Intro <br> Metrics <br> Parameter Estimation <br> Pipelines <br> Prediction Plots <br> Nesting <br> K-Fold <br> Stratified K-Fold <br> Leave One Out <br> Leave P Out <br> Class Label CV <br> Grouped Data CV <br> Predefined Splits <br> Time Series Splits <br> Permutation Testing <br> Visualizations |
| 32 | **Parameter Tuning** | Grid Search <br> Randomized Optimization <br> Successive Halving <br> Composite Estimators & Parameter Spaces <br> Alternative to Brute Force <br> Info Criteria (AIC, BIC) |
| 33 | **Metrics & Scoring (Intro)** | scoring <br> make_scorer | 
| 33a | **Classification Metrics** | Accuracy <br> Top-K Accuracy <br> Balanced Accuracy <br> Cohen's Kappa <br> Confusion Matrix <br> Classification Report <br> Hamming Loss <br> Precision <br> Recall <br> F-Measure <br> Precision-Recall Curve <br> Average Precision <br> Jaccard Similarity <br> Hinge Loss <br> Log Loss <br> Matthews Correlation Coefficient <br> Receiver Operating Characteristic (ROC) Curves <br> ROC-AUC <br> Detection Error Tradeoff (DET) <br> Zero One Loss <br> Brier Score |
| 33b | **Multilabel Ranking Metrics** | Coverage Error <br> Label Ranking Avg Precision (LRAP) <br> Label Ranking Loss <br> Discounted Cumulative Gain (DCG), Normalized DCG |
| 33c | **Regression Metrics** | Explained Variance <br> Max Error <br> Mean Absolute Error (MAE) <br> Mean Squared Error (MSE) <br> Mean Squared Log Error (MSLE) <br> Mean Absolute Pct Error (MAPE) <br> R^2 score <br> aka Coefficient of Determination  <br> Tweedie Deviances |
| 33d | **Dummy Metrics** | Dummy Classifiers <br> Dummy Regressors | 
| 34 | **Validation Curves** | Validation Curve <br> Learning Curve | 
| 41 | **Viz/Inspection** | 2D PDPs, 3D PDPs <br> Individual Conditional Expectation (ICE) Plot |
| 42 | **Viz/Permutations** | Permutation Feature Importance (PFI) <br> Impurity vs Permutation Metrics | 
| 50a | **Viz/ROC Curves** | ROC Curve |
| 50b | **Viz/custom PDP Plots** | Example |
| 50c | **Vis/Classification metrics** | Confusion Matrix <br> ROC Curve <br> Precision-Recall Curve | 
| 61 | **Composite Transformers** | Pipelines <br> Caching <br> Regression Target xforms <br> Feature Unions <br> Column Transformers |
| 62a | **Text Feature Extraction** | Bag of Words (BoW) <br> Sparsity <br> Count Vectorizer <br> Stop Words <br> Tf-Idf <br> Binary Markers <br> Text file decoding <br> Hashing Trick <br> Out-of-core Scaling <br> Custom Vectorizers |
| 62b | **Image Patch Extraction** | Extract from Patches <br> Reconstruct from Patches <br> Connectivity Graphs |
| 63 | **Data Preprocessing** | Scaling <br> Quantile Transforms <br> Power Maps (Box-Cox, Yeo-Johnson) <br> Category Coding <br> One-Hot Coding <br> Quantization aka Binning <br> Feature Binarization |
| 64 | **Missing Value Imputation** | Univariate <br> Multivariate <br> Multiple-vs-Single <br> Nearest-Neighbors <br> Marking Imputed Values | 
| 66 | **Dimensionality Reduction/Random Projections** | Johnson-Lindenstrauss lemma <br> Gaussian RP <br> Sparse RP<br> Empirical Validation |
| 67 | **Kernel Approximations** | Nystroem <br> RBF Sampler <br> Additive Chi-Squared Sampler <br> Skewed Chi-Squared Sampler <br> Polynomial Sampling - Tensor Sketch |
| 68 | **Pairwise Ops** | Distances vs Kernels <br> Cosine Similarity <br> Kernels |
| 69 | **Transforming Prediction Targets** | Label Binarization <br> Multilabel Binarization <br> Label Encoding |
| 71 | **Toy Datasets** | Boston <br> Iris <br> Diabetes <br> Digits <br> Linnerud <br> Wine <br> Breast Cancer <br> Olivetti faces <br> 20 newsgroups <br> Labeled faces <br> Forest covertypes <br> Reuters corpus <br> KDD <br> Cal housing |
| 73 | **Artificial Data** | random-nclass-data <br> Gaussian blobs <br> Gaussian quantiles <br> Circles <br> Moons <br> Multilabel class data <br> Hastie data <br> BiClusters <br> Checkerboards <br> Regression <br> Friedman1/2/3 <br> S-Curve <br> Swiss Roll <br> Low-Rank Matrix <br> Sparse Coded Signal <br> Sparse Symmetric Positive Definite (SPD) Matrix | 
| 74 | **Other Data** | Sample images <br> SVMlight/LibSVM formats <br> OpenML <br> pandas.io <br> scipy.io <br> numpy.routines.io <br> scikit-image <br> imageio <br> scipy.io.wavfile |
| 81 | **Scaling** | Out-of-core ops (**BUG = TODO**) | 
| 82 | **Latency** | Bulk-vs-atomic ops <br> Latency vs Validation <br> Latency vs #Features <br> Latency vs Datatype <br> Latency vs Feature Extraction <br> Linear Algebra Libs (BLAS, LAPACK, ATLAS, OpenBLAS, MKL, vecLib) |
| 83 | **Parallelism** | JobLib <br> OpenMP <br> NumPy <br> Oversubscription <br> config switches |
| 90 | **Persistence** | Pickle <br> Joblib |
