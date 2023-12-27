# resources|paperswithcode
- [paperswithcode](https://paperswithcode.com)
- [datasets](https://paperswithcode.com/datasets)
- [methods](https://paperswithcode.com/methods)
- [SOTA by use case](https://paperswithcode.com/sota)
# resources|kdnuggets
- [datasets]https://www.kdnuggets.com/datasets/index.html)
- [topics](https://www.kdnuggets.com/topic)
- [cheat sheets](https://www.kdnuggets.com/cheat-sheets/index.html)
# resources|jekyll blog pdfs

## scikit-learn: install
---
- [latest release](https://scikit-learn.org/stable/install.html#installing-the-latest-release)
- [3rd party distributions](https://scikit-learn.org/stable/install.html#third-party-distributions-of-scikit-learn)
- [troubleshooting](https://scikit-learn.org/stable/install.html#troubleshooting)
- https://scikit-learn.org/stable/install.html#troubleshooting

## scikit-learn: user guide

### [linear models](https://scikit-learn.org/stable/modules/linear_model.html) [(my notebook (github))](https://github.com/bjpcjp/scikit-learn/blob/master/01_linear_models.ipynb)
---

#### [OLS (ordinary least squares)](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
---
- [example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)
- [non-negative OLS](https://scikit-learn.org/stable/modules/linear_model.html#non-negative-least-squares)
    - [example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_nnls.html#sphx-glr-auto-examples-linear-model-plot-nnls-py)
- [complexity](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares-complexity)

#### [ridge regression/classification](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
---
- [regression](https://scikit-learn.org/stable/modules/linear_model.html#regression)
  - [example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html)
  - solvers
- [classifier](https://scikit-learn.org/stable/modules/linear_model.html#classification)
  - [example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html#sphx-glr-auto-examples-linear-model-plot-ridge-path-py)
  - [example: text docs, sparse features](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py)
  - [pitfalls: coefficients](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py)
  - [complexity](https://scikit-learn.org/stable/modules/linear_model.html#ridge-complexity)
  - [ridge with built-in cross validation](https://scikit-learn.org/stable/modules/linear_model.html#setting-the-regularization-parameter-leave-one-out-cross-validation)

#### [lasso / multitask](https://scikit-learn.org/stable/modules/linear_model.html#lasso)
---
- [example: tomography](https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#sphx-glr-auto-examples-applications-plot-tomography-l1-reconstruction-py)
- [example: sparse data](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)
- [lasso can also be used for feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection)
- [regularization with _alpha_](https://scikit-learn.org/stable/modules/linear_model.html#setting-regularization-parameter)
- [regularization with CV](https://scikit-learn.org/stable/modules/linear_model.html#using-cross-validation)
- [feature selection using info criteria (AIC or BIC)](https://scikit-learn.org/stable/modules/linear_model.html#information-criteria-based-model-selection)

- [multitask lasso](https://scikit-learn.org/stable/modules/linear_model.html#multi-task-lasso)
  - [example: joint feature detection](https://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html#sphx-glr-auto-examples-linear-model-plot-multi-task-lasso-support-py)
  - [math (placeholder; use parent url)]()

#### [elastic-net / multitask](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)
---
- [example: sparse signals](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)
- [example: lasso vs elastic-net](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py)
- [multitask](https://scikit-learn.org/stable/modules/linear_model.html#multi-task-elastic-net)

#### [least angle regression (LARS) / lasso](https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression)
---
- [lars lasso](https://scikit-learn.org/stable/modules/linear_model.html#lars-lasso)
- [example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html#sphx-glr-auto-examples-linear-model-plot-lasso-lars-py)
- [algorithm](https://scikit-learn.org/stable/modules/linear_model.html#mathematical-formulation)

#### [orthogonal matching persuit (OMP)](https://scikit-learn.org/stable/modules/linear_model.html#orthogonal-matching-pursuit-omp)
---
- [example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_omp.html#sphx-glr-auto-examples-linear-model-plot-omp-py)

#### [bayes regression](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression)
---
- [description](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression)
- [example: curve fitting](https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge_curvefit.html#sphx-glr-auto-examples-linear-model-plot-bayesian-ridge-curvefit-py)
- [auto relevance determination (ARD)](https://scikit-learn.org/stable/modules/linear_model.html#automatic-relevance-determination-ard)
  
#### [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/01a_logistic_regression.ipynb)
---
- [logistic function](https://en.wikipedia.org/wiki/Logistic_function)
- [binary](https://scikit-learn.org/stable/modules/linear_model.html#binary-case)
- [multinomial](https://scikit-learn.org/stable/modules/linear_model.html#multinomial-case)
- [solvers](https://scikit-learn.org/stable/modules/linear_model.html#solvers): “lbfgs”, “liblinear”, “newton-cg”, “newton-cholesky”, “sag” and “saga”
- [example: L1 penalty & sparsity](https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html#sphx-glr-auto-examples-linear-model-plot-logistic-l1-l2-sparsity-py)
- [example: L1 regularization path](https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#sphx-glr-auto-examples-linear-model-plot-logistic-path-py)
- [example: multinomial vs one-vs-rest LR](https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html#sphx-glr-auto-examples-linear-model-plot-logistic-multinomial-py)
- [example: multiclass sparse LR - 20newsgroups](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py)
- [example: MNIST classification](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-mnist-py)
  
#### [generalized linear models (GLMs)](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models)
---
- [probability distributions: poisson, tweedie, gamma](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models)
- [usage: tweedie regressor](https://scikit-learn.org/stable/modules/linear_model.html#usage)
- [example: poisson regression, non-normal loss](https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html#sphx-glr-auto-examples-linear-model-plot-poisson-regression-non-normal-loss-py)
- [example: tweedie regression, insurance claims](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html#sphx-glr-auto-examples-linear-model-plot-tweedie-regression-insurance-claims-py)
- [tips](https://scikit-learn.org/stable/modules/linear_model.html#practical-considerations)

#### [stochastic gradient descent (SGD)](https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd)
---
- [reference](https://scikit-learn.org/stable/modules/sgd.html#sgd)
  
#### [perceptron](https://scikit-learn.org/stable/modules/linear_model.html#perceptronhttps://scikit-learn.org/stable/modules/linear_model.html#perceptronhttps://scikit-learn.org/stable/modules/linear_model.html#perceptron)
---
#### [passive-aggressive algos](https://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms)
---
#### [robustness, outliers, modeling errors](https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors)
---
- [concepts](https://scikit-learn.org/stable/modules/linear_model.html#different-scenario-and-useful-concepts)
- [RANSAC estimator](https://scikit-learn.org/stable/modules/linear_model.html#ransac-random-sample-consensus)
- [RANSAC example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py)
- [RANSAC example: fitting](https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py)
- [theil-sen estimator](https://scikit-learn.org/stable/modules/linear_model.html#theil-sen-estimator-generalized-median-based-estimator)
- [theil-sen regression example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_theilsen.html#sphx-glr-auto-examples-linear-model-plot-theilsen-py)
- [theil-sen fitting](https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py)
- [theil-sen math background](https://scikit-learn.org/stable/modules/linear_model.html#theoretical-considerations)
- [huber regression](https://scikit-learn.org/stable/modules/linear_model.html#huber-regression)
- [huber regression tips](https://scikit-learn.org/stable/modules/linear_model.html#notes)
- [huber regression example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_huber_vs_ridge.html#sphx-glr-auto-examples-linear-model-plot-huber-vs-ridge-py)

#### [quantile regression](https://scikit-learn.org/stable/modules/linear_model.html#quantile-regression)
---
- [example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html#sphx-glr-auto-examples-linear-model-plot-quantile-regression-py)

#### [extending linear models to polynomials](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
---

#### [discriminants]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/02_discriminant_analysis.ipynb)
---
- [dimensionality reduction - LDA]()
- [LDA & QDA math background]()
- [LDA dimension reduction math background]()
- [shrinkage & covariance]()
- [estimators]()

#### [kernel ridge regression (KRR)]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/03_kernel_ridge_regression.ipynb)
---

#### [support vector machines]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/04_support_vector_machines.ipynb)
---
- [classifiers]()
- [regressions]()
- [density estimates | novelty detection]()
- [complexity]()
- [tips]()
- [kernels]()
- [math]()
- [implementation]()

#### [stochastic gradient descent (SGD)]()  [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/05_stochastic_gradient_descent.ipynb)
---
- [classifiers]()
- [regressions]()
- [one-class SVM]()
- [sparse data]()
- [complexity]()
- [stopping]()
- [tips]()
- [math]()
- [implementation]()

#### [nearest neighbors]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/06_nearest_neighbors.ipynb)
---
- [unsupervised]()
- [classifiers]()
- [regressions]()
- [algorithms]()
- [nearest centroid]()
- [transformer]()
- [components analysis]()

#### [gaussians]() [(jupyter notebook - github)\)](https://github.com/bjpcjp/scikit-learn/blob/master/07_gaussians.ipynb)
---
- [regressions]()
- [classifiers]()
- [examples]()
- [kernels]()

#### [cross decomposition]() [(jupyter notebook - github)]()
---
- [PLS - canonical]()
- [PLS - SVD]()
- [PLS - regressions]()
- [canonical correlation analysis (CCA)]()

#### [naive bayes]()  [(jupyter notebook - github)]()
---
- [gaussian NB]()
- [multinomial NB]()
- [complement NB]()
- [bernoulli NB]()
- [categorical NB]()
- [tip: out-of-core fitting]()

#### [decision trees]() [(jupyter notebook - github)]()
---
- [classifiers]()
- [regressions]()
- [multiple outputs]()
- [complexity]()
- [tips]()
- [core algorithms]()
- [math]()
- [missing values]()
- [minimal cost-complexity pruning]()

#### [ensembles]() [(jupyter notebook - github)]()
---
- [gradient-boosted trees]()
- [random forests etc]()
- [bagging]()
- [voting classifiers]()
- [voting regressors]()
- [stacking]()
- [adaboost]()

#### [multiple class | output algos]() [(jupyter notebook - github)]()
---
- [multiclass classifiers]()
- [multilabel classifiers]()
- [multiclass-multioutput classifiers]()
- [multioutput regressors]()

#### [feature selection]() [(jupyter notebook - github)]()
---
- [removing low-variance features]()
- [univariate]()
- [recursive]()
- [select from model]()
- [sequential]()
- [pipelines]()

#### [semi-supervised algos]() [(jupyter notebook - github)]()
---
- [self training]()
- [label propagation]()

#### [isotonic regression]() [(jupyter notebook - github)]()
---

#### [probability calibration]() [(jupyter notebook - github)]()
---
- [calibration curves]()
- [classifier calibration]()
- [tips]()

#### [supervised neural nets]() [(jupyter notebook - github)]()
- [multilayer perceptrons]()
- [classifiers]()
- [regressions]()
- [regularization]()
- [algorithms]()
- [complexity]()
- [math]()
- [tips]()
- [warm starts]()
  
### __Unsupervised learning__

#### [gaussian mixtures]() [(jupyter notebook - github)]()
---
- [gaussian mixtures (GMs)]()
- [variational bayes GMs]()
  
#### [manifolds]() [(jupyter notebook - github)]()
---
- [intro]()
- [isomaps]()
- [locally linear embedding (LLE)]()
- [modified LLE]()
- [hessian eigenmapping]()
- [spectral embedding]()
- [local tangent space alignment (LTSA)]()
- [multidimensional scaling (MDS)]()
- [t-distributed stochastic neighbor embedding (t-SNE)]()
- [tips]()
  
#### [clustering]() [(jupyter notebook - github)]()
---
- [overview]()
- [k-means]()
- [affinity propagation]()
- [mean shift]()
- [spectral clustering]()
- [hierarchical]()
- [DBSCAN]()
- [HDBSCAN]()
- [OPTICS]()
- [BIRCH]()
- [evaluation]()
  
#### [biclustering]() [(jupyter notebook - github)]()
---
- [spectral co-clustering]()
- [spectral biclustering]()
- [evaluation]()
  
#### [matrix factorization | signal decomposition]() [(jupyter notebook - github)]()
---
- [principal component analysis (PCA)]()
- [kernel PCA]()
- [truncated SVD, latent semantic analysis (LSA)]()
- [dictionary learning]()
- [factor analysis]()
- [independent component analysis (ICA)]()
- [non-negative matrix factorization (NNMF)]()
- [latent dirichlet allocation (LDA)]()
  
#### [covariance]() [(jupyter notebook - github)]()
---
- [empirical]()
- [shrunk]()
- [sparse inverse covariance]()
- [robust estimates]()
  
#### [novelties & outliers]() [(jupyter notebook - github)]()
---
- [overview]()
- [novelty detection]()
- [outlier detection]()
- [local outlier factor]()
  
#### [density estimates]() [(jupyter notebook - github)]()
---
- [histograms]()
- [kernel density estimates (KDEs)]()
  
#### [neural nets]() [(jupyter notebook - github)]()
---
- [restricted boltzmann machines (RBMs)]()
  
### __model selection | evaluation | inspection | vizualization__

#### [cross validation (CV)](https://scikit-learn.org/stable/modules/cross_validation.html)  [(jupyter notebook - github)]()
---
- [computation](https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics)
- [iterators](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators)
- [shuffling](https://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling)
- [CV & model selection](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-and-model-selection)
- [permutation test scoring](https://scikit-learn.org/stable/modules/cross_validation.html#permutation-test-score)
  
#### [hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html) [(jupyter notebook - github)]()
---
- [grid search (exhaustive)](https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)
- [random parameter optimization](https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization)
- [successive halving](https://scikit-learn.org/stable/modules/grid_search.html#searching-for-optimal-parameters-with-successive-halving)
- [tips](https://scikit-learn.org/stable/modules/grid_search.html#tips-for-parameter-search)
- [alternatives to brute-force search](https://scikit-learn.org/stable/modules/grid_search.html#alternatives-to-brute-force-parameter-search)
  
#### [metrics|scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) [(jupyter notebook - github)]()
---
- [__scoring__](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
- [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [multilabel ranking metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics)
- [regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [clustering metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#clustering-metrics)
- [dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
  
#### [validation/learning curves](https://scikit-learn.org/stable/modules/learning_curve.html) [(jupyter notebook - github)]()
---
- [validation curves](https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve)
- [learning curves](https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve)

#### [ inspections]() [(jupyter notebook - github)]()
---
- [partial dependence & individual conditional expectation plots]()
  - partial dependence plots (PDPs)
  - individual conditional expectation (ICE) plots
  - math
  - computation
- [permutation feature importance plots]()
  - algorithm outline
  - relation to impurity-based importance in trees
  - misleading values
    
#### [viz utilities](https://scikit-learn.org/stable/visualizations.html#visualizations) [(jupyter notebook - github)]()
---
- [example: ROC curve](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_roc_curve_visualization_api.html#sphx-glr-auto-examples-miscellaneous-plot-roc-curve-visualization-api-py)
- [example: partial dependence plot (PDP)]()
- [example: display objects]()
- [classifier calibration comparisons]()
- [utilities list](https://scikit-learn.org/stable/visualizations.html#available-plotting-utilities)
  
#### [data transformers]() [(jupyter notebook - github)]()
---
- [pipelines]()
  - [chaining estimators]()
  - [regression target transforms]()
  - ["featureUnion": composite features]()
  - ["columnTransformer": heterogeneous data]()
  - [visualizations]()

#### [feature extraction]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/62_feature_extraction.ipynb)
  - [from dicts]()
  - [feature hashing]()
  - [text FE]()
  - [image FE]()
    
#### [data preprocessing]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/63_preprocessing.ipynb)
---
  - [standardization (mean removal, variance scaling]()
  - [non-linear transforms]()
  - [normalization]()
  - [encoding categories]()
  - [discretization]()
  - [missing values]()
  - [polynomial feature generation]()
  - [custom transformers]()
    
#### [missing value imputation]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/64_imputation.ipynb)
---
  - [univariate vs multivariate]()
  - [univariate]()
  - [multivariate]()
  - [nearest neighbors]()
  - [constant # of features]()
  - [marking imputed values]()
  - [NaN handling]()
    
#### [dimension reduction (unsupervised)]()
---
  - [PCA]()
  - [random projections]()
  - [feature agglomeration]()
    
#### [random projections]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/66_random_projections.ipynb)
---
  - [johnson-lindenstrauss lemma]()
  - [gaussian RPs]()
  - [sparse RPs]()
  - [inverse transforms]()
    
#### [kernel approximations]() [(jupyter notebook - github](https://github.com/bjpcjp/scikit-learn/blob/master/67_kernel_approximation.ipynb)
---
  - [nystroem method]()
  - [radial basis function (RBF) kernel]()
  - [additive chi squared kernel]()
  - [skewed chi squared kernel]()
  - [polynomal kernel approximation via Tensor sketch]()
  - [math background]()
    
#### [pairwise ops]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/68_pairwise_ops.ipynb)
---
  - [cosine similarity]()
  - [linear kernel]()
  - [polynomial kernel]()
  - [sigmoid kernel]()
  - [RBF kernel]()
  - [laplacian kernel]()
  - [chi-squared kernel]()
    
#### [prediction target transforms]() [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/69_transform_prediction_targets.ipynb)
---
  - [label binarization]()
  - [label encoding]()
  
#### [example (toy) datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/71_example_datasets.ipynb)
---
- [iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset)
- [diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- [digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- [linnerud](https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset)
- [wine](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)
- [breast cancer](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)

#### [real-world datasets](https://scikit-learn.org/stable/datasets/real_world.html)
---
- [olivetti faces](https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset)
- [20 newsgroups (text)](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
- [labeled faces in the wild](https://scikit-learn.org/stable/datasets/real_world.html#the-labeled-faces-in-the-wild-face-recognition-dataset)
- [forest covertypes](https://scikit-learn.org/stable/datasets/real_world.html#forest-covertypes)
- [RCV1](https://scikit-learn.org/stable/datasets/real_world.html#rcv1-dataset)
- [kddcup99](https://scikit-learn.org/stable/datasets/real_world.html#kddcup-99-dataset)
- [cal housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
  
#### [artificial data generators](https://scikit-learn.org/stable/datasets/sample_generators.html) [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/73_artificial_data_generators.ipynb)
---
- [classification | clustering](https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-classification-and-clustering)
- [regressions](https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-regression)
- [manifolds](https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-manifold-learning)
- [decomposition](https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-decomposition)

#### [other datasets](https://scikit-learn.org/stable/datasets/loading_other_datasets.html) [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/74_other_datasets.ipynb)
---
- [sample images](https://scikit-learn.org/stable/datasets/loading_other_datasets.html#sample-images)
- [svmlight|libsvm formats](https://scikit-learn.org/stable/datasets/loading_other_datasets.html#datasets-in-svmlight-libsvm-format)
- [openml.org](https://scikit-learn.org/stable/datasets/loading_other_datasets.html#downloading-datasets-from-the-openml-org-repository)
- [external datasets](https://scikit-learn.org/stable/datasets/loading_other_datasets.html#loading-from-external-datasets)
  
#### [scaling](https://scikit-learn.org/stable/computing/scaling_strategies.html) [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/81_scaling.ipynb)
  - [out-of-core learning](https://scikit-learn.org/stable/computing/scaling_strategies.html#scaling-with-instances-using-out-of-core-learning)

#### [performance](https://scikit-learn.org/stable/computing/computational_performance.html) [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/82_performance.ipynb)
---
- [latency](https://scikit-learn.org/stable/computing/computational_performance.html#prediction-latency)
- [bulk vs atomic mode](https://scikit-learn.org/stable/computing/computational_performance.html#bulk-versus-atomic-mode)
- [example - plot latency](https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html)
- [settings - reduced validation overhead](https://scikit-learn.org/stable/computing/computational_performance.html#configuring-scikit-learn-for-reduced-validation-overhead)
- [number of features](https://scikit-learn.org/stable/computing/computational_performance.html#influence-of-the-number-of-features)
  - [plot example](https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html)
- [sparse vs dense number representations](https://scikit-learn.org/stable/computing/computational_performance.html#influence-of-the-input-data-representation)    
- [throughput](https://scikit-learn.org/stable/computing/computational_performance.html#prediction-throughput)
- [example](https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html#sphx-glr-auto-examples-applications-plot-prediction-latency-py) 
- [tips](https://scikit-learn.org/stable/computing/computational_performance.html#tips-and-tricks)
- linear algebra (BLAS & LAPACK)
- limiting working memory
- model compression
- model re-shaping
- [scikit-learn docs](https://scikit-learn.org/stable/developers/performance.html#performance-howto)
- [SciPy sparse matrix docs](https://docs.scipy.org/doc/scipy/reference/sparse.html)

#### [parallelism](https://scikit-learn.org/stable/computing/parallelism.html) [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/83_parallelism.ipynb)
---
- [intro](https://scikit-learn.org/stable/computing/parallelism.html#parallelism)
- [joblib](https://scikit-learn.org/stable/computing/parallelism.html#higher-level-parallelism-with-joblib)
- [openMP](https://scikit-learn.org/stable/computing/parallelism.html#lower-level-parallelism-with-openmp)
- [BLAS & LAPACK tweaks](https://scikit-learn.org/stable/computing/parallelism.html#parallel-numpy-and-scipy-routines-from-numerical-libraries)
- [oversubscribing](https://scikit-learn.org/stable/computing/parallelism.html#oversubscription-spawning-too-many-threads)
- [config switches](https://scikit-learn.org/stable/computing/parallelism.html#configuration-switches)
- [python API](https://scikit-learn.org/stable/computing/parallelism.html#python-api)
- [env variables](https://scikit-learn.org/stable/computing/parallelism.html#environment-variables)
      
#### [model persistence](https://scikit-learn.org/stable/model_persistence.html) [(jupyter notebook - github)](https://github.com/bjpcjp/scikit-learn/blob/master/90_persistence.ipynb)
---
- python-specific
  - issues
  - __skops__ - a more secure format
- more formats

#### [common gotchas](https://scikit-learn.org/stable/common_pitfalls.html) [(jupyter notebook - github)]()
---
- preprocessing
- data leakage
  - during preprocessing
  - how to avoid
- randomness
  - using __None__ or __RandomState__
  - common pitfalls
  - general recommendations

#### [dispatching (beta)](https://scikit-learn.org/stable/dispatching.html)
---
- array API support

## API
---
- [base classes & utilities](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base)
- [base classes](https://scikit-learn.org/stable/modules/classes.html#base-classes)
- [functions](https://scikit-learn.org/stable/modules/classes.html#base-classes)
- [probability calibration](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration)
- [clustering](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster)
- [composite estimators](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose)
- [covariance](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance)
- [cross decomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition)
- [simple dataset loaders](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)
- [dataset geneators](https://scikit-learn.org/stable/modules/classes.html#samples-generator)
- [(matrix) decomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition)
- [discriminants](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis)
- [dummy estimators](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy)
- [ensembles](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
- [exceptions & warnings](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.exceptions)
- [experimental | preliminary](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental)
- [feature extraction](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction)
- [feature selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)
- [gaussians](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process)
- [(missing data) imputation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute)
- [inspection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection)
- [isotonic regression](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic)
- [kernel approximation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation)
- [(kernel ridge) regression](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge)
- [linear classifiers](https://scikit-learn.org/stable/modules/classes.html#linear-classifiers)
- [linear regressors](https://scikit-learn.org/stable/modules/classes.html#classical-linear-regressors)
- [linear regressors - variable selection](https://scikit-learn.org/stable/modules/classes.html#regressors-with-variable-selection)
- [linear regressors - bayesian](https://scikit-learn.org/stable/modules/classes.html#bayesian-regressors)
- [linear regressors - multitask](https://scikit-learn.org/stable/modules/classes.html#multi-task-linear-regressors-with-variable-selection)
- [linear regressors - outlier robust](https://scikit-learn.org/stable/modules/classes.html#outlier-robust-regressors)
- [linear regressors - generalized](https://scikit-learn.org/stable/modules/classes.html#generalized-linear-models-glm-for-regression)
- [linear models - other](https://scikit-learn.org/stable/modules/classes.html#miscellaneous)
- [manifolds](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold)
- [metrics | classification](https://scikit-learn.org/stable/modules/classes.html#classification-metrics)
- [metrics | regression](https://scikit-learn.org/stable/modules/classes.html#regression-metrics)
- [metrics | multilabel ranking](https://scikit-learn.org/stable/modules/classes.html#multilabel-ranking-metrics)
- [metrics | clustering](https://scikit-learn.org/stable/modules/classes.html#clustering-metrics)
- [metrics | biclustering](https://scikit-learn.org/stable/modules/classes.html#biclustering-metrics)
- [metrics | distances](https://scikit-learn.org/stable/modules/classes.html#distance-metrics)
- [metrics | pairwise](https://scikit-learn.org/stable/modules/classes.html#pairwise-metrics)
- [metrics | plotting](https://scikit-learn.org/stable/modules/classes.html#id7)
- [(gaussian) mixtures](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture)
- [model selection | splitters](https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
- [model selection | parameter optimizers](https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers)
- [model selection | validators](https://scikit-learn.org/stable/modules/classes.html#model-validation)
- [model selection | visualizers](https://scikit-learn.org/stable/modules/classes.html#visualization)
- [multiclass classifiers](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass)
- [multioutput classifiers & regressors](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput)
- [naive bayes](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)
- [nearest neighbor algorithms](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
- [(simple) neural nets](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network)
- [pipelines](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline)
- [(data) preprocessors](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
- [random projection transformers](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection)
- [semi-supervised learning](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised)
- [support vector machines (SVMs)](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
- [(decision) trees](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)
- [utilities](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils)

## examples
- [release highlights](https://scikit-learn.org/stable/auto_examples/index.html#release-highlights)
- biclustering
- calibration
- classification
- clustering
- covariance
- cross decomposition
- datasets
- decision trees
- decomposition
- developing estimators
- ensembles
- examples - real world datasets
- feature selection
- gaussian mixtures
- gaussian processes
- general linear models
- inspection
- kernel approximation
- manifolds
- miscellaneous
- missing values (imputation)
- model selection
- multioutput
- nearest neighbors
- neural nets
- pipelines | composite estimators
- preprocessing
- semisupervised classification
- support vector machines
- tutorial exercise
- working with text data
  
## community (blog posts)
## more
- getting started
- tutorial
- whats new
- glossary
- development
- faq
- support
- related packages
- roadmap
- governance
- about
- github
- other versions & download