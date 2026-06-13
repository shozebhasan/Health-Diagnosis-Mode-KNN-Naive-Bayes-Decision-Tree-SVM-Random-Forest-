# Health-Diagnosis-Mode-KNN-Naive-Bayes-Decision-Tree-SVM-Random-Forest-

Course: IT-466T Machine Learning (Fall 2025)
CLO: CLO 3 – Analyze different algorithms to improve the performance of machine learning models
Dataset: Smart Healthcare and Lifestyle Prediction (smart_healthcare_dataset.csv)

# Problem Statement

A regional healthcare diagnostics provider needs an analytical profiling module to optimize its predictive pipeline. This project evaluates architectural trade-offs (dimensionality reduction, model choice, hyperparameter tuning) across multiple ML paradigms to minimize classification error before deployment.

Target variable: heart_disease (binary: 0 = no disease, 1 = disease)
Goal: Not just maximize accuracy, but analyze how and why each technique (PCA, cross-validation, hyperparameter tuning) affects accuracy, stability (variance), training time, and memory.

# Contents


ML_CCP_Solution.ipynb – complete, self-contained Jupyter notebook with all code, plots, and analysis.


# Workflow Covered


EDA – inspecting the dataset, checking missing values, dropping the constant health_risk_score column, correlation heatmap.
Preprocessing & PCA (Dimensional Sensitivity)

Feature scaling with StandardScaler.
PCA explained-variance analysis.
Ablation study: accuracy, training time, and prediction latency vs. number of retained principal components.
Final component count selected at the ~95% explained-variance threshold.



Multi-Paradigm Baseline Models – KNN (distance-based), Naive Bayes (probabilistic), SVM (margin-based), Decision Tree, and Random Forest (tree-based), all evaluated with default hyperparameters on PCA-reduced data.
K-Fold Cross Validation – 5-fold CV for each model, reporting mean and standard deviation of accuracy across folds.
Hyperparameter Tuning

Grid Search: KNN (n_neighbors, weights), Decision Tree (max_depth, min_samples_split).
Random Search: SVM (C, kernel, gamma), Random Forest (n_estimators, max_depth).



Non-Functional Testing – training time, peak memory, and CV variance measured across varying PCA feature counts.
Functional Test Cases

CV stability check across different random shuffles.
Assertion that hyperparameter search converges to a result at least as good as the default configuration.



Results Matrix & Engineering Analysis – consolidated table of baseline / CV / tuned accuracies, plus a written bias-variance discussion of every result.


# Key Results (example run)

ModelBaseline AccCV Mean ± StdTuned  

# Test Acc KNN : 0.94 varies 0.944
# Naive Bayes: 0.94 varies: 0.944
# SVM: 0.96 varies: 0.971
# Decision Tree: 0.93 varies : 0.949
# Random Forest: 0.96 varies: 0.962

PCA: 12 of 14 components retain 97% variance, with accuracy plateauing beyond 10–12 components while training time keeps increasing, illustrating the dimensionality vs. cost trade-off.

# Tools & Libraries

pandas, numpy, matplotlib, seaborn, scikit-learn (PCA, KNN, GaussianNB, SVC, DecisionTreeClassifier, RandomForestClassifier, KFold, GridSearchCV, RandomizedSearchCV)
