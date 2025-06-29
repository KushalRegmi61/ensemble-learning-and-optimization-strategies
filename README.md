# Week 7: Ensemble Learning and Optimization Strategies

**Fusemachines AI Fellowship**

This week focused on improving model performance using **ensemble learning techniques** such as Bagging, Boosting, and XGBoost, along with **hyperparameter tuning** to optimize these models. Concepts were implemented through a series of Jupyter notebooks, each targeting specific ensemble methods and optimization strategies.

## Repository Structure

```
week_7/
├── error_decomposition.ipynb
├── bagging_random_forest.ipynb
├── boosting_adaboost.ipynb
├── boosting_gradientboosting.ipynb
├── boosting_xgboost.ipynb
├── hyperparameter_tuning.ipynb
└── README.md
```



## Key Concepts Covered

* **Bias–Variance Trade-off and Error Decomposition**: Understanding how model complexity impacts underfitting, overfitting, and noise.
* **Bagging (Bootstrap Aggregation)**: Reducing variance by training multiple base learners on different data subsets.
* **Random Forest**: Using randomized decision trees in bagging, with OOB evaluation and feature importance analysis.
* **Boosting Techniques**: Sequentially training models to minimize residual errors using AdaBoost, Gradient Boosting, and XGBoost.
* **XGBoost Enhancements**: Incorporating regularization, second-order optimization, and system-level efficiencies.
* **Hyperparameter Tuning**: Systematic optimization using GridSearchCV and RandomizedSearchCV.

---

## File Overview

### `error_decomposition.ipynb`

* **Objective**: Visualize and understand the bias–variance trade-off.
* **What it does**:

  * Simulates models with varying complexity (e.g., underfit, overfit).
  * Plots learning curves to illustrate training and validation error behavior.
  * Demonstrates how error decomposes into bias², variance, and noise.
* **Key takeaway**: Helps diagnose and interpret model performance issues.


### `bagging_implementation.ipynb`

* **Objective**: Develop a hands-on understanding of Bagging and Random Forest through both custom and scikit-learn implementations.
* **What it does**:

  * Implements **Bagging from scratch** with decision trees and bootstrap sampling.
  * Applies `BaggingClassifier` on the `make_moons` dataset for comparison.
  * Demonstrates **Out-of-Bag (OOB) evaluation** for model validation without a hold-out set.
  * Experiments with **random patches** and **subpatches** to improve model generalization.
  * Implements `RandomForestClassifier`, discussing its:

    * **Merits**: Reduces overfitting, performs well out-of-the-box, supports parallelization.
    * **Demerits**: Less interpretable, slower with large forests, prone to noise overfitting.
  * Uses **feature importances** from Random Forest for **feature selection**.
* **Key takeaway**: Explores how randomness in sampling and features enhances ensemble robustness and reduces variance.

---


### `adaboost_implementation.ipynb`

* **Objective**: Improve model performance using AdaBoost with a pruned decision tree base learner.
* **What it does**:

  * Generated a synthetic classification dataset and performed cost-complexity pruning to find the optimal `ccp_alpha` for a baseline decision tree (**84% accuracy**).
  * Implemented AdaBoost with the optimized tree, boosting accuracy to **94%**.
  * Tuned AdaBoost hyperparameters (`n_estimators`, `learning_rate`) using `GridSearchCV`.
  * Applied feature scaling with `StandardScaler` and built a pipeline for preprocessing + boosting.
  * Conducted comprehensive tuning (including `ccp_alpha`), achieving **95% accuracy**.
* **Key takeaway**: AdaBoost significantly outperforms a single decision tree, especially when combined with feature scaling and hyperparameter tuning.





### `boosting_gradientboosting.ipynb`

* **Objective**: Implement and tune Gradient Boosting models.
* **What it does**:

  * Uses gradient descent to minimize loss by fitting residuals stage-wise.
  * Compares different loss functions (squared error, Huber, log-loss).
  * Visualizes decision boundaries and training convergence.
* **Key takeaway**: Shows how gradient boosting improves bias and model flexibility.

---

### `boosting_xgboost.ipynb`

* **Objective**: Use XGBoost to build optimized, scalable boosting models.
* **What it does**:

  * Applies `XGBClassifier` to classification problems.
  * Explains the use of second-order derivatives (Hessians) for optimization.
  * Implements L1/L2 regularization, shrinkage, and subsampling.
  * Analyzes training speed and model performance.
* **Key takeaway**: Illustrates the efficiency and robustness of XGBoost.

---

### `hyperparameter_tuning.ipynb`

* **Objective**: Tune ensemble model hyperparameters to improve performance.
* **What it does**:

  * Compares `GridSearchCV` and `RandomizedSearchCV` for tuning.
  * Tunes parameters like `max_depth`, `n_estimators`, and `learning_rate`.
  * Applies to Random Forest, AdaBoost, and XGBoost models.
  * Uses cross-validation scores to select best configurations.
* **Key takeaway**: Highlights the importance of model tuning for generalization.

---


## Summary

This week’s work combined theoretical foundations with practical implementation of ensemble learning. The notebooks are designed to progressively demonstrate how to reduce prediction error, increase model stability, and improve generalization through boosting and bagging methods—culminating in fine-tuned, high-performing models using techniques like XGBoost and hyperparameter search.

---

## Links


* [Main Fellowship Repository](https://github.com/KushalRegmi61/AI_Fellowship_FuseMachines)

---

