

# Week 7: Ensemble Learning and Optimization Strategies

This week focused on improving model performance through **ensemble learning techniques** like Bagging, Boosting, and **XGBoost**, paired with practical methods for **hyperparameter tuning**. The goal was to reduce bias and variance while building more accurate and generalizable models.


### Repository Structure

```
week_7/
├── bagging_implementation.ipynb
├── adaboost_implementation.ipynb
├── xgboost_implementation.ipynb
├── gridsearch_randomsearch.ipynb
```

---

## Topics Covered

### 1. Error Decomposition

* **Definition**: Broke down prediction error into **bias²**, **variance**, and **irreducible noise**.
* **Trade-off Visualization**: Used learning curves to identify overfitting and underfitting scenarios.

**Takeaway**: Understanding bias–variance decomposition helps select models with appropriate complexity for the data.


### 2. Bagging and Random Forest

* **Bagging**: Combined multiple **unstable learners** using bootstrapped datasets to reduce variance.
* **Out-of-Bag Evaluation**: Used for internal model validation without needing an extra hold-out set.
* **Random Forest**: Added row and column subsampling (random patches) for stronger de-correlation between trees.
* **Feature Importance**: Used `feature_importances_` for ranking and interpreting predictors.

>  [bagging\_implementation.ipynb](week_7/bagging_implementation.ipynb)

**Takeaway**: Bagging stabilizes model performance by reducing variance through averaged predictions from multiple base learners.


### 3. Boosting Techniques

Boosting builds strong learners sequentially by training each new model to correct the residual errors of the previous one.

#### AdaBoost

* **Core Idea**: Assigns more weight to misclassified examples and adjusts learner importance based on accuracy.
* **Loss**: Uses **exponential loss** for classification and **squared loss** for regression.

>  [adaboost\_implementation.ipynb](week_7/adaboost_implementation.ipynb)

**Takeaway**: AdaBoost improves performance by prioritizing difficult examples and adjusting learner contributions adaptively.

#### Gradient Boosting

* **Gradient Descent on Residuals**: Fits each new learner to the **negative gradient** (error) of the loss function w\.r.t. predictions.
* **Loss Functions**: Supports squared loss, logistic loss (for classification), and **Huber loss** for robustness against outliers.
* **Sequential Updates**: Each model is added in a stage-wise manner to minimize total loss using gradient-based optimization.

**Takeaway**: Gradient Boosting is a flexible and powerful approach that iteratively reduces model error using gradient descent on residuals.

#### XGBoost

* **Optimizations**:

  * Uses **second-order derivatives** (Hessian) for more accurate gradient updates via **Taylor series expansion**
  * Supports **L1 and L2 regularization** to reduce overfitting
  * Implements **shrinkage**, **column and row subsampling**
* **System Features**: Includes parallelization, out-of-core computation, and cache-aware learning for scalability.

>  [xgboost\_implementation.ipynb](week_7/xgboost_implementation.ipynb)

**Takeaway**: XGBoost scales gradient boosting with additional regularization and optimization techniques for speed and predictive power.


### 4. Hyperparameter Tuning

* **Core Idea**: Hyperparameters are the tuning knobs outside the model that significantly affect learning performance.
* **Techniques**: Used `GridSearchCV` for exhaustive search and `RandomizedSearchCV` for faster sampling-based tuning.
* **Application**: Tuned ensemble methods including Random Forest, AdaBoost, and XGBoost.

> [gridsearch\_randomsearch.ipynb](week_7/gridsearch_randomsearch.ipynb)

**Takeaway**: Effective hyperparameter tuning unlocks a model’s full potential by finding the best-performing configuration.



## Key Insight

> Ensemble methods like Bagging and Boosting improve accuracy and generalization by reducing error components, while techniques like hyperparameter tuning and regularization help scale these models to real-world applications efficiently.

--- 

## Full Fellowship Progress

[AI Fellowship Repository](https://github.com/KushalRegmi61/AI_Fellowship_FuseMachines/tree/master)


