# Week 7: Ensemble Learning and Optimization Strategies

**Fusemachines AI Fellowship**

This week focused on improving model performance using **ensemble learning techniques** such as Bagging, Boosting, and XGBoost, along with **hyperparameter tuning** to optimize these models. Concepts were implemented through a series of Jupyter notebooks, each targeting specific ensemble methods and optimization strategies.

## Repository Structure

```
essemble-learning-stragegies/
├── notebooks/
│   ├── adaboost_implementation.ipynb
│   ├── bagging_implementation.ipynb
│   └── xgboost_text_classification.ipynb
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


### [`bagging_implementation.ipynb`](/notebooks/bagging_implementation.ipynb)

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


### [`adaboost_implementation.ipynb`](/notebooks/adaboost_implementation.ipynb)

* **Objective**: Improve model performance using AdaBoost with a pruned decision tree base learner.
* **What it does**:

  * Generated a synthetic classification dataset and performed cost-complexity pruning to find the optimal `ccp_alpha` for a baseline decision tree (**84% accuracy**).
  * Implemented AdaBoost with the optimized tree, boosting accuracy to **94%**.
  * Tuned AdaBoost hyperparameters (`n_estimators`, `learning_rate`) using `GridSearchCV`.
  * Applied feature scaling with `StandardScaler` and built a pipeline for preprocessing + boosting.
  * Conducted comprehensive tuning (including `ccp_alpha`), achieving **95% accuracy**.
* **Key takeaway**: AdaBoost significantly outperforms a single decision tree, especially when combined with feature scaling and hyperparameter tuning.



--- 
### [`xgboost_text_classification.ipynb`](/notebooks/xgboost_text_classification.ipynb)

* **Objective**: Build a high-performance text classification model using XGBoost, enhanced with metadata features and robust preprocessing.

* **What it does**:

  * Loaded a clean news dataset and performed exploratory data analysis (no missing values).
  * Developed a custom `TextPreprocessor` class with advanced text cleaning (stopword removal, lemmatization, etc.).
  * Built a baseline Multinomial Naive Bayes model using `CountVectorizer`, achieving **95% F1-score**.
  * Engineered metadata features (text length, punctuation counts) and combined them with text data.
  * Created a flexible `Pipeline` using `ColumnTransformer` to process mixed feature types.
  * Trained an XGBoost model with hyperparameter tuning via `RandomizedSearchCV`, reaching **99% F1-score**.
  * Compared baseline vs. boosted model performance, showing a **4% improvement**.

* **Key takeaway**: Combining comprehensive text preprocessing, metadata features, and XGBoost with hyperparameter tuning can substantially boost performance in text classification tasks.


## Summary

This week’s work combined theoretical foundations with practical implementation of ensemble learning. The notebooks are designed to progressively demonstrate how to reduce prediction error, increase model stability, and improve generalization through boosting and bagging methods culminating in fine-tuned, high-performing models using techniques like XGBoost and hyperparameter search.

---

## Links


* [Main Fellowship Repository](https://github.com/KushalRegmi61/AI_Fellowship_FuseMachines)


