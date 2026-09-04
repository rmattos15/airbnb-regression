# New York City Airbnb Regression Analysis

## Project Summary

New York City Airbnb popularity prediction using regression models. Explores data preprocessing, feature engineering, model selection, cross-validation, hyperparameter optimization and SHAP-based model interpretation, comparing five regression approaches and achieving an R² of 56.67% on the test set.

## Overview/Motivation

Airbnb was founded in 2007 and has since grown into a global platform connecting guests with unique accommodations and experiences. With a large number of listings available across different locations, understanding which features are associated with listing popularity can provide useful insights for hosts and help inform decisions when creating new listings.

This project explores this idea by using `reviews_per_month` as a proxy for listing popularity and investigating how different listing characteristics can be used to predict it.

*Disclaimer: This project was adapted from material originally developed for CPSC 330 (Applied Machine Learning) at the University of British Columbia.*

## Dataset

The dataset used in this project is the "New York City Airbnb Open Data", obtained from [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data). It contains 48,895 observations and 15 features per listing, plus the target variable `reviews_per_month`, which can be divided into a few different categories:

* Identification Features : `id`, `name`, `host_id`, `host_name`,
* Geographical Features: `neighbourhood_group`, `neighbourhood`, `latitude`, `longitude`
* Listing Characteristics: `room_type`, `price`, `minimum_nights`, `availability_365`
* Review Features: `number_of_reviews`, `last_review`
* Host Information: `calculated_host_listings_count`

Not all features were used in the regression models due to reasons such as limited informative value (e.g., Identification Features), arbitrary transformations or high cardinality.
The dataset is approximately 7 MB and is included directly in this repository, along with a NYC photo used for the exploratory data analysis.

## Approach

We evaluate five regression models, progressively experimenting with different algorithms and approaches to improve predictive performance. Feature selection and hyperparameter optimization are also explored to further assess the models and identify the best-performing configuration. The models are compared using cross-validation before the final model is evaluated on the held-out test set.

A brief description of each model and experiment is shown below:

| Model / Experiment          | Main Experiment                          |
| --------------------------- | ---------------------------------------- |
| Baseline                    | Dummy Regressor used as the baseline     |
| Ridge                       | Linear regression with L2 regularization |
| Decision Tree               | Nonlinear tree-based regression          |
| kNN                         | Nearest-neighbor regression              |
| Random Forest               | Ensemble of decision trees               |
| Feature Selection           | RFECV used to identify relevant features |
| Hyperparameter Optimization | Grid Search used to optimize the hyperparameters | 

## Results

### Standard Results

The following results were obtained using the standard configurations of each model, without hyperparameter optimization. The models were evaluated using cross-validation on the training data, with the average training and validation $R^2$ scores reported below.

| Model         | Train $R^2$ | Validation $R^2$ |
| ------------- | ----------: | ---------------: |
| Baseline      |       0.000 |           -0.001 |
| Ridge         |       0.334 |            0.334 |
| Decision Tree |       1.000 |           -0.118 |
| kNN           |       0.594 |            0.385 |
| Random Forest |       0.933 |            0.526 |

### Feature Selection Results

The following results were obtained after applying RFECV to the models. The models were evaluated using cross-validation on the training data.

| Model         | Train $R^2$ | Validation $R^2$ |
| ------------- | ----------: | ---------------: |
| Baseline      |       0.000 |            0.000 |
| Ridge         |       0.334 |            0.334 |
| Decision Tree |       1.000 |           -0.053 |
| kNN           |       0.598 |            0.400 |
| Random Forest |       0.932 |            0.522 |

### Hyperparameter Optimization Results

The following results were obtained after optimizing the hyperparameters of the Decision Tree, kNN, and Random Forest models using Grid Search. The models were evaluated using cross-validation on the training data.

| Model         | Train $R^2$ | Validation $R^2$ |
| ------------- | ----------: | ---------------: |
| Decision Tree |       0.521 |            0.467 |
| kNN           |       0.487 |            0.424 |
| Random Forest |       0.856 |            0.546 |

### Final Model

The best-performing model was the Random Forest with its hyperparameters optimized, which was therefore selected for evaluation on the test set.

| Model                   | Test $R^2$ | Test RMSE |
| ----------------------- | ---------: | --------: |
| Optimized Random Forest |     0.5667 |    1.0742 |

## Key Findings

* One of the biggest takeaways from the supervised machine learning material was realizing how much we can do to investigate and improve a model without ever touching the test data.

* During this project, it was clear that different algorithms captured different patterns. We saw the Random Forest substantially outperform the linear Ridge model, which showed the value of nonlinear models for this dataset.

* On the other hand, feature selection using `RFECV` did not lead to improvements in model performance, even though adding `total_price` and `is_manhattan` was a useful part of the prior exploration and helped with understanding the data.

* Finally, the evaluation metrics were useful in different parts of the project. For example, comparing training with validation and test performance helped identify overfitting, while $R^2$ and RMSE provided complementary views of predictive performance.

## Requirements

The project was developed using Python. The main tools and libraries used include:

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP
* JupyterLab