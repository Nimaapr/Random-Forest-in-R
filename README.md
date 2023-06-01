# Iris Data Analysis with Random Forest Classification in R

This project conducts various feature selection methods and a Random Forest classification on the classic iris dataset. The purpose is to demonstrate different strategies for feature selection and to illustrate the impact of these strategies on the performance of a machine learning model. 

## Structure of the project

The project is structured into several parts:

1. **Read and Analyse Data**: The iris dataset is loaded and initial exploratory data analysis is conducted.

2. **Visualization**: Data is visualized to understand the distribution and relationship between different features.

3. **Feature Selection and Random Forest Classification**: Different feature selection methods are applied and their impact on a Random Forest classifier is observed. The feature selection methods used include:
    - Correlation-based feature selection
    - Univariate feature selection
    - Recursive feature elimination (RFE)
    - Tree-based feature selection

4. **Feature Extraction with PCA**: Principal Component Analysis is applied for feature extraction, followed by classification with a Random Forest classifier.

5. **Conclusion**: Results from the different approaches are summarized and key findings are presented.

## Libraries Used

The project uses several R packages, including:
- `caret`: for model training, feature selection, and data splitting.
- `randomForest`: for random forest modeling.
- `corrplot`: for plotting a correlation matrix.
- `e1071`: for skewness calculation.
- `psych`: for pair-wise correlation matrix.
- `mlbench`: for univariate feature selection.
- `DMwR`: for recursive feature elimination (RFE).
- `Boruta`: for tree-based feature selection.
- `FactoMineR` and `factoextra`: for PCA and associated visualizations.

## Results

The results highlight the importance of feature selection in improving the accuracy and performance of a machine learning model. Different feature selection methods can have varying impact on the model's performance. Feature extraction, such as PCA, is also explored in this project. The effectiveness of these methods, however, largely depends on the dataset.

For more detailed information, please refer to the R Markdown file in this repository.
