# UCI Default of Credit Card Clients Dataset analysis
Tesina for the Mathematics in Machine Learning course. <br/>
Analysis of the dataset [UCI Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients), that contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.<br/>
 
 The repository containes three folders:
 - `dataset` folder with the Default of Credit Card Clients dataset;
 - `imgs` folder, that containes all the images for the report;
 - `results` folder, in which all the results are stored.
 
 
 The script `svdd.py` containes a implementation of [Support Vector Data Description](https://www.researchgate.net/publication/226109293_Support_Vector_Data_Description) for novelty detection, while the file `visualization.py` containes some useful tools for visualizing plots and graphs.
Every algorithm used in the machine learning pipeline in the notebook `run.ipynb` is preceded by a theoretical description.
The analysis is divided into different steps:
 - Data exploration: description of the attributes and understanding of each features;
 - Data cleaning: processing of data that involves correction of errors, outlier detection, analysis of correlation;
 - Outlier detection: identification and removal of rows considered as possible outliers by different algorithm of outlier detection;
 - Correlation and dimensionality reduction: analysis of correlated features and application of [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis);
 - Class imbalancing managment: application of some tecninques for managing imbalancing in classes;
 - Classification algorithms and metrics for evaluation;
 - Comments on results achieved.
 
 All the results and the analysis can be found in the file `report.html`.
