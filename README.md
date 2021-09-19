# Welcome to the Creditworthiness Machine-Learning (ML) Model!<a id="Top-of-Page">
***
## <a id="Contents">Cotents</a>
[Overview of Analysis](#Overview-of-Analysis)<br>
[Purpose of Analysis](#Overview-of-Analysis_Purpose-of-Analysis)<br>
 - [Training Dataset](#Overview-of-Analysis_Training-Dataset)<br>
 - [Sampling Methods](#Overview-of-Analysis_Sampling-Methods)<br>
 - [Value Predictions](#Overview-of-Analysis_Value-Predictions)<br>
 - [Overview of Applied Supervised ML Methods](#Overview-of-Analysis_Overview-of-Applied-Supervised-ML-Methods)<br>
 - [Summary of ML Methods Used](#Overview-of-Analysis_Summary-of-ML-Methods-Used)<br>

[Project Layout](#Project-Layout)<br>
[Results](#Results)<br>
[Summary](#Summary)<br>
[Technologies and Resources](#Technologies-Resources)<br>
[Installation Guide](#Installation-Guide)<br>
[Contributors](#Contributors)<br>
[License](#License)<br>
[Bottom of Page](#Bottom-of-Page)<br>

***
## Overview of Analysis<a id="Overview-of-Analysis">
#### Purpose of Anaylsis<a id="Overview-of-Analysis_Purpose-of-Analysis">
This project applies supervised ML training and resampling algorithms to build an effective, accurate regression model for detecting both healthy and, the more scarce, high-risk loans (i.e., the loan status).<br>

#### Training Dataset<a id="Overview-of-Analysis_Training-Dataset">
Using supervised ML, this project utilizes a dataset of historical lending activiity from a peer-to-peer lending service to **model-fit-predict-evaluate**. The lending activity used includes the following information:<br>
 - **Loan Size**: The size of the loan
 - **Interest Rate**: The applied interest rate on the loan
 - **Borrower Income**: The borrower's annual income
 - **Debt-to-Income**: The borrower's debt-to-income ratio (DTI)
 - **Number of Account**: The borrower's total number of lending accounts
 - **Derogatory Marks**: The borrower's total number of derogatory marks on their credit score
 - **Total Debt**: The borrower's total amount of debt owed
 - **Loan Status**: The status of the borrower's loan ("healthy" vs "high-risk")<br>
      
#### Sampling Methods<a id="Overview-of-Analysis_Sampling-Methods">
This project uses the supplied financial dataset to train an initial <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression" title="sklearn.linear_model.LogisticRegression" target="_blank">Logistic Regression model</a> and compare that model with an over-sampled model.<br>
> Note, in this case, over-sampling is advantageous due to the inherent imbalance in "healthy" versus "high-risk" loans within the dataset.<br><br>
For this dataset, the "high-risk" loan status is the minority class label. The over-sampling implications are noted below in the <code>value_counts()</code> descriptoin in [Value Predictions](#Overview-of-Analysis_Value-Predictions).<br><br>

#### Value Predictions<a id="Overview-of-Analysis_Value-Predictions">
This project identifies frequency classifications as well as evaluating the accuracy of classification predictions. The notable variables are as follows:
 - <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html" title="pandas.Series.value_counts" target="_blank"><code>value_counts</code></a>: A count of the distinct labels for both the original and resampled data
 - <a id="balanced-accuracy-score" href="https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score" title="sklearn.metrics.balanced_accuracy_score" target="_blank"><code>balanced_accuracy_score</code></a>: This metric computes the balanced accuracy, which avoids inflated performance estimates on imbalanced datasets. In this binary case, this metric can be represented as follows (where TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative):<br>
\begin{equation*}
balanced\space accuracy = {\frac{1}{2}}\left({\frac{TP}{TP + FN}} + {\frac{TN}{TN + FP}}\right)
\end{equation*}
 - <a id="confusion-matrix" href="https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix" title="sklearn.metrics.confusion_matrix" target="_blank"><code>confusion_matrix</code></a>: This report provides the classification accuracy. In general terms, it provides the number of observations the model correctly classified.
 - <a id="classification-report-imbalanced" href="https://imbalanced-learn.org/dev/metrics.html#classification-report" title="imblearn.metrics.classification_report_imbalanced" target="_blank"><code>classification_report_imbalanced</code></a>: This report computes a set of metrics per class and summarizes it in a resultant table. The noteable fields in this report are:
  - **pre**: Precision, also known as the positive predictive value (PPV), measures how confident we are that the model correctly made the positive predictions. This metric is represented as follows:<br><center>$precision = TPs ÷ (TPs + FPs)$</center>
  - **rec**: Recall, also known as sensitivity, measures the number of actually "high-risk" loans taht the model correctly classified as "high-risk". This metric is represented as follows:<br><center>$recall = TPs ÷ (TPs + FNs)$</center>
  - **f1**: F1 Score, also known as the harmonic mean, can be characterized as a single summary statistic for the precision and recall. This metric is represented as follows:<br><center>$F1 = 2 × (precision × recall) ÷ (precision + recall)$</center>
  - **sup**: Support provides the number of instances for each class label ("high-risk" loans) found in the dataset.   
    
#### Overview of Applied Supervised ML Methods<a id="Overview-of-Analysis_Overview-of-Applied-Supervised-ML-Methods">
In order to identify a quantify the class label's bias, evaluate the impact, and develop an accurate model for prediction, this project performs the following high-level model-fit-predict-evaluate sequences:
1. Using the original dataset, create a logisitic regression model:<br>
This stage in the sequence provides us with a base prediction using the dataset as-is (with the class imbalance). Once a base prediction is made, we can then evaluate its [balance accuracy score](#balanced-accuracy-score), [confusion matrix](#confusion-matrix), and the [classification report imbalanced](#classification-report-imbalanced). This evaluation will provide us with necessary metrics for determining the accuracy of the trained model.<br><br>
2. Using resampled training data, create a logistic regression model:<br>
In this stage, we apply a random over-sampling technique to account for the class label imbalance for "high-risk" loans. Given that this classification is the most important, we need to be able to remove any bias in our model due to the reduced frequency of this classification in the original dataset. Once our over-sampling of the training data is complete, the new over-sampled training data is then used to generate a new resampled model and a resultant prediction using the model. Evaluation of the results in item 1 above are then repeated for the new prediction.<br><br>
3. Compare the original model with the over-sampled model:<br>
Given the two sets of accuracy reports for the above predictions, develop a quantifiable conclusion of the most accurate model.<br>

#### Summary of ML Methods Used<a id="Overview-of-Analysis_Summary-of-ML-Methods-Used">
The machine learning methods applied in this project are directly sourced from <a href="https://scikit-learn.org/stable" title="sklearn" target="_blank">scikit-learn</a>.<br><br>
Linear modeling was used as the primary method for modeling our dataset. Specifically, the <a href="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression" title="sklearn.linear_model.LogisticRegression" target="_blank">LogisticRegression</a> method was used for generating a linear model for classification. This method models the possible outcomes using a logistic function or logistic curve. A general model of the logistic curve is shown below:
<img src="img/general-logistic-curve.png" title="general logistic curve"><br>
**Resampling** was used to address the imbalanced, or under-represented, class of "high-risk" loans. The <a href="https://imbalanced-learn.org/stable/over_sampling.html#random-over-sampler" title="imblearn.over_sampling.RandomOverSampler" target="_blank">RandomOverSampler</a> method used does this by intentionally generating new samples by randomly sampling with replacment the current available samples.

## Project Layout<a id="Project-Layout">
The layout of this project is show below.<br>
.<br>
├── crypto_investments.ipynb<br>
├── data<br>
│   └── crypto_market_data.csv<br>
├── img<br>
│   ├── crypto-performance-clustering.png<br>
│   └── elbow-plot.png<br>
│   └── project_tree.png<br>
├── LICENSE<br>
├── README.md<br>
├── requirements.txt<br>
└── tree.txt<br>

***
## Results
***
## Technologies and Resources<a id="Technologies-Resources">
#### Technologies:
<a href="https://docs.python.org/release/3.8.0/" title="https://docs.python.org/release/3.8.0/"><img src="https://img.shields.io/badge/python-3.8%2B-red">
<a href="https://pandas.pydata.org/docs/" title="https://pandas.pydata.org/docs/"><img src="https://img.shields.io/badge/pandas-1.3.1-green"></a>
<a href="https://jupyter-notebook.readthedocs.io/en/stable/" title="https://jupyter-notebook.readthedocs.io/en/stable/"><img src="https://img.shields.io/badge/jupyter--notebook-5.7.11-blue"></a>
    <a href="https://hvplot.holoviz.org/user_guide/Introduction.html" title="https://hvplot.holoviz.org/user_guide/Introduction.html"><img src="https://img.shields.io/badge/hvplot-0.7.3-orange"></a>
    <a href="https://scikit-learn.org/stable/user_guide.html" title="https://scikit-learn.org/stable/user_guide.html"><img src="https://img.shields.io/badge/scikit_learn-0.24.2-green"></a><br>

***
## Installation Guide<a id="Installation-Guide">
### Project Installation
To install <a href="https://github.com/jasonjgarcia24/ml-performance-clustering.git" title="https://github.com/jasonjgarcia24/ml-performance-clustering.git">ml-performance-clustering</a>, type <code>git clone https://github.com/jasonjgarcia24/ml-performance-clustering.git</code> into bash in your prefered local directory.<br><br>
Alternatively, you can navigate to the same address (<code>https://github.com/jasonjgarcia24/ml-performance-clustering.git</code>) and download the full <code>main</code> branch's contents as a zip file to your prefered local directory.<br>

## Usage<a id="Usage">
Observe price-dislocation with <code>crypto_investments.ipynb</code>. No input variables are required.<br>

### Outputs
This tool provides several necessary visualizations for Cryptocurrency performance analysis:
1. K versus Inertia plots for both price change percentage of 24 hours versus 7 days and the PCA data:<br>
<img src="img/elbow-plot.png" title="elbow plot"><br>
These inertia plots allow us to measure the distribution of the data within a cluster given the K values.<br>

2. Scatter plots of both the original dataset and the PCA clusters following dimensionality reduction:
<img src="img/crypto-performance-clustering.png" title="crypto performance clustering">
***
## Contributors<a id="Contributors">
Currently just me :)<br>

***
## License<a id="License">
Each file included in this repository is licensed under the <a href="https://github.com/jasonjgarcia24/ml-performance-clustering/blob/2370b0c29d2c11c57d7c41d581612a5ca8c35503/LICENSE" title="LICENSE">MIT License.</a>

***
[Top of Page](#Top-of-Page)<br>
[Contents](#Contents)<br>
[Project Description](#Project-Description)<br>
[Technologies and Resources](#Technologies-Resources)<br>
[Installation Guide](#Installation-Guide)<br>
[Usage](#Usage)<br>
[Contributors](#Contributors)<br>
[License](#License)<br>
<a id="Bottom-of-Page"></a>
