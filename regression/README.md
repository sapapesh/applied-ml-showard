**Sarah's Auto MPG Data Set**

Author: Sarah Howard 
Date: April 20, 2025

**Objective:** An analysis of the UCI Auto MPG dataset using regression to predict fuel efficiency.

**Link to Notebook**  
https://github.com/sapapesh/applied-ml-showard/blob/main/regression/regression_sahoward.ipynb

**Link to Peer Review** 
https://github.com/sapapesh/applied-ml-showard/blob/main/regression/peer_review.md

# Create a virtual environment and activate the .venv
Create a virtual environment py -m venv .venv
Activate virtual environment .venv\Scripts\activate

# Import the data
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the Data set
The data set can be found at [UCI Auto-mpg dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset/data).

# Section 1. Import and Inspect the Data
Loaded the data, checked for missing values, and viewed summary statistics

# Section 2. Data Exploration and Preparation
2.1 Scatter matrixes, histograms, boxplots, count plots
2.2 Remove the outliers, encode the categorical feature
2.3 Create a new feature

# Section 3. Feature Selection and Justification
3.1 Choose features, select target variable
3.2 Define X and y

# Section 4. Train a Model (Linear Regression)

# Section 5. Improve the Model or Try Alternates (Implement Pipelines)

# Section 6. Final Thoughts & Insights
