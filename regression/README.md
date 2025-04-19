**Sarah's Auto MPG Data Set**

Author: Sarah Howard 
Date: April 20, 2025

**Objective:** An analysis of the UCI Auto MPG dataset using regression to predict fuel efficiency.

**Link to Notebook**  

**Link to Peer Review** 

# Create a virtual environment and activate the .venv
Create a virtual environment py -m venv .venv
Activate virtual environment .venv\Scripts\activate

# Import the data
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Data set
The data set can be found at [UCI Auto-mpg dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset/data).

# Section 1. Import and Inspect the Data
Loaded the data, checked for missing values, and viewed summary statistics

# Section 2. Data Exploration and Preparation

# Section 3. Feature Selection and Justification

# Section 4. Train a Model (Linear Regression)

# Section 5. Improve the Model or Try Alternates (Implement Pipelines)

# Section 6. Final Thoughts & Insights
