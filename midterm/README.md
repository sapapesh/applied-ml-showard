# Sarah's Poisonous Mushroom Data Set

**Author:** Sarah Howard 
**Date:** April 6, 2025
**Objective:** Training a classification model to determine which mushrooms are poisonous or edible.
# Create a virtual environment and activate the .venv
Create a virtual environment py -m venv .venv
Activate virtual environment .venv\Scripts\activate

# Import the data
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the Data set
The data set can be found at [UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom).
Upon reviewing the data, the variable "stalk-root" was missing data so I dropped that from our dataset as we still had 22 other variables avaiable to be able to review.


