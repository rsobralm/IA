import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


df = pd.read_csv('BankChurners.csv', delimiter=",") #abrindo o arquivo
df

for column in df.columns:
    df = df[df[column] != "Unknown"]

#https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html
#https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python

df = df.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])