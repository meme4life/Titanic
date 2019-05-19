#Linear algebra
import numpy as np

#Data processing
import pandas as pd

#Data visualization
import seaborn as sns
#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.style


#Algorithms

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
 
#Getting the data
test_df = pd.read_csv("F:/Machine Learning/titanic/test.csv")
train_df = pd.read_csv('F:/Machine Learning/titanic/train.csv')

total = train_df.isnull().sum().sort_values(ascending=False) # isnull(): Detect missing values(True = NaN)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%']) #axis=1: Total will be the column
missing_data.head(5)

train_df.columns.values

#Age and sex
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))
women = train_df[train_df['Sex']=='female'] #Assemble all rows which have female gender
men = train_df[train_df['Sex']=='male'] #Assemble all rows which have male gender
ax = sns.distplot(women[women['Survived'] ==1].Age.dropna(), bins = 18, label = survived, ax = axes[0], kde = False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False) #https://seaborn.pydata.org/generated/seaborn.distplot.html
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)# dropna(): Remove missing values
ax.set_title('Male')
ax.legend() #legend: a list that explains the symbols on a map


#Embarked, Pclass and Sex
'''https://seaborn.pydata.org/generated/seaborn.FacetGrid.map.html?highlight=facetgrid
https://seaborn.pydata.org/generated/seaborn.FacetGrid.html?highlight=facetgrid'''
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


sns.barplot(x = 'Pclass', y = 'Survived',data= train_df)
grid = sns.FacetGrid(train_df,col = 'Survived',row = 'Pclass',size = 2.2, aspect = 1.6)
grid.map(plt.hist,'Age',alpha =.5,bins = 20)
grid.add_legend()

#SibSp and Parch:
data= [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives']>0, 'not_alone' ] = 0
    dataset.loc[dataset['relatives']==0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
    
train_df['not_alone'].value_counts()

#https://medium.com/@yoonho0715/seaborn-factor-plot-params-2a3ed9cf71bc
axes = sns.factorplot('relatives','Survived',data = train_df,aspect = 2.5)