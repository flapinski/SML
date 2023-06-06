import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import scipy
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler  
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression

#Calculate the 10-fold Cross validation error of a given model trained on features cols in dataset df. 
def E_kfold(Model,cols,df):
    E_ho = []

    # Extract the features given in cols as input and the 'Lead' feature as input
    X=df.iloc[:,cols]
    y=df.Lead

    # Define the scaling algorithm and the cross-validatin split
    scaler = StandardScaler()
    cv = skl_ms.KFold(n_splits=10, random_state=1, shuffle=True)

    for train_indxs, val_indxs in cv.split(X):

        # For each iteration in the cross validation splitting, define the validation and training data
        Xtrain, Xval, ytrain, yval = X.iloc[train_indxs], X.iloc[val_indxs], y.iloc[train_indxs], y.iloc[val_indxs]

        # Fit scaler based on training data only, but transform both training and validation data.
        scaler.fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xval = scaler.transform(Xval)

        # Fit the model based on training data, make predictions and calculated the missclassification rate (E_hold-out for this split)
        if Model != 'Naive':
            Model.fit(Xtrain, ytrain)
            predictions = Model.predict(Xval)
        else:
            predictions = ['Male']*len(yval)
        E_ho.append(np.mean(predictions != yval))
    
    # Return a list of all E_hold-out
    return E_ho

import random

#Look-up table of data features
#           0 = # words f       1 = # words         2 = # words lead,
#           3 = diffwrds l/cl   4 = # male actors   5 = year,
#           6 = # fe actors     7 = # words male    8 = gross
#           9 = mean age m      10 = mean age f     11 = age l
#           12 = age cl         13 = lead
#           14 = f w prop       15 = m w prop       16 = f a prop
#           17 = diff w prop    18 = prop.ageL/CL   19 = weighted prop words m
#           20 = prop w lead    21 = prop w co-lead

#Load the data
url = 'https://raw.githubusercontent.com/flapinski/SML/main/proj/train.csv'
df = pd.read_csv(url)

# Define new features
df['F words prop'] = df.iloc[:,0]/df.iloc[:,1] #14
df['M words prop'] = df.iloc[:,7]/df.iloc[:,1] #15
df['F actors prop'] = df.iloc[:,6]/(df.iloc[:,6] + df.iloc[:,4]) #16
df['Diff words prop'] = df.iloc[:,3]/df.iloc[:,2] #17
df['prop age L/CL'] = df.iloc[:,11]/df.iloc[:,12] #18
df['weighted prop words m'] = df.iloc[:,15]/(df.iloc[:,4]/(df.iloc[:,6] + df.iloc[:,4])) #19
df['prop words L'] = df.iloc[:,2]/df.iloc[:,1] #20
df['prop words CL'] = (df.iloc[:,2]-df.iloc[:,3])/df.iloc[:,1] #21

#M6=[BaggingClassifier(random_state=1),[0,3,11,12,15,14,16]]
# Define models using fine-tuned model parameters and features
M0=[skl_lm.LogisticRegression(solver='liblinear'),[1,2,6,9,10,11,12,14,15,16,17]] 
M1=[skl_da.QuadraticDiscriminantAnalysis(),[20,14,16,18,21,19]]
M2=[skl_nb.KNeighborsClassifier(7), [10,14,15,16,17,18]]
M3=[DecisionTreeClassifier(max_leaf_nodes=75,random_state=1),[0,3,11,12,15,14,16]]
M4=[RandomForestClassifier(n_estimators=100,max_leaf_nodes=75, random_state=1),[0,3,11,12,15,14,16]] 
M5=[GradientBoostingClassifier(n_estimators=82,learning_rate=0.31),[0,2,4,5,6,7,8,9,10,11,14,15,16,17]]
M6=[AdaBoostClassifier(n_estimators=82,learning_rate=0.31),[0, 3, 6, 7, 10, 14, 15, 16, 17, 18]]
M7=[MLPClassifier(hidden_layer_sizes=(100),activation='relu', max_iter = 1000,verbose=False,tol=0.0001,random_state=1),[0, 1, 4, 5, 6, 8, 12, 14, 15, 16, 17, 18] ]
M8 = ['Naive',[1,2,3]]
# Calculate hold out errors obtained by 10-fold cross validation and store in result matrix
methods = [M0,M1,M2,M3,M4,M5,M6,M7,M8]
result = np.array([E_kfold(i[0],i[1],df) for i in methods])

# Calculate cross-validation error as the mean of the hold-out errors
E_nfold = np.array([np.mean(result[i,:]) for i in list(range(0,len(methods)))])

# Make Box plot of hold-out error distribution
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.boxplot(result.T)
plt.xticks(np.arange(len(E_nfold))+1,('LOG','QDA','KNN','CT','RF','GBC','ADA','NN','Naive'))
plt.ylabel('$E_{hold-out}$',)
plt.xlabel('Method')
plt.show()

# Print 10-fold Cross validation error of all methods
print('E_10fold:')
print(f'LOG: {E_nfold[0]:.7}')
print(f'QDA: {E_nfold[1]:.7}')
print(f'KNN: {E_nfold[2]:.7}')
print(f'CT: {E_nfold[3]:.7}')
print(f'RF: {E_nfold[4]:.7}')
print(f'GBC: {E_nfold[5]:.7}')
print(f'ADA: {E_nfold[6]:.7}')
print(f'NN: {E_nfold[7]:.7}')
print(f'Naive: {E_nfold[8]:.7}')

#Produce figures for data analysis part:

#Part 1)
x = df['Number of female actors']/(df['Number of female actors'] + df['Number of male actors'])
N = len(x)
mu = np.mean(x)
sd = np.sqrt(np.var(x, ddof = 1))

plt.hist(x, bins=16, color= 'orange', edgecolor='w')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.xlabel('Proportion of female speaking roles', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xlim([0,1])
plt.ylim([0,150])

t = sc_st.t.interval(alpha=0.99, df=N-1)
L = mu + t[0]*sd/np.sqrt(N)
U = mu + t[1]*sd/np.sqrt(N)

plt.vlines(mu, 0, 150, linestyles='solid')
plt.vlines(L, 0, 150, linestyles='dashed', colors = 'g')
plt.vlines(U, 0, 150, linestyles='dashed', colors = 'g')
fig = plt.gcf()
fig.set_size_inches(5, 4)
plt.show() 

#Part 2)
df2 = df.sort_values('Year').loc[:, ['Year', 'Number of male actors', 'Number of female actors']]
df2['Prop. of female speaking roles'] = df2['Number of female actors']/(df2['Number of male actors']+df2['Number of female actors'])
year = df2['Year']
prop = df2['Prop. of female speaking roles']

plt.plot(year, prop, 'o', color = 'c')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.xlabel('Year', fontsize=16)
plt.ylabel('Prop. of female speaking roles', fontsize=16)
plt.ylim([0,1])
fig = plt.gcf()
fig.set_size_inches(5, 4)
from scipy.stats import pearsonr
print(pearsonr(year,prop))

#Part 3)
prop = []
for i in list(range(0,len(df.Lead))):
    if df.Lead[i] == 'Female':
        prop.append((df['Number words female'][i]+df['Number of words lead'][i])/(df['Total words'][i]))
    else:
        prop.append(df['Number words female'][i]/df['Total words'][i])

gross = np.asarray(df['Gross'])
plt.plot(prop, gross, 'o', color = 'y')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.xlabel('Prop. of female words', fontsize=16)
plt.ylabel('Gross', fontsize=16)
fig = plt.gcf()
fig.set_size_inches(5, 4)
plt.show()
print(pearsonr(prop,gross))
