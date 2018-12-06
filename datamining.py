from pydoc import help
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from pandas import DataFrame
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
matplotlib.style.use('ggplot')
warnings.filterwarnings("ignore", category=FutureWarning)
filename = "my.csv"

### Reading dataset
train= pd.read_csv(filename)
test=pd.read_csv('my1.csv')

###Look up the datasets
print("Look up the train datasets")
print(train.describe())
print("Look up the test datasets")
print(test.describe())
#print(train.shape)#Dataset train contains 2432 samples
#print(test.shape) #Dataset test contains 122 samples

###Let's loot at correlation between variables
print("Let's loot at correlation between variables")
print(train.corr())
#pd.scatter_matrix(train, figsize=(6, 6))


###Plot histogram  for each numeric variable
#train.hist(bins=50, figsize=(20,15))
#plt.savefig("attribute_histogram_plots")

###Discover outliers with visualization tools %Box plot

#sns.boxplot(x=train['rooms'])
#sns.boxplot(x=train['Elitka'])
#sns.boxplot(x=train['heating'])
#sns.boxplot(x=train['price'])
#sns.boxplot(x=train['floor'])
#sns.boxplot(x=train['area'])

###Calculating Z-score to determine and remove outliers
z = np.abs(stats.zscore(train))
zt=np.abs(stats.zscore(test))

train = train[(z < 3).all(axis=1)]
test=test[(zt<3).all(axis=1)]

train=train.drop("heating", axis=1)
test=test.drop("heating", axis=1)
print('')
print("Let's loot at correlation between variables after removing outliers")
print(train.corr())
plt.show()

###Seperating dataset

X = train.iloc[:, 0:-1]
test_x=test.iloc[:, 0:-1]
y = train.iloc[:, -1]
test_y=test.iloc[:, -1]

###Build Linear Regression and traint it

logReg = LinearRegression()
logReg.fit(X,y)

pred = logReg.predict(test_x)
print('')
print('Liner Regression R squared: %.4f' % logReg.score(test_x,test_y))
###print(pred)

###Build RandomForestRegressor and traint it

forest_reg = RandomForestRegressor(random_state=55)
forest_reg.fit(X, y)
print('Random Forest R squared": %.4f' % forest_reg.score(test_x,test_y))
