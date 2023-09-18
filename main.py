#ML Project
##import pandas
import pandas as pd

##read file frpm github
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")
##look at the first 10 rows
print(df.head(10))

##create pandas dataframe
y=df['logS']
print(y)

##create pandas dataframe
X=df.drop(['logS'], axis=1)
print(X)

##import scikit-learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=100)

##Model building
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train, y_train)

##train model predicting the solubility
y_lr_train_pred= lr.predict(X_train)
y_lr_test_pred= lr.predict(X_test)

##Model evaluation
from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse= mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2= r2_score(y_train, y_lr_train_pred)

lr_test_mse= mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2= r2_score(y_test, y_lr_test_pred)

print("Linear Regression Train MSE: ", lr_train_mse)
print("Linear Regression Train R2: ", lr_train_r2)
print("Linear Regression Test MSE: ", lr_test_mse)
print("Linear Regression Test R2: ", lr_test_r2)


