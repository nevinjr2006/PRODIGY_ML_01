import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv('train.csv')
data['TotalBathrooms']=data['FullBath']+0.5*data['HalfBath']
X=data[['GrLivArea','BedroomAbvGr','TotalBathrooms']]
y=data['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
print(f"Root Mean Squared Error(RMSE):${rmse:,.2f}")
print(f"R-squared(R^2):{r2:.4f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6,color='teal')
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
