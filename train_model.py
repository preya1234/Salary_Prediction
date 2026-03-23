import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
data={
    "experience":[1,2,3,4,5],
    "salary":[20000,30000,40000,50000,60000]
}
df=pd.DataFrame(data)
X=df[['experience']]
y=df['salary']
model=LinearRegression()
model.fit(X,y)
pickle.dump(model,open('model.pkl','wb'))
'''this line saves the trained model to file named 'model.pkl' using pickle library
the 'wb' mode stands for write binary which is necessary for saving the model '''
print("Model Trained & saved")