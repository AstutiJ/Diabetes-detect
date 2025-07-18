import pandas as pd
import numpy as np
data = pd.read_csv(r'C:\Users\astut\Desktop\Diabetes_detect\diabetes.csv')
print(data.head(1))
print(data.shape)
print(data.info())
print(data.columns)

x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'Age']]
y = data[['Outcome']]

sample_data= pd.DataFrame({'Pregnancies':[6], 'Glucose':[148], 'BloodPressure':[72], 'SkinThickness':[35], 'Insulin':[0],
       'BMI':[33.6], 'Age':[50]})

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

model.fit(x,y)

Y_predicted= model.predict(x)
print("x_predict y: ", Y_predicted)

acc = model.score(x,y)
print('Acc',acc)

Y_predicted=model.predict(sample_data)
print("X_predicted y",Y_predicted)


import joblib as jb
#jb.(dumpName of trained model,name)

jb.dump(model,'Diabetes_KNN.pkl')
print("model saved")
# model saved
# to save the model 