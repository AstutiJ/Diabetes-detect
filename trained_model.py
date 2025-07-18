import joblib as jb
import pandas as pd
model=jb.load('Diabetes_KNN.pkl')
print('Model loaded successfully')
sample_data= pd.DataFrame({'Pregnancies':[6], 'Glucose':[148], 'BloodPressure':[72], 'SkinThickness':[35], 'Insulin':[0],
       'BMI':[33.6], 'Age':[50]})
Y_predicted=model.predict(sample_data)
print("X_predicted y",Y_predicted)

if (Y_predicted==0):
    print("Congraluations you don't have diabetes. Eat more sugar")
else:
    print("you have diabetes. No sugar anymore!!")
