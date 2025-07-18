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
# print(x)
# print(y)

# KNN algorithm is the nearest neighbour it is an algorithm of
# # this problem is not suitable for linear regression , ridge, laaso
# from sklearn.linear_model import LinearRegression
# model=LinearRegression()
# print("sklearn.linear_model import LinearRegression done")

# model.fit(x,y)
# # step 5 result 
# Y_prediction=model.predict(x)
# print(Y_prediction)

# print(model.predict([[1,2]]))

# test_data=pd.DataFrame({'A':[1],'B':[2]})
# test_result=model.predict(test_data)
# print(test_result)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

model.fit(x,y)

Y_predicted= model.predict(x)
print("x_predict y: ", Y_predicted)

acc = model.score(x,y)
print('Acc',acc)

# for regression accuracy is r2score
# for classification accuracy is model.score
# find the optmial value of k
# ac=[]
# n_neighbors = []
# for e in np.arange(1,786,0.1):
#     model=KNeighborsClassifier(n_neighbors=e)
#     model.fit(x,y)
#     Y_predicted=model.predict(x)
#     ac.append(model.score())
#     n_neighbors.append(e)
# A = np.array(ac)
# max_ac=np.max(A)
# max_index=np.argmax(A)
# final_alpha = n_neighbors[max_index]

# print(max_ac,"done!!")
# print(max_index,"Done!!")
# print(final_alpha,"Maximum value")

# # hyper parameter tuning
