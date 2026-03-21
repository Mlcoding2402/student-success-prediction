import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("D:\AI engineer\MACHINE LEARNING\Student.csv")

print(df)
print(df.info())
print(df.isnull().sum())
print(df.describe())

features = ['Maths','Physics','Chemistry']


df_scaler = df.copy()
X = df_scaler[features]
Y = df_scaler['Result']

X_train , X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape,X_test.shape)
 
model = LogisticRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(Y_test, Y_pred))
print("Confussion matrix:\n", confusion_matrix(Y_test, Y_pred))
