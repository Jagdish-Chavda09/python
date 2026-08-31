import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = {
    "marks" : [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    "placed" :  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['marks']]
y = df['placed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

new_student = [[65]]

prediction = model.predict(new_student)

print("Prediction:", prediction)