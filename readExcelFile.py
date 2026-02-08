import pandas as pd

data = pd.read_excel('data.xlsx')
#print(data.head())

#print("colums names", data.columns)
print(data['name'])