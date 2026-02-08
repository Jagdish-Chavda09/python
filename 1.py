import pandas as pd

# Create a tiny table (DataFrame)
data = {
    'Fruit': ['Apple', 'Banana'],
    'Price': [1.20, 0.50]
}

df = pd.DataFrame(data)

print("Pandas is working perfectly! Here is your first DataFrame:")
print(df)