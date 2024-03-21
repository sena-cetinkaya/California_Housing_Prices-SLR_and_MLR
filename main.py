# Import the libraries.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read the dataset.
data = pd.read_csv("housing.csv")

# Display the first few rows of the data.
print(data.head)

# Learning columns.
print(data.columns)

# Data types of columns.
print(data.dtypes)

# Learning the size of the dataset.
print(data.shape)

# Check for missing values.
print(data.isnull().sum())

# Summary statistics.
print(data.describe)

# Categorical data and counts in the Ocean Proximity column.
print(data['ocean_proximity'].value_counts())

# DATA VISUALIZATION
data.hist(figsize=(15, 8),bins = 50)

# ocean_proximity categories and percentages.
location = data["ocean_proximity"].value_counts()
location.plot(kind="pie", figsize=(8, 8), title="Ocean Proximity", autopct="%.1f%%")

# population based on total_rooms.
plt.figure(figsize=(8,8))
plt.scatter(data["population"], data["total_rooms"], color = 'lightcoral')
plt.title("population based on total_rooms")
plt.xlabel("population")
plt.ylabel("total_rooms")
plt.show()

# MULTIPLE LINEAR REGRESSION
data.dropna(axis=1,inplace=True)

X = data.iloc[:,0:7]
y = data['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# SUCCESS RATE OF THE MULTIPLE LINEAR REGRESSION MODEL
print("R2 Score: ", r2_score(y_test, y_pred)*100)

# SIMPLE LINEAR REGRESSION
X = data[["population"]]
y = data[["total_rooms"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# SUCCESS RATE OF THE SIMPLE LINEAR REGRESSION MODEL
print("R2 Score: ", r2_score(y_test, y_pred)*100)

# VISUALIZATION OF THE RESULT
plt.scatter(X_train, y_train, color="pink")
X_train_pred = model.predict((X_train))
plt.scatter(X_train, X_train_pred, color="lightblue")
plt.title('population - total_rooms')
plt.xlabel('population')
plt.ylabel('total_rooms')
plt.show()
