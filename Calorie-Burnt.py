import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle
from tkinter import *

def read_csv(file_path):
    return pd.read_csv("calorie.csv)

def dataset_info_statistics(data):
    print("Dataset Information:")
    print(data.info())
    print("\nBasic Statistics for Numerical Columns:")
    print(data.describe())
    print("\n")

def check_null(data):
    null_counts = data.isnull().sum()
    print("Null Values in the Dataset:")
    return null_counts

def check_duplicates(data):
    return data.duplicated().any()

def plot_graph(data):
    numerical_columns = data.select_dtypes(include=np.number).columns
    for column in numerical_columns:
        plt.figure(figsize=(5, 3))
        sns.distplot(data[column], kde=True)
        plt.title(f"Histogram for {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
        
    categorical_columns = data.select_dtypes(include='object').columns
    for column in categorical_columns:
        plt.figure(figsize=(5, 3))
        sns.countplot(data[column])
        plt.title(f'Countplot for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

def separate_features_target(data, target_column):
    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]
    return X, y

def perform_train_test_split(X, y, test_size=0.20, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test  

calories = read_csv('calories.csv')
exercise = read_csv('exercise.csv')

data = pd.merge(calories, exercise, on='User_ID')
data.head()
dataset_info_statistics(data)
check_null(data)

# plot_graph(data)
data.columns
X, y = separate_features_target(data, 'Calories')
X = X.drop(columns=['User_ID'])
X_train, X_test, y_train, y_test = perform_train_test_split(X, y, test_size=0.20, random_state=42)

# Column Transformer and Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(), ['Gender']),
    ('num', StandardScaler(), ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']),
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

from sklearn import set_config
set_config(display='diagram')
pipeline

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

r2_score(y_test, y_pred)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
cv_results.mean()

mean_absolute_error(y_test, y_pred)

def model_scorer(model_name, model):
    output = []
    output.append(model_name)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    output.append(r2_score(y_test, y_pred))
    output.append(mean_absolute_error(y_test, y_pred))
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
    output.append(cv_results.mean())
    
    return output

model_dict = {
    'log': LinearRegression(),
    'RF': RandomForestRegressor(),
    'XGBR': XGBRegressor(),
}

model_output = []
for model_name, model in model_dict.items():
    model_output.append(model_scorer(model_name, model))

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(), ['Gender']),
    ('num', StandardScaler(), ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor())
])

pipeline.fit(X, y)

sample = pd.DataFrame({
    'Gender': 'male',
    'Age': 68,
    'Height': 190.0,
    'Weight': 94.0,
    'Duration': 29.0,
    'Heart_Rate': 105.0,
    'Body_Temp': 40.8,
}, index=[0])

pipeline.predict(sample)

# Save The Model
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('pipeline.pkl', 'rb') as f:
    pipeline_saved = pickle.load(f)

result = pipeline_saved.predict(sample)
print(result)

# GUI
def show_entry():
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    p1 = str(clicked.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())

    sample = pd.DataFrame({
        'Gender': [p1],
        'Age': [p2],
        'Height': [p3],
        'Weight': [p4],
        'Duration': [p5],
        'Heart_Rate': [p6],
        'Body_Temp': [p7],
    }, index=[0])

    result = pipeline.predict(sample)
    print(result)
    Label(master, text="Amount of Calories Burnt").grid(row=13)
    Label(master, text=result[0]).grid(row=14)

master = Tk()
master.title("Calories Burnt Prediction using Machine Learning")
label = Label(master, text="Calories Burnt Prediction", bg="black", fg="white").grid(row=0, columnspan=2)

Label(master, text="Select Gender").grid(row=1)
Label(master, text="Enter Your Age").grid(row=2)
Label(master, text="Enter Your Height").grid(row=3)
Label(master, text="Enter Your Weight").grid(row=4)
Label(master, text="Duration").grid(row=5)
Label(master, text="Heart Rate").grid(row=6)
Label(master, text="Body Temp").grid(row=7)

clicked = StringVar()
options = ['male', 'female']

e1 = OptionMenu(master, clicked, *options)
e1.configure(width=15)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)

Button(master, text="Predict", command=show_entry).grid()
mainloop()
