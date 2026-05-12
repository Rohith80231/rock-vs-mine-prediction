# rock-vs-mine-prediction
```markdown
# Sonar Data Classification with Logistic Regression

## Project Overview
This project aims to classify objects detected by sonar signals as either a rock (`R`) or a mine (`M`). The dataset contains 60 numerical features representing the sonar signal patterns and a final column indicating the object type. A Logistic Regression model is used for this binary classification task.

## Dataset Description
The dataset consists of 207 samples, each with 60 features (sonar signal readings) and one target variable. The target variable is categorical, with 'R' representing a rock and 'M' representing a mine. There are 111 instances of mines and 96 instances of rocks in the dataset.

## Libraries Used
- `numpy`: For numerical operations, especially array manipulation.
- `pandas`: For data loading, manipulation, and analysis.
- `sklearn.model_selection.train_test_split`: For splitting the data into training and testing sets.
- `sklearn.linear_model.LogisticRegression`: For building the logistic regression model.
- `sklearn.metrics.accuracy_score`: For evaluating the model's performance.

## Steps and Code Explanation

### 1. Import Libraries
Essential libraries are imported at the beginning.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 2. Data Loading
The sonar data is loaded from a CSV file into a pandas DataFrame.

```python
df = pd.read_csv('/content/sonar data.csv')
```

### 3. Initial Data Exploration
- `df.shape`: Shows the number of rows and columns (207 rows, 61 columns).
- `df.head()`: Displays the first 5 rows of the DataFrame, showing the numerical features and the target variable (unnamed column 'R').
- `df.describe()`: Provides descriptive statistics of the numerical features.
- `df.columns`: Lists the column names. It's noted that the feature columns are unnamed numerical values and the target column is named 'R'.
- `df['R'].value_counts()`: Shows the distribution of the target variable (Mines vs. Rocks).
- `df.groupby('R').mean()`: Calculates the mean of each feature for both 'M' (Mine) and 'R' (Rock) classes, helping to understand feature differences between the two classes.

### 4. Data Preprocessing
- **Separating Features and Target**: The features (`X`) are separated from the target variable (`Y`). The last column ('R') is identified as the target.

```python
X = df.drop(columns='R', axis=1)
Y = df['R']
```

### 5. Train-Test Split
The data is split into training and testing sets. This is crucial for evaluating the model's performance on unseen data.
- `test_size=0.1`: 10% of the data is used for testing, and 90% for training.
- `stratify=Y`: Ensures that the proportion of 'R' and 'M' in both training and testing sets is similar to the original dataset, which is important for imbalanced datasets.
- `random_state=1`: Ensures reproducibility of the split.

```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
```

### 6. Model Training
- **Logistic Regression Initialization**: An instance of the Logistic Regression model is created.
- **Model Fitting**: The model is trained using the training features (`x_train`) and training target (`y_train`).

```python
model = LogisticRegression()
model.fit(x_train, y_train)
```

### 7. Model Evaluation
The model's performance is evaluated using the accuracy score on both the training and testing data.
- **Training Data Accuracy**: Measures how well the model learned from the training data.
- **Testing Data Accuracy**: Measures the model's generalization ability on unseen data. A significant difference between training and testing accuracy might indicate overfitting.

```python
X_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print("Accuracy training data:", training_data_accuracy)

X_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(X_test_prediction, y_test)
print("Accuracy testing data:", testing_data_accuracy)
```

### 8. Prediction System
This section demonstrates how to use the trained model to make predictions on new, single data points.
- **Input Data**: A tuple representing a new sonar signal is created.
- **Numpy Array Conversion**: The input data is converted into a NumPy array.
- **Reshaping**: The NumPy array is reshaped to indicate that it's a single instance (1 row) with 60 features. This is required because the model expects a 2D array.
- **Prediction**: The `model.predict()` method is used to classify the input data.

```python
input_data = (0.0317,0.0956,0.1321,0.1408,0.1674,0.1710,0.0731,0.1401,0.2083,0.3513,0.1786,0.0658,0.0513,0.3752,0.5419,0.5440,0.5150,0.4262,0.2024,0.4233,0.7723,0.9735,0.9390,0.5559,0.5268,0.6826,0.5713,0.5429,0.2177,0.2149,0.5811,0.6323,0.2965,0.1873,0.2969,0.5163,0.6153,0.4283,0.5479,0.6133,0.5017,0.2377,0.1957,0.1749,0.1304,0.0597,0.1124,0.1047,0.0507,0.0159,0.0195,0.0201,0.0248,0.0131,0.0070,0.0138,0.0092,0.0143,0.0036,0.0103)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
elif (prediction[0]=='M'):
  print('The object is a Mine')
else:
  print('Prediction error')

```

## Conclusion
The Logistic Regression model achieved an accuracy of approximately 84.4% on the training data and 76.2% on the test data. This indicates a reasonably good performance for classifying sonar signals. Further improvements could involve exploring other machine learning models, hyperparameter tuning, or more advanced feature engineering.
```
