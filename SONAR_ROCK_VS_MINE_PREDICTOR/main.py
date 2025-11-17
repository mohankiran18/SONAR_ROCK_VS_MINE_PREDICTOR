# SONAR ROCK VS MINE PROJECT
# SONAR_ROCK_VS_MINE_PREDICTOR
# The submarine should predict whether an object is a mine or a rock.
# The system uses sonar data to detect and classify objects.
# Workflow:
# 1. Collect sonar data (labeled as Mine or Rock)
# 2. Preprocess the data
# 3. Split data into training and test sets
# 4. Train a Logistic Regression model (binary classification)
# 5. Evaluate the model
# 6. Predict new object type using the trained model

# Step 1: Import Libraries 
import numpy as np              # For numerical operations and arrays
import pandas as pd             # For data processing (tables/dataframes)
from sklearn.model_selection import train_test_split  # For splitting data into train/test
from sklearn.linear_model import LogisticRegression   # Logistic Regression model
from sklearn.metrics import accuracy_score            # For model evaluation

# Step 2: Load and Explore the Data
# Load dataset into pandas dataframe (no header in CSV)
sonar_data = pd.read_csv('Copy of sonar data.csv', header=None)

# Check the first few rows
print(sonar_data.head())

# Number of rows and columns
print(sonar_data.shape)

# Statistical measures of the dataset
print(sonar_data.describe())

# Count of each label (M = Mine, R = Rock)
print(sonar_data[60].value_counts())

# Mean values grouped by label
print(sonar_data.groupby(60).mean())

#  Step 3: Separate Features and Labels 
x = sonar_data.drop(columns=60, axis=1)  # Features
y = sonar_data[60]                        # Labels

# Print features and labels
print(x)
print(y)

#  Step 4: Split Data into Training and Test Sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, stratify=y, random_state=1
)

# Print dataset shapes
print(x.shape, x_train.shape, x_test.shape)

# Step 5: Train Logistic Regression Model 
model = LogisticRegression()
model.fit(x_train, y_train)  # Train the model with training data

#  Step 6: Evaluate the Model 
# Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on training data : ', training_data_accuracy)

# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on test data : ', test_data_accuracy)

# Step 7: Predict on New Data
input_data = (
    0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,
    0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,
    0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,
    0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,
    0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,
    0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044
)

# Convert input data to NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array for a single prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
prediction = model.predict(input_data_reshaped)
print(prediction)

# Display the result
if (prediction[0] == 'R'):
    print('The object is a Rock')
else:
    print('The object is a mine')
