Sonar Rock vs Mine Project
Project Idea

This project uses sonar readings to detect whether an object detected by a submarine is a Rock (R) or a Mine (M).

We train a Logistic Regression model with past sonar data and then test it with new readings to predict the type of object. It’s a simple but effective binary classification problem.

Steps in the Project
1. Import Libraries

We use:

numpy for numerical operations and array handling

pandas for working with the dataset

scikit-learn for splitting data, training the model, and checking accuracy

2. Load and Explore the Data

Dataset is a CSV file (Copy of sonar data.csv) without a header

We check the first few rows, dataset shape, statistical summary, and label distribution (M vs R)

Group data by labels to see average values for each feature

3. Separate Features and Labels

Features (X): All columns except the last column

Labels (Y): Last column (60) contains M or R

4. Split Data into Training and Test Sets

Training data: 90% of the dataset

Test data: 10% of the dataset

Stratified split ensures equal distribution of Mines and Rocks

5. Train the Logistic Regression Model

Logistic Regression is used because it works well for binary classification

Model is trained on the training data

6. Check Accuracy

Training Accuracy: How well the model predicts on training data

Test Accuracy: How well the model predicts on unseen data

7. Make Predictions with New Data

Provide 60 sonar readings for a new object

Convert the input to a NumPy array and reshape for prediction

Model outputs either R (Rock) or M (Mine)

Print a clear result to the user

Example
Input: 0.0453,0.0523,... (60 values)
Output: The object is a Rock.

Notes

More data improves the model’s accuracy

Logistic Regression is a supervised learning model

This project can be extended with a GUI (Tkinter or Streamlit) to enter values more easily