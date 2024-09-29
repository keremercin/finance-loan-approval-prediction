
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Data Preprocessing
train_data['LoanAmount'].fillna(train_data['LoanAmount'].median(), inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].median(), inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)

label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    train_data[col] = label_encoder.fit_transform(train_data[col])

scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

# Feature Engineering
train_data['TotalIncome'] = train_data['ApplicantIncome'] + train_data['CoapplicantIncome']
train_data['Income_to_Loan_Ratio'] = train_data['TotalIncome'] / (train_data['LoanAmount'] + 1)
train_data[['TotalIncome', 'Income_to_Loan_Ratio']] = scaler.fit_transform(train_data[['TotalIncome', 'Income_to_Loan_Ratio']])

# Train-test split
X = train_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_data['Loan_Status']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_val)
logreg_acc = accuracy_score(y_val, y_pred_logreg)

# Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
rf_acc = accuracy_score(y_val, y_pred_rf)

# Gradient Boosting Model
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_val)
gb_acc = accuracy_score(y_val, y_pred_gb)

# Print results
print(f'Logistic Regression Accuracy: {logreg_acc}')
print(f'Random Forest Accuracy: {rf_acc}')
print(f'Gradient Boosting Accuracy: {gb_acc}')
