# %%
# Data Collection, Data Cleaning & Data Manipulation
import numpy as np
import pandas as pd
from sklearn import datasets

# Data Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Data Transformation
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats

# Models Building
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

## Classification Problems
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

## Regression Problems
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Explainbale AI (XAI)

import lime.lime_tabular
import shap
from sklearn.linear_model import LinearRegression
import pickle

# Unsupervised Learning: Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, MeanShift, SpectralClustering
from sklearn.metrics import adjusted_rand_score
import warnings 
warnings.filterwarnings('ignore')
# %%
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="D:/DS/Semester_1/MachineLearning/TermProject/Churn_Dataset/Telecom_Churn_Dataset.csv"):
        self.data = pd.read_csv(path)
        #Changing Datatypes
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce').astype(float)
        # Drop id as it is not relevant
         # Impute missing values of BMI
        self.data.TotalCharges = self.data.TotalCharges.fillna(0)
        self.data.drop(["customerID"], axis=1, inplace=True)
        # Mapping the Target Variable
        target_mapping = {"Yes": 1, "No": 0}
        self.data['Churn'] = self.data['Churn'].map(target_mapping)

    def standardize(self):

        # Standardization 
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        features = self.data.drop('Churn', axis=1)
        target = self.data['Churn']

        # Separate the numerical and categorical columns
        numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = features.select_dtypes(include=['object']).columns

        # Define the feature scaling transformer
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])

        # Define the column transformer to apply different transformations to numerical and categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols)
                # Add more transformers if needed for categorical or other types of columns
            ])

        # Apply the preprocessing steps to the features only
        transformed_data = preprocessor.fit_transform(features)

        # If you want to convert the transformed data back to a DataFrame
        self.data = pd.DataFrame(transformed_data, columns=numerical_cols)
        self.data[categorical_cols] = features[categorical_cols]  # Include the categorical columns as they are
        self.data['Churn'] = target  # Include the target variable in the transformed DataFrame
        
    def preprocess_data(self):
        # One-hot encode all categorical columns
        categorical_cols = ["gender",
                            "Partner",
                            "Dependents",
                            "PhoneService",
                            "MultipleLines",
                            "InternetService",
                            "OnlineBackup",
                            "OnlineSecurity",
                            "DeviceProtection",
                            "TechSupport",
                            "StreamingTV",
                            "StreamingMovies",
                            "Contract",
                            "PaperlessBilling",
                            "PaymentMethod"]
        
        encoded = pd.get_dummies(self.data[categorical_cols],
                            prefix=categorical_cols,
                            dtype='uint8')  # Set dtype to 'uint8'
        
        # Update data with new columns
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(categorical_cols, axis=1, inplace=True)
        

    def get_data_split(self):
        X = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]
        return train_test_split(X, y, test_size=0.20, random_state=2021)
    
    def oversample(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over
# %%
import matplotlib.pyplot as plt


# Load data
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data

# Show head
print(data.shape)
data.head()
# %%
#  Show general statistics
data_loader.data.info()
# %%
# Show histogram for all columns
columns = data.columns
for col in columns:
    print("col: ", col)
    data[col].hist()
    plt.show()
# %%
# Show Standardize dataframe
data_loader.standardize()
data_loader.data.head()
#%%
data_loader.data.info()
# %%
data_loader.preprocess_data()
data_loader.data.head()
# %%
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
print(X_train.shape)
print(X_test.shape)
# %%
# Feature Importance
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Get feature importance scores
feature_importances = clf.feature_importances_

# Create a dataframe to store feature importance
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the dataframe by importance scores in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance scores
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
# %%
from interpret.glassbox import (LogisticRegression,
                                ClassificationTree, 
                                ExplainableBoostingClassifier)
from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from interpret.blackbox import LimeTabular
import interpret
from interpret import show

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
# %%
import lime
import lime.lime_tabular
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score

# Assuming you have your features and labels in X and y respectively

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = data_loader.get_data_split()

# Create a Gradient Boosting Classifier with specified hyperparameters
gbc = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None, learning_rate=0.1, loss='log_loss', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_iter_no_change=None, random_state=4432, subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0, warm_start=False)

# Train the model on the training data
gbc.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = gbc.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Calculate the recall score
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate the precision score
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate ROC curve and AUC score
y_prob = gbc.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc)

# %%
# Initilize Lime for Tabular data
lime = LimeTabular(model=gbc.predict_proba, 
                   data=X_train, 
                   random_state=1)
# Get local explanations
lime_local = lime.explain_local(X_test[-20:], 
                                y_test[-20:], 
                                name='LIME')

show(lime_local)
# %%
from lime import lime_tabular
# Train a logistic regression model
lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=1000, multi_class='auto', n_jobs=None, penalty='l2', random_state=4733, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
lr.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")
auc = roc_auc_score(y_test, y_pred)
print("AUC Score:", auc)


# %%
# Initilize Lime for Tabular data
lime = LimeTabular(model=gbc.predict_proba, 
                   data=X_train, 
                   random_state=1)
# Get local explanations
lime_local = lime.explain_local(X_test[-3:], 
                                y_test[-3:], 
                                name='LIME')

show(lime_local)
##filename = 'Churn_LIME.sav'
#pickle.dump(lime_local, open(filename, 'wb'))

#%%
from lime.lime_tabular import LimeTabularExplainer

# Initialize Lime for Tabular data
lime = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=["0", "1"], random_state=1)

# Get local explanations
lime_local = lime.explain_local(X_test[-10:], y_test[-10:], name='LIME')

# Get the predicted probabilities
predicted_probabilities = lime_local.predict_proba

# Loop through the instances and retrieve the predicted probabilities individually
for i, instance in enumerate(lime_local.data):
    predicted_probability = predicted_probabilities[i]
    # Display or use the predicted probability as desired in your Streamlit dashboard
    print(f"Instance {i+1} Predicted Probability: {predicted_probability}")

# Save the LIME local explanations
filename = 'Churn_LIME_Predict.bin'
pickle.dump(lime_local, open(filename, 'wb'))




#%%
from lime.lime_tabular import LimeTabularExplainer

# Initialize Lime for Tabular data
lime = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=["0", "1"], random_state=1)

# Get local explanations
num_instances = 10
for i in range(num_instances):
    explanation = lime.explain_instance(X_test.iloc[i], gbc.predict_proba, num_features=10)
    # Access the predicted probabilities
    predicted_probabilities = explanation.predict_proba
    # Display or use the predicted probabilities as desired in your Streamlit dashboard
    print(f"Instance {i+1} Predicted Probability: {predicted_probabilities[1]}")

# Save the Lime local explanations
filename = 'Churn_LIME_Predict.bin'
pickle.dump(explanation, open(filename, 'wb'))






# %%
import dice_ml
data_dice = dice_ml.Data(dataframe=data_loader.data, 
                         # For perturbation strategy
                         continuous_features=[ 
                                              'tenure',
                                              'MonthlyCharges',
                                              'TotalCharges'], 
                         outcome_name='Churn')
# Model
model_dice = dice_ml.Model(model=gbc, 
                        # There exist backends for tf, torch, ...
                        backend="sklearn")
explainer = dice_ml.Dice(data_dice, 
                         model_dice, 
                         # Random sampling, genetic algorithm, kd-tree,...
                         method="random")
print ('Done')



# %% Create explanation
# Generate CF based on the blackbox model
input_datapoint = X_test[90:91]
cf = explainer.generate_counterfactuals(input_datapoint, 
                                  total_CFs=1, 
                                  desired_class="opposite")
# Visualize it
cf.visualize_as_dataframe(show_only_changes=True)


# %% Create feasible (conditional) Counterfactuals
input_datapoint = X_test[1047:1049]
features_to_vary=[
                 'tenure',
                 'MonthlyCharges',
                'TotalCharges']

permitted_range={'tenure':[1,250],
                'TotalCharges':[0, 1000]}
# Now generating explanations using the new feature weights
cf = explainer.generate_counterfactuals(input_datapoint, 
                                  total_CFs=1, 
                                  desired_class="opposite",
                                  permitted_range=permitted_range,
                                  features_to_vary=features_to_vary)
# Visualize it
cf.visualize_as_dataframe(show_only_changes=True)
# Save the Dice Counterfactuals object
pickle.dump(cf, open('dice_counterfactuals.bin', 'wb'))



# %%
from dice_ml.utils import helpers
#from dice_ml.model_interfaces import TensorflowKerasModel
from dice_ml import Dice

new = pd.DataFrame(data_loader.data)

# Specify the continuous features and the outcome name
continuous_features = [
                 'tenure',
                 'MonthlyCharges',
                'TotalCharges']
outcome_name = 'Churn'
# Create an instance of the Data class
data_dice = dice_ml.Data(dataframe=new, continuous_features=continuous_features, outcome_name=outcome_name)

rf_dice = dice_ml.Model(model=gbc, 
                        # There exist backends for tf, torch, ...
                        backend="sklearn")
explainer = dice_ml.Dice(data_dice, 
                         rf_dice, 
                         # Random sampling, genetic algorithm, kd-tree,...
                         method="random")

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %%
import shap
import numpy as np

explainer = shap.TreeExplainer(rf)
# Calculate shapley values for test data
start_index = 1
end_index = 2
shap_values = explainer(X_test[start_index:end_index])
X_test[start_index:end_index]
# class 0 = contribution to class 1
# class 1 = contribution to class 2
print(shap_values[0].shape)
shap_values

# %%
shap.initjs()
# Force plot
prediction = rf.predict(X_test[start_index:end_index])[0]
print(f"The RF predicted: {prediction}")
shap.force_plot(explainer.expected_value[1],
                shap_values[1],
                X_test[start_index:end_index]) # for values
# Feature summary
shap.summary_plot(shap_values, X_test)
# %%
