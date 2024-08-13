#!/usr/bin/env python
# coding: utf-8

# # Loan Default Detection
# 
# Dataset from Kaggle: https://www.kaggle.com/datasets/nanditapore/credit-risk-analysis/data
# 
# This is a credit risk analysis dataset we found on Kaggle. This dataset provides essential information about loan applicants and their characteristics, including their loan rate, income, age, credit length, etc.
# This dataset provides a simplified view of the factors contributing to credit risk, presenting an excellent opportunity for us to apply our machine learning analysis in determining whether a loan applicant is likely to default.
# 
# 
# Column Descriptions:
#     
# * ID: Unique identifier for each loan applicant.
# * Age: Age of the loan applicant.
# * Income: Income of the loan applicant.
# * Home: Home ownership status (Own, Mortgage, Rent).
# * Emp_Length: Employment length in years.
# * Intent: Purpose of the loan (e.g., education, home improvement, medical, etc.).
# * Amount: Loan amount applied for.
# * Rate: Interest rate on the loan.
# * Status: Loan approval status (Fully Paid, Charged Off, Current).
# * Percent_Income: Loan amount as a percentage of income.
# * Default: Whether the applicant has defaulted on a loan previously (Yes, No).
# * Cred_Length: Length of the applicant's credit history.

# ## Import Data

# In[1]:


import pandas as pd


# In[2]:


# Load the dataset
df = pd.read_csv('/Users/cengwenqi/Library/CloudStorage/OneDrive-UCIrvine/credit_risk.csv')
df.head()


# In[3]:


print(df.info())


# ## Data Transformation

# In[4]:


# We use pd.get_dummies function to transform our categorical columns using dummy variables

df_encoded = pd.get_dummies(df, columns=["Home", "Intent"], drop_first=True)
df_encoded['Default'] = [1 if i == "Y" else 0 for i in df['Default']]
df_encoded.head()


# In[5]:


df_encoded.isnull().sum()


# In[6]:


# We use SimpleImputer to fill in the missing values with mean
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)


# In[7]:


df_imputed.isnull().sum()


# In[8]:


df_cleaned = df_imputed.drop(["Id"], axis=1)
df_cleaned.head()


# In[9]:


occ = df_cleaned['Default'].value_counts()
print( occ)

print(len(df))
print(occ/ len(df))


# In[10]:


import matplotlib.pyplot as plt

occ.plot(kind='bar', color='tab:gray', alpha = 0.7)
plt.title('Histogram of Class Column')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # Keep the class labels horizontal for readability
plt.show()


# ## Data Resampling

# In[11]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score,
)

from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV



# In[35]:


# Split data into feature(X) and target(Y)
X =df_cleaned.drop('Default', axis=1)
Y = df_cleaned['Default']


# Create training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Training set: ", len(X_train))
print("Testing set: ", len(X_test))


# In[36]:


# Resample the training data
method = SMOTE()
X_resampled, Y_resampled = method.fit_resample(X_train, Y_train)


# In[37]:


original = Y_train.value_counts()
resampled = Y_resampled.value_counts()


# In[183]:


def plot_histograms(data1, data2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original dataset
    ax[0].bar(data1.index, data1.values, color='tab:gray', alpha=0.7)
    ax[0].set_title('Original Training Dataset')
    ax[0].set_xticks([0, 1])
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Count')

    # Resampled dataset
    ax[1].bar(data2.index, data2.values, color='tab:olive', alpha=0.7)
    ax[1].set_title('Resampled Training Dataset')
    ax[1].set_xticks([0, 1])
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

# Plot the histograms
plot_histograms(original, resampled)


# ## Benchmark Model Training

# ### Logistice Regression

# In[39]:


# Fit a logistic regression model to our data
log_model = LogisticRegression()
log_model.fit(X_resampled, Y_resampled)

# Get prediction
log_model_pred = log_model.predict(X_test)


# ##### Classification Metrix

# In[40]:


# Classification report
print(f"Classification Report:\n{classification_report(Y_test, log_model_pred)}")
print(f"Accuracy: {accuracy_score(Y_test, log_model_pred)}")


# In[41]:


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(Y_test, log_model_pred, cmap = plt.cm.Oranges, normalize = None, display_labels = ['0', '1'])


# ### Rondom Forest

# In[42]:


# Fit a random forest model to our data
rf_model = RandomForestClassifier()
rf_model.fit(X_resampled, Y_resampled)

# Get prediction
rf_model_pred = rf_model.predict(X_test)


# ##### Classification Metrix

# In[43]:


# Classification report
print(f"Classification Report:\n{classification_report(Y_test, rf_model_pred)}")
print(f"Accuracy: {accuracy_score(Y_test, rf_model_pred)}")


# In[44]:


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(Y_test, rf_model_pred, cmap = plt.cm.Oranges, normalize = None, display_labels = ['0', '1'])


# ### XGBoost

# In[45]:


# Fit a xgboost model to our data
xg_model = XGBClassifier()
xg_model.fit(X_resampled, Y_resampled)

# Get prediction
xg_model_pred = xg_model.predict(X_test)


# ##### Classification Metrix

# In[46]:


# Classification report
print(f"Classification Report:\n{classification_report(Y_test, xg_model_pred)}")
print(f"Accuracy: {accuracy_score(Y_test, xg_model_pred)}")


# In[47]:


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(Y_test, xg_model_pred, cmap = plt.cm.Oranges, normalize = None, display_labels = ['0', '1'])


# ## ROC for Benchmark Models

# In[184]:


pred1 = rf_model.predict_proba(X_test)
pred2 = log_model.predict_proba(X_test)
pred3 = xg_model.predict_proba(X_test)


# In[185]:


fpr1, tpr1, _ = roc_curve(Y_test, pred1[:, 1])
fpr2, tpr2, _ = roc_curve(Y_test, pred2[:, 1])
fpr3, tpr3, _ = roc_curve(Y_test, pred3[:, 1])

auc1 = auc(fpr1, tpr1)
auc2 = auc(fpr2, tpr2)
auc3 = auc(fpr3, tpr3)


# In[186]:


plt.figure()

plt.plot(fpr1, tpr1, color='tab:blue', lw=2, label='Random Forest (AUC = %0.2f)' % auc1, alpha = 0.7)
plt.plot(fpr2, tpr2, color='tab:green', lw=2, label='Logistic Regression (AUC = %0.2f)' % auc2, alpha = 0.7)
plt.plot(fpr3, tpr3, color='tab:orange', lw=2, label='XGBoost (AUC = %0.2f)' % auc3, alpha = 0.7)

plt.plot([0, 1], [0, 1], color='tab:gray', lw=2, linestyle='--', alpha = 0.7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Benchmark Models')
plt.legend(loc="lower right")
plt.show()


# 

# ## Improvement on Benchmark Models

# In[79]:


from sklearn.preprocessing import StandardScaler

# Scale the features to ensure they have similar scales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ### Logistic Regression

# In[87]:


# Utilize lasso (l1) regularization and the C parameters
new_log_model = LogisticRegression(penalty='l1', C=0.05, solver='liblinear', random_state=42)
new_log_model.fit(X_train_scaled, Y_train)


# In[93]:


# Get prediction
new_log_model_pred = new_log_model.predict(X_test_scaled)


# #### Classification Metrix

# In[94]:


# Classification report
print(f"Classification Report:\n{classification_report(Y_test, new_log_model_pred)}")
print(f"Accuracy: {accuracy_score(Y_test, new_log_model_pred)}")


# In[95]:


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(Y_test, new_log_model_pred, cmap = plt.cm.Oranges, normalize = None, display_labels = ['0', '1'])


# ### Random Forest

# #### Grid Search

# In[115]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, Y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)


# Best Parameters: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}

# In[136]:


# Input the optimal parameters in the model
new_rf_model = RandomForestClassifier(class_weight={0:1, 1:8}, max_depth = 15, min_samples_leaf = 1, min_samples_split= 5, n_estimators = 200, random_state=42)
new_rf_model.fit(X_train_scaled, Y_train)


# In[137]:


# Get prediction
new_rf_model_pred = new_rf_model.predict(X_test_scaled)


# #### Classification Metrix

# In[138]:


# Classification report
print(f"Classification Report:\n{classification_report(Y_test, new_rf_model_pred)}")
print(f"Accuracy: {accuracy_score(Y_test, new_rf_model_pred)}")


# In[139]:


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(Y_test, new_rf_model_pred, cmap = plt.cm.Oranges, normalize = None, display_labels = ['0', '1'])


# ### XGBoost

# In[143]:


# This makes it so the "1" class is more weighted than the 0 since our model
# has trouble predicting the "1" class
scale_pos_weight = (len(Y_train) - sum(Y_train)) / sum(Y_train)


# In[144]:


# Implementing our new model on the resampled and scaled data
new_xgb_model = XGBClassifier(learning_rate = 0.05, scale_pos_weight=3, random_state=42, max_depth = 4)
new_xgb_model.fit(X_train_scaled, Y_train)



# In[146]:


# Getting our new prediction
new_xgb_model_pred = new_xgb_model.predict(X_test_scaled)


# #### Classification Metrix

# In[147]:


# Classification report
print(f"Classification Report:\n{classification_report(Y_test, new_xgb_model_pred)}")
print(f"Accuracy: {accuracy_score(Y_test, new_xgb_model_pred)}")


# In[148]:


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(Y_test, new_xgb_model_pred, cmap = plt.cm.Oranges, normalize = None, display_labels = ['0', '1'])


# ## ROC for Improved Models

# In[153]:


pred1 = new_rf_model.predict_proba(X_test_scaled)
pred2 = new_log_model.predict_proba(X_test_scaled)
pred3 = new_xgb_model.predict_proba(X_test_scaled)


# In[154]:


fpr1, tpr1, _ = roc_curve(Y_test, pred1[:, 1])
fpr2, tpr2, _ = roc_curve(Y_test, pred2[:, 1])
fpr3, tpr3, _ = roc_curve(Y_test, pred3[:, 1])

auc1 = auc(fpr1, tpr1)
auc2 = auc(fpr2, tpr2)
auc3 = auc(fpr3, tpr3)


# In[155]:


plt.figure()

plt.plot(fpr1, tpr1, color='tab:blue', lw=2, label='Random Forest (AUC = %0.2f)' % auc1, alpha = 0.7)
plt.plot(fpr2, tpr2, color='tab:green', lw=2, label='Logistic Regression (AUC = %0.2f)' % auc2, alpha = 0.7)
plt.plot(fpr3, tpr3, color='tab:orange', lw=2, label='XGBoost (AUC = %0.2f)' % auc3, alpha = 0.7)

plt.plot([0, 1], [0, 1], color='tab:gray', lw=2, linestyle='--', alpha = 0.7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Improved Models')
plt.legend(loc="lower right")
plt.show()


# ## Ensemble Method

# ### Define the three classifiers to use in the ensemble

# In[174]:


from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(penalty='l1', C=0.05, solver='liblinear', random_state=42)
clf2 = RandomForestClassifier(class_weight={0:1, 1:8}, max_depth = 15, min_samples_leaf = 1, min_samples_split= 5, n_estimators = 200, random_state=42)
clf3 = XGBClassifier(learning_rate = 0.05, scale_pos_weight=3, random_state=42, max_depth = 4)


# #### Hard voting

# In[175]:


# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('xgb', clf3)], voting='hard')

ensemble_model.fit(X_train_scaled, Y_train)


# In[176]:


ensemble_model_pred = ensemble_model.predict(X_test_scaled)


# #### Classification Metrix

# In[177]:


# Classification report
print(f"Classification Report:\n{classification_report(Y_test, ensemble_model_pred)}")
print(f"Accuracy: {accuracy_score(Y_test, ensemble_model_pred)}")


# In[178]:


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(Y_test, ensemble_model_pred, cmap = plt.cm.Oranges, normalize = None, display_labels = ['0', '1'])


# ### Adjust Weights within the Voting Classifier

# In[179]:


ensemble_model_weight = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('xgb', clf3)], voting='soft', weights=[1, 1, 5], flatten_transform=True)

ensemble_model_weight.fit(X_train_scaled, Y_train)


# In[180]:


ensemble_model_pred_weight= ensemble_model_weight.predict(X_test_scaled)


# In[181]:


# Classification report
print(f"Classification Report:\n{classification_report(Y_test, ensemble_model_pred_weight)}")
print(f"Accuracy: {accuracy_score(Y_test, ensemble_model_pred_weight)}")


# ## Conclusion

# | Model | Precision for Class 1 (Default) | Recall for Class 1 (Default) | F1 for Class 1 (Default) | Testing Accuracy
# | :---: | :-----------------------------: | :--------------------------: | :----------------------: | :--------------: |
# | Logistic Regression | 0.52 | 0.28 | 0.37 | 0.82|
# | Random Forest | 0.52 | 0.84 | 0.64 | 0.84 |
# | XGBoost | 0.52 | 0.87 | 0.65 | 0.83 |
# | Ensemble Model (hard voting) | 0.52 | 0.83 | 0.64 | 0.83 | 
# | Ensemble Model (adjust weights within voting) |0.52 | 0.84 | 0.64 | 0.83

# If we only focus on testing accuracy, all five models perform similarly. However, in predicting loan defaults, it is more important to prioritize recall since the cost of missing an actual default is usually higher than the cost of incorrectly predicting a loan default.
# 
# XGBoost has the highest recall and F1 score. This is reasonable because logistic regression models the relationship between the features and the log odds of the target variable as a linear combination of features, limiting its performance when the relationship is not linear. This limitation is reflected in the table's low recall and F1-score of the logistic regression model. The Random Forest model, also based on decision trees, builds trees independently in parallel. However, XGBoost improves on this by boosting the training process to correct the errors of previous trees.
# 
# In conclusion, although building an ensemble model usually results in better performance than individual models, in this case, XGBoost outperforms the others and is likely the most suitable model. This practice leverages the knowledge and skills I acquired in the UCI BANA273 Machine Learning class, providing me with a more comprehensive understanding of real-world applications.

# 

# 
