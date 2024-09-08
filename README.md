# 💰 Loan Default Detection
This report is a coursework project that employs different algorithms to detect individual loan defaults. Curious about how lenders decide whether to approve loans, I chose this topic to leverage my knowledge and skills and explore this question through real-world applications.

# Introduction
### Datasource: https://www.kaggle.com/datasets/nanditapore/credit-risk-analysis/data
### Column Descriptions:
* ID: Unique identifier for each loan applicant.
* Age: Age of the loan applicant.
* Income: Income of the loan applicant.
* Home: Home ownership status (Own, Mortgage, Rent).
* Emp_Length: Employment length in years.
* Intent: Purpose of the loan (e.g., education, home improvement, medical, etc.).
* Amount: Loan amount applied for.
* Rate: Interest rate on the loan.
* Status: Loan approval status (Fully Paid, Charged Off, Current).
* Percent_Income: Loan amount as a percentage of income.
* Default: Whether the applicant has defaulted on a loan previously (Yes, No).
* Cred_Length: Length of the applicant's credit history.
<img width="816" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/b7100421-f502-4b1d-9121-8d96c9b3e641">

### Exploratory Data Analysis (EDA) on Tableau
<img width="700" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/03304c95-04a0-4acf-95b2-4c62f2fd75e1">


# Data Transformation and Resampling Using Python
- Class 1 (Default) and Class 0 (Non Default)
- Split data into 80% training (26,064) and 20% testing (6,517).
- Due to the high imbalance of class, I used SMOTE to resample data in the training set to avoid bias toward the majority class.
<img width="700" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/eaabe6d6-de18-45c8-8684-613fe06df78e">

# Trained Benchmark Models with Preprocessing
### Logistic Regression 
* Add LASSO regularization to minimize the impact of variables that have little influence on the model.
<img width="500" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/b6697ec8-9e92-4fae-b156-a06d9d197856">

### Random Forest
* Use GridSearch to find out the optimal parameters, including 'n_estimators', 'max_depth', 'min_samples_split', etc.
<img width="500" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/e3b7ef69-0891-4c15-a62a-d52926d7cdb5">

### XGBoost
* Add learning rate and max_depth. I did not adjust too many hyperparameters because I wanted to compare the difference between this model and the improved Random Forest model.
 <img width="500" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/a4f882a9-3372-4e08-b497-b4ec736d3002">

### Before and After Improvement
<img width="400" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/3000d17d-8f86-43c9-a650-34c2e2eab110"> <img width="400" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/efa1e23b-1f1d-4d34-a179-db00f9afd063">

### Ensemble Model 
* Combine the 3 models above with the voting method.
<img width="500" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/9d97fe6c-b1b4-4b64-bd66-b052942122d8">

# Conclusion
<img width="935" alt="image" src="https://github.com/wenchitseng/loan_default_detection/assets/145182368/1c154b6f-352b-4c88-a8dd-cc9fd41f0123">

If we only focus on testing accuracy, all five models perform similarly. However, in predicting loan defaults, it is more important to prioritize recall since the cost of missing an actual default is usually higher than the cost of incorrectly predicting a loan default.

**XGBoost has the highest recall and F1 score.** From the cost-metrix below, we achieved a 62% reduction(Total cost: $8,008,985 -> $3,160,103) in potential cost from false predictions using XGBoost model.








This practice leverages the knowledge and skills I acquired in the UCI BANA273 Machine Learning class, providing me with a more comprehensive understanding of real-world applications.






