# Claxton-Competions-Credit-Default-Prediction
A Credit Default projects based on the previous Transactions
# **Data Science Competition: Predicting Probability of Default**
# **Problem Statement**
Financial institutions face significant risks due to loan defaults. Accurately predicting the
probability of default (PD) on loans is critical for risk management and strategic planning. In this
competition, participants are tasked with developing a predictive model that estimates the
probability of default on loans using historical loan data.
# **Dataset Description**
The provided dataset contains historical information about borrowers, including various features
that may impact the probability of default.

# **Competition Tasks**
The competition runs from June 16, 2024 - July 20, 2024. Your submission should cover the following
aspects:
## **1. Data Cleaning:**
o Clean the dataset and explain your decisions regarding techniques used.
## **2. Basic EDA (Exploratory Data Analysis):**
o Explore the dataset to gain insights into feature distributions, correlations, and
potential patterns.

o Visualize key relationships and summarize your findings.

o Discuss any interesting observations.
## **3. Feature Selection:**
o Select relevant features for model training.

o Justify your feature selection methods (e.g., statistical tests, domain knowledge).
## **4. Hyperparameter Tuning:**
o Optimize hyperparameters for your chosen machine learning model(s).

o Explain the rationale behind your choices.
## **5. Cross Validation:**
o Implement cross-validation to assess model performance.

o Describe the cross-validation strategy.
o Report evaluation metrics.
## **6. Feature Scaling and Transformation:**
o Apply appropriate scaling and transformation techniques.'

o Discuss why you chose specific transformations.
# **7. Model Building:**
o Train at least 5 models.

o Explain your choice of algorithm.

o Discuss any model assumptions and limitations.
## **8. Model Evaluation:**
o Evaluate your model(s) on a separate validation set.

o Interpret performance metrics.
## **9. Endpoint Development for Inference:**
o Create API endpoints using Fast API.'

o Implement endpoints for model training and inference.

o Provide clear documentation on how to use these endpoints.
## **10. Data Drift Detection:**
o Implement a mechanism for detecting data drift in the deployed model.

o Explain why monitoring data drift is crucial for model maintenance.
## **11. Model Analysis:**
o Interpret model coefficients or feature importances.

o Investigate instances where the model performs poorly.

o Analyze the model for biases in predictions.

o Explain how the model is making predictions.

o Clearly communicate the limitations of the model, acknowledging situations where
it might not perform well.

o Propose potential enhancements or future directions for model improvement.







Analysis of Loan Default Factors

We could find that only three columns had missing values and eeach had a percenatge missing of less than 5% of the total number of observations in the dataset
>> Taking the percentages independently, 4.14+0.1+0.6 wont make it even 5 %, so next time in data cleaning, we consider droping these observations from those columns



We could find out that most data types are object data types with 11 features
>> followed by numeric data types with 9 features
>> lastly bool data types with the only one feeature












Loan Amount Distribution

The distribution of loan amounts reveals a pronounced right skew, characterized by a sharp peak near zero and a long tail extending towards higher values. The mode is concentrated around $10,000 to $20,000, indicating that most loans are relatively small. The majority of loans fall between $0 and $50,000, highlighting a tendency towards smaller loan amounts. However, the long tail signifies the presence of some very high loan amounts, with outliers extending beyond $200,000. Potential causes of loan default in this context include:

    High Loan Amounts: Borrowers taking out very large loans might face greater financial strain, increasing the likelihood of default.
    Inadequate Assessment: Financial institutions might not have adequately assessed the repayment capacity of borrowers taking out larger loans.
    Purpose of Loans: Large loan amounts could be linked to riskier investments or expenditures, which might not yield the expected returns, leading to default.

Number of Defaults Distribution

The distribution of the number of defaults reveals distinct, discrete peaks at zero, one, and two defaults, indicating a categorical nature of the data. The majority of cases have zero defaults, evidenced by the highest peak at zero, followed by a notable peak at one default, and a smaller but distinct peak at two defaults. This suggests that most individuals do not default on their loans, with a smaller proportion defaulting once, and an even smaller group defaulting twice. Potential causes of default here include:

    Financial Instability: Borrowers with one or two defaults may experience periods of financial instability or insufficient income to cover repayments.
    Credit History: Poor credit history or previous defaults can lead to difficulty in managing new loan obligations.
    Behavioral Factors: Repeat defaulters might have patterns of financial behavior that predispose them to default, such as poor budgeting or overreliance on credit.

Outstanding Balance Distribution

The distribution of outstanding balances is unimodal and right-skewed, with a sharp peak around $40,000, indicating this as the most common balance amount. Most outstanding balances are concentrated near this central value, but the distribution shows a long tail towards higher balances, with some instances exceeding $100,000. Potential causes of loan default include:

    High Outstanding Balances: Borrowers with high outstanding balances might struggle to manage large monthly payments, leading to default.
    Economic Changes: Economic downturns can disproportionately affect those with higher balances, as they may find it more challenging to adjust their repayments.
    Credit Management: Poor credit management and lack of financial planning might result in higher outstanding balances and increased default risk.

Distribution of Interest Rates

The distribution of interest rates shows a concentration of observations at lower rates (10%-17.5%) with outliers at higher rates (20%-30%). Potential causes of loan default include:

    High-Interest Rates: Borrowers with higher interest rates are often considered higher risk, making them more susceptible to default.
    Economic Vulnerability: Higher rates might be linked to borrowers in more volatile economic conditions or with less stable income.
    Mispriced Risk: If lenders underprice the risk associated with higher interest rates, there might be an accumulation of systemic risk, increasing the default probability during economic downturns.

Age Distribution

The distribution of age follows a typical bell-shaped curve, indicating a normal distribution of the population across different age groups. The graph displays a gradual increase in the count of individuals from the younger ages, reaching a peak around the 40-50 age range, and then gradually declining towards the older ages. This pattern suggests that the majority of the population falls within the middle-age cohort. Potential causes of loan default related to age distribution include:

    Young Borrowers: Younger borrowers (typically aged 20-30) may have less stable income or be in early stages of their careers, increasing default risk.
    Middle-Aged Borrowers: This group (typically aged 40-50) might have multiple financial responsibilities, such as mortgages, children's education, etc., which can strain their ability to repay loans.
    Older Borrowers: Borrowers approaching retirement (typically aged 60 and above) might have fixed incomes, limiting their ability to meet loan repayments, especially if they have not adequately planned for their financial needs in retirement.

Correlation Analysis

    Loan Amount and Outstanding Balance: A moderately strong positive correlation (0.56) indicates that larger loan amounts are associated with higher outstanding balances.
    Number of Defaults: A strong positive correlation (1.0) between the number of defaults and a related feature suggests they measure the same characteristic.
    Age: A strong positive correlation (1.0) between age and a related feature implies they represent the same age-related information.
    Salary: A moderate positive correlation with loan amount (0.54) and outstanding balance (0.34) suggests that higher salaries are associated with larger loan amounts and higher outstanding balances.
    Interest Rate: A low positive correlation with loan amount (0.22) and outstanding balance (0.15) indicates a relatively weak relationship between these variables.
    Other Correlations: Most other correlations are relatively low, suggesting the variables are largely independent or have only a weak relationship with each other.

Gender Distribution

The gender distribution among borrowers shows females accounting for 32,685 borrowers, males at 32,287, and a smaller "other" category at 35,028. This gender breakdown can provide valuable insights for assessing potential loan default patterns. Understanding the socioeconomic factors and behavioral differences that may impact repayment ability across genders could inform risk management strategies and help identify areas of potential bias or inequality in the loan approval process. By closely examining the relationship between gender and loan default, more inclusive and effective credit risk management practices can be developed to ensure fair and responsible lending while also optimizing the overall portfolio performance.
Loan Status Distribution

The distribution of loan status shows two distinct categories: "Did not default" with 85,134 loans, and "Defaulted" with 14,866 loans. This distribution provides valuable insight into portfolio performance and risk management. The significantly higher number of loans in the "Did not default" category suggests that the majority of borrowers were able to fulfill their repayment obligations, indicating a relatively healthy loan portfolio. However, the presence of 14,866 defaulted loans warrants closer examination to identify patterns or risk factors contributing to defaults. This analysis can enhance credit risk assessment, loan underwriting, and portfolio management strategies to mitigate future defaults and maintain overall stability.
Marital Status Distribution

The distribution of marital status among borrowers shows that the largest group is "married" with 44,710 borrowers, followed by "divorced" at 26,465, "single" at 25,698, and "other" at 3,127. This breakdown provides valuable insights into the relationship between marital status and loan repayment behavior. Key considerations include:

    Married Borrowers: May have greater financial stability and access to joint resources, positively impacting their ability to repay loans.
    Divorced Borrowers: May face unique financial challenges that could increase their risk of default.
    Single Borrowers: May have different income and expenditure patterns that should be accounted for.
    Other Category: May represent unique circumstances requiring further investigation.

By analyzing the distribution of marital status and its potential correlation with loan performance, credit risk models can be refined, underwriting processes tailored, and appropriate risk mitigation strategies implemented, leading to a more robust and sustainable loan portfolio.
Job Type Distribution

The distribution of job types among borrowers reveals that the largest group is "Engineer" with 16,524 borrowers, followed by "Nurse" with 15,284, "Analyst" with 13,204, "Doctor" with 12,186, and so on. Understanding the relationship between job type and loan repayment behavior can help assess credit risk more accurately. Key considerations include:

    Stable Incomes: Certain job types may be associated with higher and more stable incomes, positively impacting repayment ability.
    Economic Vulnerability: Some occupations may be more vulnerable to economic downturns or industry-specific challenges, increasing default risk.
    Concentration Risks: Identifying any concentration risks within specific job categories can help diversify the loan portfolio.
    Default Rates: Analyzing default rates and repayment patterns across different job types can refine credit assessment models and underwriting criteria.

By examining the distribution of job types and their correlation with loan performance, more informed decisions can be made, optimizing risk management practices and enhancing the overall health and sustainability of the loan portfolio.
Loan Amount Outliers

Several prominent outliers in the "loan_amount" feature, with some loan amounts reaching as high as around $200,000, are significantly larger than the main bulk of the loan amounts, which cluster around a much lower range. These large outliers could signal higher credit risk and a greater likelihood of loan defaults, suggesting that certain borrowers have been granted credit that exceeds their ability to repay or are engaging in riskier financial activities. Closely examining these outliers and understanding the factors contributing to their high loan amounts could provide valuable insights for assessing credit risk and developing appropriate risk management strategies.
Outstanding Balance Outliers

Several prominent outliers in the "outstanding_balance" feature, with some balances reaching as high as around $130,000, stand out from the main bulk of the data, which clusters around a much lower range. These large outliers could indicate higher risk and a greater likelihood of loan defaults within the portfolio, suggesting that certain borrowers have accumulated debts that exceed their ability to repay. Closely examining these outliers and understanding the factors contributing to their high outstanding balances could provide valuable insights for risk assessment and portfolio management strategies, helping to mitigate potential losses and maintain a healthy and sustainable loan portfolio.
Interest Rate Outliers

There are clear outliers in the "interest_rate" feature, with some rates reaching as high as around 28%. These outliers could indicate higher risk associated with certain borrowers or loan arrangements, suggesting that the lender has identified these borrowers as higher-risk due to factors like poor credit history or unstable income. Alternatively, the outliers could represent unique loan structures or special circumstances warranting elevated interest rates. Closely examining these outliers and understanding the underlying reasons behind them could provide valuable insights for risk assessment, pricing strategies, and risk mitigation measures, helping to maintain a balanced and sustainable loan portfolio.









Feature Engineering

Adding these new columns to the dataset enhances the ability to capture and analyze relationships between various factors, providing more granular insights into loan performance and risk assessment. Here’s a detailed explanation of each column and the reasoning behind its addition:
Loan-to-Income Ratio



AllData['LoanIncomeRatio'] = AllData['loan_amount'] / AllData['salary']

Reason: This ratio measures the proportion of a borrower’s income that is allocated towards the loan amount. A higher loan-to-income ratio may indicate a higher financial burden on the borrower, increasing the likelihood of default. This metric helps in assessing the borrower’s ability to manage and repay the loan based on their income.
Outstanding Balance-to-Income Ratio



AllData['OutstandingBalanceIncomeRatio'] = AllData['outstanding_balance'] / AllData['salary']

Reason: This ratio evaluates the proportion of a borrower’s income required to cover their outstanding balance. A higher ratio suggests that the borrower has a significant portion of their income tied up in repaying existing debt, which could indicate higher financial stress and risk of default.
Interest-to-Income Ratio



AllData['interestToIncomeRatio'] = AllData['interest_rate'] / AllData['salary']

Reason: This ratio assesses the proportion of a borrower’s income that goes towards paying interest. High interest-to-income ratios can indicate that the borrower might struggle to meet their interest obligations, especially if their income is relatively low compared to the interest rates they are paying.
Remaining Term to Age Ratio



AllData['remainingTermToAgeRatio'] = AllData['remaining term'] / AllData['age']

Reasons: This ratio compares the remaining term of the loan to the borrower’s age. It helps in understanding how the loan term fits within the borrower’s life stage. For example, a high ratio for an older borrower might indicate potential difficulties in repaying the loan as they approach retirement.
Default Rate



AllData['defaultRate'] = AllData['number_of_defaults'] / AllData['remaining term']

Reasoning: This rate calculates the frequency of defaults over the remaining term of the loan. A higher default rate indicates that the borrower has a history of defaults and might continue to default in the future. This metric is crucial for assessing the risk associated with the borrower’s repayment behavior over time.
Loan Amount and Interest Rate Interaction



AllData['loanInterestInteraction'] = AllData['loan_amount'] * AllData['interest_rate']

Reason: This interaction term captures the combined effect of loan amount and interest rate. High values may indicate loans that are both large and carry high-interest rates, potentially signifying higher risk loans. This feature helps in identifying borrowers who may be under significant financial stress due to both high loan amounts and high-interest rates.

Adding these columns to the dataset provides a more comprehensive view of the borrower’s financial situation and potential risk factors, enabling more precise and informed decision-making in credit risk assessment and loan management.




Data Cleaning 

In the data cleaning process, several columns were dropped for various reasons. 'Unnamed: 0' and 'loan_id' were removed as they are identifiers that do not contribute to the analysis or modeling. Duplicate columns like 'number_of_defaults.1' and 'age.1' were also dropped to avoid redundancy. The 'country' column was excluded as it may not add significant value or could introduce unnecessary complexity depending on the dataset's context. Other columns underwent specific actions to make them suitable for analysis and modeling. For instance, 'currency', 'sex', 'gender', 'is_employed', 'job', 'location', and 'marital_status' were label encoded to convert categorical data into a numerical format. 'Loan Status' was label encoded and null values removed to ensure data integrity, as it is a key variable in loan analysis. Missing values in 'salary' and 'loan_amount' were filled with the mean and median, respectively, to maintain data completeness without skewing the results. 'Number_of_defaults' was standardized to normalize the data for modeling purposes. Additional steps included removing duplicates to prevent data distortion, handling outliers in numerical columns to avoid skewed results, dropping columns with more than 50% missing values for reliability, and standardizing column names for consistency. These measures ensure a clean, robust dataset ready for accurate analysis and predictive modeling.


Feature engineering


In the feature engineering process, a comprehensive approach combining both filter and wrapper methods was employed to identify the most relevant features for the predictive modeling of loan defaults. The process involved several steps to ensure that the selected features have the highest predictive power. Here’s an explanation of the criteria for selection and the final features identified:
Criteria for Feature Selection

    Filter Methods:
        ANOVA F-value (SelectKBest with f_classif): This method selects features based on the ANOVA F-value between the feature and the target variable, identifying features with the most significant relationships.
        Mutual Information (SelectKBest with mutual_info_classif): This method measures the dependency between each feature and the target variable, selecting features that provide the most information about the target.
        Correlation: Features were evaluated based on their Pearson correlation with the target variable, and features with higher absolute correlation values were considered more significant.

    Wrapper Methods:
        Recursive Feature Elimination (RFE): This method recursively removes the least important features based on the model’s performance until the optimal number of features is reached.
        SelectFromModel (Random Forest): This method selects features based on their importance as determined by a RandomForestClassifier.
        Feature Importance (Random Forest): RandomForestClassifier was used to rank features based on their importance scores derived from the model.

    Hybrid Method Aggregation:
        The results from all the above methods were aggregated. Each feature's scores from different methods were averaged to compute a comprehensive importance score.
        The top features were selected based on their aggregated scores, ensuring a balanced consideration of multiple evaluation criteria.

Final Features

After applying the above methods and aggregating the results, the following features were selected as the top contributors to the predictive model:

    Interest Rate (Score: 18.20): The interest rate charged on the loan is a crucial determinant of the loan's affordability and the borrower's likelihood to default.
    Age (Score: 14.42): The borrower's age can be indicative of their financial stability and experience in managing debt.
    Loan-Interest Interaction (Score: 11.96): The interaction between the loan amount and interest rate captures the combined effect of these two critical factors on loan repayment.
    Loan Amount (Score: 10.81): The total amount borrowed can influence the borrower's ability to repay the loan, especially relative to their income and financial obligations.
    Number of Defaults (Score: 9.43): The borrower's past default history is a strong predictor of future default risk.
    Loan-to-Income Ratio (Score: 9.07): This ratio measures the loan amount relative to the borrower’s income, indicating the burden of the loan on their financial resources.
    Default Rate (Score: 8.55): The frequency of defaults over the remaining term provides insights into the borrower's reliability.
    Salary (Score: 4.40): The borrower’s income is a primary factor in their ability to meet loan payments.
    Remaining Term to Age Ratio (Score: 3.90): This ratio provides a perspective on how much time the borrower has left to repay the loan relative to their age, affecting their repayment strategy.
    Outstanding Balance (Score: 3.67): The current unpaid balance of the borrower’s other debts can impact their ability to service additional loans.

Explanation of Final Features

    Interest Rate: Reflects the cost of borrowing and impacts the total repayment amount, influencing the likelihood of default.
    Age: Older borrowers may have more stable financial histories, while younger borrowers might have higher default risks.
    Loan-Interest Interaction: Captures the compounded effect of borrowing costs and the principal amount on default risk.
    Loan Amount: Larger loans may be harder to repay, especially if not proportionate to the borrower's income.
    Number of Defaults: Past behavior is a strong predictor of future actions, making this a critical feature.
    Loan-to-Income Ratio: High ratios indicate higher financial strain, increasing default probability.
    Default Rate: Frequency of defaults can indicate borrower reliability.
    Salary: Higher income typically reduces the risk of default as borrowers are better able to meet repayment obligations.
    Remaining Term to Age Ratio: Aligns loan repayment period with borrower’s lifecycle, affecting their repayment capacity.
    Outstanding Balance: Indicates overall debt burden, impacting the borrower’s ability to manage new loans.













