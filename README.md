
## CUSTOMER-CHURN-PREDICTION-AND-ANALYSIS-MACHINE LEARNING MODEL

#### Overview

This repository contains a Jupyter Notebook that analyses customer churn and implements predictive models to improve retention strategies. The analysis explores various machine-learning techniques to predict customer churn based on historical data.

In this project, I aim to find the likelihood of a customer leaving the organization, the key indicators of churn as well as the retention strategies that can be implemented to avert this problem.


- Create the project environment (Repository, virtual environment, version control)
1. Business Understanding
    - Problem Statement
    - Goal and Objectives
    - Stakeholders
    - Key Metrics and Success Criteria
    - Hypothesis (Null and Alternate)
    - Analytical Questions
    - Scope and Constraints
    - More Information about the project if applicable
2. Data Understanding
    1. Importation
    2. Load Dataset
    3. EDA
        1. Data Quality Assessment & Exploring data (info, duplicated, null values, describe)
        2. Univariate
        3. Bi-variate
        4. Multi-variate Analysis
    4. Answer Analytical questions with visualizations
    5. Test hypothesis
    6. Give your insights
3. Data Preparation
    1. Split data set into X, y
    2. Split data set into training and evaluation
    3. Feature Engineering (Creating New Features, (binning & bucketing), Handling Missing Data, Encoding, Standardization, 
        Normalization, Scaling)
      a. Create a pipeline to preprocess the data
        1. Separate impute features into numeric and categorical for different pipelines
        2. Handle missing values using imputation Techniques
        3. Scaling or normalising numeric features
        4. Encode categorical features
        5. Transformations for skewed data (log, power, custom, etc)
        6. Balance dataset (depending on what you see)
    4. Encode label/target
4. Modeling 
    1. Fit data to the pipeline and train Model
        a.Train Model 1 - Distance-based model
        b.Train Model 2 - Gradient Descent model
        c.Train Model 3 - Tree-based model
        c.Train Model 4 - Neural Network
5. Evaluation
    1. Advanced model Evaluation and Visualizing Model Performance: (ROC curves, AUC, precision-recall curves, and confusion matrices).
    2. Feature Importance and Selection
    3. Hyperparameter Tuning: (GridSearchCV, RandomizedSearchCV, or Bayesian optimization.)
    4. Final Model Comparison & Selection: (Consider trade-offs and choose most appropriate)
    5. Retrain Model with best parameters
    5. Business Impact Assessment and Documentation of the Model (How does what you choose affect the business?)
    6. Model Persistence
6. Deployment (Not applicable in this project)
    1. Deployment Strategy
    2. Model Monitoring and Maintenance

## SETUP
It is recommended to have a Jupyter Noteotebook or any other standard code editor on your local machine.
You can go ahead and install the required packages locally to your computer.

It would be best for you to run a Python version above 3.0. You can download the required python version for mac [here](https://www.python.org/downloads/macos/)

Use these recommended steps to set up your local machine for this project:

#### Create the Python's virtual environment :
This will isolate the required libraries of the project to avoid conflicts.
Choose any of the line of code that will work on your local machine.

     python3 -m venv venv
     python -m venv venv


#### Activate the Python's virtual environment :
This will ensure that the Python kernel & libraries will be those of the created isolated environment.

     - for Windows : 
                  venv\Scripts\activate

     - for Linux & MacOS :
                  source venv/bin/activate


Upgrade Pip :
Pip is the installed libraries/packages manager. Upgrading Pip will give an to up-to-date version that will work correctly.

     python -m pip install --upgrade pip

#### Install the required libraries/packages :
Some libraries and packages are required for this project. These libraries and packages are listed in the requirements.txt file.
The text file will allow you to import these libraries and packages into the python's scripts and notebooks without any issue.

     python -m pip install -r requirements.txt 


#### Project Structure and Description

The telecommunication industry is fiercely competitive, with companies striving to capture and retain customer loyalty. 
One major challenge they face is customer churn, which is the rate at which customers discontinue their services and move to competitors. Customer churn can significantly affect a company's revenue and profitability, as acquiring new customers is generally more expensive than keeping existing ones. 
To tackle this issue in this project, I aim to build a classification model to analyze customer churn for a company. 
By understanding the factors influencing customer churn, the company can implement strategies to improve customer retention and increase profit margins. 
The project follows the CRISP-DM methodology to ensure a structured and thorough approach.

•	Business Understanding

•	Data Understanding

•	Data Preparation

•	Modeling

•	Evaluation

•	Deployment


 #### Business Goals and Objective

The main goal of this churn analysis project is to assist the telecommunication company in reducing customer churn and improving customer retention through data-driven insights and predictive modeling, ultimately leading to increased revenue and profitability.


 #### Problem Statement
The telecommunication company is experiencing a significant rate of customer churn, which negatively impacts its revenue and profitability. Despite efforts to retain customers, the company lacks a comprehensive understanding of the factors driving churn and the effectiveness of current retention strategies. This project aims to identify the key drivers of customer churn and develop predictive models to forecast churn risk, enabling the company to implement targeted interventions to retain customers and improve overall business performance.


 #### Stakeholders

1.	**Executive Management**

Interested in overall business performance, revenue growth, and profitability. They need high-level insights and strategic recommendations based on the churn analysis to make informed decisions.

2.	**Customer Service and Support Teams**

Focused on improving customer satisfaction and retention. They require detailed analysis of churn drivers and actionable recommendations to enhance customer interactions and reduce churn rates.

3.	**Marketing and Sales Departments**

Responsible for customer acquisition and retention strategies. They need insights into customer behavior, preferences, and churn predictors to design effective marketing campaigns and loyalty programs.

4.	**Data Science and Analytics Teams**

Tasked with conducting the churn analysis and developing predictive models. They require access to comprehensive data, advanced analytical tools, and collaboration with other departments to ensure accurate and actionable insights.

5.	**Product Development Teams**

Involved in improving and innovating the company’s service offerings. They need feedback on how product features and service quality impact customer retention and what changes could reduce churn.

6.	**Finance Department**

Concerned with the financial implications of customer churn. They need to understand the cost of churn, the potential ROI of retention strategies, and how churn affects financial forecasts.

7.	**IT and Infrastructure Teams**

Responsible for maintaining and improving the technology systems that support customer interactions and data analysis. They need to ensure that the necessary infrastructure is in place to support data-driven initiatives for churn reduction.


 #### Key Metrics and Success Criteria

1.	**Churn Rate**

The percentage of customers who discontinue their service within a specified period.

Success Criteria: A measurable reduction in the churn rate over time, indicating improved customer retention.

2.	**Customer Lifetime Value (CLV)**

The total revenue a company can expect from a customer over the duration of their relationship.

Success Criteria: An increase in CLV, reflecting better customer retention and higher profitability.

3.	**Retention Rate**

The percentage of customers who continue to use the service over a specified period.

Success Criteria: An increase in retention rate, demonstrating the effectiveness of retention strategies.

4.	**Customer Satisfaction Score (CSAT)**

A metric that measures customer satisfaction with a company’s products or services.

Success Criteria: Higher CSAT scores, indicating improved customer experiences and satisfaction.

5.	**Net Promoter Score (NPS)**

A metric that measures customer loyalty and the likelihood of customers recommending the company’s services to others.

Success Criteria: An increase in NPS, showing stronger customer loyalty and positive word-of-mouth.

6.	**Predictive Model Accuracy**

The accuracy of the predictive models in identifying customers at risk of churning.

Success Criteria: High accuracy rates (e.g., precision, recall, F1 score) for the predictive models, ensuring reliable identification of at-risk customers.

7.	**Reduction in Customer Acquisition Costs (CAC)**

The cost associated with acquiring new customers.

Success Criteria: A decrease in CAC, indicating that retaining existing customers is more cost-effective than acquiring new ones.

8.	**Engagement Metrics**

Metrics that measure customer interactions with the company’s services (e.g., usage frequency, active user rates).

Success Criteria: Increased engagement metrics, reflecting higher customer involvement and satisfaction with the services.

By tracking these key metrics, the project can measure the effectiveness of churn reduction strategies and ensure alignment with the business objective of enhancing customer retention and profitability.


 #### Hypothesis

**Null Hypothesis (H0)**: Monthly charges, total charges, and tenure do not have a significant impact on the churn rate.

**Alternative Hypothesis (H1)**: Monthly charges, total charges, and tenure have a significant impact on the churn rate.

 #### Analytical Questions

1.	What are the primary factors driving churn among different genders, and how do they differ?

Conduct a multivariate analysis to identify and compare the key factors contributing to churn for male and female customers. Consider variables such as service quality, customer service interactions, billing issues, and usage patterns. Use techniques like logistic regression or decision trees to determine the relative importance of these factors.

1.   Is there a significant association between gender and churn?
 
Assess whether there is a statistically significant relationship between a customer's gender and their likelihood of churning.

3.	Is there a correlation between senior citizens and churn? 

Analyze whether senior citizens have higher churn rates compared to non-senior citizens.

4.	Do paperless billing and payment methods influence churn?

Investigate if the use of paperless billing and different payment methods (e.g., electronic check, mailed check) has an impact on customer churn rates.

5.	Does the presence of dependents affect monthly charges?

Examine if customers with dependents tend to pay higher or lower monthly charges compared to those without dependents.

6.  Which gender pays more monthly charges?
  
Determine whether male or female customers have higher average monthly charges.

7.  Does the presence of dependents affect customer churn?

Evaluate if having dependents influences the likelihood of a customer churning.

8.  Which gender exhibited the highest churn?

Identify which gender, male or female, has a higher rate of customer churn.





 #### Scope and Constraints

 #### Scope

1.	**Data Collection and Preparation**

Gather and clean relevant customer data, including demographics, service usage, billing information, and customer service interactions.

Ensure data is comprehensive, accurate, and up-to-date.

2.	**Exploratory Data Analysis (EDA)**

Conduct initial data analysis to identify patterns, trends, and anomalies in customer behavior related to churn.

Visualize data to uncover insights and generate hypotheses.

3.	**Predictive Modeling**

Develop and validate predictive models to identify customers at risk of churning.

Utilize machine learning techniques such as logistic regression, decision trees, random forests, and neural networks.

4.	**Churn Driver Analysis**

Identify and quantify the key factors contributing to customer churn.

Use statistical and machine learning methods to determine the relative importance of different churn drivers.

5.	**Actionable Insights and Recommendations**

Provide data-driven insights to stakeholders on how to reduce churn and improve customer retention.

Develop targeted intervention strategies based on model predictions and churn driver analysis.

6.	**Implementation and Monitoring**

Support the implementation of recommended retention strategies.
Monitor the effectiveness of these strategies over time and adjust as needed based on continuous feedback and analysis.

 #### Constraints

1.	**Data Quality and Availability**

Limited access to high-quality, comprehensive data may hinder the accuracy and reliability of the analysis and predictive models.

Incomplete or missing data can affect the robustness of insights and recommendations.

2.	**Resource Limitations**

Limited availability of skilled data scientists and analysts may impact the project's timeline and depth of analysis.

Budget constraints may restrict the tools and technologies available for data processing and analysis.

3.	**Time Constraints**

The project must be completed within a specified timeframe, which may limit the scope of analysis and model development.

Tight deadlines can impact the thoroughness of data cleaning, model validation, and result interpretation.

4.	**Stakeholder Engagement**

Ensuring consistent and active engagement from all relevant stakeholders is crucial for the project's success.

Lack of stakeholder involvement can result in misaligned objectives and ineffective implementation of recommendations.

5.	**Data Privacy and Security**

Adhering to data privacy regulations (e.g., GDPR) and ensuring the security of sensitive customer information is paramount.

Compliance requirements may limit the type of data that can be used and shared.

6.	**Model Interpretability**

Complex predictive models may be difficult for non-technical stakeholders to understand and trust.

Ensuring model transparency and interpretability is essential for stakeholder buy-in and successful implementation.

7.	**External Factors**

Market conditions, competitive actions, and regulatory changes can influence customer behavior and churn rates, impacting the project's outcomes.

The analysis must account for external factors that may affect the generalizability and applicability of the insights and recommendations.


 #### Exploratory Data Analysis (EDA)

1.	**Data Overview**

•	Load the dataset and examine the structure, types of variables, and basic statistics.

•	Check for missing values and potential outliers.

2.	**Data Visualization**

•	Visualize the distribution of churn across different genders.

•	Plot the relationship between monthly charges and churn.

•	Analyze the impact of dependents on churn using bar charts.


•	Examine the correlation between senior citizen status and churn.

3.	**Statistical Analysis**

•	Perform chi-square tests to assess the significance of categorical variables like gender and dependents on churn.

•	Conduct t-tests or ANOVA to compare monthly charges across different categories.


 #### Issues with the Data

1.	**Missing Values**

•	Identify columns with missing values.

•	Decide on appropriate imputation techniques or removal strategies.

2.	**Outliers**

•	Detect outliers in numerical columns like monthly charges.

•	Determine if outliers should be treated or removed based on their impact on the analysis.


3.	**Data Types**

•	Ensure all columns have the correct data types for analysis.

•	Convert categorical variables to appropriate formats if needed.

 #### Handling Identified Issues

1.	**Imputation**

•	Use mean, median, or mode imputation for missing values depending on the data distribution.

•	Consider using advanced techniques like KNN imputation for more accurate estimates.

2.	**Outlier Treatment**

•	Apply transformations or remove outliers if they significantly skew the data.

•	Use robust statistical methods to minimize the impact of outliers.

3.	**Data Type Conversion**

•	Convert categorical variables to numerical formats using techniques like one-hot encoding for analysis.

•	Ensure date columns, if any, are in the proper datetime format for time-based analyses.


By addressing these questions and issues, we aim to gain a comprehensive understanding of the factors influencing customer churn and build a robust classification model to predict churn effectively.




## Resources availability
#### Data for this project

In this project, the dataset resides in 3 places. For ease of access and security, the datasets were made available there in the GitHub Repository

#### First Data Set
The data called LP2_Telco_churn_First_3000 was extracted from a database.

#### Second Data Set
The data called Telco-churn-second-2000.xlsx was found in OneDrive which is my test dataset

#### Third Data Set
The third part of the data called LP2_Telco-churn-last-2000.csv. is hosted on a GitHub Repository

#### Column names and description

1.	Gender -- Whether the customer is a male or a female

2.	SeniorCitizen -- Whether a customer is a senior citizen or not

3.	Partner -- Whether the customer has a partner or not (Yes, No)

4.	Dependents -- Whether the customer has dependents or not (Yes, No)

5.	Tenure -- Number of months the customer has stayed with the company

6.	Phone Service -- Whether the customer has a phone service or not (Yes, No)

7.	MultipleLines -- Whether the customer has multiple lines or not

8.	InternetService -- Customer's internet service provider (DSL, Fiber Optic, No)

9.	OnlineSecurity -- Whether the customer has online security or not (Yes, No, No Internet)

10.	OnlineBackup -- Whether the customer has online backup or not (Yes, No, No Internet)

11.	DeviceProtection -- Whether the customer has device protection or not (Yes, No, No internet service)

12.	TechSupport -- Whether the customer has tech support or not (Yes, No, No internet)

13.	StreamingTV -- Whether the customer has streaming TV or not (Yes, No, No internet service)

14.	StreamingMovies -- Whether the customer has streaming movies or not (Yes, No, No Internet service)

15.	Contract -- The contract term of the customer (Month-to-Month, One year, Two year)

16.	PaperlessBilling -- Whether the customer has paperless billing or not (Yes, No)

17.	Payment Method -- The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))

18.	MonthlyCharges -- The amount charged to the customer monthly

19.	TotalCharges -- The total amount charged to the customer

20.	Churn -- Whether the customer churned or not (Yes or No)


#### Feature Engineering
Map boolean columns to 'Yes'/'No' and then to 1/0.
Encode categorical variables using one-hot encoding.
Scale numerical features.
Data Transformation
Use pipelines for consistent and efficient preprocessing of both training and test data.

#### Modeling and Evaluation
Model Training
We train several models, including:

Logistic Regression
Random Forest Classifier
Support Vector Machine
Decision Tree Classifier
Model Evaluation
Evaluate models using metrics such as accuracy, precision, recall, F1-score, and ROC AUC score.

#### Cross-Validation
Perform cross-validation to ensure the model's robustness and avoid overfitting.

#### Model Selection
Based on evaluation metrics, the Logistic Regression model was selected as the best-performing model.

#### Deployment
Model Saving
Save the trained Logistic Regression model using joblib.

python
Copy code
import joblib
joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl', compress=3)
Model Loading and Prediction
Load the saved model and use it to predict churn on new data.

python

#### Load the trained model

loaded_model = joblib.load('logistic_regression_model.pkl')

#### Apply the preprocessing pipeline to the test data

test_data_processed = preprocessor.transform(test_data)

#### Use the loaded model to make predictions

test_data['Churn'] = loaded_model.predict(test_data_processed)

#### Save the predictions to a new file

test_data.to_csv('test_data_with_predictions.csv', index=False)

#### Print the first few rows of the test data with predictions

print(test_data.head())

#### Conclusion
The Logistic Regression model provided the best performance in predicting customer churn. This model can be used to identify high-risk customers, allowing for targeted retention strategies.

#### Files

main.ipynb: Jupyter Notebook containing the entire workflow.
logistic_regression_model.pkl: Saved Logistic Regression model.
train_data.csv: Cleaned and preprocessed training data.
test_data_with_predictions.csv: Test data with predicted churn values.

#### Dependencies

Python 3.x
Pandas
Numpy
Scikit-learn
Seaborn
Matplotlib
Joblib

To install the required packages, use the following command:

bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib


#### Power BI Deployment 🌟
I take my analysis to the next level with Power BI's Python scripting tool:

Find the Power BI dashboard [Power BI](https://app.powerbi.com/groups/me/reports/ef2b9364-48bf-4d65-909f-81af5d324499/ReportSection?experience=power-bi)
<img width="742" alt="image" src="https://github.com/Wisdom-EdemD-rah/Customer-Churn-Prediction-and-Analysis/assets/159122419/a3fa7735-8b86-4c7d-8cb8-e52fc053aa97">

This dashboard provides a comprehensive overview of customer churn analysis for a telecommunications company. Here's a breakdown of each component:

Average Monthly Charge: Displays the average monthly charge for customers, which is $64.50.

Churners: Shows the total number of customers who have churned, which is 675.

Non-Churners: Shows the total number of customers who have not churned, which is 1883.

Churn Rate: Displays the churn rate as a percentage, which is 26.4%.

Churners by Gender: A pie chart that breaks down the churners by gender, showing nearly equal percentages of male (49.48%) and female (50.52%) churners.

Non-Senior Citizens by Churn: A bar chart showing the number of non-senior citizens who have churned (174) versus those who have not (236).

Senior Citizens by Churn: A bar chart showing the number of senior citizens who have churned (0.5k) versus those who have not (1.6k).

Monthly Charge by Presence of Dependents: A doughnut chart displaying the distribution of monthly charges by whether customers have dependents (49.19k without dependents and 115.87k with dependents).

Payment Methods by Churn Rate & Paperless Billing: A combination chart showing the count of customers using different payment methods along with their churn rates and whether they use paperless billing. It indicates higher churn rates for electronic checks and lower for automatic payments via bank transfer or credit card.

Average of Monthly Charges by Gender: A bar chart showing the average monthly charges by gender, with females slightly lower than males.

Churners and Churn Rate by Presence of Dependents: A bar chart with a line overlay showing the count of churners and churn rates by the presence of dependents. It indicates a higher churn rate for customers without dependents (31.71%) compared to those with dependents (14.90%).

In a nutshell, this dashboard helps in visualizing key metrics and trends related to customer churn, such as demographic influences, payment methods, and the impact of dependents, which can inform strategies to reduce churn and improve customer retention.


#### Key Insights and Conclusion
Analyzing customer churn and implementing retention strategies offer valuable insights into customer behavior and business performance enhancement. Here are the key takeaways from this analysis:

• Churn Distribution: The analysis of churned versus non-churned customers revealed that a significant portion of customers remained loyal during the observation period.

• Feature Importance: Important factors such as contract type, monthly charges, and tenure significantly influence customer churn. Businesses should prioritize these factors when developing retention strategies.

• Dealing with Imbalanced Data: Addressing imbalanced data using techniques like SMOTE and class weight balancing improved model training by creating a more representative dataset and enhancing predictive accuracy.

• Model Performance: Machine learning models, particularly Random Forest and Logistic Regression, showed promising accuracy in predicting customer churn. Fine-tuning model parameters and rigorous evaluation were critical for optimizing their performance.

• Retaining High-Value Customers: Identifying high-value customers prone to churn enables businesses to offer personalized retention incentives, fostering loyalty and reducing churn rates.

• Challenges Faced: Imbalanced Data: Managing imbalanced data required careful application of SMOTE and class weight techniques to avoid bias and ensure accurate predictions.

• Hyperparameter Tuning: Optimizing model parameters was challenging due to the complexity of models like Random Forest and Linear Regression. Techniques such as grid search and cross-validation were essential for finding optimal settings.

• Interpreting Insights: Understanding relationships between features and churn in complex models posed challenges. Analyzing feature importance helped mitigate this issue.

• Next Steps: Continuous Monitoring: Regularly monitoring churn patterns and analyzing new data will provide insights into evolving customer behaviours, enabling proactive adjustments to retention strategies.

• Personalized Offers: Implementing targeted offers based on customer behaviour can enhance engagement and loyalty.

• Advanced Analytics: Leveraging advanced techniques such as deep learning or ensemble methods can further improve churn prediction accuracy.

• Feedback Integration: Establishing a feedback loop from customer interactions helps refine models and strategies over time.

• Customer-Centric Approach: Prioritizing customer feedback and tailored experiences fosters strong customer relationships and loyalty.

In conclusion, analyzing customer churn provides actionable insights for businesses to retain customers and enhance overall performance. By leveraging advanced analytics and continuously refining strategies, businesses can effectively reduce churn and cultivate lasting customer loyalty.


#### Acknowledgements
I would like to extend my sincere appreciation to the Azubi Africa Data Analytics and Science team for their exceptional support and guidance throughout this project. I am especially grateful to Portia Bentum, Glen Nii-Noi Anum, and Racheal Appiah-Kubi for their ongoing encouragement, valuable insights, and expertise. Their contributions were crucial to the successful completion of this analysis, and their commitment to creating a collaborative and innovative environment has been truly inspiring. Thank you for your steadfast support and for being an essential part of this journey.



#### Author 👨‍💼
Name	Wisdom Edem Drah

Article Link [here](https://medium.com/@drahwizi06/95d4d0aa8963)

[Github](https://github.com/Wisdom-EdemD-rah/Customer-Churn-Prediction-and-Analysis)


#### 📝 License
This project is licensed under the [MIT License](https://github.com/Wisdom-EdemD-rah/Customer-Churn-Prediction-and-Analysis/blob/main/LICENSE).

#### 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

Fork the repository.
Create a new branch: git checkout -b feature-name.
Commit your changes: git commit -m 'Add feature'.
Push to the branch: git push origin feature-name.
Create a pull request.

#### Acknowledgements 🙏
I would like to express my gratitude to the [Azubi Africa Data Analyst Program] (https://www.azubiafrica.org/data-analytics) for their support and for offering valuable projects as part of this program. Not forgetting my Teachops on this project Rachael Appiah-Kubi & Glen Nii-Noi Anum

#### 📧 Contact
For questions or feedback, please contact [Wisdom Edem Drah](https://github.com/Wisdom-EdemD-rah?tab=repositories)

Connect with me on [LinkedIn Here](https://www.linkedin.com/in/wisdom-edem-drah-/) 
