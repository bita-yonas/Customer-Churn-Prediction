# 📊 Customer Churn Prediction with Machine Learning

🚀 **[Live Web App](https://customer-churn-prediction-bitaniayonas.replit.app)**: Click to explore the deployed web app.

This project leverages advanced machine learning techniques to predict which customers are likely to stop using a service, also known as customer churn. The solution analyzes various customer behavioral and demographic data points—such as age, account balance, and activity history—to identify patterns associated with churn. By training multiple models, including **XGBoost, Random Forest, and SVM**, the project selects the best-performing model for deployment. The final model is deployed through an interactive **web app**, providing real-time predictions and actionable insights. In addition to predictions, the app offers personalized explanations of the factors driving each customer’s risk of churning. Businesses can use these insights to implement targeted retention strategies, boosting customer satisfaction and loyalty through automated engagement, such as personalized email campaigns.
  
---

## 🛠️ Features

- **Data Preprocessing:** Handle missing values, encode categorical data, and normalize numerical features.
- **EDA (Exploratory Data Analysis):** Visualize data distributions and analyze correlations to gain insights.
- **Model Training:** Train **5 different machine learning models** for churn prediction.
- **Hyperparameter Tuning:** Use **grid search** to optimize the models.
- **Web App Deployment:** Serve the best-performing model through a web application for real-time predictions.
- **Churn Explanation:** Provide feature importance insights and personalized recommendations.
- **Email Generation:** Generate engaging emails to encourage at-risk customers to stay.

---

## 🚀 Tech Stack

- **Python**: Programming language for data processing and model building.
- **Pandas & NumPy**: Libraries for data manipulation and numerical operations.
- **Scikit-Learn & XGBoost**: Tools for machine learning and model building.
- **Seaborn & Matplotlib**: Visualization libraries for EDA.
- **Streamlit**: For deploying the web app and interacting with predictions.
- **OpenAI API**: For generating explanations and personalized customer emails.
- **Groq API**: For interacting with models via Groq's accelerated AI API services.
- **Jupyter Notebook**: For iterative development and experimentation.
- **Pickle**: For saving and loading trained machine learning models.

---

## 🏆 Results

The models were evaluated using the following metrics:

- **Accuracy**: Overall correctness of the model.
- **Precision**: Percentage of predicted churn cases that were correct.
- **Recall**: Ability to detect actual churn cases.
- **F1-score**: A harmonic mean of precision and recall, balancing both metrics.

Below are the key performance metrics for the trained models:

### XGBoost Classifier
- **Accuracy**: 85.4%
- **Precision (Class 1 - Churn)**: 67%
- **Recall (Class 1 - Churn)**: 50%
- **F1-score (Class 1 - Churn)**: 0.58

### Random Forest Classifier
- **Accuracy**: 86.4%
- **Precision (Class 1 - Churn)**: 75%
- **Recall (Class 1 - Churn)**: 46%
- **F1-score (Class 1 - Churn)**: 0.57

### Decision Tree Classifier
- **Accuracy**: 78.5%
- **Precision (Class 1 - Churn)**: 46%
- **Recall (Class 1 - Churn)**: 51%
- **F1-score (Class 1 - Churn)**: 0.48

### K-Nearest Neighbors (KNN)
- **Accuracy**: 82.4%
- **Precision (Class 1 - Churn)**: 59%
- **Recall (Class 1 - Churn)**: 36%
- **F1-score (Class 1 - Churn)**: 0.44

### Support Vector Classifier (SVM)
- **Accuracy**: 85.6%
- **Precision (Class 1 - Churn)**: 77%
- **Recall (Class 1 - Churn)**: 38%
- **F1-score (Class 1 - Churn)**: 0.51

### Gaussian Naive Bayes
- **Accuracy**: 81.8%
- **Precision (Class 1 - Churn)**: 56%
- **Recall (Class 1 - Churn)**: 38%
- **F1-score (Class 1 - Churn)**: 0.45

### Voting Classifier (Ensemble Model)
- **Accuracy**: 85.3%
- **Precision (Class 1 - Churn)**: 63%
- **Recall (Class 1 - Churn)**: 59%
- **F1-score (Class 1 - Churn)**: 0.61

---

The **Random Forest Classifier** and **XGBoost Classifier** achieved the highest accuracies, with XGBoost also providing better precision for churn cases. Ultimately, the **XGBoost model** was selected for deployment due to its balance of accuracy and interpretability.
