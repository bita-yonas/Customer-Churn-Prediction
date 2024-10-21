import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import utils as ut
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get('GROQ_API_KEY')
)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load models at the beginning
xgboost_model = load_model('xgb_model.pkl')
native_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('bt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifiers_model = load_model('voting_Clf.pkl')
xgboost_SMOTE_model = load_model('xgboost_featureEngineered.pkl')

def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_of_products, has_credit_history, is_active_member, estimated_salary):
    input_dict = {
        'credit_score': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': int(has_credit_history),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

def calculate_customer_percentiles(df, selected_customer):
    metrics = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    percentiles = {}
    for metric in metrics:
        percentile = np.percentile(df[metric], (df[metric] < selected_customer[metric]).mean() * 100)
        percentiles[metric] = round(percentile, 2)
    return percentiles

def make_prediction(input_df, input_dict):
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig_probs = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig_probs, use_container_width=True)
        st.write(f"The customer has a **{avg_probability:.2%}** probability of churning.")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability

def explain_prediction(probability, input_dict, surname):
    prompt = f"""
    You are an expert data scientist at a bank, where you specialize in 
    interpreting and explaining predictions of machine learning models. 

    Your machine learning model has predicted that a customer named {surname} 
    has a {round(probability * 100, 1)}% probability of churning based on the 
    information provided below.

    Here is the customer's information:
    {input_dict}

    Here are the machine learning model's top 10 most important features for predicting churn:

      Feature            | Importance 
    |--------------------|------------|
    | NumOfProducts      | 0.323888   |
    | IsActiveMember     | 0.164146   |
    | Age                | 0.109550   |
    | Geography_Germany  | 0.091373   |
    | Balance            | 0.052786   |
    | Geography_France   | 0.046463   |
    | Gender_Female      | 0.045283   |
    | Geography_Spain    | 0.036855   |
    | CreditScore        | 0.035005   |
    | EstimatedSalary    | 0.032655   |
    | HasCrCard          | 0.031940   |
    | Tenure             | 0.030054   |
    | Gender_Male        | 0.000000   |


    {pd.set_option('display.max_columns', None)}

    here are the summary statistic for churned customer:
    {df[df['Exited'] == 1].describe()}

    here are the summary statistic for non-churned customer:
    {df[df['Exited'] == 0].describe()}


   
       - Generate a concise, 3-sentence explanation about the customer’s likelihood to stay or leave, based on key patterns and behavioral insights:  

        - If the customer has over a 40% risk of churning: Highlight specific behaviors or trends from the customer's profile that resemble those of churned customers. Focus on potential pain points, behavioral shifts, or unmet needs that may be driving the risk.  

        - If the customer has less than a 40% risk of churning**: Emphasize the positive aspects of the customer’s engagement, loyalty patterns, or alignment with satisfied customers. Highlight factors that suggest stability, satisfaction, or ongoing commitment.  

        -Use the provided customer information, behavioral trends, summary statistics, and feature importances to guide your explanation.  
        -Do not reference the churn probability, machine learning models, or technical features explicitly.** Provide an insightful, natural explanation that focuses on the customer’s experience and engagement.  
    
    """

    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}]
    )

    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""
    As a manager at HS Bank, you are committed to fostering customer loyalty and retaining valued clients through tailored offers.  
    You’ve identified that a customer named {surname} has a {round(probability * 100, 1)}% chance of leaving the bank.  

    Below is the customer’s information:
    {input_dict}

    Additionally, here are key factors contributing to the risk of churn:
    {explanation}

    Write a personalized, engaging email to the customer, encouraging them to continue their relationship with the bank.  
    The email should:
    - Acknowledge the customer’s value to the bank.
    - Use positive, empathetic language.
    - Present personalized incentives in bullet points to address potential concerns and boost loyalty.
    - End with a friendly, actionable call-to-action encouraging them to stay in touch.

    Ensure the tone is warm, professional, and customer-focused.
    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return raw_response.choices[0].message.content

st.title("📊 Customer Churn Prediction")
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)

df = pd.read_csv("churn.csv")
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a Customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    st.markdown("## 📋 Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=int(selected_customer["CreditScore"]))
        location = st.selectbox("Location", ["Spain", "France", "Germany"], index=["Spain", "France", "Germany"].index(selected_customer['Geography']))
        gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer["Gender"] == "Male" else 1)
        age = st.number_input("Age", min_value=18, max_value=100, value=int(selected_customer["Age"]))
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=int(selected_customer["Tenure"]))

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=float(selected_customer["Balance"]))
        num_products = st.number_input("Number of Products", min_value=0, max_value=10, value=int(selected_customer["NumOfProducts"]))
        has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer["HasCrCard"]))
        is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer["IsActiveMember"]))
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(selected_customer["EstimatedSalary"]))

    input_df, input_dict = prepare_input(
        credit_score, location, gender, age, tenure, balance,
        num_products, has_credit_card, is_active_member, estimated_salary
    )

    
    if st.button("Predict Churn"):
        input_df, input_dict = prepare_input(
            credit_score, location, gender, age, tenure, balance,
            num_products, has_credit_card, is_active_member, estimated_salary
        )

        
        avg_probability = make_prediction(input_df, input_dict)

       
        st.subheader("📊 Customer Percentiles Across Metrics")
        percentiles = calculate_customer_percentiles(df, selected_customer)
        fig_percentiles = ut.create_percentile_bar_chart(percentiles)
        st.plotly_chart(fig_percentiles, use_container_width=True)

        
        explanation = explain_prediction(avg_probability, input_dict, selected_customer["Surname"])
        st.subheader("📝 Explanation of Predictions")
        st.markdown(explanation)

        
        email = generate_email(avg_probability, input_dict, explanation, selected_customer["Surname"])
        st.subheader("📧 Personalized Email")
        st.markdown(email)
