import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#Load necessary files
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

def load_customer_segmentation_model():
    with open('customer_segmentation.pkl', 'rb') as file:
        return pickle.load(file)

def load_label_encoder(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def load_column():
    with open('column_names.pkl', 'rb') as file:
        return pickle.load(file)

def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        return pickle.load(file)

def load_stockcode_description():
    with open('stockcode_description.pkl', 'rb') as file:
        return pickle.load(file)
    
def load_country_list():
    with open('country_list.pkl', 'rb') as file:
        return pickle.load(file)
    
def load_encoded_stockcode():
    with open('label_encoder_stockcode.pkl', 'rb') as file:
        return pickle.load(file)
    
def load_encoded_description():
    with open('label_encoder_description.pkl', 'rb') as file:
        return pickle.load(file)
    
def load_encoded_customerid():
    with open('label_encoder_customerid.pkl', 'rb') as file:
        return pickle.load(file)
    
def load_encoded_country():
    with open('label_encoder_country.pkl', 'rb') as file:
        return pickle.load(file)

#Load models and encoders
model = load_model()
rfm_model = load_customer_segmentation_model()
le_stockcode = load_encoded_stockcode()
le_description = load_encoded_description()
le_customerid = load_encoded_customerid()
le_country = load_encoded_country()
column_names = load_column()
scaler = load_scaler()

#Load the grouping of stockcode and description
stockcode_description = load_stockcode_description()

#List of countries for the dropdown
countries = load_country_list()

#Add unseen labels dynamically
def add_unseen_label(encoder, value):
    if value not in encoder.classes_:
        new_classes = np.append(encoder.classes_, value)
        encoder.classes_ = new_classes
    return encoder.transform([value])[0]

#Prediction function for total sales
def prediction(model, x):
    x = x.reindex(columns=column_names, fill_value=0)  
    return model.predict(x)

#Customer segmentation prediction function
def predict_segmentation(rfm_values, scaler, model):
    rfm_scaled = scaler.transform([rfm_values])
    return model.predict(rfm_scaled)

#Main Streamlit app
def main():
    #Create tabs
    tab1, tab2 = st.tabs(["Prediction for Total Sales", "Customer Segmentation Analysis"])

    #Predict Total Sales tab
    with tab1:
        st.header("Prediction for Total Sales")

        #Input field for StockCode
        stockcode_input = st.text_input("Enter the Stock Code:")

        #Automatically retrieve the corresponding description
        description_input = ""
        if stockcode_input:
            if stockcode_input in stockcode_description:
                #Fetch the first description associated with the StockCode
                description_input = stockcode_description[stockcode_input][0]
                st.write(f"Product Name: {description_input}")
            else:
                st.warning("StockCode not found in the database.")
        
        #Input fields for UnitPrice, Quantity, CustomerID, and Year, Month
        unitprice_input = st.text_input("Enter the Unit Price (Pounds):")
        quantity_input = st.text_input("Enter the Quantity Sold:")
        customerid_input = st.text_input("Enter the Customer ID:")

        #Country dropdown for selection
        country_input = st.selectbox("Select the Country:", countries)

        year_input = st.text_input("Enter the Year:")
        month_input = st.text_input("Enter the Month (1-12):")

        #Perform validation and prediction
        if st.button("Predict Total Sales"):
            try:
                #Convert inputs
                unit_price = float(unitprice_input)
                quantity = int(quantity_input)
                year = int(year_input)
                month = int(month_input)

                #Handle unseen labels for StockCode, Description, CustomerID, and Country
                stockcode_encoded = add_unseen_label(le_stockcode, stockcode_input)
                description_encoded = add_unseen_label(le_description, description_input)
                customerid_encoded = add_unseen_label(le_customerid, customerid_input)
                country_encoded = add_unseen_label(le_country, country_input)

                #Create input DataFrame
                X = pd.DataFrame([[stockcode_encoded, description_encoded, unit_price, quantity, customerid_encoded, country_encoded, year, month]],
                                columns=['StockCode', 'Description', 'UnitPrice', 'Quantity', 'CustomerID', 'Country', 'Year', 'Month'])
                
                #Predict total sales
                result = prediction(model, X)
                st.success(f"The Predicted Total Sales is: {result[0]}")

            except ValueError as e:
                st.error(f"Error: {e}")

    #Customer Segmentation tab
    with tab2:
        st.header("Customer Segmentation")

        recency_input = st.number_input("Enter Recency:", min_value=0)
        frequency_input = st.number_input("Enter Frequency:", min_value=0)
        monetary_input = st.number_input("Enter Monetary Value:", min_value=0)

        if st.button("Predict Customer Segment"):
            rfm_values = np.array([recency_input, frequency_input, monetary_input])
            segment = predict_segmentation(rfm_values, scaler, rfm_model)
            st.write(f"Customer belongs to segment: {segment[0]}")

            #Plot Customer Segmentation
            st.subheader("Customer Segmentation Visualization")

            #Generate data for visualization
            rfm = pd.DataFrame({
                'Recency': [recency_input],
                'Frequency': [frequency_input],
                'Monetary': [monetary_input],
                'Segment': [segment[0]]
            })

            #Reload data again to gain a more accurate data
            X = pd.read_csv('onlineRetail.csv')  
            X['TotalSales'] = X['Quantity'] * X['UnitPrice']
            X['InvoiceDate'] = pd.to_datetime(X['InvoiceDate'], format='%d/%m/%Y %H:%M')
            last_date = pd.to_datetime(X['InvoiceDate']).max()
            rfm_existing = X.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (last_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'TotalSales': 'sum'
            }).reset_index()
            rfm_existing.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
            rfm_existing['Segment'] = rfm_model.predict(scaler.transform(rfm_existing[['Recency', 'Frequency', 'Monetary']])) 

            #plotting cluster
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=rfm_existing, x='Recency', y='Monetary', hue='Segment', palette='viridis', alpha=0.7)
            plt.scatter(rfm['Recency'], rfm['Monetary'], color='red', s=100, label='New Customer')
            plt.title('Customer Segmentation')
            plt.xlabel('Recency')
            plt.ylabel('Monetary Value')
            plt.legend()
            st.pyplot(plt)

if __name__ == '__main__':
    main()
