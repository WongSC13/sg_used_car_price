import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import pickle

# loading the saved models
car_price_model = pickle.load(open('sg_used_car_cbc.pkl', 'rb'))

st.write("""
# Used Car Price Prediction App
This app predicts the price of a used car based on its features!
""")

# CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # predictions
    predictions = car_price_model.predict(input_df)
    input_df['Prediction'] = predictions

    # show predictions
    st.write(input_df)


# show form
st.title("Used Car Price Prediction using ML")

col1, col2, col3, col4, col5, col6 = st.columns(6)  

with col1:
    DEPRE_VALUE_PER_YEAR = st.slider('DEPRE_VALUE_PER_YEAR', 0, 150000, 7090)
    MILEAGE_KM = st.slider('MILEAGE_KM', 0, 400000, 43206)
    COE_FROM_SCRAPE_DATE = st.slider('COE_FROM_SCRAPE_DATE', 0, 150000, 50991)
    DAYS_OF_COE_LEFT = st.slider('DAYS_OF_COE_LEFT', 0, 5000, 2562)
      
with col2:
    CAR_AGE_YEARS = st.slider('CAR_AGE_YEARS', 0, 20, 3)
    DEREG_VALUE_FROM_SCRAPE_DATE = st.slider('DEREG_VALUE_FROM_SCRAPE_DATE', 0, 500000, 50952)
    OMV = st.slider('OMV', 0, 500000, 19990)
    ARF = st.slider('ARF', 0, 600000, 19990)
        
with col3:
    ENGINE_CAPACITY_CC = st.slider('ENGINE_CAPACITY_CC', 0, 3000, 1598)
    ROAD_TAX_PER_YEAR = st.slider('ROAD_TAX_PER_YEAR', 0, 15000, 242)
    CURB_WEIGHT_KG = st.slider('CURB_WEIGHT_KG', 1000, 2000, 1215)
    NO_OF_OWNERS = st.slider('NO_OF_OWNERS', 0, 10, 1)
    
with col4:
    TRANSMISSION_Auto = st.slider('TRANSMISSION_Auto', 0, 1, 1)
    TRANSMISSION_Manual = st.slider('TRANSMISSION_Manual', 0, 1, 0)
    VEHICLE_TYPE_Hatchback = st.slider('VEHICLE_TYPE_Hatchback', 0, 1, 0)
    VEHICLE_TYPE_Luxury_Sedan = st.slider('VEHICLE_TYPE_Luxury Sedan', 0, 1, 0)
    
with col5:
    VEHICLE_TYPE_MPV = st.slider('VEHICLE_TYPE_MPV', 0, 1, 0)
    VEHICLE_TYPE_Mid_Sized_Sedan = st.slider('VEHICLE_TYPE_Mid-Sized Sedan', 0, 1, 1)
    VEHICLE_TYPE_SUV = st.slider('VEHICLE_TYPE_SUV', 0, 1, 0)
    VEHICLE_TYPE_Sports_Car = st.slider('VEHICLE_TYPE_Sports_Car', 0, 1, 0)
    
with col6:
    VEHICLE_TYPE_Stationwagon = st.slider('VEHICLE_TYPE_Stationwagon', 0, 1, 0)

'''Let us create a dictionary to hold our inputs'''
inputs = {
        'DEPRE_VALUE_PER_YEAR': DEPRE_VALUE_PER_YEAR,
        'MILEAGE_KM': MILEAGE_KM,
        'COE_FROM_SCRAPE_DATE': COE_FROM_SCRAPE_DATE,
        'DAYS_OF_COE_LEFT': DAYS_OF_COE_LEFT,
        'CAR_AGE_YEARS': CAR_AGE_YEARS,
        'DEREG_VALUE_FROM_SCRAPE_DATE': DEREG_VALUE_FROM_SCRAPE_DATE,
        'OMV': OMV,
        'ARF': ARF,
        'ENGINE_CAPACITY_CC': ENGINE_CAPACITY_CC,
        'ROAD_TAX_PER_YEAR': ROAD_TAX_PER_YEAR,
        'CURB_WEIGHT_KG': CURB_WEIGHT_KG,
        'NO_OF_OWNERS': NO_OF_OWNERS,
        'TRANSMISSION_Auto': TRANSMISSION_Auto,
        'TRANSMISSION_Manual': TRANSMISSION_Manual,
        'VEHICLE_TYPE_Hatchback': VEHICLE_TYPE_Hatchback,
        'VEHICLE_TYPE_Luxury Sedan': VEHICLE_TYPE_Luxury_Sedan,
        'VEHICLE_TYPE_MPV': VEHICLE_TYPE_MPV,
        'VEHICLE_TYPE_Mid-Sized Sedan': VEHICLE_TYPE_Mid_Sized_Sedan,
        'VEHICLE_TYPE_SUV': VEHICLE_TYPE_SUV,
        'VEHICLE_TYPE_Sports_Car': VEHICLE_TYPE_Sports_Car,
        'VEHICLE_TYPE_Stationwagon': VEHICLE_TYPE_Stationwagon
        }

'''Convert the inputs dictionary into a dataframe'''
input_df = pd.DataFrame([inputs])


# Update the user input features according to the trained model's features
model_features = ['DEPRE_VALUE_PER_YEAR', 'MILEAGE_KM', 'COE_FROM_SCRAPE_DATE',
                   'DAYS_OF_COE_LEFT', 'CAR_AGE_YEARS',
                   'DEREG_VALUE_FROM_SCRAPE_DATE', 'OMV', 'ARF', 'ENGINE_CAPACITY_CC',
                   'ROAD_TAX_PER_YEAR', 'CURB_WEIGHT_KG', 'NO_OF_OWNERS',
                   'TRANSMISSION_Auto', 'TRANSMISSION_Manual', 'VEHICLE_TYPE_Hatchback',
                   'VEHICLE_TYPE_Luxury Sedan', 'VEHICLE_TYPE_MPV',
                   'VEHICLE_TYPE_Mid-Sized Sedan', 'VEHICLE_TYPE_SUV',
                   'VEHICLE_TYPE_Sports Car', 'VEHICLE_TYPE_Stationwagon']

for feature in model_features:
    if feature not in input_df.columns:
        input_df[feature] = 0

# Reorder the columns in the same way as the trained model's features
input_df = input_df[model_features]

st.subheader('User Input parameters')
st.write(input_df)

# Make a prediction using the loaded CBC model
prediction = car_price_model.predict(input_df)

st.subheader('Price Prediction')
st.write(prediction)