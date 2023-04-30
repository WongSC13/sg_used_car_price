import streamlit as st
import pandas as pd
from sklearn import ensemble
import pickle

# loading the saved models
car_price_model = pickle.load(open(r'sg_used_car_model_gbr.pkl', 'rb'))

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

col1, col2, col3, col4 = st.columns(4)  

with col1:
    DEPRE_VALUE_PER_YEAR = st.text_input('DEPRE_VALUE_PER_YEAR', '11320')
    CAR_AGE_YEARS = st.text_input('CAR_AGE_YEARS', '1.99')
    ENGINE_CAPACITY_CC = st.text_input('ENGINE_CAPACITY_CC', '1991')
  
with col2:
    MILEAGE_KM = st.text_input('MILEAGE_KM', '29717')
    DEREG_VALUE_FROM_SCRAPE_DATE = st.text_input('DEREG_VALUE_FROM_SCRAPE_DATE', '69556')
    ROAD_TAX_PER_YEAR = st.text_input('ROAD_TAX_PER_YEAR', '1202')

with col3:
    COE_FROM_SCRAPE_DATE = st.text_input('COE_FROM_SCRAPE_DATE', '49012')
    OMV = st.text_input('OMV', '34564')
    CURB_WEIGHT_KG = st.text_input('CURB_WEIGHT_KG', '1644')
    
with col4:
    DAYS_OF_COE_LEFT = st.text_input('DAYS_OF_COE_LEFT', '2922')
    ARF = st.text_input('ARF', '40390')
    NO_OF_OWNERS = st.text_input('NO_OF_OWNERS', '1')

st.sidebar.subheader('TRANSMISSION')
TRANSMISSION_Auto= st.sidebar.checkbox('Auto', value=True)
TRANSMISSION_Manual= st.sidebar.checkbox('Manual', value=False)
 
st.sidebar.subheader('VEHICLE TYPE')
VEHICLE_TYPE_Hatchback= st.sidebar.checkbox('Hatchback', value=False)
VEHICLE_TYPE_Luxury_Sedan= st.sidebar.checkbox(' Luxury_Sedan ', value=True)
VEHICLE_TYPE_MPV= st.sidebar.checkbox(' MPV ', value=False)
VEHICLE_TYPE_Mid_Sized_Sedan = st.sidebar.checkbox(' Mid_Sized_Sedan ', value=False)
VEHICLE_TYPE_SUV= st.sidebar.checkbox(' SUV ', value=False)
VEHICLE_TYPE_Sports_Car= st.sidebar.checkbox(' Sports_Car ', value=False)
VEHICLE_TYPE_Stationwagon= st.sidebar.checkbox(' Stationwagon ', value=False)


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