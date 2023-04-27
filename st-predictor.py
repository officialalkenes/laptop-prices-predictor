import pickle

import numpy as np
import pandas as pd
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

st.title("Laptop Price Prediction App")
st.markdown("""
This app Estimates the prices of your Laptops based on preferred specification
            """)


@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

filename = 'testing.csv'
df = load_data(filename)



company = df.groupby('Company')
type_name = df.groupby("TypeName")
ram = df.groupby("Ram")
weight = df.groupby("Weight")
touchscreen = df.groupby("TouchScreen")
ips = df.groupby("IPS")
ppi = df.groupby("PPI")
cpu_brand = df.groupby("Cpu_brand")
hdd = df.groupby("HDD")
ssd = df.groupby("SSD")
gpu_brand = df.groupby("Gpu brand")
os = df.groupby("os")



@st.cache_data
def predict(value_arrays):
    numpy_array = np.asarray(value_arrays)
    reshaped_array = numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_array)
    st.write(f'based on your inputs Your preferred Laptop price is estimated at {str(prediction)} Euros.')

# Sidebar - Company selection
sorted_company = sorted( df['Company'].unique())
sorted_ram = sorted( df['Ram'].unique())
sorted_type_name = sorted( df['TypeName'].unique())
sorted_weight = sorted(df['Weight'].unique())
sorted_touch = sorted(df['TouchScreen'].unique())
sorted_ips = sorted(df['IPS'].unique())
sorted_ppi = sorted(df['PPI'].unique())
sorted_cpu = sorted(df['Cpu_brand'].unique())
sorted_hdd = sorted(df['HDD'].unique())
sorted_ssd = sorted(df['SSD'].unique())
sorted_gpu = sorted(df['Gpu brand'].unique())
sorted_os = sorted(df['os'].unique())

os_sel = st.selectbox('Choose Preferred Operating Software', sorted_os)
company_sel = st.selectbox('Pick Company', sorted_company)
ram_sel = st.selectbox('Select Ram (GB)', sorted_ram)
type_sel = st.selectbox('Select Laptop Type', sorted_type_name)
weight_sel = st.selectbox('Preferred Weight (KG)', sorted_weight)
touch_sel = st.selectbox('Touch Screen (1=Yes, 0=No)', sorted_touch)
hdd_sel = st.selectbox('HDD Memory (GB)', sorted_hdd)
ssd_sel = st.selectbox('SSD Memory (GB)', sorted_ssd)
# touch_sel = st.selectbox('Touch Screen (1=Yes, 0=No)', sorted_touch)
ips_sel = st.selectbox('IPS (1=Yes, 0=No)', sorted_ips)
ppi_sel = st.selectbox('Pixel Per Inch', sorted_ppi)
gpu_sel = st.selectbox('Select GPU Brand', sorted_gpu)
cpu_sel = st.selectbox('Choose CPU Brand', sorted_cpu)


if st.button("Predict Now"):
    selections = [company_sel, type_sel, ram_sel, weight_sel, touch_sel, ips_sel, ppi_sel, cpu_sel, hdd_sel, ssd_sel, gpu_sel,  os_sel]
    predictions = predict(selections)

# 	Company	TypeName	Ram	Weight	TouchScreen	IPS	PPI	Cpu_brand	HDD	SSD	Gpu brand	os
# Sidebar - Company selection
# sorted_company_unique = sorted( df['GICS Sector'].unique() )
# selected_sector = st.sidebar.select('Sector', sorted_company_unique, sorted_company_unique)


# # Sidebar - Company selection
# sorted_company_unique = sorted( df['GICS Sector'].unique() )
# selected_sector = st.sidebar.select('Sector', sorted_company_unique, sorted_company_unique)


# # Sidebar - Company selection
# sorted_company_unique = sorted( df['GICS Sector'].unique() )
# selected_sector = st.sidebar.select('Sector', sorted_company_unique, sorted_company_unique)


