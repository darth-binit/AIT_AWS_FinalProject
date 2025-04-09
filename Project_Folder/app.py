import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import os
import boto3
from S3 import download_and_extract_model_artifacts  # Import your function
import sklearn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# Download & Extract Artifacts
# --------------------------

# Download and extract model artifacts from S3.
# This ensures that every time the app starts, it pulls the latest model.
model_dir = download_and_extract_model_artifacts()

# Set local paths for the artifacts from the extracted folder
brand_means_path = os.path.join(model_dir, "brand_means.pkl")
model_path       = os.path.join(model_dir, "random_forest_model.pkl")
scaler_path      = os.path.join(model_dir, "scaler.pkl")
onehot_path      = os.path.join(model_dir, "onehotencode.pkl")

# Load the artifacts
brand_means = joblib.load(brand_means_path)
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
onehot = joblib.load(onehot_path)

# --------------------------
# Download Data File from S3 if needed
# --------------------------
data_dir = os.path.join(BASE_DIR, "data")
data_file_path = os.path.join(data_dir, "Out_287.csv")
if not os.path.exists(data_file_path):
    s3 = boto3.client("s3", region_name="us-east-2")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading data file to {data_file_path} ...")
    s3.download_file("finalprojectait2025", "data/Out_287.csv", data_file_path)
df = pd.read_csv(data_file_path)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.head(5))

## Identify columns
cat_col = df.select_dtypes(include='object').columns.tolist()
num_col = df.select_dtypes(include='number').columns.tolist()

def show_centered_image(image_path, brand):
    import base64

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{encoded_string}" width="400" alt="{brand} Logo">
        <p style="font-weight: bold; font-size: 18px;">{brand} Visual</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Define a prediction function that uses the loaded artifacts.
# -----------------------------
def predict(val):
    # Preprocess the input DataFrame similar to training
    val["brand_encoded"] = val["brand"].map(brand_means).fillna(brand_means.mean())
    val = val.drop(columns=["brand"])

    # One-hot encode categorical features. Here we assume that during training the onehot encoder was fit
    # on columns: ["fuel", "seller_type", "transmission"]
    cat_cols = ["fuel", "seller_type", "transmission"]
    encoded_cat = onehot.transform(val[cat_cols])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=onehot.get_feature_names_out(), index=val.index)

    # Scale numerical features. Ensure your input DataFrame has the same numerical columns as used in training.
    num_cols = ['year', 'km_driven', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    scaled_num = scaler.transform(val[num_cols])
    scaled_num_df = pd.DataFrame(scaled_num, columns=num_cols, index=val.index)

    # Combine features
    X = pd.concat([scaled_num_df, encoded_cat_df, val[["brand_encoded"]]], axis=1)
    print("Final features:", X.head())

    # Make prediction and reverse log transformation if model was trained with log1p transformation
    prediction_log = model.predict(X)
    prediction = np.expm1(prediction_log)
    return prediction

# -----------------------------
# Streamlit App Code
# -----------------------------
st.set_page_config(layout="wide")

def streamlit_menu():
    st.markdown(
        "<h1 style='text-align: center; color: #355F10;'>Binit Automobiles Solution</h1>",
        unsafe_allow_html=True
    )
    selected = option_menu(
        menu_title=None,
        options=["Home | Descriptive ", "Predictive Analytics"],
        icons=["bi bi-activity", "bi bi-clipboard-data", "bi bi-pie-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#C6DDB6"},
            "icon": {"color": "#072810", "font-size": "25px"},
            "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px", "--hover-color": "#eee", "padding": "10px"},
            "nav-link-selected": {"background-color": "#355F10"},
        }
    )
    return selected

select = streamlit_menu()

if select == "Home | Descriptive ":
    col1, col2, col3 = st.columns([1, 0.13, 1])
    with col2:
        icon_url = os.path.join(BASE_DIR, "auto-automobile-car-pictogram-service-traffic-transport--2.png")
        st.image(icon_url, width=50)
    with col1:
        st.markdown(
            "<div style='background-image: linear-gradient(to right, #428142, #AAD4AA); font-size: 20px; padding: 8px; text-align: center;'>Numerical Features Distribution</div>",
            unsafe_allow_html=True
        )
        feat_select = st.selectbox('Select Feature', df[num_col].columns)
        fig = px.histogram(
            df,
            x=feat_select,
            nbins=30,
            marginal="violin",
            histnorm="density",
            opacity=0.75,
            color_discrete_sequence=['#379037']
        )
        fig.update_layout(xaxis_title=feat_select, yaxis_title="Density", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.markdown(
            "<div style='background-image: linear-gradient(to left, #428142, #AAD4AA); font-size: 20px; padding: 8px; text-align: center;'>Categorical Features Count</div>",
            unsafe_allow_html=True
        )
        cat_feat_select = st.selectbox('Select Feature', df[cat_col].columns)
        colors = ['#379037', '#71B971', '#9DD39D']
        feat_val = df[cat_feat_select].value_counts().values
        total = sum(feat_val)
        percentages = [f'{(v/total)*100:.2f}%' for v in feat_val]
        trace = go.Bar(x=df[cat_feat_select].value_counts().index, y=df[cat_feat_select].value_counts().values,
                        marker=dict(color=colors), hovertext=percentages)
        layout = go.Layout()
        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('\n\n')
    st.markdown(
        "<div style='background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;'>Correlation | Numerical Features</div>",
        unsafe_allow_html=True
    )
    st.markdown('\n\n\n\n')
    fig, ax = plt.subplots(figsize=(30, 16))
    num_corr = df[num_col].corr()
    mask = np.triu(np.ones_like(num_corr, dtype=bool))
    sns.heatmap(num_corr, mask=mask, xticklabels=num_corr.columns, yticklabels=num_corr.columns, annot=True, linewidths=.3,
                cmap='Greens', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

if select == "Predictive Analytics":
    colp1, colp2 = st.columns([1, 1])
    with colp2:
        prediction_placeholder = st.empty()
        st.session_state['prediction_result'] = ''
        prediction_placeholder.markdown(
            "<div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>"
            "<h2 style='text-align: center'>Prediction</h2>"
            "<p style='font-size: 24px; text-align: center'>{}</p></div>".format(st.session_state['prediction_result']),
            unsafe_allow_html=True
        )
        st.markdown('\n\n')
        predict_btn = st.button('Predict', use_container_width=True)
        brand = st.selectbox('Brand', df['brand'].unique())
        colpp1, colpp2, colpp3 = st.columns([1, 1, 1])
        with colpp1:
            year = st.selectbox('Year', sorted(df['year'].unique()))
            seats = st.selectbox('Seats', sorted(df['seats'].unique()))
            fuel = st.selectbox('Fuel Type', df['fuel'].unique())
        with colpp2:
            seller_type = st.selectbox('Seller Type', df['seller_type'].unique())
            mileage = st.number_input('Mileage', min_value=5, max_value=42, value=15)
            kms_driven = st.number_input('Kilometer Driven', min_value=1000, max_value=2500000, value=1000)
        with colpp3:
            owner = st.number_input('Ownership', min_value=1, max_value=4, value=1)
            engine = st.number_input('Engine', min_value=500, max_value=4000, value=1400)
            max_power = st.number_input('Maximum Power', min_value=30, max_value=400, value=85)
        transmission = st.selectbox('Transmission', df['transmission'].unique())
    usr_data = {
        'brand': [brand],
        'year': [year],
        'km_driven': [kms_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    }
    usr_data = pd.DataFrame(usr_data)

    with colp1:
        sel_cols = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner',
                                        'mileage', 'engine', 'max_power', 'seats']
        fuel_map = {'Diesel': 2, 'Petrol': 1}
        seller_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
        transmission_map = {'Automatic': 2, 'Manual': 1}

        temp_df = df.drop(columns=['brand','selling_price'])
        temp_df = temp_df.replace(fuel_map)
        temp_df = temp_df.replace(seller_map)
        temp_df = temp_df.replace(transmission_map)

        scaler_m = MinMaxScaler()
        scaled = scaler_m.fit(temp_df)

        mod_usr_data = usr_data.replace(fuel_map)
        mod_usr_data = mod_usr_data.replace(seller_map)
        mod_usr_data = mod_usr_data.replace(transmission_map)
        mod_usr_data = mod_usr_data.drop(columns=['brand'])


        scaled_data = scaler_m.transform(mod_usr_data[sel_cols])

        scaled_df = pd.DataFrame(scaled_data, columns=sel_cols)

        fig = px.line_polar(scaled_df, scaled_df.values.reshape(-1), theta=sel_cols, line_close=True)
        fig.update_traces(fill='toself', line_color='green')
        st.plotly_chart(fig, use_container_width=True)

        brand_lower = brand.lower()
        image_path = os.path.join(BASE_DIR, "brand_img",
                                  f"{brand_lower}.png")  # assuming your images are in /images directory
        # if os.path.exists(image_path):
        #     st.image(image_path, caption=f"{brand} Visual", width=400)
        if os.path.exists(image_path):
            show_centered_image(image_path, brand)

    if predict_btn:
        pred_val = predict(usr_data)[0]
        pred_val = np.round(pred_val, decimals=2)
        formatted_val = f"{pred_val:,.2f}"
        st.session_state['prediction_result'] = formatted_val
        prediction_placeholder.markdown(
            "<div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>"
            "<h2 style='text-align: center'>Prediction | Price </h2>"
            "<p style='font-size: 24px; text-align: center'>{}</p></div>".format(st.session_state['prediction_result']),
            unsafe_allow_html=True
        )
        print(formatted_val)

