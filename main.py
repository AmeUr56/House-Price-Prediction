import streamlit as st
from numpy import array
from pandas import DataFrame
from joblib import load

property_type_encoder = load("artifacts/property_type_encoder.pkl")
city_encoder = load("artifacts/city_encoder.pkl")
purpose_encoder = load("artifacts/purpose_encoder.pkl")
scaler = load("artifacts/scaler.pkl")
estimator = load("artifacts/estimator.pkl")

linkedin_profile_badge = """
<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
<div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="ameur-b-25a155247" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://dz.linkedin.com/in/ameur-b-25a155247?trk=profile-badge">Ameur B.</a></div>
"""

# Page Configurations
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏘️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide hamburger menu and footer
hide_streamlit_style = """
    
"""

# Inject custom CSS
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏘️ House Price Prediction")
st.sidebar.markdown("<a href='https://www.linkedin.com/in/ameur-b-25a155247/' target='_blank'><h1>Linkedin</h1></a>",unsafe_allow_html=True)
st.sidebar.markdown("<a href='https://x.com/Ame44i' target='_blank'><h1>X (Twitter)</h1></a>",unsafe_allow_html=True)

# Main
st.title("🏘️ House Price Prediction")
st.subheader("Predict house prices💲based on size, location, number of rooms, etc.")

property_type_col,city_col,baths_col,purpose_col,bedrooms_col,area_col = st.columns([3]*6)

with st.form("form"):
    with property_type_col:
        property_type_val = st.selectbox(label="Property Type",options=['House', 'Upper Portion', 'Flat', 'Penthouse', 'Lower Portion',
       'Room', 'Farm House'])
    
    with city_col:
        city_val = st.selectbox(label="City",options=['Karachi', 'Islamabad', 'Lahore', 'Rawalpindi', 'Faisalabad'])
                
    with purpose_col:
        purpose_val = st.selectbox(label="Purpose",options=['For Rent', 'For Sale'])
        
    with bedrooms_col:
        bedrooms_val = st.number_input(label="Bedrooms",min_value=1,value=1,step=1)
    
    with baths_col:
        baths_val = st.number_input(label="Baths",min_value=1,value=1,step=1)
    
    with area_col:
        area_val = st.number_input(label="Area",min_value=0.01,value=1.00,step=0.01)

    if st.form_submit_button(label="Predict"):
        values = array([property_type_val,city_val,baths_val,bedrooms_val,area_val,purpose_val]).reshape(1,-1)
        df = DataFrame(values,columns=["property_type","city","baths","bedrooms","Area_in_Marla","purpose"])
        df.bedrooms = df.bedrooms.astype(int)
        df.baths = df.baths.astype(int)
        df.Area_in_Marla = df.Area_in_Marla.astype(float)
         
        # Feature Engineering
        df['bedroom_area'] = df['Area_in_Marla'] / df['bedrooms']
        df['bath_area'] = df['Area_in_Marla'] / df['baths']

        # Text Encoding
        df.property_type = property_type_encoder.transform(df.property_type.values.reshape(-1,1))

        df.city = city_encoder.transform(df.city.values.reshape(-1,1))
        
        df = df.reset_index().drop("index",axis=1).join(DataFrame(purpose_encoder.transform(df[['purpose']]), columns=purpose_encoder.get_feature_names_out(['purpose'])))
        df.drop("purpose",axis=1,inplace=True)

        # Feature Scaling
        df = scaler.transform(df)

        # Prediction
        pred = estimator.predict(df)
        
        st.subheader(f"{pred[0]:,.2f}💲")