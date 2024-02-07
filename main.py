import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Menambahkan logo dan nama situs di bagian atas
st.set_page_config(page_title="Restaurant Bill Estimator", page_icon=":fork_and_knife:")

st.title("Restaurant Bill Estimator")

st.image('img/cctv.png', width=350, use_column_width=True)

# Menambahkan emoticon restaurant berukuran besar pada sidebar
st.sidebar.markdown('<div style="text-align:center"><img src="https://emojicdn.elk.sh/🍽" width="150" height="150">', unsafe_allow_html=True)

st.sidebar.title("SIDEBAR")

page = st.sidebar.selectbox("Select Page", ["Tools","Information", "Dataset", "Summary"])

if page == "Tools":

    dt_model = pickle.load(open('final_DT_model.pkl', "rb"))
    xgb_model = pickle.load(open('final_XGB_model.pkl', 'rb')) 
    ann_model = load_model('final_ANN_model.h5')
    scaler = pickle.load(open('final_scaler_saved.pkl', "rb"))

    # Penjelasan singkat tentang alat pada sidebar
    st.sidebar.markdown("This tool helps estimate restaurant bills based on various factors.")

    st.markdown("This tool will analysed the time spent and bill amount relationship of customers in the restaurant. It applied 12 ML, and 1 ANN model and deployed them on Streamlit.")

    # Pilihan fitur contoh
    # Splitting the input fields into two columns
    col1, col2 = st.columns(2)

    # Column 1
    with col1:
        meal_type = st.selectbox("Meal Type", ['Breakfast', 'Lunch', 'Dinner'])
        table_location = st.selectbox("Table Location", ['Window', 'Center', 'Patrio'])
        weather_condition = st.selectbox("Weather Condition", ['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
        day = st.selectbox("Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        gender = st.radio("Gender", ['Male', 'Female', 'Other'])

    # Column 2
    with col2:
        age_group = st.selectbox("Age Group", ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        number_of_people = st.number_input("Number of People", min_value=1, value=1)
        time_spent = st.number_input("Time Spent (minutes)", min_value=30)
        reservation = st.radio("Reservation", [1, 0])
        live_music = st.radio("Live Music", [1, 0])

    # Additional input fields outside of columns

    customer_satisfaction = st.slider('Customer Satisfaction', 1, 5, 3)

    # Menyimpan data yang dimasukkan pengguna sebagai kamus
    user_data = {
        "Number of People": number_of_people,
        "Meal Type": meal_type,
        "Table Location": table_location,
        "Weather Condition": weather_condition,
        "Day": day,
        "Time Spent (minutes)": time_spent,
        "Gender": gender,
        "Reservation": reservation,
        "Age Group": age_group,
        "Live Music": live_music,
        "Customer Satisfaction": customer_satisfaction
    }

    # Mengonversi data yang dimasukkan pengguna menjadi DataFrame
    df_user = pd.DataFrame.from_dict([user_data])

    st.subheader("Entered Information:")
    st.table(df_user.style.set_table_styles([{
        'selector': 'td',
        'props': [
            ('background-color', '#2a2a2a'), # Warna latar belakang
            ('color', 'white'), # Warna teks
            ('font-family', 'Arial') # Jenis huruf
        ]
    }]))

    # Tambahkan kolom yang diharapkan oleh model Anda di sini.
    columns_expected_by_model = [
        'Number of People', 'Time Spent (minutes)',
        'Customer Satisfaction', 'Live Music_True', 'Reservation_True',
        'Meal Type_Dinner', 'Meal Type_Lunch', 'Day_Monday', 'Day_Saturday',
        'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday',
        'Gender_Male', 'Gender_Other', 'Table Location_Patio',
        'Table Location_Window', 'Age Group_26-35', 'Age Group_36-45',
        'Age Group_46-55', 'Age Group_56-65', 'Age Group_65+',
        'Weather Condition_Rainy', 'Weather Condition_Snowy',
        'Weather Condition_Sunny'
    ]
    df_user = pd.get_dummies(df_user).reindex(columns=columns_expected_by_model, fill_value=0)

    # Melakukan penskalaan data
    df_user_scaled = scaler.transform(df_user)

    # Melakukan prediksi dengan model Decision Tree
    dt_prediction = dt_model.predict(df_user_scaled)

    # Melakukan prediksi dengan model XGBoost
    xgb_prediction = xgb_model.predict(df_user_scaled)

    # Melakukan prediksi dengan model ANN
    ann_prediction = ann_model.predict(df_user_scaled)

    # Menampilkan perkiraan kepada pengguna
    st.success("Estimated Bill Amount with Decision Tree Model:   ${:.2f} ".format(int(dt_prediction[0])))
    st.success("Estimated Invoice Amount with XGBoost Model:   ${:.2f}".format(int(xgb_prediction[0])))
    st.success("Estimated Invoice Amount with ANN Model:   ${:.2f}".format(int(ann_prediction[0][0])))

elif page == "Information":
    st.header("Information")

    # Penjelasan singkat tentang alat pada sidebar
    st.sidebar.markdown("This page provides an overview of the analysis conducted on the restaurant dataset. It includes descriptive statistics and visualizations to explore various aspects of the data.")
    
    st.markdown("Descriptive Statistics")
                
    st.image('img/outlier_check.png', caption='Outliers Check', width=500, use_column_width=True)
    

    # Splitting the input fields into two columns
    col1, col2 = st.columns(2)

    # Column 1
    with col1:
        st.image('img/pairplot.png', caption='Pairplot', width=350, use_column_width=True)
        st.image('img/scatterplot.png', caption='On days with live music, individuals tend to spend more on their bills.', width=350, use_column_width=True)
        st.image('img/correlation.png', caption='Correlation', width=350, use_column_width=True)
    

    # Column 2
    with col2:
        st.image('img/barchart.png', caption='Bar Charts', width=350, use_column_width=True)
        st.image('img/scatterplot_timespend_billamount.png', caption='As the time spent by customers in the restaurant increases, the amount they spend on their bills also tends to increase.', width=350, use_column_width=True)
    
    
    
elif page == "Dataset":
    st.header("Dataset Explanation")

    # Penjelasan singkat tentang alat pada sidebar
    st.sidebar.markdown("This page provides an explanation of the dataset used in the analysis. It includes information about the features and their meanings.")
    
    df = pd.read_csv("data/restaurant_data.csv")

        # Menampilkan head dari DataFrame
    st.write("Restaurant Data:")
    st.write(df.head())

    st.markdown("""
    
    In this study; The relationship between the time customers spent in the restaurant and the amount they paid was analyzed using Machine Learning and Artificial Neural Network (ANN) methods. It is planned to obtain information such as the time customers spend in the restaurant, the table they sit at and the number of people at the table, from the security camera. The data used and their explanations are as follows:

    **Day:** The day the customer came to the restaurant

    **Entry Time:** The time the customer enters the restaurant

    **Exit Time:** The time the customer leaves the restaurant

    **Meal Type:** Breakfast, Lunch, Dinner

    **Number of People:** How many people the customer came to the restaurant with (How many people he ate at the table with)

    **Time Spent (minutes):** How long the customer spent in the restaurant

    **Bill Amount ($):** How much the customer paid (account amount paid per person)

    **Gender:** Male, Female, Other

    **Table Location:** Window, Patio (Veranda, courtyard), Center

    **Reservation:** Information whether the customer has made a reservation in advance.

    **Customer Satisfaction:** The degree of satisfaction the customer feels with the food he ate and the service he received ((points between 1-5)

    **Live Music:** Information about whether live music is performed in the restaurant while eating

    **Age Group:** Customer's age group. 18-25; 26-35; 36-45; 46-55; 56-65; 65+

    **Weather Condition:** Cloudy, Sunny, Sbowy, Rainy

    To this data set; You can access it at [this link](https://www.kaggle.com/datasets/ismetgocer/time-spent-bill-amount-data-of-restaurants).
    """)

elif page == "Summary":  
    st.header("Summary of Data Analysis and Modeling Process Followed in This Study")

    st.sidebar.markdown("This page summarizes the data analysis and modeling process followed in this study. It includes information about data preparation, model development, final models, and predictions.")

    st.markdown('<div style="text-align:center"><img src="https://emojicdn.elk.sh/🗒️" width="200" height="200"></div>', unsafe_allow_html=True)

    st.markdown("""
    This document summarizes the steps of the data analysis and modeling process performed using artificial learning (ANN) and machine learning (ML) techniques.

    ## Data Preparation

    1. **Encoding**: We transformed the data to express it as 0s and 1s.
    2. **Train-Test Split**: We divided the data set into training and test sets.
    3. **Scaling**: We performed scaling to make the values smaller.

    ## Model Development

    1. **ANN Model**: We built and trained the Artificial Neural Network model.
    2. **Machine Learning Methods**: We made predictions using 12 different ML methods. While we achieved high scores in the training set, we observed low scores in the test set. This indicates overfitting.
    3. **GridsearchCV**: To solve the overfitting problem, we determined the best hyperparameters by applying GridsearchCV for Decision Tree (DT) and XGBoost (XGB) models.

    ## Final Model

    1. **Data Re-Preparation**: We read the data from scratch and re-applied the encoding, labeling and scaling processes. The scaling process has also been exported to be used for new incoming data in Streamlit.
    2. **Final Models**: We trained Final DT and Final XGB models using the entire dataset and the best hyperparameters identified.
    3. **Saving Model**: We can export final ML models with pickle or joblib.
    4. **Final ANN Model**: Using the entire dataset, the Final ANN model was trained with 1500 Epoch and exported in h5 or keras format.

    ## Prediction

    1. **Data Preparation and Scaling**: We reloaded the exported scalers and models. Scaling was applied for the selected observation (row).
    2. **Model Predictions**: Predictions were made with Final DT, Final XGB and Final ANN models.
    3. **Prediction for New Observation**: Data for a new observation coming from outside was entered manually, and encoding and scaling processes were applied. This data was estimated with Final DT, Final XGB and Final ANN models.
    """)