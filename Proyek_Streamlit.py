#!/usr/bin/env python
# coding: utf-8

# # UAS Project Streamlit: 
# - **Nama:** Marutha Wira Yuda
# - **Dataset:** https://www.kaggle.com/datasets/ismetgocer/time-spent-bill-amount-data-of-restaurants
# - **URL Website:** [Di isi jika web streamlit di upload]

# ## 1. Menentukan Pertanyaan Bisnis

# - Bagaimana variabel-variabel seperti waktu yang dihabiskan, jumlah tagihan, dan faktor-faktor lainnya berhubungan dalam konteks restoran? Apakah ada pola atau tren yang dapat membantu meningkatkan pengalaman pelanggan atau efisiensi operasional?
# - Apakah ada faktor tertentu seperti jenis makanan, lokasi meja, atau kondisi cuaca yang secara signifikan mempengaruhi kepuasan pelanggan? Bagaimana variabel-variabel ini dapat dioptimalkan untuk meningkatkan tingkat kepuasan pelanggan?

# ## 2. Import Semua Packages/Library yang Digunakan

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import applications

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

# ## 3. Data Wrangling

# ### 3.1. Gathering Data

# In[2]:


df = pd.read_csv("data/restaurant_data.csv")
df.head(3)

# ### 3.2. Cleaning Data

# In[3]:


df.drop(['Entry Time', 'Exit Time'], axis=1, inplace=True)
df.head(2)

# ## 4. Exploratory Data Analysis (EDA)

# ### 4.1. Information

# In[4]:


df.info()

# In[5]:


df.head(2)

# ### 4.2 Null Check

# In[6]:


df.isna().sum()

# ### 4.3. Shape

# In[7]:


df.shape

# ### 4.4. Unique Value

# In[8]:


def get_unique_values(df):
    
    output_data = []

    for col in df.columns:

        # If the number of unique values in the column is less than or equal to 10
        if df.loc[:, col].nunique() <= 10:
            # Get the unique values in the column
            unique_values = df.loc[:, col].unique()
            # Append the column name, number of unique values, unique values, and data type to the output data
            output_data.append([col, df.loc[:, col].nunique(), unique_values, df.loc[:, col].dtype])
        else:
            # Otherwise, append only the column name, number of unique values, and data type to the output data
            output_data.append([col, df.loc[:, col].nunique(),"-", df.loc[:, col].dtype])

    output_df = pd.DataFrame(output_data, columns=['Column Name', 'Number of Unique Values', ' Unique Values ', 'Data Type'])

    return output_df

get_unique_values(df)

# ### 4.5. Column Names

# In[9]:


df.columns

# ### 4.6. Descriptive Statistics

# In[10]:


df.describe().T

# ### 4.7. Correlation

# In[11]:


# I selected only numeric columns and I am examining the correlation among them
df.select_dtypes(include=[np.number]).corr()

# In[12]:


sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot = True);

# ### 4.8. Pairplot

# In[13]:


sns.pairplot(df, kind = "reg", diag_kind = "kde", diag_kws={"color":"red"}, plot_kws={"line_kws":{"color":"red"}});

# ### 4.9. Outliers Check

# In[14]:


# Let's draw boxplots and histplots for checking distributions of features;
index=0
for feature in df.select_dtypes('number').columns:
    index+=1
    plt.figure(figsize=(40,40))
    plt.subplot((len(df.columns)),2,index)
    sns.boxplot(x=feature,data=df,whis=3) 
        
    plt.tight_layout()
    
    plt.show()

# ### 4.10. Bar Charts

# In[15]:


# Amounts paid according to categorical data
# List of categorical columns
categorical_columns = ['Day', 'Meal Type', 'Gender', 'Table Location', 'Reservation', 'Customer Satisfaction', 'Live Music', 'Age Group', 'Weather Condition']

# Creating a bar chart showing the average 'Bill Amount ($)' for each categorical column
plt.figure(figsize=(20, 20))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)  # 3x3 grid layout
    barplot = sns.barplot(x=column, y='Bill Amount ($)', data=df, ci=None)  # Confidence interval removed
    plt.title(f'Average Bill Amount by {column}')
    plt.xticks(rotation=45)
    
    # Displaying values on top of the bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.2f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 10), 
                         textcoords='offset points')

plt.tight_layout()
plt.show()

# In[16]:


# Amounts of time spent according to categorical data
# List of categorical columns
categorical_columns = ['Day', 'Meal Type', 'Gender', 'Table Location', 'Reservation', 'Customer Satisfaction', 'Live Music', 'Age Group', 'Weather Condition']

# Creating a bar chart showing the average 'Time Spent (minutes)' for each categorical column
plt.figure(figsize=(20, 20))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)  # 3x3 grid layout
    barplot = sns.barplot(x=column, y='Time Spent (minutes)', data=df, ci=None)  # Confidence interval removed
    plt.title(f'Time Spent (minutes) by {column}')
    plt.xticks(rotation=45)
    
    # Displaying values on top of the bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.2f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 10), 
                         textcoords='offset points')

plt.tight_layout()
plt.show()

# ### 4.11. Scatterplot

# In[17]:


sns.scatterplot(x="Time Spent (minutes)", y= "Bill Amount ($)", data = df, hue = "Live Music");

# **Insight:** On days with live music, individuals tend to spend more on their bills.

# In[18]:


# Time Spent (minutes) ile Bill Amount ($) arasındaki ilişkiyi scatter plot ile inceleme ve eğilim çizgisi ekleme
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Time Spent (minutes)', y='Bill Amount ($)', data=df)
sns.regplot(x='Time Spent (minutes)', y='Bill Amount ($)', data=df, scatter=False, color='red')  # Kırmızı eğilim çizgisi
plt.title('Relationship between Time Spent and Bill Amount')
plt.show()

# **Insight:** As the time spent by customers in the restaurant increases, the amount they spend on their bills also tends to increase.

# ## 5. Visualization & Explanatory Analysis

# ### Pertanyaan 1:
# Apakah terdapat korelasi antara waktu yang dihabiskan dan jumlah tagihan? Dapatkah visualisasi membantu mengidentifikasi pola-pola tertentu, seperti apakah waktu makan malam cenderung memiliki tagihan yang lebih tinggi?
# 

# ### Pertanyaan 2:
# Bagaimana distribusi kepuasan pelanggan berdasarkan jenis makanan atau lokasi meja? Dapatkah visualisasi membantu menyoroti faktor-faktor tertentu yang berkaitan dengan kepuasan pelanggan?

# ### 5.1 Deep Learning

# #### 5.1.1. Encoding
# 
# Untuk mengonversi variabel kategori seperti hari, kondisi cuaca, dll. menjadi nilai numerik dalam kumpulan data.

# In[19]:


df.head(2)

# In[20]:


df.columns

# In[21]:


# Applying "One-Hot Encoding" to data where we believe the order is not important
df = pd.get_dummies(df, columns=['Live Music', 'Reservation', 'Meal Type', 'Day', 'Gender', 'Table Location', 'Age Group', 'Weather Condition'], drop_first=True)

# In[22]:


df.head(2)

# In[23]:


# Converting boolean data type to integer

for column in df.columns:
    if df[column].dtype == 'bool':
        df[column] = df[column].astype(int)

# simpan pengkodeannya

# In[24]:


# Save the encoded dataset as a pickle file
# I did this once initially, and after exporting the encoding process, I read the data from the beginning
# and skipped this part to preserve the entire df. The reason for saving this is to be able to encode new data
# during the prediction stage. Since the new incoming data won't have the "Bill Amount ($)" data, 
# I dropped it and saved the encoded version.
"""df.drop(["Bill Amount ($)"], axis=1).to_pickle('encoded_data.pkl')
df.to_pickle('encoded_data.pkl')"""

# ##### Muat Ulang Data dan Siapkan untuk Analisis
# 
# Pada baris di atas, kami menghapus "Jumlah Tagihan ($)", tetapi kami memerlukannya untuk analisis. Oleh karena itu, mari kita baca datanya dari awal dan ulangi proses pengkodeannya.

# In[25]:


# Let's reload the data
df = pd.read_csv("data/restaurant_data.csv") 

# Drop entry-exit times
df.drop(['Entry Time', 'Exit Time'], axis=1, inplace=True)

# Perform encoding
df = pd.get_dummies(df, columns=['Live Music', 'Reservation', 'Meal Type', 'Day', 'Gender', 'Table Location', 'Age Group', 'Weather Condition'], drop_first=True)

# Convert the "Boolean" data type generated during encoding to integer
for column in df.columns:
    if df[column].dtype == 'bool':
        df[column] = df[column].astype(int)

df.head(2)

# #### 5.1.2. Labelling

# In[26]:


X = df.drop(["Bill Amount ($)"], axis = 1)
y = df["Bill Amount ($)"]

# #### 5.1.3. Split Train & Test

# In[27]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42)

# #### 5.1.4. Scalling the Data

# In[28]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = MinMaxScaler()

# In[29]:


X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# #### 5.1.5. Membuat Model 

# ##### 5.1.5.1. Training Model

# In[30]:


X_train.shape

# In[31]:


# Setting up the DL model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

seed = 101

tf.random.set_seed(seed)  # Ensures reproducibility by generating random numbers around a certain seed.

model = Sequential()  # We will use a layered, sequential structure.

# 1st Hidden Layer
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))  # X_train.shape[1] specifies the number of features used.
# model.add(Dropout(0.2))  # Turn off/zero out 20% of neurons for better learning.

# 2nd Hidden Layer
model.add(Dense(32, activation='relu'))  # There will be 32 neurons in the hidden layer.
# model.add(Dropout(0.2))  # Using this led to worse scores.

# 3rd Hidden Layer
model.add(Dense(16, activation='relu'))  # ReLU activation function is commonly used in intermediate layers.
# model.add(Dropout(0.2))

# 4th Hidden Layer
model.add(Dense(8, activation='relu'))  # Reducing the number of neurons towards the end is a best practice.
# model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1))  # For regression, Dense should be 1!!!

# Define the optimizer
optimizer = Adam(lr=0.003)  # Default learning rate value is 0.001. This can be set to 0.002 / 0.003. We are using Adam optimizer here, Gradient Descent can also be used.

# Compile the model
model.compile(optimizer='adam', loss='mse')  # This line is correct! For regression analysis, it's important that loss='mse'.

# Early Stop
early_stop = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=25)  # "patience=25" means wait for 25 epochs, if a better score does not come, stop.
# Patience is usually given as 15, 20, 25, etc.
# mode = "auto" means better when the loss value drops.

# Model Summary
model.summary()

# ##### 5.1.5.2. Fit the Model

# In[32]:


model.fit(x = X_train, y = y_train, validation_split = 0.15, batch_size = 128, epochs = 100) # If the epoch is 1000, it takes 20-25 minutes
# validation_split = 0.15 shows how much of the data in the train data we evaluate as validation. Cross-Valuiidation ratio

# ##### 5.1.5.3. Model History

# In[33]:


pd.DataFrame(model.history.history)

# ##### 5.1.5.4. Evaluasi Model

# In[34]:


loss_df = pd.DataFrame(model.history.history)# Bu iki egri birbirine yakin olmalidir. 
loss_df.plot();

# In[35]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    score = r2_score(actual, pred)
    return print("r2_score:", score, "\nmae:", mae, "\nmse:", mse, "\nrmse:", rmse)

# In[36]:


model.evaluate(X_test, y_test, verbose=0) # Loss amount in the test data

# In[37]:


y_pred = model.predict(X_test)

# In[38]:


eval_metric(y_test, y_pred)  # Run the model again without making any changes, the result may improve
# The reason for this is that the weights and biases are randomly selected.
# If ANN acts randomly, it's called "Hallucination," run the model again, it may get better.
# If it still doesn't improve, you can increase the epochs and modify other parameters.

# **Hasil:** Hasil ANN tidak sesuai harapan. Kami akan melanjutkan dengan Machine Learning (ML) tradisional.

# ### 5.2. Machine Learning
# 
# Di sini, 12 metode ML berbeda akan dijalankan secara bersamaan

# In[39]:


from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# In[40]:


ridge=Ridge().fit(X_train, y_train)
lasso=Lasso().fit(X_train, y_train)
enet=ElasticNet().fit(X_train, y_train)
knn=KNeighborsRegressor().fit(X_train, y_train)
ada=AdaBoostRegressor().fit(X_train, y_train)
svm=SVR().fit(X_train, y_train)
dtc=DecisionTreeRegressor().fit(X_train, y_train)
rf=RandomForestRegressor().fit(X_train, y_train)
xgb=XGBRegressor().fit(X_train, y_train)
gbm=GradientBoostingRegressor().fit(X_train, y_train)
lgb=LGBMRegressor().fit(X_train, y_train) # LightGBM
catbost=CatBoostRegressor().fit(X_train, y_train)

# #### 5.2.1. Get Scores for the Training Data

# In[41]:


models=[ridge,lasso,enet,knn,ada,svm,dtc,rf,xgb,gbm,lgb,catbost]

def ML(y,models):
    r2_score=models.score(X_train, y_train)
    return r2_score

# In[42]:


for i in models:
     print(i,"Algorithm succed rate :", ML("Bill Amount ($)",i))

# #### 5.2.2. Get Scores for the Test Data

# In[43]:


# We will get scores for the test data;
def ML(y, models):
    r2_score = models.score(X_test, y_test)
    return r2_score

# In[44]:


for i in models:
     print(i,"Algorithm succed rate :",ML("Bill Amount ($)",i))

# **Komentar:** Keberhasilan metode "DecisionTreeRegressor", "XGBRegressor", dan "GradientBoostingRegressor" sangat tinggi pada data pelatihan namun menurun pada data pengujian. Hal ini menunjukkan adanya overfitting. Untuk mengatasinya mari kita lakukan GridSearchCV.

# #### 5.2.3. Determine Optimal Hyperparameters with GridSearchCV

# ##### 5.2.3.1. Perform GridSearchCV for DecisionTreeRegressor

# In[45]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Set up the parameter grid
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2", None]
}

# Create the DecisionTreeRegressor model
dt_model = DecisionTreeRegressor(random_state=101)

# Create the GridSearchCV object
grid_dt_model = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=10, n_jobs=-1)

# Train the model
grid_dt_model.fit(X_train, y_train)

# Print the best score and parameters
print("Best Score:", grid_dt_model.best_score_)
print("Best Parameters:", grid_dt_model.best_params_)

# In[46]:


best_grid_model = grid_dt_model.best_estimator_

# Best Score: 0.77081479548352
# 
# Best Parameters: {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2}

# ##### 5.2.3.2. Perform GridSearchCV for XGBoostRegressor

# In[47]:


from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators": [100, 300, 500], 'max_depth': [3, 5, 6, 7], "learning_rate": [0.05, 0.1, 0.2],
              "subsample": [0.5, 1], "colsample_bytree": [0.5, 1]}

xgb_model = XGBRegressor(booster='gblinear', random_state=101, silent=True, objective="reg:squarederror")
grid_xgb_model = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10, n_jobs=-1)

# Train the model
grid_xgb_model.fit(X_train, y_train)

# Print the best score and parameters
print("Best Score:", grid_xgb_model.best_score_)
print("Best Parameters:", grid_xgb_model.best_params_)

# Best Score: 0.7696271078817388
# 
# Best Parameters: {'colsample_bytree': 0.5, 'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.5}

# In[48]:


best_XGB_grid_model=grid_xgb_model.best_estimator_

# ### 5.3. Final Model
# 
# Model akhir menggunakan semua data.

# In[50]:


# To avoid any mistakes, we start by reading the data again and performing the encoding process

# Let's read the data again;
df = pd.read_csv("data/restaurant_data.csv") 

# Drop entry and exit times
df.drop(['Entry Time', 'Exit Time'], axis=1, inplace=True)

# Perform encoding
df = pd.get_dummies(df, columns=['Live Music', 'Reservation', 'Meal Type', 'Day', 'Gender', 'Table Location', 'Age Group', 'Weather Condition'], drop_first=True)

# Convert the "Boolean" data type generated in encoding to integer
for column in df.columns:
    if df[column].dtype == 'bool':
        df[column] = df[column].astype(int)
df.head(2)


# #### 5.3.1. Labelling

# In[51]:


X = df.drop(["Bill Amount ($)"], axis = 1)
y = df["Bill Amount ($)"]

# #### 5.3.2. Scaling

# In[52]:


# Let's import the libraries we will use in the scaling process
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = MinMaxScaler()


# In[53]:


X= scaler.fit_transform(X)

# In[54]:


# Let's save the scaler externally. Because in Streamlit, when predicting new data with the Final models, we will need to scale them with the scales we used.
import pickle
pickle.dump(scaler, open("final_scaler_saved.pkl", 'wb'))  # write binary

# #### 5.3.3. Final DT Model

# In[55]:


# Let's predict the final DT model using the optimal hyperparameters we defined above and the entire data set
from sklearn.tree import DecisionTreeRegressor
final_DT_model = DecisionTreeRegressor(max_depth=5, max_features='auto', min_samples_leaf=1, min_samples_split=2).fit(X, y)

# ##### 5.3.3.1. Menyimpan Model

# **IMPORTANT: ML models need to be saved using joblib or pickle.**
# 
# **Save with pickle**
# import pickle
# pickle.dump(final_DT_model, open("final_DT_model.pkl", 'wb'))
# 
# **Alternatively (both do the same operation)**
# with open('final_DT_model.pkl', 'wb') as file:
#     pickle.dump(final_DT_model, file)
# 
# **Load the model saved with pickle**
# with open('final_DT_model.pkl', "rb") as file:
#     loaded_DT_model = pickle.load(file)
# 
# **----------------------------------------**
# 
# **Save with joblib**
# from joblib import dump
# dump(final_DT_model, 'final_DT_model.joblib')
# 
# **Load the model saved with joblib**
# from joblib import load
# loaded_DT_model_joblib = load('final_DT_model.joblib')
# 
# **----------------------------------------**
# 
# **DL models (Keras-Tensorflow) should be saved as h5 or keras!**
# 
# **Save as h5**
# final_DT_model.save("final_DT_model.h5")
# 
# **Load the model saved as h5**
# from tensorflow.keras.models import load_model
# loaded_DT_model_h5 = load_model("final_DT_model.h5")
# 
# **----------------------------------------**
# 
# **Save as Keras**
# final_DT_model.save("final_DT_model.keras")
# 
# **Load the model saved as Keras**
# from tensorflow.keras.models import load_model
# loaded_DT_model_keras = load_model("final_DT_model.keras")
# 

# In[56]:


# Save the Final DT model
import pickle
pickle.dump(final_DT_model, open("final_DT_model.pkl", 'wb'))

# In[57]:


# #from joblib import dump (This is another way, but we haven't used it for now)

# Save your model
#dump(final_DT_model, 'final_DT_model.joblib')

# ##### 5.3.3.2. SHAP

# In[58]:


import shap

explainer = shap.Explainer(final_DT_model.predict, X)
shap_values = explainer(X)

# In[59]:


df_features= df.drop('Bill Amount ($)', axis = 1)
df_features.head(2)

# In[60]:


shap.summary_plot(shap_values, features=X, feature_names=df_features.columns)

# **SHAP (SHapley Additive exPlanations)** analysis is used particularly for interpreting predictions of machine learning models, such as decision trees. SHAP values quantify the contribution and importance of each feature to a prediction in a numerical manner. When interpreting the resulting SHAP analysis graph, consider the following key points:
# 
# **Importance of Features:** SHAP values illustrate the magnitude of each feature's contribution to the model prediction. Features in your graph are arranged based on their contributions. The feature at the top is generally the one that has the largest impact on model predictions.
# 
# **Positive and Negative Effects:** The SHAP values of each feature indicate whether that feature has a positive or negative effect on the prediction. Positive values indicate that the feature increases the prediction (e.g., directing towards the positive class in a classification model), while negative values indicate a decrease in the prediction (directing towards the negative class).
# 
# **Value Distribution:** By examining the distribution of SHAP values for each feature, you can understand how your model responds to different feature values. A wide distribution indicates that the model exhibits significant variations for different values of that feature.
# 
# **Coloring:** SHAP graphs often use coloring for each feature value. This signifies the direction of the effect of feature values on predictions. For instance, red may indicate positive effects when high values contribute positively, while blue may indicate positive effects when low values contribute positively.
# 
# **Interactions:** Some SHAP graphs also display interactions between features, showing how two features together impact the model. This helps you understand the joint effect of two features on the model.
# 
# SHAP analysis assists in understanding which features are crucial for your model and how these features contribute to predictions. This information can be utilized to enhance model performance, make feature selections, or explain model decisions to end-users.

# **KOMENTAR:** Jumlah yang akan dibayarkan pelanggan dipengaruhi secara signifikan oleh "pengeluaran waktu", dengan dampak positif jika tinggi dan dampak negatif jika rendah. Dampak negatif dari rendahnya waktu yang dihabiskan lebih terasa. Oleh karena itu, upaya harus dilakukan untuk mendorong pelanggan menghabiskan lebih banyak waktu di restoran.
# 
# Tingkat “kepuasan pelanggan” yang tinggi berpengaruh positif dan signifikan terhadap jumlah yang harus dibayar, sedangkan “kepuasan pelanggan” yang rendah mempunyai dampak negatif namun dalam jumlah yang lebih kecil. Pola serupa juga berlaku untuk "musik live".

# In[61]:


feature_names=df_features.columns
feature_names

# In[62]:


# It shows the absolute values of the effects.

shap.plots.bar(shap_values)

# In[63]:


y.value_counts().sum() # will display the total number of people.

# In[64]:


shap.plots.waterfall(shap_values[1999])
# Here, entering one less than the number of observations is intended to use their index numbers. 
# While there are 2000 people, the index number of the 2000th person is 1999. 
# The waterfall plot below will provide information about the person with the index number 1999 in the list.

# In[65]:


df.columns

# **KOMENTAR:** Warna biru menunjukkan efek negatif, warna merah menunjukkan efek positif.
# **Faktor yang secara positif mempengaruhi jumlah pelanggan yang paling banyak membayar adalah 'Waktu yang Dihabiskan (menit)', sedangkan rendahnya kepuasan pelanggan dan sedikitnya jumlah orang dalam grup berpengaruh negatif terhadap jumlah tagihan.**

# ##### 5.3.3.3. Feature Importance

# In[66]:


df_f_i = pd.DataFrame(index=df_features.columns, data = final_DT_model.feature_importances_,
                     columns = ["feature Importance"]).sort_values("feature Importance")
df_f_i

# In[67]:


final_DT_model.feature_importances_

# In[68]:


sns.barplot(x=df_f_i.index, y = "feature Importance", data = df_f_i)
plt.xticks(rotation = 90)
plt.tight_layout()

# #### 5.3.4. Final XGB Model

# In[69]:


# Let's predict the final XGB model using the optimum hyperparameters we specified above and the entire dataset;
from xgboost import XGBRegressor
final_XGB_model = XGBRegressor(colsample_bytree=0.5, learning_rate=0.2, max_depth=3, n_estimators=500, subsample=0.5).fit(X, y)

# ##### 5.3.4.1. Save the Final XGB Model

# In[70]:


# Save the final XGB model
import pickle
pickle.dump(final_XGB_model, open("final_XGB_model.pkl", 'wb'))

# In[71]:


#from joblib import dump

# Save your model
#dump(final_XGB_model, 'final_XGB_model.joblib')

# ##### 5.3.4.2. SHAP

# In[72]:


import shap
explainer = shap.Explainer(final_XGB_model.predict, X)
shap_values = explainer(X)

# In[73]:


shap.summary_plot(shap_values, features=X, feature_names=df_features.columns)

# In[74]:


# Display the absolute values of the effects.
shap.plots.bar(shap_values)

# In[75]:


y.value_counts().sum()  # Show the number of individuals in the analysis.

# In[76]:


shap.plots.waterfall(shap_values[1999]) 
# Here, the purpose of entering one less than the number of observations is to use their index numbers.
# When there are 2000 individuals, the index number of the 2000th individual is 1999.
# The following plot will generate information about the individual with index number 1999 in the list.

# In[77]:


df.columns

# **KOMENTAR:** Faktor yang paling berpengaruh positif terhadap jumlah yang harus dibayar adalah 'Waktu yang Dihabiskan (menit)' dengan nomor indeks 1.
# 
# **Faktor dengan indeks nomor 3, 'Kepuasan Pelanggan', dapat mempengaruhi besarannya baik secara positif maupun negatif. Dengan kata lain, ini berarti bahwa akun akan terkena dampak negatif ketika 'Kepuasan Pelanggan' rendah dan terkena dampak positif ketika 'Kepuasan Pelanggan' tinggi.**
# 
# **Meskipun faktor dengan indeks nomor 4, 'Live Music_True,' tampaknya tidak memiliki dampak yang signifikan dalam Pohon Keputusan (DT), faktor tersebut diamati memiliki pengaruh positif terhadap jumlah di XGBoost (XGB).**

# ##### 5.3.4.3. Feature Importance

# In[78]:


df_f_i = pd.DataFrame(index=df_features.columns, data = final_XGB_model.feature_importances_,
                     columns = ["feature Importance"]).sort_values("feature Importance")
df_f_i

# In[79]:


final_XGB_model.feature_importances_

# In[80]:


sns.barplot(x=df_f_i.index, y = "feature Importance", data = df_f_i)
plt.xticks(rotation = 90)
plt.tight_layout()

# **KOMENTAR:** Berdasarkan grafik ini, **variabel yang paling mempengaruhi jumlah yang harus dibayar di XGBoost (XGB) adalah, secara berurutan: waktu yang dihabiskan di restoran, keberadaan musik live, kepuasan pelanggan, dan adanya reservasi. Sementara itu, dalam Decision Trees (DT), faktor-faktor lain yang tampaknya tidak penting masih mempunyai pengaruh yang kecil terhadap jumlah yang harus dibayarkan.**

# #### 5.3.5. Final ANN Model

# In[81]:


# Setting up the DL model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

seed = 101

tf.random.set_seed(seed)  # Setting a seed for reproducibility of random number generation.

ANN_model = Sequential()  # Using a layered, sequential structure for the model.

# 1. Hidden Layer
ANN_model.add(Dense(32, input_dim=X.shape[1], activation='relu'))  # 'X.shape[1]' specifies the number of features used.
# ANN_model.add(Dropout(0.2))  # Turn off 20% to enhance learning.

# 2. Hidden Layer
ANN_model.add(Dense(32, activation='relu'))  # 32 neurons in the hidden layer.
# ANN_model.add(Dropout(0.2))  # Using dropout deteriorated the scores.

# 3. Hidden Layer
ANN_model.add(Dense(16, activation='relu'))  # ReLU activation function is commonly used in intermediate layers.
# ANN_model.add(Dropout(0.2))

# 4. Hidden Layer
ANN_model.add(Dense(8, activation='relu'))  # Reducing the number of neurons towards the end is a best practice.
# ANN_model.add(Dropout(0.2))

# Output Layer
ANN_model.add(Dense(1))  # For regression, 'Dense' should be 1!!!

# Define the optimizer
optimizer = Adam(lr=0.003)  # Default learning rate is 0.001. Can be set to 0.002 / 0.003. Using Adam optimizer here.

# Compile the model
ANN_model.compile(optimizer='adam', loss='mse')  # For regression analysis, it's important to have 'loss='mse''. 

# Early Stop
early_stop = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=25)  # "patience=25" means wait for improvement for 25 epochs.
# 'patience' is usually set to values like 15, 20, 25.
# 'mode="auto"' means better when the loss value decreases.

# Model Summary
ANN_model.summary()

# In[82]:


ANN_model.fit(x = X, y = y, batch_size = 128, epochs = 2000)

# In[83]:


# To save the model in the h5 format:
ANN_model.save("final_ANN_model.h5")


# In[84]:


# To save the model in the Keras format (not used for now):
# ANN_model.save("final_ANN_model.keras")

# ##### 5.3.5.1. Eval Metrics

# In[85]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    score = r2_score(actual, pred)
    return print("r2_score:", score, "\nmae:", mae, "\nmse:", mse, "\nrmse:", rmse)

# In[86]:


# Scores for Decision Tree (DT)
y_pred = final_DT_model.predict(X)
eval_metric(y, y_pred)


# In[87]:


# scores for XGB
y_pred = final_XGB_model.predict(X)
eval_metric(y, y_pred)

# In[88]:


# scores for ANN
y_pred = ANN_model.predict(X)
eval_metric(y, y_pred)

# **KOMENTAR:** Tampaknya model yang paling sukses adalah model ANN.

# ### 5.4. Prediction

# #### 5.4.1. Loading Scaler and Models 

# In[89]:


from tensorflow.keras.models import load_model
import pickle

final_scaler = pickle.load(open("final_scaler_saved.pkl", "rb"))
DT_model = pickle.load(open('final_DT_model.pkl', "rb"))
XGB_model = pickle.load(open('final_XGB_model.pkl', "rb"))
ANN_model = load_model('final_ANN_model.h5')                      

# #### 5.4.2. Prediction
# 
# Di sini saya menggunakan pengkodean data.
# 
# Jika kita akan membaca data dari awal, maka perlu dilakukan proses pengkodean dengan cara membaca ulang **encoded_data.pkl**.

# In[90]:


df.head(1)

# In[91]:


# Let's make an observation from the current data set we have;

Customer_A = df.drop('Bill Amount ($)', axis = 1).iloc[0:1, :]   # We are currently attracting the person with index 0. If we write 2:3, you will attract the customer with index number 2 (3rd place on the list).
Customer_A

# In[92]:


# Let's apply scaling to this data;
Customer_A_Scaled = final_scaler.transform(Customer_A)
Customer_A_Scaled

# ##### 5.4.2.1. Prediction with DT

# In[93]:


DT_model.predict(Customer_A_Scaled)

# In[94]:


# Account actually paid
df.iloc[0][2]

# **Komentar:** Pelanggan dengan indeks 0 sebenarnya membayar $117,08. Model DT memperkirakannya sebesar 99,91 USD.

# ##### 5.4.2.2. Prediction with XGB

# In[95]:


XGB_model.predict(Customer_A_Scaled)

# **Komentar:** Pelanggan dengan indeks 0 sebenarnya membayar 117,66 USD. Model XGB memperkirakannya seharga 98,8 USD.

# ##### 5.4.2.3. Prediction with ANN

# In[96]:


ANN_model.predict(Customer_A_Scaled)

# **Komentar:** Pelanggan dengan indeks 0 sebenarnya membayar $117,08. Model ANN memperkirakannya pada 104,46 USD.

# ##### 5.4.2.4. Prediction Using New Observed Values

# In[97]:


df.columns

# In[98]:


data = {
    "Day": ['Monday'],
    "Meal Type": ['Dinner'],
    "Number of People": [2],
    "Time Spent (minutes)": [125],
    "Gender": ['Male'],
    "Table Location": ['Window'],
    "Reservation": [1],
    "Customer Satisfaction": [4],
    "Live Music": [1],
    "Age Group": ['18-25'],
    "Weather Condition": ['Cloudy']
}

# Create a new DataFrame
df_new = pd.DataFrame(data)
df_new

# In[99]:


# WE ARE ENCODING df_new

encoded_data = pd.read_pickle('encoded_data.pkl')

# Apply the same encoding to df_new
df_new_encoded = pd.get_dummies(df_new, columns=['Live Music', 'Reservation', 'Meal Type', 'Day', 'Gender', 'Table Location', 'Age Group', 'Weather Condition'], drop_first =True)

# Converting the data type of the converted columns to int
for column in df_new_encoded.columns:
     if df_new_encoded[column].dtype == 'bool':
        df_new_encoded[column] = df_new_encoded[column].astype(int)
        
# Ensure that the columns in df_new_encoded are the same as in the original encoding
# This is important to make sure the order and presence of columns are consistent

df_new_encoded = df_new_encoded.reindex(columns=encoded_data.columns, fill_value=0)
df_new_encoded

# **Penjelasan:** Saat melakukan pengkodean, ia memberikan nilai 0 pada data yang dianggap ada di kumpulan data asli tetapi tidak dapat ditemukan di kumpulan data baru.
# 
# Kita tahu bahwa "Jumlah Tagihan ($)" bukan 0. Kami bahkan mencoba menebaknya. Jadi mari kita hilangkan kolom ini. Mari kita minta model untuk menemukan nilai ini.

# In[100]:


df_new_encoded.drop(['Bill Amount ($)'], axis=1, inplace=True)
df_new_encoded

# Saat pertama kali menyimpan bagian "encoded_data.pkl"; Karena ini menampilkan musik live dan reservasi sebagai 0, itu mengkodekannya sebagai 0 meskipun kami mengatakan ada musik live dan reservasi.

# In[101]:


# SCALING

df_new_encoded_Scaled = final_scaler.transform(df_new_encoded)
df_new_encoded_Scaled

# In[102]:


DT_model.predict(df_new_encoded_Scaled)

# In[103]:


XGB_model.predict(df_new_encoded_Scaled)

# In[104]:


ANN_model.predict(df_new_encoded_Scaled)

# **KOMENTAR:** Pelanggan ini diharapkan membayar 89,1 USD pada akun sesuai dengan model ANN.

# ## Conclusion

# - Berdasarkan analisis, terlihat bahwa waktu yang dihabiskan memiliki korelasi positif dengan jumlah tagihan. Oleh karena itu, strategi untuk meningkatkan waktu tinggal pelanggan mungkin dapat berdampak positif pada pendapatan restoran.
# 
# - Dari hasil analisis, jenis makanan dan lokasi meja mungkin memiliki pengaruh signifikan pada kepuasan pelanggan. Restoran dapat mempertimbangkan untuk mengoptimalkan pengalaman pelanggan dengan menyesuaikan penawaran makanan atau mengelola tata letak meja.
