import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pydeck as pdk

st.title('🤖 AppoML')

st.info('This is app builds a machine learning model for predicting the species of penguin')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')


# Sidebar for user interaction
st.sidebar.header('User Input Features')

penguins = df
# Display the dataset
st.header('Penguins Dataset')
st.write(penguins.head())

# Display basic information about the dataset
st.subheader('Basic Information')
st.write(penguins.describe())

# Show number of missing values
st.subheader('Missing Values')
st.write(penguins.isnull().sum())

# Replace missing values (if needed)
penguins.fillna(method='ffill', inplace=True)

# Select features to visualize
st.sidebar.subheader('Choose Features to Visualize')
x_axis = st.sidebar.selectbox('Select X-axis Feature', penguins.columns, index=0)
y_axis = st.sidebar.selectbox('Select Y-axis Feature', penguins.columns, index=1)
hue_feature = st.sidebar.selectbox('Select Hue (Categorical)', ['species', 'island', 'sex'])

# Scatterplot of the selected features
st.subheader(f'Scatterplot: {x_axis} vs {y_axis}')
fig = px.scatter(penguins, x=x_axis, y=y_axis, color=hue_feature, title=f'{x_axis} vs {y_axis}')
st.plotly_chart(fig)

# Correlation heatmap
st.subheader('Correlation Heatmap')
corr_matrix = penguins.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Histograms
st.subheader('Histograms')
feature_hist = st.sidebar.selectbox('Select Feature for Histogram', penguins.columns, index=3)
fig_hist = px.histogram(penguins, x=feature_hist, color=hue_feature, title=f'Histogram of {feature_hist}')
st.plotly_chart(fig_hist)

# Pairplot for multi-feature relationships
st.subheader('Pairplot')
if st.checkbox('Show Pairplot'):
    st.write("Generating Pairplot...")
    fig_pairplot = sns.pairplot(penguins, hue=hue_feature)
    st.pyplot(fig_pairplot)
  
# Input features
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))
  
  # Create a DataFrame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins


# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y


# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
