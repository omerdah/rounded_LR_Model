import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import joblib
import numpy as np

# Custom class for rounded Linear Regression
class RoundedLinearRegression(LinearRegression):
    def predict(self, X):
        predictions = super().predict(X)
        rounded_predictions = np.round(predictions)
        return rounded_predictions

# Load the saved model
model = pickle.load(open('rounded_LR_model.pkl', 'rb'))
stats_per_season_and_area = pd.read_excel('stats_per_season_and_area.xlsx',index_col=None)
df_mappings = pd.read_excel('feature_mappings.xlsx',index_col=None)
X = ['מועד זריעה', "מס' דונם", 'לחות יחסית ממוצע לילה', 'עננות ממוצע יום', 'השקיה_1.0', 'סוג הקרקע_2.5', 'סוג הקרקע_3.0', 'כרב/גידול קודם_2.0', 'אופן הדברה_1.0', 'אופן הדברה_3.0', 'סוג זבל_0.0', 'סוג זבל_1.0', 'טיפוס תירס_2.0', 'טיפוס תירס_3.0']
# Define the mapping dictionaries
month_mapping = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 1,
    6: 2,
    7: 3,
    8: 1,
    9: 1,
    10: 2,
    11: 3,
    12: 3
}
stats_per_season_and_area = stats_per_season_and_area[['עונת גידול','אזור','לחות יחסית ממוצע לילה', 'עננות ממוצע יום']]
meteo_feats = ['לחות יחסית ממוצע לילה', 'עננות ממוצע יום']
all_features = ['השקיה','סוג הקרקע','כרב/גידול קודם','אופן הדברה','סוג זבל','טיפוס תירס', "מס' דונם", 'לחות יחסית ממוצע לילה', 'עננות ממוצע יום','מועד זריעה','עונת גידול','אזור']
categorial_feats = ['השקיה','סוג הקרקע','כרב/גידול קודם','אופן הדברה','סוג זבל','טיפוס תירס']
# season_mapping = {'סתיו': 0, 'אביב': 1, 'אביב-קיץ': 2}
season_mapping = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0
}
watering = {
   'קונוע' :1,
   'טפטוף' :2,
   'משולב' :3
}
sorgum = {
   'ללא' :0,
   'מזרח' :1,
   'דרום' :2,
   'מערב' :3,
   'צפון' :4,
   'לא ידוע' :5
}
mission = {
   'שוק' :1,
   'תעשייה' :2,
   'תחמיץ' :3,
   'פופקורן' :4
}
soil = {
   'קלה' :1,
   'בינונית' :2,
   'בינונית-כבדה' :2.5,
   'כבדה' :3
}
formar_crop = {
   'קטנית' :1,
   'שושניים' :2,
   'סוככים' :3,
   'דגניים' :4,
   'שחור' :5
}
spray = {
   'קרקע' :1,
   'אוויר' :2,
   'משולב' :3
}
pre_process = {
   'חריש' :1,
   'דיסקוס' :2,
   'קלטור' :3
}

fertile = {
   'ללא' :0,
   'קומפוסט' :1,
   'זבל חצרות' :2,
   'זבל עוף' :3,
   'טריפל' :4,
   'לא ידוע' :5
}
corn_type = {
   'מספוא' :1,
   'מתוק' :2,
   'סופר-מתוק' :3,
   'פופקורן' :4
}
confidor = {
   'ללא' :0,
   'יישום בזריעה' :1,
   'יישום בזריעה ו30 יום לפני אסיף' :2
}
# Function to preprocess the input data
def preprocess_input(data):
    data['מועד זריעה'] = pd.to_datetime(data['מועד זריעה']).dt.dayofyear
    
    data['השקיה'] = data['השקיה'].map(watering)
    data['סוג הקרקע'] = data['סוג הקרקע'].map(soil)
    data['כרב/גידול קודם'] = data['כרב/גידול קודם'].map(formar_crop)
    data['אופן הדברה'] = data['אופן הדברה'].map(spray)
    data['סוג זבל'] = data['סוג זבל'].map(fertile)
    data['טיפוס תירס'] = data['טיפוס תירס'].map(corn_type)
    
    tmp = data.drop(columns = ['אזור'])
    tmp = data.drop(columns = ['עונת גידול'])
    tmp = tmp.astype('float64')
    tmp = pd.get_dummies(tmp, columns=categorial_feats, prefix=categorial_feats, prefix_sep='_')
    # Realign new data columns with training data columns
    missing_cols = set(X) - set(tmp.columns)
    for col in missing_cols:
        tmp[col] = 0

    # Ensure the order of columns is the same as in the training data
    tmp = tmp[X]
    return tmp

def procces_meteo(data):
    data['עונת גידול'] = pd.to_datetime(data['מועד זריעה']).dt.month.map(season_mapping)
    # Extract the season and area values from the first DataFrame
    season = data['עונת גידול'].values[0]
    area = data['אזור'].values[0]

    # Filter the second DataFrame to match the season and area values
    filtered_df2 = stats_per_season_and_area[(stats_per_season_and_area['עונת גידול'] == season) & (stats_per_season_and_area['אזור'] == area)].copy()

    # Get the extra column names from df2
    extra_columns = ['לחות יחסית ממוצע לילה', 'עננות ממוצע יום']

    # Add the extra columns to the first DataFrame
    for col in extra_columns:
        data[col] = filtered_df2[col].values[0]
    return data
        
# Create the Streamlit app
def main():
    # Set page layout to center alignment
    st.set_page_config(layout="centered")
    st.title('חיזוי מספר הריסוסים כנגד גדודנית פולשת בתירס')
    st.write('הזן את כלל הקלטים המופיעים מטה ולחץ על כפתור התחזית')

    # Get the feature names that need validation from df_mappings
    input_names = ['השקיה','סוג הקרקע','כרב/גידול קודם','אופן הדברה','סוג זבל','טיפוס תירס','אזור','עונת גידול']
    with_no_meteo = list(set(all_features)-set(meteo_feats))
    inputs = []
    for feature_name in with_no_meteo:
        if feature_name == 'מועד זריעה':
            min_date = pd.to_datetime('today').date()
            max_date = pd.to_datetime('2030-12-31').date()
            input_value = st.date_input(feature_name, min_value=min_date, max_value=max_date)
        elif feature_name in input_names:
            valid_values = list(df_mappings[feature_name].dropna().unique())

            if isinstance(valid_values, list):
                input_value = st.selectbox(feature_name, valid_values)
            else:
                input_value = st.number_input(feature_name, value=0.0)
        else:
            input_value = st.number_input(feature_name, value=0.0)
        
        inputs.append(input_value)
        
    # Create a button to trigger the prediction
    if st.button('חיזוי'):
        
        input_data = pd.DataFrame([inputs], columns=with_no_meteo)
        
        # Preprocess the input data  
        preprocessed_data = procces_meteo(input_data)
        preprocessed_data = preprocess_input(preprocessed_data)
        # Make predictions
        prediction = model.predict(preprocessed_data)

        # Display the prediction
        st.write('חיזוי:', prediction)
    
# Run the app
if __name__ == '__main__':
    main()
