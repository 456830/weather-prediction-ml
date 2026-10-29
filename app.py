
import streamlit as st
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🌦️ Weather Prediction System")

# -----------------------------------
# LOAD DATASET
# -----------------------------------

data = pd.read_csv("weather_data.csv")

st.subheader("Weather Dataset")
st.dataframe(data)

# -----------------------------------
# DATA CLEANING
# -----------------------------------

st.subheader("Data Cleaning")

st.write("Missing values before cleaning:")
st.write(data.isnull().sum())

# Remove duplicate rows
data = data.drop_duplicates()

# Convert required columns to numeric
numeric_columns = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove rows containing invalid/missing values
data = data.dropna()

st.write("Dataset after cleaning:")
st.dataframe(data)

# -----------------------------------
# WEATHER VISUALIZATION
# -----------------------------------

st.subheader("Weather Data Visualization")

st.line_chart(
    data[["Temperature", "Humidity", "Pressure", "WindSpeed"]]
)

# -----------------------------------
# FEATURES AND TARGET
# -----------------------------------

X = data[["Humidity", "Pressure", "WindSpeed"]]
y = data["Temperature"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# LINEAR REGRESSION
# -----------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = linear_mse ** 0.5

# -----------------------------------
# RANDOM FOREST
# -----------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = rf_mse ** 0.5

# -----------------------------------
# MODEL COMPARISON
# -----------------------------------

st.subheader("Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "R² Score": [
        linear_r2,
        rf_r2
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "MSE": [
        linear_mse,
        rf_mse
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ]
})

st.dataframe(comparison)

# -----------------------------------
# BEST MODEL
# -----------------------------------

if linear_r2 >= rf_r2:
    best_model = linear_model
    best_name = "Linear Regression"
else:
    best_model = rf_model
    best_name = "Random Forest"

st.success(f"Best Model: {best_name}")

# -----------------------------------
# MODEL PERFORMANCE
# -----------------------------------

st.subheader("Model Performance")

st.write(f"Linear Regression R² Score: {linear_r2:.2f}")
st.write(f"Linear Regression MAE: {linear_mae:.2f} °C")
st.write(f"Linear Regression RMSE: {linear_rmse:.2f} °C")

st.write(f"Random Forest R² Score: {rf_r2:.2f}")
st.write(f"Random Forest MAE: {rf_mae:.2f} °C")
st.write(f"Random Forest RMSE: {rf_rmse:.2f} °C")

# -----------------------------------
# USER INPUT
# -----------------------------------

st.subheader("Enter Weather Details")

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

pressure = st.number_input(
    "Pressure (hPa)",
    value=1012.0
)

windspeed = st.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    value=8.0
)

# -----------------------------------
# PREDICTION
# -----------------------------------

if st.button("Predict Temperature"):

    input_data = pd.DataFrame(
        [[humidity, pressure, windspeed]],
        columns=["Humidity", "Pressure", "WindSpeed"]
    )

    prediction = best_model.predict(input_data)

    st.success(
        f"Predicted Temperature: {prediction[0]:.2f} °C"
  )




# ---------------------------------------
# LIVE CURRENT WEATHER WITH STATE AND CITY
# ---------------------------------------

st.title("🌍 Live Current Weather")
st.write("Select an Indian state and enter a city name.")

indian_states = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Delhi"
]

selected_state = st.selectbox(
    "Select State",
    indian_states,
    index=indian_states.index("Uttar Pradesh")
)

city_name = st.text_input(
    "Enter City Name",
    value="Greater Noida"
)

if st.button("Get Current Weather"):

    if not city_name.strip():
        st.warning("Please enter a city name.")

    else:
        try:
            # STEP 1: Find city coordinates
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": city_name.strip(),
                    "count": 10,
                    "language": "en",
                    "countryCode": "IN"
                },
                timeout=20
            )

            geo_response.raise_for_status()

            locations = geo_response.json().get("results", [])

            # Match city with selected state
            matching_locations = [
                location for location in locations
                if location.get("admin1", "").casefold()
                == selected_state.casefold()
            ]

            if not matching_locations:
                st.error(
                    "City not found in the selected state. "
                    "Please check the city and state names."
                )

            else:
                place = matching_locations[0]

                latitude = place["latitude"]
                longitude = place["longitude"]

                # STEP 2: Fetch current weather
                weather_response = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current": (
                            "temperature_2m,"
                            "relative_humidity_2m,"
                            "surface_pressure,"
                            "wind_speed_10m"
                        ),
                        "timezone": "auto"
                    },
                    timeout=20
                )

                weather_response.raise_for_status()

                weather_data = weather_response.json()
                current = weather_data["current"]

                temperature = current["temperature_2m"]
                humidity = current["relative_humidity_2m"]
                pressure = current["surface_pressure"]
                wind_speed = current["wind_speed_10m"]

                # STEP 3: Display live weather
                st.subheader(
                    f"Current Weather: {place['name']}, "
                    f"{selected_state}, India"
                )

                col1, col2 = st.columns(2)

                col1.metric("Temperature", f"{temperature} °C")
                col2.metric("Humidity", f"{humidity} %")

                col1.metric("Pressure", f"{pressure} hPa")
                col2.metric("Wind Speed", f"{wind_speed} km/h")

                st.caption(
                    f"Weather source: Open-Meteo | "
                    f"Reported time: {current['time']}"
                )

                # STEP 4: ML estimated temperature
                if (
                    "best_model" in globals()
                    and best_model is not None
                ):
                    input_data = pd.DataFrame(
                        [[humidity, pressure, wind_speed]],
                        columns=[
                            "Humidity",
                            "Pressure",
                            "WindSpeed"
                        ]
                    )

                    prediction = best_model.predict(input_data)

                    st.success(
                        f"ML Estimated Temperature: "
                        f"{prediction[0]:.2f} °C"
                    )

                    st.info(
                        "Live temperature is retrieved from Open-Meteo. "
                        "ML estimated temperature is a separate "
                        "prediction from your trained model."
                    )

                else:
                    st.warning(
                        "The trained ML model is not available. "
                        "Please check your model training section."
                    )

        except requests.exceptions.RequestException as error:
            st.error(
                f"Could not fetch weather data. "
                f"Check your internet connection. Details: {error}"
            )

        except (KeyError, ValueError, TypeError) as error:
            st.error(
                f"Could not process the weather response: {error}"
            )
