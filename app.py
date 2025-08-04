import streamlit as st
import googlemaps
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import os
import folium
from streamlit_folium import st_folium
from functools import lru_cache
import geocoder
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Google Maps API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=API_KEY)

# Constants
R = 6371  # Earth's radius in km

def deg_to_rad(degrees):
    return degrees * (np.pi / 180)

def calculate_distance(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Load and process dataset
df = pd.read_csv('https://raw.github.com/Vasanthkumar5648/food-delivery-with-live/main/deliverytime.txt')
df['distance'] = df.apply(lambda row: calculate_distance(
    row['Restaurant_latitude'],
    row['Restaurant_longitude'],
    row['Delivery_location_latitude'],
    row['Delivery_location_longitude']
), axis=1)

X = df[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]]
y = df["Time_taken(min)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Keras regression model
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Cache Google Maps API requests
@lru_cache(maxsize=1000)
def get_distance_duration_cached(origin_lat, origin_lng, dest_lat, dest_lng):
    origin = f"{origin_lat},{origin_lng}"
    destination = f"{dest_lat},{dest_lng}"
    try:
        result = gmaps.distance_matrix(origins=origin, destinations=destination, mode='driving', departure_time='now')
        element = result['rows'][0]['elements'][0]
        if element['status'] == 'OK':
            distance_km = element['distance']['value'] / 1000
            duration_min = element['duration_in_traffic']['value'] / 60
            return distance_km, duration_min
        else:
            return None, None
    except Exception as e:
        st.error(f"Google Maps API Error: {e}")
        return None, None

# Default location
try:
    g = geocoder.ip('me')
    default_location = g.latlng if g.latlng else [28.6139, 77.2090]
except:
    default_location = [28.6139, 77.2090]

city_coords = {
    "New Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Bangalore": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Chennai": [13.0827, 80.2707]
}

# UI
st.title("🚚 Real-Time Food Delivery Prediction")

age = st.number_input("Delivery Partner Age", min_value=18, max_value=65, value=30)
rating = st.slider("Delivery Partner Rating", 0.0, 5.0, 4.5, 0.1)

st.subheader("📍 Choose Restaurant Starting Location")
city = st.selectbox("Choose a city or use map", ["Use my current location"] + list(city_coords.keys()))

if city != "Use my current location":
    default_location = city_coords[city]

# Map
m = folium.Map(location=default_location, zoom_start=12)
folium.Marker(default_location, tooltip="Select Restaurant").add_to(m)
output = st_folium(m, height=400, width=700)
coords = output.get("last_clicked")

if coords:
    res_lat = coords["lat"]
    res_lng = coords["lng"]
    st.success(f"Restaurant Selected: ({res_lat:.5f}, {res_lng:.5f})")

    st.subheader("🚚 Enter Delivery Location")
    use_same_location = st.checkbox("Use same location as delivery location")

    if use_same_location:
        del_lat, del_lng = res_lat, res_lng
    else:
        del_lat = st.number_input("Delivery Latitude", format="%.6f")
        del_lng = st.number_input("Delivery Longitude", format="%.6f")

    if st.button("🔍 Predict Delivery Time"):
        dist, _ = get_distance_duration_cached(res_lat, res_lng, del_lat, del_lng)
        if dist is not None:
            features = np.array([[age, rating, dist]])
            pred = model.predict(features)
            st.success(f"📦 Estimated Delivery Time: **{pred[0][0]:.2f} minutes**")

            # Route map
            route_map = folium.Map(location=[res_lat, res_lng], zoom_start=13)
            folium.Marker([res_lat, res_lng], tooltip="Restaurant", icon=folium.Icon(color='green')).add_to(route_map)
            folium.Marker([del_lat, del_lng], tooltip="Customer", icon=folium.Icon(color='red')).add_to(route_map)
            folium.PolyLine([(res_lat, res_lng), (del_lat, del_lng)], color="blue", weight=2.5, opacity=1).add_to(route_map)
            st.subheader("📍 Delivery Route")
            st_folium(route_map, height=400, width=700)
        else:
            st.warning("Could not retrieve distance or traffic data.")
else:
    st.info("Click on the map to select the restaurant location.")
