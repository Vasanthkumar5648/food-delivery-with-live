import streamlit as st
import googlemaps
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import os
import folium
from streamlit_folium import st_folium
from functools import lru_cache

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Load model
model = tf.keras.models.load_model("model.h5")

# Google Maps API client
gmaps = googlemaps.Client(key=API_KEY)

# Cache Google API calls (avoid hitting quota)
@lru_cache(maxsize=1000)
def get_distance_duration_cached(origin_lat, origin_lng, dest_lat, dest_lng):
    origin = f"{origin_lat},{origin_lng}"
    destination = f"{dest_lat},{dest_lng}"
    try:
        result = gmaps.distance_matrix(origins=origin,
                                       destinations=destination,
                                       mode='driving',
                                       departure_time='now')
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

# UI
st.title("🚗 Real-Time Food Delivery Time Prediction")

# Input: Age and Rating
age = st.number_input("Delivery Partner Age", min_value=18, max_value=65, value=30)
rating = st.slider("Delivery Partner Rating", 0.0, 5.0, 4.5, 0.1)

# Input: Location
st.subheader("📍 Select Restaurant and Delivery Locations on the Map")

# Default coordinates
default_location = [28.6139, 77.2090]  # Delhi, India

# Create interactive map
m = folium.Map(location=default_location, zoom_start=12)

folium.Marker(default_location, tooltip="Drag or click on map to select").add_to(m)
output = st_folium(m, height=400, width=700)

# Extract clicked coordinates
coords = output.get("last_clicked")

if coords:
    st.write(f"Selected coordinates: {coords}")
    res_lat = coords["lat"]
    res_lng = coords["lng"]
    del_lat = st.number_input("Delivery Latitude", format="%.6f")
    del_lng = st.number_input("Delivery Longitude", format="%.6f")

    if st.button("Predict Delivery Time"):
        dist, dur = get_distance_duration_cached(res_lat, res_lng, del_lat, del_lng)
        if dist is not None:
            features = np.array([[age, rating, dist, dur]])
            pred = model.predict(features)
            st.success(f"📦 Estimated Delivery Time: **{pred[0][0]:.2f} minutes**")

            # Show route on map
            route_map = folium.Map(location=[res_lat, res_lng], zoom_start=13)
            folium.Marker([res_lat, res_lng], tooltip="Restaurant", icon=folium.Icon(color='green')).add_to(route_map)
            folium.Marker([del_lat, del_lng], tooltip="Customer", icon=folium.Icon(color='red')).add_to(route_map)
            folium.PolyLine([(res_lat, res_lng), (del_lat, del_lng)], color="blue", weight=2.5, opacity=1).add_to(route_map)

            st.subheader("📍 Route Overview")
            st_folium(route_map, height=400, width=700)
        else:
            st.warning("Could not fetch distance or traffic data. Please check inputs.")
else:
    st.info("Click on the map to select the restaurant location.")
