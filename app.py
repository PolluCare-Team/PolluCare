import streamlit as st
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim 
from geopy.distance import geodesic 

# --- Antarmuka Streamlit ---
st.set_page_config(
    page_title="PolluCare",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/PolluCare-Team/PolluCare/',
        'Report a bug': 'https://github.com/PolluCare-Team/PolluCare/', # Ganti dengan link repo Anda
        'About': 'PolluCare: Prediksi Kualitas Udara & Saran Kesehatan'
    }
)

st.title("PolluCare: Prediksi Kualitas Udara & Saran Kesehatan")
st.markdown(
    "Selamat datang di PolluCare! Aplikasi ini didukung AI untuk membantu Anda memahami kualitas udara di lokasi Anda "
    "dan memberikan saran kesehatan yang dipersonalisasi. Mari jaga kesehatan pernapasan kita bersama!"
)
st.markdown("---")

# --- Konfigurasi ---
OPENWEATHER_API_KEY = "a38218a0a52f61f3e854c91d96906638"

# Konfigurasi Gemini API Key Anda
GEMINI_API_KEY = "AIzaSyDRys2TDy1B2-mHYQwkDUj1nTfogxnjL3Q"

# Path ke model DNN TensorFlow Anda
MODEL_PATH = "MLProject/models/air_quality_dnn_model.h5"

# Pemetaan manual untuk kategori AQI.
AQI_CATEGORY_MAP = {
    0: 'Baik',
    1: 'Berbahaya',
    2: 'Sedang',
    3: 'Tidak Sehat',
    4: 'Tidak Sehat untuk Kelompok Sensitif',
    5: 'Sangat Tidak Sehat'
}

AQI_COLOR_MAP = {
    'Baik': '#d4edda',       
    'Sedang': '#fff3cd',    
    'Tidak Sehat untuk Kelompok Sensitif': '#f8d7da', 
    'Tidak Sehat': '#f8d7da',   
    'Sangat Tidak Sehat': '#f0e0f8', 
    'Berbahaya': '#e0c0d8'     
}

AQI_TEXT_COLOR_MAP = {
    'Baik': '#155724', 
    'Sedang': '#856404', 
    'Tidak Sehat untuk Kelompok Sensitif': '#721c24', 
    'Tidak Sehat': '#721c24', 
    'Sangat Tidak Sehat': '#4F2864', 
    'Berbahaya': '#5B0B3A' 
}

AQI_EMOJI_MAP = {
    'Baik': '😊',
    'Sedang': '😐',
    'Tidak Sehat untuk Kelompok Sensitif': '😷',
    'Tidak Sehat': '😷',
    'Sangat Tidak Sehat': '🤢',
    'Berbahaya': '☠️'
}

FEATURES_USED_IN_TRAINING = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']

@st.cache_resource
def load_gemini_model(api_key):
    if not api_key:
        st.error("Kunci API Gemini tidak ditemukan. Harap masukkan di sidebar atau konfigurasi.")
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Gagal memuat model Gemini: {e}")
        st.warning("Pastikan kunci API Gemini Anda valid dan nama model yang digunakan benar.")
        return None

GEMINI_MODEL = load_gemini_model(GEMINI_API_KEY)

@st.cache_resource
def load_dnn_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model DNN: {e}")
        st.warning("Pastikan file model DNN ada di path yang benar.")
        return None

model_dnn = load_dnn_model(MODEL_PATH)

@st.cache_data(ttl=3600)
def get_coordinates(city_name_param, api_key):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name_param}&limit=1&appid={api_key}"
    try:
        response = requests.get(geo_url)
        response.raise_for_status()
        data = response.json()
        if data:
            # Access the first item in the list
            return data[0]['lat'], data[0]['lon']
    except requests.exceptions.RequestException as e:
        st.error(f"Error mengambil koordinat untuk {city_name_param}: {e}")
    except (IndexError, KeyError):
        st.error(f"Tidak dapat menemukan koordinat untuk {city_name_param}. Pastikan nama kota sudah benar.")
    return None, None

@st.cache_data(ttl=3600)
def get_city_from_coords(lat, lon, api_key):
    reverse_geo_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={api_key}"
    try:
        response = requests.get(reverse_geo_url)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            return data[0].get('name')
    except requests.exceptions.RequestException as e:
        st.error(f"Error mengambil nama kota dari koordinat: {e}")
    except (IndexError, KeyError):
        st.error("Format respon reverse geocoding tidak sesuai yang diharapkan atau data tidak ditemukan.")
    return None

@st.cache_data(ttl=600)
def get_air_pollution_data(lat, lon, api_key):
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(pollution_url)
        response.raise_for_status()
        data = response.json()
        if data and data['list']:
            components = data['list'][0]['components']
            return {
                "CO": components.get('co', 0.0),
                "Ozone": components.get('o3', 0.0),
                "NO2": components.get('no2', 0.0),
                "PM25": components.get('pm2_5', 0.0)
            }
    except requests.exceptions.RequestException as e:
        st.error(f"Error mengambil data polutan: {e}")
    except KeyError:
        st.error("Format respon data polutan tidak sesuai yang diharapkan.")
    return None

def generate_health_advice(aqi_category, pollutant_values, city_name_param, user_info=None):
    if not GEMINI_MODEL:
        return "Maaf, fitur saran kesehatan tidak tersedia karena masalah dengan model AI."

    prompt = f"""
    Sebagai tenaga medis dan ahli kualitas udara, berikan saran tindakan dan rekomendasi spesifik yang ringkas, persuasif, dan mudah dipahami.
    Fokuslah pada tindakan yang dibutuhkan berdasarkan kondisi kualitas udara dan polutan, serta profil pengguna.

    Saran harus disajikan sebagai satu paragraf yang mengalir, tidak poin-poin panjang, dan gunakan bahasa yang hangat serta peduli,
    seperti seorang tenaga medis yang berbicara kepada pasien. Jangan lupa menyapa terlebih dahulu dengan panggilan Bapak/Ibu.

    Gunakan data berikut untuk menghasilkan saran:
    - Kota: {city_name_param}
    - Prediksi Kategori Kualitas Udara (AQI): {aqi_category}
    - Konsentrasi polutan saat ini (berikan nilai dalam µg/m³):
        - Karbon Monoksida (CO): {pollutant_values.get('CO', 'N/A')} µg/m³
        - Ozon (O3): {pollutant_values.get('Ozone', 'N/A')} µg/m³
        - Nitrogen Dioksida (NO2): {pollutant_values.get('NO2', 'N/A')} µg/m³
        - Partikulat (PM2.5): {pollutant_values.get('PM25', 'N/A')} µg/m³
    """

    if user_info and (user_info.get('medical_condition', 'Tidak ada') != 'Tidak ada' or user_info.get('activity_preference', 'Tidak disebutkan') != 'Tidak disebutkan'):
        prompt += f"""
    Informasi tambahan tentang pengguna:
    - Usia: {user_info.get('age', 'N/A')} tahun
    - Kondisi medis: {user_info.get('medical_condition', 'Tidak ada')}
    - Kebiasaan: {user_info.get('activity_preference', 'Tidak disebutkan')}
    """

    try:
        response = GEMINI_MODEL.generate_content(prompt, request_options={"timeout": 60})
        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    return part.text
            return "Respon dari Gemini tidak dalam format teks yang diharapkan."
        else:
            return "Respon dari Gemini tidak dalam format teks yang diharapkan."
    except Exception as e:
        st.error(f"Error saat memanggil Gemini API untuk saran kesehatan: {e}")
        return "Maaf, tidak dapat menghasilkan saran kesehatan saat ini."


@st.cache_data(ttl=7200)
def search_nearby_hospitals(latitude, longitude, radius_km=10, limit=5):
    """
    Mencari rumah sakit terdekat menggunakan Overpass API (OpenStreetMap) dengan filter yang lebih baik.
    Args:
        latitude (float): Latitude lokasi pengguna.
        longitude (float): Longitude lokasi pengguna.
        radius_km (int): Radius pencarian dalam kilometer.
        limit (int): Jumlah maksimal rumah sakit yang ingin diambil.
    Returns:
        list: Daftar string informasi rumah sakit (nama, alamat, jarak).
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="hospital"]["name"](around:{radius_km*1000},{latitude},{longitude});
      way["amenity"="hospital"]["name"](around:{radius_km*1000},{latitude},{longitude});
      relation["amenity"="hospital"]["name"](around:{radius_km*1000},{latitude},{longitude});
    );
    out center;
    """

    try:
        response = requests.post(overpass_url, data=overpass_query)
        response.raise_for_status() # Tangani HTTP errors (4xx, 5xx)
        data = response.json()

        found_hospitals_raw = []
        if data and data['elements']:
            user_coords = (latitude, longitude)
            
            for element in data['elements']:
                lat_hosp = element.get('lat')
                lon_hosp = element.get('lon')
                
                if 'center' in element: # Untuk way/relation, ambil koordinat dari center
                    lat_hosp = element['center'].get('lat')
                    lon_hosp = element['center'].get('lon')

                if lat_hosp is not None and lon_hosp is not None:
                    hosp_coords = (lat_hosp, lon_hosp)
                    distance = geodesic(user_coords, hosp_coords).km
                    
                    name = element.get('tags', {}).get('name', 'Nama Tidak Diketahui').strip()
                    
                    # Coba ambil alamat lebih lengkap
                    tags = element.get('tags', {})
                    address_parts = []
                    if tags.get('addr:housenumber'): address_parts.append(tags['addr:housenumber'])
                    if tags.get('addr:street'): address_parts.append(tags['addr:street'])
                    if tags.get('addr:subdistrict'): address_parts.append(tags['addr:subdistrict'])
                    if tags.get('addr:city'): address_parts.append(tags['addr:city'])
                    if tags.get('addr:postcode'): address_parts.append(tags['addr:postcode'])
                    
                    address = ", ".join(address_parts) if address_parts else "Alamat tidak tersedia"
                    
                    # 1. Pastikan nama tidak kosong atau generik
                    if not name or name == 'Nama Tidak Diketahui' or name.lower() in ['unknown', 'hospital']:
                        continue
                    
                    # 2. Filter apotek atau entri yang jelas bukan rumah sakit berdasarkan nama
                    if "apotek" in name.lower() or "klinik" in name.lower() and "rumah sakit" not in name.lower():
                        continue # Lewati apotek atau klinik murni (yang tidak mengandung "rumah sakit")
                    
                    # 3. Hindari duplikasi yang terlihat mirip (sensitif terhadap huruf kecil/besar dan spasi)
                    # Menggunakan set untuk nama yang sudah ditambahkan (normalisasi nama)
                    normalized_name = name.lower().replace(" ", "")
                    if normalized_name in [h[0].lower().replace(" ", "") for h in found_hospitals_raw]:
                        continue

                    found_hospitals_raw.append((name, address, distance, normalized_name)) # Simpan nama asli dan nama normalisasi
            
            # Sort by distance
            found_hospitals_raw.sort(key=lambda x: x[2]) # Sort berdasarkan jarak (indeks 2)

            hospitals = []
            added_hospitals_normalized_names = set()

            for name, address, distance, normalized_name in found_hospitals_raw:
                if normalized_name in added_hospitals_normalized_names:
                    continue
                
                hospitals.append(f"- **{name}**, Alamat: {address}, Jarak: {distance:.2f} km")
                added_hospitals_normalized_names.add(normalized_name)

                if len(hospitals) >= limit:
                    break
            
            return hospitals if hospitals else ["Tidak ditemukan rumah sakit yang relevan di sekitar lokasi ini."]
        else:
            return ["Tidak ditemukan rumah sakit di sekitar lokasi ini."]

    except requests.exceptions.RequestException as e:
        st.error(f"Error saat memanggil Overpass API: {e}")
        return ["Tidak dapat menemukan informasi rumah sakit saat ini karena masalah koneksi atau server Overpass API."]
    except Exception as e:
        st.error(f"Terjadi kesalahan tidak terduga saat mencari rumah sakit: {e}")
        return ["Terjadi kesalahan saat mencari rumah sakit."]

# --- Antarmuka Streamlit: Input Utama ---
st.subheader("📍 Masukkan Lokasi Anda")

# Opsi input lokasi
location_input_method = st.radio(
    "Pilih metode input lokasi:",
    ("Ketik nama kota", "Pilih lokasi dari peta")
)

city_input = ""
selected_lat = None
selected_lon = None

if location_input_method == "Ketik nama kota":
    city_input = st.text_input("Nama Kota", placeholder="Contoh: Pekanbaru", key="city_input_text")
elif location_input_method == "Pilih lokasi dari peta":
    st.markdown("Klik pada peta untuk memilih lokasi:")
    m = folium.Map(location=[-0.7893, 113.9213], zoom_start=5)

    m.add_child(folium.LatLngPopup())

    map_data = st_folium(m, width=700, height=500, key="map_input")

    if map_data and map_data.get('last_clicked'):
        selected_lat = map_data['last_clicked']['lat']
        selected_lon = map_data['last_clicked']['lng']
        st.info(f"Koordinat yang dipilih: Lintang {selected_lat:.4f}, Bujur {selected_lon:.4f}")

        # Reverse geocoding to get city name from coordinates
        detected_city = get_city_from_coords(selected_lat, selected_lon, OPENWEATHER_API_KEY)
        if detected_city:
            city_input = detected_city
            st.success(f"Lokasi yang terdeteksi: **{city_input}**")
        else:
            st.warning("Tidak dapat mendeteksi nama kota dari koordinat yang dipilih. Silakan coba lagi atau masukkan nama kota secara manual.")
            city_input = ""
st.markdown("---")


st.subheader("🩺 Profil Kesehatan Anda")
col1, col2, col3 = st.columns(3)
with col1:
    user_age = st.text_input("Usia Anda", placeholder="Cth: 23", key="user_age")
with col2:
    user_medical_condition = st.text_input("Riwayat Penyakit", placeholder="Cth: Asma, Pneumonia", key="user_medical")
with col3:
    user_activity_preference = st.text_input("Rencana Aktivitas Luar Ruangan", placeholder="Cth: Jogging, Bekerja", key="user_activity")

user_info = {}
if user_age:
    try:
        user_info['age'] = int(user_age)
    except ValueError:
        st.warning("Usia tidak valid. Harap masukkan angka.")
        user_info['age'] = 'N/A'
else:
    user_info['age'] = 'N/A'

user_info['medical_condition'] = user_medical_condition if user_medical_condition else 'Tidak ada'
user_info['activity_preference'] = user_activity_preference if user_activity_preference else 'Tidak disebutkan'


if st.button("Dapatkan Prediksi & Saran Kesehatan 🚀", key="predict_button"):
    target_lat = None
    target_lon = None
    display_city_name = ""

    if location_input_method == "Ketik nama kota":
        if not city_input:
            st.warning("Mohon masukkan nama kota terlebih dahulu.")
            st.stop()
        target_lat, target_lon = get_coordinates(city_input, OPENWEATHER_API_KEY)
        display_city_name = city_input
    elif location_input_method == "Pilih lokasi dari peta":
        if selected_lat is None or selected_lon is None:
            st.warning("Mohon pilih lokasi di peta terlebih dahulu.")
            st.stop()
        target_lat = selected_lat
        target_lon = selected_lon
        detected_city = get_city_from_coords(target_lat, target_lon, OPENWEATHER_API_KEY)
        if detected_city:
            display_city_name = detected_city
        else:
            display_city_name = f"Lokasi yang Anda pilih ({target_lat:.2f}, {target_lon:.2f})" # Fallback if city name not found
            st.warning("Tidak dapat menemukan nama kota untuk koordinat yang dipilih. Hasil akan ditampilkan berdasarkan koordinat.")


    if not model_dnn:
        st.error("Terjadi masalah saat memuat model prediksi kualitas udara. Mohon coba lagi nanti.")
        st.stop()
    elif not GEMINI_MODEL:
        st.error("Terjadi masalah saat memuat layanan saran AI. Mohon periksa kunci API Gemini Anda atau coba lagi nanti.")
        st.stop()
    elif target_lat is None or target_lon is None:
        st.error("Tidak dapat menentukan koordinat lokasi. Harap pastikan input lokasi valid.")
        st.stop()
    else:
        with st.spinner(f"Menganalisis kualitas udara di {display_city_name} dan menyiapkan saran..."):
            pollutant_data = get_air_pollution_data(target_lat, target_lon, OPENWEATHER_API_KEY)

            if pollutant_data:
                st.markdown("---")

                st.subheader("📊 Data Polutan Terkini")

                # Membuat DataFrame input dengan data mentah
                mapped_pollutant_data = {
                    'CO AQI Value': pollutant_data.get('CO', 0.0),
                    'Ozone AQI Value': pollutant_data.get('Ozone', 0.0),
                    'NO2 AQI Value': pollutant_data.get('no2', 0.0),
                    'PM2.5 AQI Value': pollutant_data.get('PM25', 0.0)
                }
                input_df = pd.DataFrame([mapped_pollutant_data])

                # Display metrics in a single row
                col_co, col_o3, col_no2, col_pm25 = st.columns(4)

                with col_co:
                    st.metric(label="Karbon Monoksida (CO) (µg/m³)", value=f"{pollutant_data.get('CO', 0.0):.2f}")
                with col_o3:
                    st.metric(label="Ozon (O3) (µg/m³)", value=f"{pollutant_data.get('Ozone', 0.0):.2f}")
                with col_no2:
                    st.metric(label="Nitrogen Dioksida (NO2) (µg/m³)", value=f"{pollutant_data.get('NO2', 0.0):.2f}")
                with col_pm25:
                    st.metric(label="Partikulat (PM2.5) (µg/m³)", value=f"{pollutant_data.get('PM25', 0.0):.2f}")

                # Bagian Prediksi DNN
                try:
                    input_for_prediction = input_df[['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']].values
                    predictions_proba = model_dnn.predict(input_for_prediction)
                    predicted_class_index = np.argmax(predictions_proba, axis=1)[0]
                    aqi_category = AQI_CATEGORY_MAP.get(predicted_class_index, "Unknown Category")

                    st.markdown("---")

                    st.subheader("Hasil Prediksi Kualitas Udara ✨")

                    # Ambil warna latar belakang dan warna teks yang sesuai
                    category_bg_color = AQI_COLOR_MAP.get(aqi_category, '#E0E0E0') # Default abu-abu terang
                    category_text_color = AQI_TEXT_COLOR_MAP.get(aqi_category, '#000000') # Default hitam
                    category_emoji = AQI_EMOJI_MAP.get(aqi_category, '❓')

                    st.markdown(
                        f"<div style='background-color:{category_bg_color}; padding: 20px; border-radius: 10px; text-align: center;'>"
                        f"<h3><span style='color:{category_text_color};'>{aqi_category} {category_emoji}</span></h3>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown("---")

                    # Bagian Generasi Saran dengan Gemini AI
                    st.subheader("👩‍⚕️ Saran Kesehatan dari Tenaga Medis AI")

                    # Box kedua: Saran dari Gemini - Menggunakan format yang sama dengan box pertama
                    with st.spinner("Menyiapkan rekomendasi kesehatan yang dipersonalisasi..."):
                        health_advice = generate_health_advice(aqi_category, pollutant_data, display_city_name, user_info)

                        st.markdown(
                            f"<div style='background-color:#e0f7fa; padding: 20px; border-radius: 10px; border-left: 5px solid #00acc1; text-align: justify'>"
                            f"<p style='color: #00acc1; font-weight: bold;'>Halo! Perkenalkan saya Pollucare, asisten kesehatan Anda. </p>"
                            f"<p style='color: black'>{health_advice}</p>" # Konten saran dari Gemini
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    st.markdown("---")

                    # Bagian Rekomendasi Rumah Sakit Terdekat
                    if user_info.get('medical_condition') != 'Tidak ada' and aqi_category in ['Tidak Sehat', 'Tidak Sehat untuk Kelompok Sensitif', 'Sangat Tidak Sehat', 'Berbahaya']:
                        st.subheader("🏥 Rekomendasi Rumah Sakit Terdekat")
                        with st.spinner("Mencari rumah sakit terdekat..."):
                            if target_lat is not None and target_lon is not None:
                                nearby_hospitals = search_nearby_hospitals(target_lat, target_lon)
                                if nearby_hospitals:
                                    st.info("Berikut adalah beberapa rumah sakit terdekat yang dapat Anda pertimbangkan:")
                                    for i, hospital in enumerate(nearby_hospitals):
                                        st.markdown(f"{hospital}")
                                else:
                                    st.warning("Tidak dapat menemukan informasi rumah sakit terdekat saat ini.")
                            else:
                                st.warning("Koordinat lokasi tidak tersedia untuk mencari rumah sakit.")
                        st.markdown("---")

                    # Disclaimer
                    st.warning(
                        "**Penting:** Saran ini dihasilkan oleh kecerdasan buatan dan bersifat umum. "
                        "**Selalu konsultasikan dengan dokter atau tenaga medis profesional** untuk nasihat kesehatan yang lebih spesifik dan sesuai kondisi Anda."
                    )

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi atau menghasilkan saran: {e}")
                    st.info("Pastikan input data sesuai dengan format yang diharapkan model DNN dan Gemini API. "
                            "Jika ini terus terjadi, periksa log konsol untuk detail error.")
            else:
                st.warning(f"Tidak dapat mengambil data polutan udara untuk **{display_city_name}**. Pastikan nama kota benar atau data polusi tidak tersedia.")
