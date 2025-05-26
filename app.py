import streamlit as st
import requests
import os

# Anda HARUS mengganti ini dengan kunci API OpenWeatherMap Anda
# Lebih baik menggunakan variabel lingkungan untuk keamanan
OPENWEATHER_API_KEY = "a38218a0a52f61f3e854c91d96906638" # Ambil dari variabel lingkungan

# URL endpoint FastAPI model Anda
# Jika berjalan secara lokal, mungkin http://localhost:8000/predict
# Jika di-deploy ke Docker Compose, bisa nama service dari FastAPI (misalnya, 'fastapi_service:8000/predict')
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:8000/predict")

def get_coordinates(city_name, api_key):
    """
    Mengambil lintang dan bujur dari nama kota menggunakan OpenWeatherMap Geocoding API.
    """
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
    response = requests.get(geo_url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
    return None, None

def get_air_pollution_data(lat, lon, api_key):
    """
    Mengambil data kualitas udara dari lintang dan bujur menggunakan OpenWeatherMap Air Pollution API.
    """
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(pollution_url)
    if response.status_code == 200:
        data = response.json()
        if data and data['list']:
            components = data['list'][0]['components']
            return {
                # UBAH KUNCI DI SINI agar sesuai dengan skema FastAPI
                "CO": components.get('co', 0.0),
                "Ozone": components.get('o3', 0.0),
                "NO2": components.get('no2', 0.0),
                "PM25": components.get('pm2_5', 0.0)
            }
    return None

st.title("Prediksi Kualitas Udara (AQI) Berdasarkan Kota")

st.header("Masukkan Nama Kota")

city_input = st.text_input("Nama Kota")

if st.button("Prediksi Kualitas Udara"):
    if not OPENWEATHER_API_KEY:
        st.error("Kunci API OpenWeatherMap tidak ditemukan. Harap atur variabel lingkungan OPENWEATHER_API_KEY.")
    elif city_input:
        with st.spinner("Mencari data kota dan polusi..."):
            lat, lon = get_coordinates(city_input, OPENWEATHER_API_KEY)

            if lat is not None and lon is not None:
                st.write(f"Koordinat {city_input}: Lintang {lat}, Bujur {lon}")
                pollution_data = get_air_pollution_data(lat, lon, OPENWEATHER_API_KEY)

                if pollution_data:
                    st.subheader("Data Polusi yang Diambil:")
                    st.json(pollution_data)

                    # Panggil API prediksi model Anda
                    try:
                        response = requests.post(MODEL_API_URL, json=pollution_data)
                        if response.status_code == 200:
                            result = response.json()
                            prediction = result.get("prediction")
                            st.success(f"Prediksi Kualitas Udara untuk {city_input}: **{prediction}**")
                        else:
                            st.error(f"Gagal melakukan prediksi dari model. Kode status: {response.status_code}, Pesan: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Tidak dapat terhubung ke API prediksi model. Pastikan API model berjalan dan URL-nya benar.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memanggil API model: {e}")
                else:
                    st.warning(f"Tidak dapat mengambil data kualitas udara untuk {city_input}. Pastikan nama kota benar dan ada data polusi tersedia.")
            else:
                st.error(f"Tidak dapat menemukan koordinat untuk kota '{city_input}'. Harap periksa ejaan.")
    else:
        st.warning("Silakan masukkan nama kota.")