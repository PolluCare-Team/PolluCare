# PolluCare: Health-Aware Air Quality App with Personal Score

PolluCare is a web application designed to predict air quality and provide personalized health recommendations based on users' location and health profiles. Leveraging a Long Short-Term Memory (LSTM) machine learning model, PolluCare analyzes geographical, meteorological, and pollutant data (e.g., PM2.5, PM10, CO) to forecast air quality and calculate a tailored health risk score. This score considers factors like age, medical conditions, and habits, delivering actionable advice such as limiting outdoor activities or wearing a mask. Aimed at improving well-being and supporting vulnerable groups, PolluCare contributes to sustainability and clean air initiatives in Indonesia.

## Table of Contents

- [Project Description](#project-description)
- [Project Team](#project-team)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Plan](#project-plan)
- [Market Analysis](#market-analysis)
- [Advisor's Comments](#advisors-comments)
- [Additional Resources](#additional-resources)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Air pollution is a pressing health issue, especially in urban areas and for vulnerable populations such as those with asthma, heart disease, or respiratory disorders. Research from Dede Anwar Musadad (BRIN) highlights stroke, ischemic heart disease, diabetes mellitus, chronic obstructive pulmonary disease, and neonatal disorders as the top five illnesses linked to air pollution in Indonesia, with eastern provinces like West Sulawesi and Maluku Utara bearing the heaviest burden in 2019. The WHO also reports 3.2 million deaths globally in 2020 from indoor air pollution, including 237,000 children under five. Current air quality systems often lack personalized alerts. PolluCare bridges this gap by using an LSTM model to predict air quality and provide customized health recommendations, enhancing public health and supporting Indonesia’s clean air policies.

## Project Team

- **Group ID**: LAI25-RM098
- **Theme**: Sustainability and Well-being
- **Advisor**: Yahya Putra Pradana
- **Group Members**:
  - A345YBF016 – Aditya Pratama – Universitas 17 Agustus 1945 Surabaya
  - A528YBM041 – Akbar Widianto – Politeknik Negeri Medan
  - A007XBF052 – Amelia Gizzela Sheehan Auni – Universitas Dian Nuswantoro
  - A319YBF512 – Yohanssen Pradana Pardede – Universitas Sumatera Utara

## Features

- **Air Quality Prediction**: Forecasts air quality using an LSTM model based on environmental data.
- **Personalized Health Risk Score**: Calculates a unique score based on age, medical conditions, and habits.
- **Health Recommendations**: Offers tailored advice to mitigate air pollution risks.
- **Location Input**: Supports city name entry or map-based selection.
- **Health Profile Input**: Allows users to input age, medical conditions, and outdoor plans.
- **Nearby Hospital Recommendations**: Suggests nearby hospitals during poor air quality for at-risk users.

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: Flask
- **Machine Learning**: TensorFlow (LSTM model)
- **APIs**: OpenWeather Air Pollution API, Overpass API (hospital data)
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Folium, Geopy, Google Generative AI

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PolluCare-Team/PolluCare.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd PolluCare
   ```
3. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Set up API keys**:
   - Add your [OpenWeather API key](https://openweathermap.org/api) to the app.
   - Add your [Google Gemini API key](https://cloud.google.com/gemini) to the app.

## Usage

1. **Run the app**:
   ```bash
   streamlit run app.py
   ```
2. **Input your location**:
   - Enter a city name or select a location on the map.
3. **Enter your health profile**:
   - Provide age, medical conditions, and planned activities.
4. **Get results**:
   - Click "Dapatkan Prediksi & Saran Kesehatan" to view air quality predictions and health advice.

## Project Plan

| No | Task | Week 1 | Week 2 | Week 3 | Week 4 | Week 5 | Week 6 | Week 7 |
|----|------|--------|--------|--------|--------|--------|--------|--------|
| 1 | Project planning and plan development | X | | | | | | |
| 2 | Dataset exploration and model selection | | X | | | | | |
| 3 | Data preprocessing | | | X | | | | |
| 4 | Model training and evaluation | | | | X | | | |
| 5 | Model implementation | | | | | X | | |
| 6 | Real-time weather API integration | | | | | | X | |
| 7 | Model testing | | | | | | | X |
| 8 | Model refinement and system finalization | | | | | | | X |
| 9 | Collect project brief, slides, and video | | | | | | | X |

## Market Analysis

### Target Market
- **Demographics**: All ages, focusing on adults, the elderly, and parents with young children.
- **Professions**: Workers in high-pollution areas (e.g., construction, traffic police).
- **Hobbies**: Outdoor enthusiasts (e.g., athletes, nature lovers).
- **Others**: Health-conscious individuals, environmentalists, policymakers.

### Comparison with Similar Services
- Apps like AirVisual and Plume Labs offer air quality data but lack health personalization (60-70% similarity).
- **PolluCare’s Uniqueness**: Personalized health risk scores and specific recommendations.

### SWOT Analysis
- **Strengths**: Personalized insights, accurate LSTM predictions, real-time data, focus on vulnerable groups.
- **Weaknesses**: Relies on accurate user health data, technical integration challenges, ongoing model updates needed.
- **Opportunities**: Rising pollution awareness, government clean air support, health organization partnerships.
- **Threats**: Competition, data privacy concerns, API downtime risks.

## Advisor's Comments

"Sesi berjalan dengan lancar dan menyenangkan. Peserta aktif untuk berdiskusi. Secara project sudah terbilang on-track, namun perlu memperhatikan kembali hal-hal yang bersifat teknis."

## Additional Resources

- **Video Demo**: [Watch here](https://drive.google.com/file/d/1Rj3PwFHrYZ7pSiYNL-q6Q_0fetBYihL8/view?usp=sharing)
- **Dataset**: [Insert dataset link]
- **Deployment**: [Insert deployment link]
- **GitHub Repository**: [PolluCare-Team](https://github.com/PolluCare-Team)
- **Presentation Video**: [Insert video link]
- **Presentation Slides**: [Insert slides link]

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your updates.
4. Push to your branch.
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
