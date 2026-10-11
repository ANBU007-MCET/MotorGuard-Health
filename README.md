# MotorGuard Health ⚙️

### AI-Based Motor Health Trend Predictor

MotorGuard Health is a Python-based machine learning dashboard that predicts motor temperature using operating hours, motor load, and ambient temperature. It provides interactive visualizations and model evaluation metrics to explore motor operating conditions and support predictive maintenance learning.

## 🚀 Live Demo

**[🌐 Open MotorGuard Health Dashboard](https://motor-guard-health-anbu.streamlit.app/)**

Explore the interactive dashboard to view motor temperature predictions, analyze temperature trends, and evaluate model performance.

## ✨ Features

- 🌡️ Motor temperature prediction using Linear Regression
- 🎛️ Interactive motor operating condition controls
- ⚠️ Temperature-based warning indicator
- 📈 Temperature trend visualization
- 📊 Model evaluation using MAE, RMSE, and R²
- 🗂️ Dataset preview and CSV downloads
- 🖥️ Interactive dashboard built with Streamlit

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
MotorGuard-Health/
├── app.py
├── requirements.txt
├── test_app.py
├── .gitignore
└── README.md
```

## 💻 How to Run Locally

Follow these steps to run MotorGuard Health on your computer.

### 1. Clone the Repository

```bash
git clone https://github.com/ANBU007-MCET/MotorGuard-Health.git
```

### 2. Open the Project Folder

```bash
cd MotorGuard-Health
```

### 3. Install the Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## 📊 Dataset

The current version uses synthetic demonstration data generated using Python. The data is used to demonstrate temperature prediction, visualization, and model evaluation.

## 📈 Model Evaluation

The project uses **Linear Regression** to predict motor temperature based on operating conditions.

The model is evaluated using the following metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted temperatures.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors.
- **R² Score:** Indicates how well the model explains variation in the target values.

The reported performance reflects the demonstration dataset and should not be treated as evidence of real-world motor prediction accuracy.

## ⚠️ Limitations

- The current dataset contains synthetic demonstration data rather than real motor sensor readings.
- The model predicts temperature and does not diagnose actual motor faults.
- Real-world deployment requires real sensor data, appropriate validation, and manufacturer-specific operating limits.
- The dashboard is an educational prototype and is not a substitute for industrial monitoring or safety systems.

## 🔮 Future Improvements

- Integration with real motor sensors
- Labeled motor fault classification
- Predictive maintenance alerts
- Historical motor data monitoring
- Real-time IoT integration
- Comparison of multiple machine learning models
- Improved validation using real-world datasets

## 👨‍💻 Contact

**Name:** Anbu Selvan

**Email:** [kanbuselvan2007@gmail.com](mailto:kanbuselvan2007@gmail.com)

**LinkedIn:** [www.linkedin.com/in/anbu-selvan-k](https://www.linkedin.com/in/anbu-selvan-k/)

Feel free to connect with me for collaboration, feedback, or discussions about machine learning and predictive maintenance.

---

⭐ If you find this project interesting, consider starring the repository!
