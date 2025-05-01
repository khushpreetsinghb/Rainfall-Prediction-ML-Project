# Rainfall Prediction using Machine Learning

[![Live Demo](https://img.shields.io/badge/View%20Live%20Site-%2300C7B7?style=for-the-badge&logo=vercel&logoColor=white)](https://rainfall-prediction-ml-project.vercel.app/)

## Project Description
The project aims to predict whether it will rain tomorrow or not based on historical weather data using Machine Learning algoritms.  
The prediction is done using the algorithms **Logistic Regression** and **K-Nearest Neighbors (KNN)**.

The project was developed as a part of the final semester MCA (Master of Computer Applications) in Distance Education course at **Lovely Professional University (LPU)**.

In addition to the ML model, a fully responsive **React + Vite** website was built to present the code, results, and visualizations.

## Tools Used
- Python
- Google Colab
- React.js + Vite (for the website)
- Machine Learning (Logistic Regression, KNN)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Bootstrap

## Features
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building (Logistic Regression and KNN)
- Model Evaluation (Confusion Matrix, Classification Report)
- Visualization of Results (Correlation Heatmap, Accuracy of both Models)

## Project Structure
```
Rainfall-Prediction-ML-Project/
│
├── rainfall-prediction-app/            # React + Vite website presenting the ML project
│ ├── public/                           # Static files (HTML, favicon)
│ ├── src/                              # React components, pages, assets
│ ├── package.json                      # Project dependencies
│ ├── vite.config.js                    # Vite configuration
│ └── ...                               # Other Vite build files
├── Rainfall_Prediction_Project.ipynb   # Google Colab Notebook with ML implementation
├── weather.csv                         # Weather Dataset used for training and testing
├── Weather Prediction Report.docx      # Final project report (LPU submission)
├── README.md                           # # Project documentation
```

## How to Run the Project

### For the ML Model
1. Clone the repository or download the files.
2. Open the `Rainfall_Prediction_Project.ipynb` notebook.
3. Run all cells in sequence in Google Colab or Jupyter Notebook.
4. Interpret the output and prediction results.

### For the React Website
1. Navigate to `rainfall-prediction-app/`
2. Run the following commands:
```
npm install
npm run dev
```

## Results
- Logistic Regression attained an accuracy of around **97%**.
- K-Nearest Neighbors attained an accuracy of around **83%**.

## Project Link
**GitHub Repository**: [(https://github.com/khushpreetsinghb/Rainfall-Prediction-ML-Project)]

## University Details
- **University**: Lovely Professional University
- **Program**: Master of Computer Applications (MCA)
- **Year**: 2025

## Acknowledgment
This project is submitted as a part of the academic requirement under the guidance of respected faculty members at Lovely Professional University.
