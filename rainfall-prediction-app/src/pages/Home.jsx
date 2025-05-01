import React from 'react'
import Section from '../components/Section'
import CodeBlock from '../components/CodeBlock'
import OutputBlock from '../components/OutputBlock'
import corrHeatmap from '../assets/Correlation Matrix Heatmap.png'
import cmLogistic from '../assets/Logistic Regression Confusion Matrix.png'
import cmKnn from '../assets/KNN Confusion Matrix.png'

const Home = () => {
    return (
        <div>
            <h1 className="text-center">Machine Learning for Rainfall Prediction</h1>
            <p className="lead text-center">A project by Khushpreet Singh (MCA) 22307370127</p>
            <hr />

            <Section title="Abstract">
                <p>
                    This project entitled <strong>"Machine Learning for Rainfall Prediction"</strong> attempts to use historical weather records to forecast rainfall probability with the help of classification techniques.
                </p>
                <p>
                    This study was performed based on a Kaggle dataset, which involves a number of weather-related features such as humidity, temperature, and wind speed.
                </p>
                <p>
                    We used two popular machine learning algorithms: Logistic Regression and K-Nearest Neighbors (KNN) to predict rainfall based on the input features. The code was implemented using Python in a Google Colab platform, using libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
                </p>
                <p>
                    The data was preprocessed and cleaned and then explored through exploratory data analysis to find the relevance of features. Training and testing were done for both algorithms, and performance metrics such as accuracy and confusion matrix were used for evaluation.
                </p>
                <p>
                    The results verify that machine learning could be a useful tool in performing weather forecasting activities, and the study provides stimulus for further improvement through more advanced models or real-time inputs.
                </p>
            </Section>

            <Section title="Dataset Description">
                <p>
                    <strong>Source:</strong> Kaggle - Rain in Australia dataset<br />
                    <strong>Size:</strong> ~145,000 entries with 24 weather-related columns
                </p>

                <p><strong>Key Features Used:</strong></p>
                <ul>
                    <li>MinTemp: Minimum temperature</li>
                    <li>MaxTemp: Maximum temperature</li>
                    <li>Rainfall: Amounnt of rainfall recorded</li>
                    <li>WindGustSpeed, Humidity3pm, Pressure9am</li>
                    <li>WindDir9am, WindDir3pm: Wind direction readings</li>
                    <li>RainToday: Whether it rained today (Yes/No)</li>
                    <li><strong>RainTomorrow:</strong> Target column (Yes/No)</li>
                </ul>

                <p>
                    The target variable was encoded to binary format:
                    <code> RainTomorrow = 1 (Yes)</code> and <code>0 (No)</code>.
                </p>
            </Section>

            <Section title="Tools and Techniques">
                <ul>
                    <li><strong>Programming Language:</strong> Python</li>
                    <li><strong>Platform:</strong> Google Colab</li>
                    <li><strong>Libraries Used:</strong></li>
                    <ul>
                        <li><strong>Pandas</strong> - loading and manipulation of data</li>
                        <li><strong>NumPy</strong> - numerical computations</li>
                        <li><strong>Matplotlib & Seaborn</strong> - data visualization</li>
                        <li><strong>Scikit-learn</strong> - preprocessing, model building and evaluation</li>
                    </ul>
                </ul>
            </Section>

            <Section title="Data Preprocessing and Model Training">

                <CodeBlock
                    title="1. Load Dataset"
                    code={`import pandas as pd
data = pd.read_csv("weather.csv")
data.head()
data.info()`}
                />
                <OutputBlock
                    title="Dataset Loaded (Sample + Info)"
                    output={`<class 'pandas.core.frame.DataFrame'>
RangeIndex: 366 entries, 0 to 365
Data columns (total 22 columns):
    #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
0   MinTemp        366 non-null    float64
1   MaxTemp        366 non-null    float64
2   Rainfall       366 non-null    float64
3   Evaporation    366 non-null    float64
4   Sunshine       363 non-null    float64
5   WindGustDir    363 non-null    object 
6   WindGustSpeed  364 non-null    float64
7   WindDir9am     335 non-null    object 
8   WindDir3pm     365 non-null    object 
9   WindSpeed9am   359 non-null    float64
10  WindSpeed3pm   366 non-null    int64  
11  Humidity9am    366 non-null    int64  
12  Humidity3pm    366 non-null    int64  
13  Pressure9am    366 non-null    float64
14  Pressure3pm    366 non-null    float64
15  Cloud9am       366 non-null    int64  
16  Cloud3pm       366 non-null    int64  
17  Temp9am        366 non-null    float64
18  Temp3pm        366 non-null    float64
19  RainToday      366 non-null    object 
20  RISK_MM        366 non-null    float64
21  RainTomorrow   366 non-null    object 
dtypes: float64(12), int64(5), object(5)
memory usage: 63.0+ KB`}
                />

                <CodeBlock
                    title="2. Handle Missing Values"
                    code={`data.dropna(inplace=True)

# Impute NaN values using the mean
for column in x_train.columns:
x_train[column] = x_train[column].fillna(x_train[column].mean())
x_test[column] = x_test[column].fillna(x_train[column].mean()) # use train set mean for test set`}
                />
                <OutputBlock
                    title="Missing Values Handled"
                    output={`Rows after dropna: 328
Remaining columns: 22
Missing values filled in x_train and x_test with column-wise mean.`}
                />

                <CodeBlock
                    title="3. Encode Categorical Features"
                    code={`from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feature in category_feature:
    data[feature] = le.fit_transform(data[feature])`}
                />
                <OutputBlock
                    title="Categorical Features Encoded"
                    output={`Encoded features:
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

Example: 'Yes' → 1, 'No' → 0`}
                />

                <CodeBlock
                    title="4. Feature Scaling"
                    code={`from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)`}
                />
                <OutputBlock
                    title="Features Scaled"
                    output={`Feature scaling applied using StandardScaler.
Mean: ~0, Std Dev: ~1`}
                />

                <CodeBlock
                    title="5. Logistic Regression Training"
                    code={`from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_scaled, y_train.squeeze())`}
                />
                <OutputBlock
                    title="Logistic Regression Model Trained"
                    output={`Model: LogisticRegression()
Training complete.`}
                />

                <CodeBlock
                    title="6. K-Nearest Neighbors Training"
                    code={`from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(x_train_scaled, y_train)`}
                />
                <OutputBlock
                    title="KNN Model Trained"
                    output={`Model: KNeighborsClassifier(n_neighbors=4)
Training complete.`}
                />

            </Section>

            <Section title="Model Evaluation and Results">

                <CodeBlock
                    title="7. Evaluate Logistic Regression"
                    code={`from sklearn.metrics import classification_report, confusion_matrix
y_pred_lr = lr.predict(x_test_scaled)
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))`}
                />

                <OutputBlock
                    title="Logistic Regression - Classification Report"
                    output={`              precision    recall  f1-score   support
           0       0.96      1.00      0.98        54
           1       1.00      0.83      0.91        12

    accuracy                           0.97        66
   macro avg       0.98      0.92      0.95        66
weighted avg       0.97      0.97      0.97        66`}
                />

                <OutputBlock
                    title="Logistic Regression - Confusion Matrix"
                    output={`[[54  0]
 [ 2 10]]`}
                />

                <CodeBlock
                    title="8. Evaluate K-Nearest Neighbors (KNN)"
                    code={`y_pred_knn = neigh.predict(x_test_scaled)
print(classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))`}
                />

                <OutputBlock
                    title="KNN - Classification Report"
                    output={`              precision    recall  f1-score   support
           0       0.83      1.00      0.91        54
           1       1.00      0.08      0.15        12
         
    accuracy                           0.83        66
   macro avg       0.92      0.54      0.53        66
weighted avg       0.86      0.83      0.77        66`}
                />

                <OutputBlock
                    title="KNN - Confusion Matrix"
                    output={`[[54  0]
 [11  1]]`}
                />
            </Section>

            <Section title="Accuracy Comparison">

                <div className="card mb-4">
                    <div className="card-header bg-info text-dark">
                        Accuracy of Different Machine Learning Models
                    </div>
                    <div className="card-body p-0">
                        <table className="table table-bordered mb-0 text-center">
                            <thead className="table-light">
                                <tr>
                                    <th>Model</th>
                                    <th>Accuracy</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Logistic Regression</strong></td>
                                    <td>97%</td>
                                </tr>
                                <tr>
                                    <td><strong>K-Nearest Neighbors (KNN)</strong></td>
                                    <td>83%</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div className="alert alert-success mt-3" role="alert">
                    <strong>Conclusion:</strong> Logistic Regression achieved higher accuracy compared to KNN for the Rainfall Prediction dataset.
                </div>

            </Section>

            <Section title="Visualizations and Insights">

                <div className="row">
                    <div className="col-md-12 mb-4">
                        <div className="card">
                            <img src={corrHeatmap} className="card-img-top" alt="Correlation Heatmap" />
                            <div className="card-body">
                                <h5 className="card-title">Correlation Heatmap</h5>
                                <p className="card-text">
                                    This heatmap illustrates the correlation between various weather characteristics. MaxTemp and Evaporation show a moderate positive correlation with Rainfall, while MinTemp has a weak positive correlation. Sunshine and Pressure both have a negative correlation with Rainfall, where Sunshine shows a moderate negative correlation and Pressure has a weak negative correlation. Humidity also has a moderate positive correlation with Rainfall.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="col-md-6 mb-4">
                        <div className="card">
                            <img src={cmLogistic} className="card-img-top" alt="Confusion Matrix - Logistic Regression" />
                            <div className="card-body">
                                <h5 className="card-title">Confusion Matrix - Logistic Regression</h5>
                                <p className="card-text">
                                    The Logistic Regression model performed better than the KNN model with greater precision and recall for both rain and no-rain prediction. The Logistic Regression model correctly classified 10 out of 12 rain cases.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="col-md-6 mb-4">
                        <div className="card">
                            <img src={cmKnn} className="card-img-top" alt="Confusion Matrix - KNN" />
                            <div className="card-body">
                                <h5 className="card-title">Confusion Matrix - K-Nearest Neighbors</h5>
                                <p className="card-text">
                                    KNN performed poorly in identifying rain days, accurately classifying only 1 out of 12 rain cases. It was precise for 'No Rain' with a value of 1.0 but had a poor recall of 0.08 for 'Rain'.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

            </Section>

            <Section title="Conclusion and Future Scope">

                <p>
                    This project successfully illustrates how simple machine learning algorithms such as Logistic Regression and K-Nearest Neighbors can be used to forecast rainfall using historical weather data. The performance was evaluated in terms of accuracy scores and confusion matrices, where Logistic Regression performed better than KNN on this dataset.
                </p>

                <p>
                    The results support the capability of machine learning in climate-based forecasting applications. These models can assist in agricultural planning, resource management, and disaster readiness.
                </p>

                <h5 className="mt-4">Future Scope:</h5>
                <ul>
                    <li>Integration with live weather APIs for real-time predictions</li>
                    <li>Testing with more advanced models such as Random Forests, SVM, and Neural Networks</li>
                    <li>Including multiclass forecasting of weather (e.g., light rain, heavy rain, dry, etc.)</li>
                    <li>Building a complete functional forecasting dashboard or mobile app</li>
                </ul>

            </Section>

            <Section title="References">

                <ol>
                    <li>
                        Weather Dataset - <a href="https://www.kaggle.com/datasets/zaraavagyan/weathercsv" target="_blank" rel="noopener noreferrer">
                            Kaggle Dataset
                        </a>
                    </li>
                    <li>
                        Young Sin (2024), “Weather Predictions Model” - <a href="https://www.kaggle.com/code/yoongsin/weather-predictions-model" target="_blank" rel="noopener noreferrer">
                            Kaggle Project
                        </a>
                    </li>
                    <li>
                        1000Projects.org (2022), “Predict the Forest Fires Python Project using Machine Learning Techniques” - <a href="https://1000projects.org/predict-the-forest-fires-python-project-using-machine-learning-techniques.html" target="_blank" rel="noopener noreferrer">
                            Project Link
                        </a>
                    </li>
                    <li>
                        P. S. R. Kumar, R. V. S. Kumar, M. K. R. Kumar, and R. S. Kumar (2023), “Prediction of Rainfall Analysis Using Logistic Regression and Support Vector Machine” - <a href="https://www.researchgate.net/publication/369985746_Prediction_of_Rainfall_Analysis_Using_Logistic_Regression_and_Support_Vector_Machine" target="_blank" rel="noopener noreferrer">
                            Journal of Physics
                        </a>
                    </li>
                    <li>
                        S. Sharma, A. Gupta, and R. Patel (2024), “Comparative Analysis of Different Rainfall Prediction Models” - <a href="https://www.sciencedirect.com/science/article/pii/S2590123024003475" target="_blank" rel="noopener noreferrer">
                            A Case Study
                        </a>
                    </li>
                </ol>

            </Section>


            <footer className="bg-dark text-white text-center py-3 mt-5">
                <small>&copy; 2025 Rainfall Prediction ML Project - Developed by Khushpreet Singh</small>
            </footer>

        </div>
    )
}

export default Home