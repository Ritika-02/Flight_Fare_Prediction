# Flight Fare Prediction

Welcome to the Flight Fare Prediction project! ✈️ This project uses machine learning to predict flight fares based on various factors. Whether you're a traveler planning your next adventure or an enthusiast exploring data science, this project takes you on a journey through the skies.

## Project Overview

- **Objective:** Predict flight fares using machine learning techniques.
- **Tech Stack:** Python, Flask, Scikit-Learn, AWS
- **Dataset:** [Link to Dataset](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh)


## Features

- **Machine Learning Models:** Trained models for accurate fare predictions.
- **Interactive Web App:** Explore and predict flight fares with a user-friendly interface.
- **AWS Deployment:** Accessible globally with cloud deployment.

## Project Structure

```plaintext
.
├── artifacts
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── test.csv
│   └── train.csv
├── github
│   └── workflows
│       └── .gitkeep
├── logs
├── notebooks
│   ├── data
│   │   └── .gitkeep
│   └── EDA
├── src
│   └── Flight_Fare_Prediction
│       ├── components
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   └── model_trainer.py
│       ├── pipelines
│       │   ├── __init__.py
│       │   ├── prediction_pipeline.py
│       │   └── training_pipeline.py
│       └── utils
│           └── __init__.py
├── static
│   ├── pic.webp
│   └── styles.css
├── templates
│   ├── form.html
│   ├── index.html
│   └── result.html
├── .gitignore
├── app.py
├── init_setup.sh
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── template.py
└── test.py
