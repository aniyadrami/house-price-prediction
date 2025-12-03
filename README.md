## House Price Prediction (ML)
This project uses machine learning to predict housing prices based on key features such as income, house age, number of rooms, and population. Using a linear regression model, the project demonstrates how data can be used to estimate real-estate value in the U.S. housing market.

## Project Goals
- Analyze a real housing dataset  
- Explore relationships between housing features and price  
- Build a predictive model using **Linear Regression**  
- Evaluate model accuracy using **MAE** and **R² score**

## Dataset
The dataset includes the following columns:

- `Avg Area Income`  
- `Avg Area House Age`  
- `Avg Area Number of Rooms`  
- `Avg Area Number of Bedrooms`  
- `Area Population`  
- `Price` (target variable)

Dataset source: Kaggle 

##  Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-Learn   

## How It Works
1. Load and clean the dataset  
2. Select important features  
3. Split the data into training + testing sets  
4. Train a **Linear Regression** model  
5. Make predictions  
6. Evaluate results  

## Model Evaluation
The model prints:
- **MAE (Mean Absolute Error)** – average error in price predictions  
- **R² Score** – how well the model fits the data  

Higher R² = better predictions.

## Results Summary
The linear regression model shows strong predictive power on this dataset, demonstrating how simple ML techniques can effectively model real-world housing prices.
