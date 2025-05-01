# House Price Prediction 
Predict house prices based on size, location, number of rooms, etc.

# Machine Learning Workflow
## 1. Get & Explore Data
After finding the optimal data for the task(**House Price Prediction**), it were saved and loaded with Pandas, and checking its charastristics like size, attributes, their types,...

## 2. Prepare Data
After loading and studing the data, we start our preparation by **cleaning the data**, then **feature engineering** with new features like the area of each bedroom, after that we **encoded the text** attributes with different techniques like ordinal and one hot encoding and finally scaling of the numerical features.

## 3. Shortlist Promising Models
So the data is ready, can you guess whats next?
of course training, so we expiremented with different models: 
- **Linear Regression**
- **Polynomial Regression**
- **Support Vector Machines** with different kernel types.
- **K-Nearest Neighbors**
- **Decision Tree**
- **Random Forest**

and evaluating with **R squared** metric on the **training and validation datasets**. 

## 4. Fine-Tunning
We select the most promising models and then we fine-tune them simply with **Grid Search** method.

## 5. Ensemble Method
Then also expirementing with some ensemble methods like **XGBoost** and **LGBM**, and then using the ensemble methods like **Voting** and **stacking** to acheive the highest possible performance possible.

## 6. Evaluation
With **Cross Validation** technique we evaluate the final model on the **test data**
and we reached **0.82 R2**, amazinggg.

# Streamlit App
An interactive web app to try the model.

# License
This project is licensed under the MIT License. Copyright (c) 2025 Ameur.