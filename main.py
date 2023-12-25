from fastapi import FastAPI, Form, Query
import joblib
from pydantic import BaseModel
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning

 # Suppress DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

 
pipeline = joblib.load('model/xgb.joblib')
pipeline_1 = joblib.load('model\dt.joblib')



app = FastAPI(
    title="Customer Churn Analysis API"
)

class SepsisFeatures(BaseModel):
    TENURE: object
    MONTANT: float
    FREQUENCE_RECH: float
    REVENUE: float
    ARPU_SEGMENT: float
    FREQUENCE: float
    DATA_VOLUME: float
    ON_NET: float
    ORANGE: float
    TIGO: float
    REGULARITY: int
    FREQ_TOP_PACK:float
    

    # Define the available models
models = {
    "xgb": pipeline,
    "dt": pipeline_1,
    }


@app.get('/')
def home():
   return "Customer churn"


@app.get('/info')
def appinfo():
    return 'Customer churn prediction: This is my interface'
 
 
# Define the prediction endpoint
@app.post('/predict_churn')
#def predict_sepsis(sepsis_features: SepsisFeatures):
def predict_sepsis(
    churn_features: SepsisFeatures,
      selected_model: str = Query("Xgb", description="Select the model for prediction")
):
    
    # Convert input features to a DataFrame
    df = pd.DataFrame([churn_features.model_dump()])
    
     # Check if the specified model is valid
    if selected_model not in models:
        return {"error": "Invalid model specified"}
 
   # Perform prediction using the selected pipeline
    selected_pipeline = models[selected_model]
    prediction = selected_pipeline.predict(df)
    #encoder_prediction = encoder.inverse_transform([prediction])[0]
    
     # Get the probability scores
    try:
        probabilities = selected_pipeline.predict_proba(df)
        # Assuming binary classification, use the probability of the positive class
        probability_score = probabilities[0][1]
    except AttributeError:
        probability_score = None
    # Convert numpy.float32 to regular float
    probability_score = float(probability_score) if probability_score is not None else None

    return {"model_used": selected_model,"probability_score": probability_score}
    
   




