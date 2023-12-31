import gradio as gr
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# key lists

expected_inputs = ['TENURE','MONTANT','FREQUENCE_RECH','REVENUE','ARPU_SEGMENT','FREQUENCE','DATA_VOLUME','ON_NET','ORANGE','TIGO','REGULARITY','FREQ_TOP_PACK']
numerics = ['MONTANT','FREQUENCE_RECH','REVENUE','ARPU_SEGMENT','FREQUENCE','DATA_VOLUME','ON_NET','ORANGE','TIGO','REGULARITY','FREQ_TOP_PACK']
categoricals = ['TENURE']
# Load the model and pipeline

# Define helper functions
# Function to load the pipeline
def load_pipeline(file_path="model/dt.joblib"):
    with open(file_path, "rb") as file:
        pipeline = joblib.load(file)
    return pipeline


# Import the model
# Import the model using the load_pipeline function
model = load_pipeline("model/dt.joblib")

# Instantiate the pipeline
dt_pipeline= load_pipeline()
#decision_tree_pipeline = load_pipeline()

scaler = None
encoder = None

# Function to process inputs and return prediction


def predict_customer_attrition(*args, pipeline=dt_pipeline,model=model, scaler= scaler, encoder=encoder):
    # Convert inputs into a dataframe
    input_data = pd.DataFrame([args], columns=expected_inputs)
       # Print the input data
    print("Input Data:")
    print(input_data)

    # Make the prediction 
    model_output = pipeline.predict(input_data)

    if model_output == "Yes":
        prediction = 1
    else:
        prediction = 0


    # Return the prediction
    return {"Prediction: Customer is likely to Churn": prediction,
            "Prediction: Customer is likely not to Churn": 1 - prediction}
    
  
      # Set up interface
    # Inputs

TENURE= gr.Dropdown(label = "What is the duration of your network?", choices = ['I 18-21 month' ,'K > 24 month', 'G 12-15 month' ,'J 21-24 month','H 15-18 month', 'F 9-12 month' ,'E 6-9 month' ,'D 3-6 month'],value ='I 18-21 month')
MONTANT= gr.Slider(label= "What is your top-amount?", minimum= 15, maximum= 800, value= 10, interactive= True)
FREQUENCE_RECH= gr.Slider(label= "What is number of times does you refilled your bundle?", minimum= 5, maximum= 200, value= 10, interactive= True)
REVENUE = gr.Slider(label="What is your monthly income", minimum = 100,maximum=10000, value=200,interactive=True)
ARPU_SEGMENT = gr.Slider(label ="What is your income over 90 days / 3",minimum = 1000,maximum=500000,value=1000,interactive=True)
FREQUENCE= gr.Slider(label= "How do you how often do you use the service ",minimum = 10,maximum=200,value= 5,interactive=True)
DATA_VOLUME = gr.Slider(label= "How many times do you have connections", minimum =20,maximum = 1000,value= 2,interactive=True)
ON_NET = gr.Slider(label = "How many time do you do inter expresso call", minimum = 5,maximum= 1000,value=3,interactive=True)
ORANGE = gr.Slider(label = "How many time do you use orange to make calls(tigo)",minimum=5,maximum =100,value=2,interactive=True)
TIGO= gr.Slider(label= "How many time do you use tigo networks", minimum=6,maximum=100,value =5,interactive=True)
REGULARITY= gr.Slider(label= "How many number of times the are you active for 90 days",minimum=5,maximum=100,value=2,interactive=True)
FREQ_TOP_PACK= gr.Slider(label="How many number of times does you been activated to the top pack packages",minimum=10,maximum=1000,value=5,interactive=True)

# Create the Gradio interface
gr.Interface(inputs=[TENURE,MONTANT,FREQUENCE_RECH,REVENUE,ARPU_SEGMENT,FREQUENCE,DATA_VOLUME,ON_NET,ORANGE,TIGO,REGULARITY,FREQ_TOP_PACK],
             fn=predict_customer_attrition,
             outputs= gr.Label("Awaiting Submission...."),
             title="Telecommunication Customer Attrition Prediction App",
             description="This app was created by Santorini during our Project 4 ", live=True).launch(inbrowser=True, show_error=True)


