import joblib
import pandas as pd
from fastapi import FastAPI,Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title = 'Jobs API')

# CORS lets the dashboard (a web page) call this API from the browser.
# "*" allows any website, which is fine for local use but should be restricted when deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_headers = ["*"],
    allow_methods = ["*"]
)

# Send the cleaned dataframe to the dashboard as a list of rows
@app.get('/data')
def data():
    df = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Jobs_Project\models\data_frame.pkl")
    return Response(df.to_json(orient = 'records'),media_type = 'application/json')


# Load the trained model and the column order it was trained with (done once, when the server starts)
model = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Jobs_Project\data\model.pkl")
columns = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Jobs_Project\models\model_columns.pkl")


# The data a request must contain. FastAPI checks the types and rejects bad requests.
# Text features are one-hot encoded: 1 for the selected category and 0 for the others.
class CompanyInput(BaseModel):
    company:int
    post_date_days:int
    company_size:int
    job_title_data_analyst:int
    job_title_data_engineer:int
    job_title_data_scientist:int
    job_title_machine_learning_engineer:int
    seniority_level_junior:int
    seniority_level_lead:int
    seniority_level_midlevel:int
    seniority_level_senior:int
    status_hybrid:int
    status_on_site:int
    status_remote:int
    status_unknown:int
    industry_education:int
    industry_energy:int
    industry_finance:int
    industry_healthcare:int
    industry_logistics:int
    industry_manufacturing:int
    industry_retail:int
    industry_technology:int
    ownership_private:int
    ownership_public:int


@app.post('/predict')
def category_predict(data: CompanyInput):

    # Turn the request into a one-row dataframe
    df = pd.DataFrame([data.model_dump()])
    # Put the columns in the same order as during training (missing columns are filled with 0)
    df = df.reindex(columns = columns,fill_value = 0)

    # Predicted class (1 = high salary category, 0 = low) and the probability of class 1
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {'Prediction':prediction,'Probability':probability}