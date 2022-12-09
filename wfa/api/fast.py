from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from wfa.ml_logic.registry import load_model
from wfa.ml_logic import model
from wfa.utils.get_new_images import get_new_image, split_tiles
import pandas as pd
#from colorama import Fore, Style
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model=load_model()

# http://127.0.0.1:8000/watchingfromabove/prediction?address=35%av%Joseph%Monier&year_of_interest=2020&historical_year=2018

@app.get('/watchingfromabove/prediction')
def predict(address:str='Paris',
            year_of_interest:str='2020',
            historical_year:str='2019'):
    """
    With the address provided by user, we are able to return the satellite
    image of current year, as well as the satellit image of historical year
    (option among 2017,2018,2019,2020)
    We will also return the evolution in % of changes
    """
    user_input  = pd.DataFrame(dict(
        address=[address],
        year_of_interest=[year_of_interest],
        historical_year=[historical_year]
    ))
    year_of_interest_image = get_new_image(address,year_of_interest)
    historical_year_image = get_new_image(address,historical_year)

    X_current = split_tiles(year_of_interest_image)
    X_history = split_tiles(historical_year_image)

    cat_pred1 = model.predict_new_images(app.state.model, X_current);
    cat_pred2 = model.predict_new_images(app.state.model, X_history);
    cat_pred1 = np.ndarray.tolist(cat_pred1)
    cat_pred2 = np.ndarray.tolist(cat_pred2)

    #print(Fore.BLUE + f"\nLoad model {cat_pred2} stage from mlflow..." + Style.RESET_ALL)
    return {'current_year':cat_pred1, 'historical_year':cat_pred2}


@app.get("/")
def root():
    return dict(greeting="Hello")
