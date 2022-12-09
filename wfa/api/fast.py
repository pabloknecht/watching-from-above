from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from wfa.ml_logic.registry import load_model
from wfa.ml_logic import model
from wfa.utils.get_new_images import get_new_image, split_tiles
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
            year_1:str='2020',
            year_2:str='2019'):
    """
    With the address provided by user, we are able to return the satellite
    image of current year, as well as the satellit image of historical year
    (option among 2017,2018,2019,2020,Google)
    """

    year_1_image = get_new_image(address,year_1)
    year_2_image = get_new_image(address,year_2)

    X_image_1 = split_tiles(year_1_image)
    X_image_2 = split_tiles(year_2_image)

    cat_pred1 = model.predict_new_images(app.state.model, X_image_1);
    cat_pred2 = model.predict_new_images(app.state.model, X_image_2);
    cat_pred1 = np.ndarray.tolist(cat_pred1)
    cat_pred2 = np.ndarray.tolist(cat_pred2)

    #print(Fore.BLUE + f"\nLoad model {cat_pred2} stage from mlflow..." + Style.RESET_ALL)
    return {'current_year':cat_pred1, 'historical_year':cat_pred2}

@app.get("/reloadmodel")
def root():
    app.state.model=load_model()
    return dict(greeting="Model reloaded")

@app.get("/")
def root():
    return dict(greeting="Hello")
