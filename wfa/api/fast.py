from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from wfa.ml_logic.model import load_model
from wfa.utils.get_new_images import get_s2maps_data
from wfa.ml_logic import model
from wfa.utils.image_viz import plot_classified_images
from wfa.utils.get_new_images import get_new_image, split_tiles, address_to_coord
import pandas as pd
import matplotlib.pyplot as plt

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
def predict(address:str,
            year_of_interest:str,
            historical_year:str):
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

#return array, and plot on frontend

#    plt.figure()
#    plt.subplot(1,2,1)
#    plot_classified_images(X_history, cat_pred2)
#    plt.title(f"Satellite image of year {historical_year}")
#    plt.subplot(1,2,2)
#    plot_classified_images(X_current, cat_pred1)
#    plt.title(f"Satellite image of year {year_of_interest}")
#    plt.show()


@app.get("/")
def root():
    return dict(greeting="Hello")
