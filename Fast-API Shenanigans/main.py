from enum import Enum
from fastapi import FastAPI

app = FastAPI()

class AvailableCuisines(str, Enum):
    indian = "indian"
    chinese = "chinese"
    italian = "italian"

food_items = {
    'indian':["Samosa","Dosa"],
    'chinese':["Spagheti", "Rice"],
    'italian':["Pizza","Ravioli" ]
}

path = "/get_items/{cuisine}"


valid_cuisines = food_items.keys()


@app.get(path)
async def get_items(cuisine: AvailableCuisines):
    return food_items.get(cuisine)

coupon_codes = {
    1: "10%",
    2: "20%",
    3: "30%"
}

path_coupons = "/get_coupons/{code}"

@app.get(path_coupons)
async def get_items(code: int):