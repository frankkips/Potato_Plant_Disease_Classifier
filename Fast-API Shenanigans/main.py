from fastapi import FastAPI , File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from fastapi.middleware.cors import CORSMiddleware



origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

model_path = 'Version_One_Model'
model = tf.keras.models.load_model(model_path)

CLASS_NAMES = ["Early blight", "Late blight", "Healthy"]




# endpoint = "http://localhost:8605/v1/models/potato_model:predict"

@app.get("/ping")
async def ping():
    return "Hello I'm Alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    prediction = model.predict(image_batch)
    # json_data = {
    #     "instances": image_batch.tolist()
    # }
    # response = requests.post(endpoint,json=json_data)
    # prediction = response.json()["predictions"][0]

    # prediction_class = CLASS_NAMES[np.argmax(prediction)]
    # confidence = np.max(prediction)
    prediction_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return {
        'class': prediction_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8000)