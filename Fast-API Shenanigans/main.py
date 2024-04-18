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

model_potato_path = 'Version_One_Model'
modelpotato = tf.keras.models.load_model(model_potato_path)

model_path = 'Leaf_Model'
model = tf.keras.models.load_model(model_path)             


CLASS_POTATO = ["Early Blight", "Late Blight", "Healthy"]
CLASS_NAMES = ["Leaf", "Not Potato"]

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
    prediction_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    if prediction_class == 'Leaf':
        prediction = modelpotato.predict(image_batch)
        prediction_class = CLASS_POTATO[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        return {
        'class': prediction_class,
        'confidence': float(confidence)
    }
    else:
        return {
            'class': prediction_class,
            'confidence': float(confidence)
        }

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8000)