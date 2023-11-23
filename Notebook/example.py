from fastapi import FastAPI , File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

model_path = 'Version_One_Model'
model = tf.keras.models.load_model(model_path)

CLASS_NAMES = ["Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy"]

app = FastAPI()

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

    pass

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port = 8000)