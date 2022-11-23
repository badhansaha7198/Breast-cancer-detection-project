from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

Model = tf.keras.models.load_model("../model/1")


Class_name = ["benign", "malignant"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/breast_cancer")
async def breast_cancer(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = Model.predict(img_batch)
    prediction_class = Class_name[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': prediction_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)