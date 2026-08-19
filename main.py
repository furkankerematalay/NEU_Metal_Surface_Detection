from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import io
import builtins

builtins.preprocess_input = preprocess_input

app= FastAPI(title= "Industrial Steel Defect Detection API", version="1.0")

SAVED_MODEL_PATH = "model_resnet50_fine_tuned.keras"

model = load_model(
    SAVED_MODEL_PATH,
    custom_objects={'preprocess_input': preprocess_input},
    compile = False,
    safe_mode=False
)

CLASS_NAMES = ['crazing', 'inclusion','patches','pitted_surface','rolled-in scale','scratches']


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 1. Read raw byte stream from the incoming network request
        image_bytes = await file.read()

        # 2. Convert bytes to a physical image using Pillow (PIL)
        image = Image.open(io.BytesIO(image_bytes))

        # 3. Ensure RGB format and resize to 200x200 (Model's required input shape)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((200, 200))

        # 4. Convert image to a Numpy Array (Mathematical matrix)
        image_array = np.array(image)

        # 5. Expand dimensions to create a batch of 1: (1, 200, 200, 3)
        image_array = np.expand_dims(image_array, axis=0)

        # ==========================================
        # 3. CPU INFERENCE & JSON RESPONSE
        # ==========================================
        # 6. Feed the image matrix to the neural network
        prediction_probabilities = model.predict(image_array, verbose=0)

        # 7. Extract the highest probability and its corresponding class index
        predicted_index = np.argmax(prediction_probabilities[0])
        confidence_score = float(np.max(prediction_probabilities[0]))

        detected_defect = CLASS_NAMES[predicted_index]

        # 8. Return the result in a universal JSON format
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "detected_defect": detected_defect,
            "confidence_score_percentage": round(confidence_score * 100, 2)
        })

    except Exception as e:
        # Prevent server crash on bad inputs; return an error JSON instead
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
