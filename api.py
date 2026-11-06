from fastapi import FastAPI, File, UploadFile
from teachable_machine import TeachableMachine
from PIL import Image
import io
import tempfile
import os
import traceback

app = FastAPI()

# Загружаем модель при старте (ОБЯЗАТЕЛЬНО проверьте пути!)
try:
    model = TeachableMachine(
        model_path="keras_model.h5",  # Убедитесь, что путь правильный
        labels_file_path="labels.txt"
    )
    print("✅ Модель успешно загружена!")
except Exception as e:
    print(f"❌ ОШИБКА ПРИ ЗАГРУЗКЕ МОДЕЛИ: {e}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Читаем картинку
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            image.save(temp_path, format="JPEG")

        try:
            # 3. Отправляем в модель
            result, _ = model.classify_and_show(temp_path)

            class_name = result["class_name"]
            clean_class = class_name[2:] if class_name[0].isdigit() else class_name

            return {
                "class": clean_class.strip(),
                "confidence": float(round(result["class_confidence"], 4))
            }
        finally:
            # Удаляем файл
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        # Если ошибка - возвращаем её текст
        error_msg = traceback.format_exc()
        print(f"❌ ОШИБКА ПРИ ПРЕДСКАЗАНИИ:\n{error_msg}")
        return {"error": str(e), "details": error_msg}