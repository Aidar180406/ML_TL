import streamlit as st
import requests
from PIL import Image
import io

st.title("Эксперимент - Хоккей")
st.write("Загрузите изображение, и модель определит класс.")

# Загрузка файла
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Показываем картинку пользователю
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    if st.button("Распознать"):
        with st.spinner("Отправка на сервер..."):
            # Отправляем файл на наш FastAPI сервер
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                response.raise_for_status()
                data = response.json()

                # Выводим результат
                st.success(f"**Результат:** {data['class']}")
                st.info(f"**Уверенность:** {data['confidence'] * 100:.2f}%")

            except requests.exceptions.ConnectionError:
                st.error("Ошибка: Сервер API не запущен. Запустите `uvicorn api:app` в другом терминале.")