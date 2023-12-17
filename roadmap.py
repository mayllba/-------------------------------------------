import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
import numpy as np
from PIL import Image

def load_image(image_path, target_size=(224, 224)):
    """ Загрузка и предварительная обработка изображения """
    image = Image.open(image_path).convert('RGB')  # Преобразование в RGB
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Нормализация
    return image

def predict_road_map(model, image):
    """ Предсказание карты дорог с помощью модели """
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    return prediction

def process_images_in_folder(folder_path, model, target_size=(224, 224)):  # Обновленный размер изображения для ResNet50
    """ Обработка всех изображений в папке """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Фильтрация по расширениям файлов
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path, target_size)
            road_map = predict_road_map(model, image)

            # Сохранение или отображение результата
            road_map_image = Image.fromarray((road_map * 255).astype(np.uint8))
            road_map_image.save(os.path.join(folder_path, f"road_map_{filename}"))  # Сохранение
            # road_map_image.show()  # Отображение

# Загрузка предварительно обученной модели ResNet50
model = ResNet50(weights='imagenet')

# Путь к папке со спутниковыми изображениями
folder_path = 'C:/Users/elkin/OneDrive/Документы/проект/image/'

# Обработка всех изображений в папке
process_images_in_folder(folder_path, model)