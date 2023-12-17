import os
import roadmap as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

# Загрузка предварительно обученной модели ResNet50
model = ResNet50(weights='imagenet')

def classify_image(img_path, model):
    # Загрузка и предобработка изображения
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Классификация изображения
    preds = model.predict(x)
    # Декодирование предсказаний
    print('Predicted:', decode_predictions(preds, top=3)[0])

    urban_classes = ['street_sign', 'traffic_light', 'parking_meter', 'skyscraper', 'taxi', 
                    'bus', 'fire_hydrant', 'pay_phone', 'bench', 'fountain', 
                    'manhole_cover', 'pedestrian_crossing', 'traffic_light']

    # Проверка, является ли изображение городским
    # На самом деле вам нужно будет определить, какие классы соответствуют городам в ImageNet
    for _, class_name, _ in decode_predictions(preds, top=3)[0]:
        if class_name in urban_classes:  # Примерные названия классов
            return True
    return False

# Путь к папке с изображениями
directory = 'C:/Users/elkin/OneDrive/Документы/проект/image/'

# Проверка каждого изображения в папке
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
        file_path = os.path.join(directory, filename)
        if not classify_image(file_path, model):
            os.remove(file_path)  # Удаление, если не город
            print(f"Изображение {filename} было удалено.")

print("Процесс классификации и удаления завершен.")