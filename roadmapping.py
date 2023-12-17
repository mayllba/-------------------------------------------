import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import sobel

# Функция для загрузки и предобработки изображения
def load_and_process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    gray_image = rgb2gray(image)
    edges = sobel(gray_image)
    return edges

# Функция для сегментации изображения
def segment_image(edges, threshold=0.1):
    return (edges > threshold).astype(np.uint8)

# Функция для извлечения контуров из сегментированного изображения
def get_contours(segmented):
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Функция для обхода всех изображений в папке и их обработки
def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Поддерживаемые форматы изображений
            full_path = os.path.join(folder_path, filename)
            try:
                edges = load_and_process_image(full_path)
                segmented = segment_image(edges)
                contours = get_contours(segmented)

                # Визуализация результата
                plt.figure(figsize=(10, 10))
                plt.imshow(segmented, cmap='gray')
                for contour in contours:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
                plt.title(f'Processed: {filename}')
                plt.axis('off')
                plt.show()
                
            except ValueError as e:
                print(e)

# Замените 'path_to_your_folder' на путь к папке с вашими изображениями
folder_path = r'C:/Users/elkin/OneDrive/Документы/проект/image/'
process_images_in_folder(folder_path)