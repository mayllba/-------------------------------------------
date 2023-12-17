import cv2
 
image = cv2.imread('C:/Users/elkin/OneDrive/Изображения/Снимки экрана/Снимок экрана 2023-11-18 155922.png')
if image is None:
    print("Ошибка: изображение не найдено. Проверьте путь к файлу.")
else:
    cv2.imshow("lubertsy", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()