import cv2
import qrcode
import numpy as np

def makeqr(data, name):
    qr = qrcode.QRCode()
    # добавить данные в QR-код
    qr.add_data(data)
    # компилируем данные в массив QR-кода
    qr.make()
    # распечатать форму изображения
    print("The shape of the QR image:", np.array(qr.get_matrix()).shape)
    # переносим массив в реальное изображение
    img = qr.make_image(back_color='yellow', fill_color='blue')
    # сохраняем в файл
    img.save(name)

def readqr(file):
    img = cv2.imread(file)
    # инициализируем детектор QRCode cv2
    detector = cv2.QRCodeDetector()
    data, bbox, straight_qrcode = detector.detectAndDecode(img)
    return data, bbox, straight_qrcode, img

def drawqrborders(bbox, img):
    if bbox is not None:
        print(f"QRCode data:\n{data}")
        # отображаем изображение с линиями
        # длина ограничивающей рамки
        n_lines = len(bbox[0])
        print(f"n_lines: {n_lines}")
        for i in range(n_lines):
            # рисуем все линии
            point1 = (int(bbox[0][i][0]), int(bbox[0][i][1]))
            point2 = (int(bbox[0][(i+1) % n_lines][0]), int(bbox[0][(i+1) % n_lines][1]))
            print(point1, type(point1[0]), type(point1[1]))
            print(point2, type(point2[0]), type(point2[1]))
            cv2.line(img, point1, point2, color=(255, 0, 0), thickness=2)
    else:
        raise IndexError('Empty input')
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectVideo():
    cap = cv2.VideoCapture(0)
    # инициализируем детектор QRCode cv2
    detector = cv2.QRCodeDetector()
    while True:
        _, img = cap.read()
        # обнаружить и декодировать
        data, bbox, _ = detector.detectAndDecode(img)
        # проверяем, есть ли на изображении QRCode
        if bbox is not None:
            # отображаем изображение с линиями
            for i in range(len(bbox)):
                # рисуем все линии
                cv2.line(img, tuple(bbox[i][0]), tuple(bbox[(i + 1) % len(bbox)][0]), color=(255, 0, 0), thickness=2)
            if data:
                print("[+] QR Code detected, data:", data)
        # отобразить результат
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

file = '2koda_test.png'
result = readqr(file)
data, bbox, straight_qrcode, img = result
print(f"result: {result}")
drawqrborders(bbox, img)