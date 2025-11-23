import cv2 as cv
import numpy as np
from readyqr import drawqrborders

class QrHandler():
    def detect(self, img):
        print(len(img), len(img[0]))
        for y in range(len(img)):
            for x in range(len(img[0])):
                if (img[y, x] < [50, 50, 50]).all():
                    print('black pixel!')
                    square_length = self._get_square_length(img, y, x)
                    if square_length != -5 and self._is_has_lil_square(img, y, x, square_length):
                        print(f'maybe x: {x} y: {y}')
                        for y_2 in range(y + square_length, len(img)):
                            if (img[y_2, x] < [50, 50, 50]).all():
                                square_length_2 = self._get_square_length(
                                    img, y_2, x)
                                if square_length_2 != -5 and self._is_has_lil_square(img, y_2, x, square_length_2):
                                    if square_length_2 in range(square_length - 3, square_length + 3):
                                        qr_size = y_2 - y
                                        square_length_3 = self._get_square_length(
                                            img, y, x + qr_size)
                                        if square_length_3 != -5 and self._is_has_lil_square(img, y, x + qr_size, square_length_3):
                                            if square_length_3 in range(square_length - 3, square_length + 3):
                                                print('ALLRIGHT')
                                                return img[y: y + qr_size + square_length, x: x + qr_size + square_length], (((y,x),(y,x+qr_size + square_length),(y + qr_size + square_length, x),(y+qr_size + square_length, x+qr_size + square_length)))

    def _is_black_point(self, img, y, x, inaccuracy):
        y_2 = y + inaccuracy
        if y_2 >= len(img):
            y_2 = len(img) - 1
        x_2 = x + inaccuracy
        if x_2 >= len(img[0]):
            x_2 = len(img[0]) - 1
        for y in range(y - inaccuracy, y_2):
            for x in range(x - inaccuracy, x_2):
                if (img[y, x] < [50, 50, 50]).all():
                    return True
        return False

    def _get_square_length(self, img, y, x):
        square_length = 0
        for x_i in range(x, len(img[0])):
            if (img[y, x_i] > [50, 50, 50]).all():
                break
            square_length += 1
        if square_length >= 6:
            if self._is_black_point(img, y + square_length, x + square_length, 3) and self._is_black_point(img, y + square_length, x, 3):
                return square_length
        return -5

    def _is_has_lil_square(self, img, y, x, square_length):
        lil_square_length = 0
        y = y + square_length // 2
        x = x + square_length // 2
        have_white = False
        print(f"x: {x}, y: {y}")
        print(f"square: {square_length}")
        print(f"x+square: {x + square_length//2}, max: {len(img[0])}")
        print(f"y+square: {y + square_length//2}, max: {len(img)}")
        print('=========================================')
        for x_lil in range(x, x + square_length//2 - 1):
            if (img[y, x_lil] > [50, 50, 50]).all():
                have_white = True
                break
            lil_square_length += 1
        if have_white:
            have_white = False
            lil_square_length_y = 0
            for y_lil in range(y, y + square_length//2 - 1):
                if (img[y_lil, x] > [50, 50, 50]).all():
                    have_white = True
                    break
                lil_square_length_y += 1
            if have_white and (lil_square_length_y in range(lil_square_length - 3, lil_square_length + 3)):
                if self._is_black_point(img, y + lil_square_length, x + lil_square_length, 3):
                    return True
        return False

#первый аргумент этой функции - наименование вашего изображения в одной папке с исполняемым файлом
#img = cv.imread('newqr2.png', cv.IMREAD_COLOR)
qr_handler = QrHandler()
result = qr_handler.detect("1.png")
print(result)
#drawqrborders(bbox, img)

#cv.imshow(winname='test', mat=img)
#cv.waitKey(0)

