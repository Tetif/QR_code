# === FILE: qr_reader/utils.py ===
"""
Набор вспомогательных функций для отладки.
"""
import cv2
import numpy as np


def draw_polys(img, quads, color=(0,255,0), thickness=2):
    out = img.copy()
    for q in quads:
        pts = q.reshape((-1,1,2)).astype(int)
        cv2.polylines(out, [pts], True, color, thickness)
    return out


def draw_points(img, points, color=(0,0,255), r=4):
    out = img.copy()
    for (x,y) in points:
        cv2.circle(out, (int(x),int(y)), r, color, -1)
    return out


def debug_binarization(img, bin_img, filename='binary_debug.png'):
    """Сохраняет отладочное изображение бинаризации"""
    # Показываем оригинал и бинаризованное изображение рядом
    h, w = img.shape[:2]
    debug_img = np.zeros((h, w * 2, 3), dtype=np.uint8)

    # Оригинал
    if len(img.shape) == 3:
        debug_img[:, :w] = img
    else:
        debug_img[:, :w] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Бинаризованное
    debug_img[:, w:] = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(filename, debug_img)