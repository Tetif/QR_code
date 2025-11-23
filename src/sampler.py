# === FILE: qr_reader/sampler.py (УЛУЧШЕННАЯ ВЕРСИЯ) ===
import numpy as np
import cv2

def sample_modules(warped_gray: np.ndarray, dimension: int, module_pixels: int = 10) -> np.ndarray:
    """
    Улучшенный sampler:
    - применяет CLAHE при низком контрасте,
    - строит бинаризацию adaptive + Otsu,
    - применяет морфологическую операцию для заполнения мелких разрывов,
    - внутри каждого патча комбинирует статистику бинарного патча и среднее серого,
      чтобы надёжнее отличать 'полутоновые' чёрные от белых.
    """
    dst_size = warped_gray.shape[0]
    module_size = dst_size / dimension

    # 1) усиление контраста при необходимости
    if np.std(warped_gray) < 40:
        warped_enhanced = cv2.createCLAHE(clipLimit=2.0).apply(warped_gray)
    else:
        warped_enhanced = warped_gray

    # 2) две бинаризации: adaptive + otsu (на усилённом)
    binary_adaptive = cv2.adaptiveThreshold(
        warped_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    _, binary_otsu = cv2.threshold(warped_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) небольшое морфологическое закрытие, чтобы заполнить тонкие разрывы
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)

    modules = np.zeros((dimension, dimension), dtype=np.uint8)

    # Параметры принятия решения
    sample_ratio = 0.6           # центральная доля модуля, используется для устойчивости
    low_std_threshold = 12.0     # если std в патче меньше, патч "размытый" — полагаемся на среднее
    bias = -6                    # смещение порога (в градусах интенсивности) для более "чёрного" решения;
                                 # отрицательное -> склоняемся к тому, чтобы считать темные пиксели черными

    # Глобальный Otsu порог (int)
    global_otsu = int(np.mean(binary_otsu == 0) * 0 + 127)  # placeholder, но мы используем локальный порог ниже
    # на самом деле значение Otsu можно взять так:
    otsu_thresh_val, _ = cv2.threshold(warped_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh_val = int(otsu_thresh_val)

    for r in range(dimension):
        for c in range(dimension):
            sample_size = max(1, int(module_size * sample_ratio))

            # округлённые координаты центра для стабильности
            cx = int(round((c + 0.5) * module_size))
            cy = int(round((r + 0.5) * module_size))

            y0 = max(0, cy - sample_size // 2)
            y1 = min(dst_size, cy + sample_size // 2 + 1)
            x0 = max(0, cx - sample_size // 2)
            x1 = min(dst_size, cx + sample_size // 2 + 1)

            # Патчи из бинарных карт и из усилённого серого
            patch_bin = binary_adaptive[y0:y1, x0:x1]
            patch_otsu = binary_otsu[y0:y1, x0:x1]
            patch_gray = warped_enhanced[y0:y1, x0:x1]

            if patch_gray.size == 0:
                val = 255
            else:
                # статистики
                black_pixels_bin = int(np.sum(patch_bin == 0))
                white_pixels_bin = int(np.sum(patch_bin == 255))
                total = black_pixels_bin + white_pixels_bin

                mean_gray = float(np.mean(patch_gray))
                std_gray = float(np.std(patch_gray))

                # если патч контрастный — полагаемся на majority бинарного патча
                if total > 0 and std_gray >= low_std_threshold:
                    # усилить роль чёрных: если ratio_black >= 0.45 считаем чёрным
                    if black_pixels_bin / total >= 0.45:
                        val = 0
                    else:
                        val = 255
                else:
                    # "размытый" патч — полагаемся на среднее значение серого и Otsu порог с bias
                    # Чем ниже mean_gray относительно otsu, тем вероятнее это чёрный модуль
                    thr = otsu_thresh_val + bias
                    val = 0 if mean_gray < thr else 255

            # Сохраняем модуль: 1 = черный, 0 = белый
            modules[r, c] = 1 if val == 0 else 0

    return modules
