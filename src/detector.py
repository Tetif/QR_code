# === FILE: qr_reader/detector.py (FIXED) ===
import cv2
import numpy as np
from typing import List, Tuple


def preprocess(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Используем билатеральный фильтр для сохранения границ
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # Adaptive threshold с большим размером блока
    bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 25, 10)
    return bin_img, gray


def find_contours_candidates(bin_img: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Находит контуры-кандидаты на позиционные метки"""
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    all_quads = []

    if contours is None or len(contours) == 0:
        return candidates, all_quads

    for i, contour in enumerate(contours):
        # Аппроксимируем контур
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Проверяем, что это четырехугольник
        if len(approx) == 4:
            all_quads.append(approx)

            # Проверяем вложенность (позиционные метки обычно имеют вложенные контуры)
            if hierarchy is not None and hierarchy[0][i][2] != -1:
                child_idx = hierarchy[0][i][2]
                if child_idx != -1 and child_idx < len(contours):
                    # Проверяем, что дочерний контур тоже квадрат
                    child_contour = contours[child_idx]
                    child_epsilon = 0.02 * cv2.arcLength(child_contour, True)
                    child_approx = cv2.approxPolyDP(child_contour, child_epsilon, True)

                    if len(child_approx) == 4:
                        # Проверяем соотношение площадей
                        area_parent = cv2.contourArea(approx)
                        area_child = cv2.contourArea(child_approx)

                        if area_parent > 0 and 0.20 < (area_child / area_parent) < 0.50:
                            candidates.append(approx)

    return candidates, all_quads


def is_valid_finder_pattern(contour, bin_img) -> bool:
    """Проверяет базовые критерии для finder pattern"""
    x, y, w, h = cv2.boundingRect(contour)

    if w < 20 or h < 20:  # Слишком маленький
        return False

    # Проверяем соотношение сторон
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio > 1.5:
        return False

    # Дополнительная проверка плотности (вырезаем сильно пустые/слишком заполненные)
    mask = np.zeros(bin_img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    area = cv2.contourArea(contour)

    # Инвертируем bin_img: теперь чёрные -> 255, белые -> 0
    inv = cv2.bitwise_not(bin_img)  # эквивалент 255 - bin_img
    masked = cv2.bitwise_and(inv, mask)  # оставляем только внутри контура
    mask_area = cv2.countNonZero(masked)

    # Если почти нет черных пикселей внутри - не паттерн
    if area <= 0 or mask_area < 0.05 * area:
        return False

    return True


def find_finder_patterns(bin_img: np.ndarray) -> List[np.ndarray]:
    """Находит три позиционные метки QR-кода"""
    candidates, all_quads = find_contours_candidates(bin_img)

    # Фильтруем кандидатов
    valid_candidates = []
    for candidate in candidates:
        if is_valid_finder_pattern(candidate, bin_img):
            valid_candidates.append(candidate)

    print(f"[detector] found candidate finder patterns: {len(valid_candidates)}")

    # Если нашли >=3 кандидатов, выбираем три с наибольшей площадью (наиболее стабильный выбор)
    if len(valid_candidates) >= 3:
        valid_candidates.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        chosen = valid_candidates[:3]
        print(f"[detector] chosen top-3 areas: {[cv2.contourArea(c) for c in chosen]}")
        return chosen

    # Fallback: если не нашли по вложенности, используем все четырехугольники
    if len(valid_candidates) < 3 and len(all_quads) >= 3:
        print("Using fallback: all quadrilaterals")
        all_quads.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        return all_quads[:3]

    return valid_candidates


def order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает точки в порядке: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype='float32')

    # Сумма координат: наименьшая - TL, наибольшая - BR
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR

    # Разность координат: наименьшая - TR, наибольшая - BL
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect


def calculate_center(contour: np.ndarray) -> Tuple[float, float]:
    """Вычисляет центр контура"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        # Fallback: используем среднее координат
        cx = float(contour[:, 0, 0].mean())
        cy = float(contour[:, 0, 1].mean())
    return (cx, cy)


def find_qr_corners(finders: List[np.ndarray], img_shape: Tuple[int, int] = None, bin_img: np.ndarray = None) -> np.ndarray:
    """Находит углы QR-кода по трём позиционным меткам.
    Возвращает точки в порядке: TL, TR, BR, BL (float32).
    
    Алгоритм работает как OpenCV: использует внешние углы finder patterns и расширяет их наружу.
    """
    if len(finders) < 3:
        raise ValueError("Need at least 3 finder patterns")

    # Шаг 1: Подготовка данных о finder patterns
    finder_data = []
    for finder in finders:
        x, y, w, h = cv2.boundingRect(finder)
        center = calculate_center(finder)
        contour_points = finder.reshape(-1, 2).astype(float)

        finder_data.append({
            'contour': finder,
            'contour_points': contour_points,
            'center': center,
            'bbox': (x, y, w, h),
            'size': max(w, h)
        })

    # Шаг 2: Выбираем 3 finder patterns (если их больше)
    if len(finder_data) == 3:
        selected = finder_data
    else:
        # Выбираем три самые удаленные друг от друга
        centers = np.array([f['center'] for f in finder_data], dtype='float32')
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append((i, j, dist))

        distances.sort(key=lambda x: x[2], reverse=True)
        selected_indices = set()
        for i, j, _ in distances:
            selected_indices.add(i)
            selected_indices.add(j)
            if len(selected_indices) >= 3:
                break

        selected = [finder_data[i] for i in list(selected_indices)[:3]]

    # Шаг 3: Определяем позиции finder patterns (TL, TR, BL)
    centers = np.array([f['center'] for f in selected], dtype='float32')

    # TL: минимальная сумма координат (x + y минимальна)
    sum_coords = centers.sum(axis=1)
    tl_idx = np.argmin(sum_coords)
    tl_center = centers[tl_idx]

    # Остальные два
    remaining = [i for i in range(len(centers)) if i != tl_idx]

    if len(remaining) >= 2:
        # Вычисляем углы относительно TL
        vec1 = centers[remaining[0]] - tl_center
        vec2 = centers[remaining[1]] - tl_center

        # Более горизонтальный (меньше |dy/dx|) - это TR
        # Более вертикальный (больше |dy/dx|) - это BL
        angle1 = abs(vec1[1] / vec1[0]) if vec1[0] != 0 else float('inf')
        angle2 = abs(vec2[1] / vec2[0]) if vec2[0] != 0 else float('inf')

        if angle1 < angle2:
            tr_idx = remaining[0]
            bl_idx = remaining[1]
        else:
            tr_idx = remaining[1]
            bl_idx = remaining[0]
    else:
        tr_idx = remaining[0] if len(remaining) > 0 else 1
        bl_idx = remaining[1] if len(remaining) > 1 else 2

    # Шаг 4: Находим углы QR-кода
    # Используем точки контура finder patterns, которые находятся дальше всего от центра QR
    # в нужных направлениях, и расширяем их минимально

    qr_center = np.mean(centers, axis=0)
    avg_size = np.mean([f['size'] for f in selected])

    def find_qr_corner_from_finder(finder_info, finder_center, qr_center, direction):
        """Находит угол QR-кода из контура finder pattern в заданном направлении"""
        contour_points = finder_info['contour_points']

        # Нормализуем направление
        direction = np.array(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        # Ищем точку контура, которая находится дальше всего в нужном направлении от центра QR
        best_point = None
        best_score = -float('inf')

        for pt in contour_points:
            vec = pt - qr_center
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                continue

            vec_norm = vec / dist
            # Скалярное произведение показывает, насколько точка в нужном направлении
            score = np.dot(vec_norm, direction) * dist

            if score > best_score:
                best_score = score
                best_point = pt

        # Fallback: используем центр finder pattern
        if best_point is None:
            best_point = finder_center

        return best_point.astype('float32')

    # Находим углы из контуров finder patterns
    tl = find_qr_corner_from_finder(selected[tl_idx], centers[tl_idx], qr_center, [-1, -1])
    tr = find_qr_corner_from_finder(selected[tr_idx], centers[tr_idx], qr_center, [1, -1])
    bl = find_qr_corner_from_finder(selected[bl_idx], centers[bl_idx], qr_center, [-1, 1])

    # Шаг 5: Вычисляем четвертый угол (BR) используя геометрию параллелограмма
    vec_horizontal = tr - tl
    vec_vertical = bl - tl
    br = tl + vec_horizontal + vec_vertical

    # Шаг 6: Минимальное расширение наружу от центра QR-кода
    # Расширяем только немного, чтобы углы были точно на краю QR-кода
    # Используем небольшое расширение на основе размера finder pattern
    expansion = 0  # 25% размера finder pattern - минимальное расширение

    def expand_from_qr_center(corner, qr_center, expansion):
        """Расширяет угол от центра QR-кода наружу"""
        vec = corner - qr_center
        dist = np.linalg.norm(vec)
        if dist > 0:
            return corner + vec * (expansion / dist)
        return corner

    tl = expand_from_qr_center(tl, qr_center, expansion)
    tr = expand_from_qr_center(tr, qr_center, expansion)
    bl = expand_from_qr_center(bl, qr_center, expansion)
    br = expand_from_qr_center(br, qr_center, expansion)

    # Шаг 7: Ограничиваем углы границами изображения
    if img_shape is not None:
        h, w = img_shape
        tl = np.clip(tl, [0, 0], [w-1, h-1])
        tr = np.clip(tr, [0, 0], [w-1, h-1])
        bl = np.clip(bl, [0, 0], [w-1, h-1])
        br = np.clip(br, [0, 0], [w-1, h-1])

    return np.array([tl, tr, br, bl], dtype='float32')


def expand_corners(corners: np.ndarray, expansion: float = 0.1) -> np.ndarray:
    """Немного расширяет область обрезки чтобы захватить весь QR-код"""
    center = np.mean(corners, axis=0)

    expanded = corners.copy()
    for i in range(4):
        direction = corners[i] - center
        expanded[i] = corners[i] + direction * expansion

    return expanded

def perspective_transform(img: np.ndarray, src_pts: np.ndarray, output_size: int = 210) -> np.ndarray:
    """Выполняет перспективную трансформацию"""
    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (output_size, output_size))

    return warped


def create_clean_qr_image(warped_gray: np.ndarray, dimension: int, output_size: int = None) -> np.ndarray:
    """
    Создает четкий бинаризованный QR-код заданного размера.

    Args:
        warped_gray: Выровненное grayscale изображение QR-кода
        dimension: Размерность QR-кода в модулях (21, 25, 29, и т.д.)
        output_size: Размер выходного изображения в пикселях (по умолчанию dimension * 10)

    Returns:
        Бинаризованное изображение QR-кода (0 = черный, 255 = белый)
    """
    if output_size is None:
        output_size = dimension * 1  # 10 пикселей на модуль по умолчанию
    
    # Усиление контраста
    if np.std(warped_gray) < 40:
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(warped_gray)
    else:
        enhanced = warped_gray
    
    # Бинаризация - пробуем несколько методов и выбираем лучший
    # Метод 1: Adaptive threshold
    binary_adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Метод 2: Otsu
    _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Метод 3: Комбинированный (majority vote для каждого модуля)
    warped_size = warped_gray.shape[0]
    module_size = warped_size / dimension
    
    # Создаем финальное изображение
    clean_qr = np.zeros((dimension, dimension), dtype=np.uint8)
    
    for r in range(dimension):
        for c in range(dimension):
            # Центр модуля
            cx = int((c + 0.5) * module_size)
            cy = int((r + 0.5) * module_size)
            
            # Область для семплирования (60% размера модуля)
            sample_size = max(1, int(module_size * 0.6))
            y0 = max(0, cy - sample_size // 2)
            y1 = min(warped_size, cy + sample_size // 2 + 1)
            x0 = max(0, cx - sample_size // 2)
            x1 = min(warped_size, cx + sample_size // 2 + 1)
            
            if y0 >= y1 or x0 >= x1:
                clean_qr[r, c] = 0  # Черный по умолчанию
                continue
            
            # Берем патчи из обоих бинарных изображений
            patch_adaptive = binary_adaptive[y0:y1, x0:x1]
            patch_otsu = binary_otsu[y0:y1, x0:x1]
            patch_gray = enhanced[y0:y1, x0:x1]
            
            # Majority vote из обоих бинаризаций
            black_adaptive = np.sum(patch_adaptive == 0)
            black_otsu = np.sum(patch_otsu == 0)
            total = patch_adaptive.size
            
            # Если оба метода согласны, используем их
            if black_adaptive / total > 0.5 and black_otsu / total > 0.5:
                clean_qr[r, c] = 0  # Черный
            elif black_adaptive / total < 0.5 and black_otsu / total < 0.5:
                clean_qr[r, c] = 1  # Белый
            else:
                # Если не согласны, используем среднее значение серого
                mean_val = np.mean(patch_gray)
                otsu_thresh, _ = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                clean_qr[r, c] = 0 if mean_val < otsu_thresh else 1
    
    # Преобразуем в изображение нужного размера (0 = черный, 255 = белый)
    # Увеличиваем каждый модуль до нужного размера
    clean_image = np.zeros((output_size, output_size), dtype=np.uint8)
    pixel_per_module = output_size // dimension
    
    for r in range(dimension):
        for c in range(dimension):
            val = 0 if clean_qr[r, c] == 0 else 255
            y0 = r * pixel_per_module
            y1 = (r + 1) * pixel_per_module
            x0 = c * pixel_per_module
            x1 = (c + 1) * pixel_per_module
            clean_image[y0:y1, x0:x1] = val
    
    return clean_image
