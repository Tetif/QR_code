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
    
    Использует внешние углы finder patterns для точного определения границ QR-кода.
    """
    if len(finders) < 3:
        raise ValueError("Need at least 3 finder patterns")

    # Вычисляем центры и размеры всех найденных меток
    # Пытаемся найти дочерние контуры (внутренние черные квадраты) для более точного определения углов
    finder_info = []
    
    # Если есть доступ к bin_img, ищем дочерние контуры
    inner_contours = {}
    if bin_img is not None:
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i, finder in enumerate(finders):
                # Находим индекс этого контура в списке всех контуров
                for j, contour in enumerate(contours):
                    if cv2.matchShapes(finder, contour, cv2.CONTOURS_MATCH_I2, 0) < 0.1:
                        # Нашли соответствующий контур, проверяем дочерний
                        if hierarchy[0][j][2] != -1:
                            child_idx = hierarchy[0][j][2]
                            if child_idx != -1 and child_idx < len(contours):
                                inner_contours[i] = contours[child_idx]
                                break
    
    for i, finder in enumerate(finders):
        x, y, w, h = cv2.boundingRect(finder)
        center = calculate_center(finder)
        
        # Используем внутренний контур (дочерний), если он есть, иначе внешний
        contour_to_use = inner_contours.get(i, finder)
        
        finder_info.append({
            'contour': finder,  # Внешний контур (для справки)
            'inner_contour': contour_to_use,  # Внутренний контур (черный квадрат)
            'center': center,
            'bbox': (x, y, w, h),
            'size': max(w, h)  # Размер finder pattern
        })

    # Находим три самые удаленные метки (вероятные углы QR-кода)
    if len(finders) == 3:
        selected_finders = finder_info
    else:
        # Выбираем три самые удаленные друг от друга
        centers = [f['center'] for f in finder_info]
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                distances.append((i, j, dist))

        distances.sort(key=lambda x: x[2], reverse=True)
        selected_indices = set()
        for i, j, dist in distances[:3]:
            selected_indices.add(i)
            selected_indices.add(j)
            if len(selected_indices) >= 3:
                break

        selected_finders = [finder_info[i] for i in list(selected_indices)[:3]]

    # Для каждой finder pattern находим внешний угол QR-кода
    # Важно: используем края черного квадрата, а не белую область вокруг
    corners_list = []
    for finder in selected_finders:
        # Используем внутренний контур (черный квадрат), если он есть
        # Это даст нам точные края черного квадрата без белой области
        inner_contour = finder.get('inner_contour', finder['contour'])
        outer_contour = finder['contour']
        
        x, y, w, h = finder['bbox']
        center = finder['center']
        
        # Предпочитаем использовать внутренний контур (дочерний)
        # Если внутренний контур есть и достаточно большой, используем его
        if 'inner_contour' in finder and inner_contour is not outer_contour:
            inner_area = cv2.contourArea(inner_contour)
            outer_area = cv2.contourArea(outer_contour)
            # Используем внутренний контур, если его площадь составляет хотя бы 30% от внешнего
            if inner_area > outer_area * 0.3:
                contour_to_use = inner_contour
            else:
                contour_to_use = outer_contour
        else:
            # Если внутреннего контура нет, используем внешний, но сдвигаем внутрь
            contour_to_use = outer_contour
        
        contour_points = contour_to_use.reshape(-1, 2).astype(float)
        
        # Находим угловые точки, но используем ближайшие к центру в каждом направлении
        # Это даст нам края черного квадрата, а не белой области
        tl_point = None
        tr_point = None
        bl_point = None
        br_point = None
        
        min_dist_tl = float('inf')
        min_dist_tr = float('inf')
        min_dist_bl = float('inf')
        min_dist_br = float('inf')
        
        # Находим точки контура в каждом квадранте, которые ближе всего к центру
        # Это будут края черного квадрата (внутренние края контура)
        relative_points = contour_points - center
        
        for pt, rel_pt in zip(contour_points, relative_points):
            dist = np.linalg.norm(rel_pt)
            # Квадрант определяется знаками x и y относительно центра
            if rel_pt[0] <= 0 and rel_pt[1] <= 0:  # Левый верхний квадрант
                # Ищем точку, которая ближе к центру, но все еще в этом квадранте
                # Это будет внутренний край черного квадрата
                if dist < min_dist_tl and dist > 0:
                    min_dist_tl = dist
                    tl_point = pt
            elif rel_pt[0] >= 0 and rel_pt[1] <= 0:  # Правый верхний квадрант
                if dist < min_dist_tr and dist > 0:
                    min_dist_tr = dist
                    tr_point = pt
            elif rel_pt[0] <= 0 and rel_pt[1] >= 0:  # Левый нижний квадрант
                if dist < min_dist_bl and dist > 0:
                    min_dist_bl = dist
                    bl_point = pt
            else:  # Правый нижний квадрант
                if dist < min_dist_br and dist > 0:
                    min_dist_br = dist
                    br_point = pt
        
        # Если не нашли точки (маловероятно), используем альтернативный метод
        # Используем точки, которые находятся на определенном расстоянии от центра
        # (примерно 70-80% от максимального расстояния - это будет край черного квадрата)
        if tl_point is None or tr_point is None or bl_point is None or br_point is None:
            # Находим максимальное расстояние от центра
            max_dist = max([np.linalg.norm(pt - center) for pt in contour_points])
            target_dist = max_dist * 0.75  # 75% от максимального - край черного квадрата
            
            # Ищем точки, которые ближе всего к target_dist в каждом квадранте
            for pt, rel_pt in zip(contour_points, relative_points):
                dist = np.linalg.norm(rel_pt)
                diff = abs(dist - target_dist)
                
                if rel_pt[0] <= 0 and rel_pt[1] <= 0:  # TL
                    if tl_point is None or abs(np.linalg.norm(tl_point - center) - target_dist) > diff:
                        tl_point = pt
                elif rel_pt[0] >= 0 and rel_pt[1] <= 0:  # TR
                    if tr_point is None or abs(np.linalg.norm(tr_point - center) - target_dist) > diff:
                        tr_point = pt
                elif rel_pt[0] <= 0 and rel_pt[1] >= 0:  # BL
                    if bl_point is None or abs(np.linalg.norm(bl_point - center) - target_dist) > diff:
                        bl_point = pt
                else:  # BR
                    if br_point is None or abs(np.linalg.norm(br_point - center) - target_dist) > diff:
                        br_point = pt
        
        # Финальный fallback: используем углы bounding rect, но сдвинутые внутрь
        # Сдвигаем на 10-15% внутрь, чтобы исключить белую область
        shrink_factor = 0.12  # Сдвигаем на 12% внутрь
        if tl_point is None:
            tl_point = np.array([x + w * shrink_factor, y + h * shrink_factor], dtype='float32')
        else:
            tl_point = tl_point.astype('float32')
        if tr_point is None:
            tr_point = np.array([x + w * (1 - shrink_factor), y + h * shrink_factor], dtype='float32')
        else:
            tr_point = tr_point.astype('float32')
        if bl_point is None:
            bl_point = np.array([x + w * shrink_factor, y + h * (1 - shrink_factor)], dtype='float32')
        else:
            bl_point = bl_point.astype('float32')
        if br_point is None:
            br_point = np.array([x + w * (1 - shrink_factor), y + h * (1 - shrink_factor)], dtype='float32')
        else:
            br_point = br_point.astype('float32')
        
        corners_list.append({
            'tl': tl_point,
            'tr': tr_point,
            'bl': bl_point,
            'br': br_point,
            'center': center
        })

    # Улучшенное определение позиции каждой finder pattern
    centers_array = np.array([f['center'] for f in corners_list], dtype='float32')
    
    # Используем более точный алгоритм определения углов
    # 1. Находим центр всех finder patterns
    center_all = np.mean(centers_array, axis=0)
    
    # 2. Определяем углы относительно центра
    angles = []
    for i, center in enumerate(centers_array):
        vec = center - center_all
        # Вычисляем угол от центра (atan2 дает угол от -pi до pi)
        angle = np.arctan2(vec[1], vec[0])
        angles.append((i, angle, center))
    
    # 3. Сортируем по углу (от -pi до pi)
    angles.sort(key=lambda x: x[1])
    
    # 4. Определяем позиции: TL (верхний левый), TR (верхний правый), BL (нижний левый)
    # Углы должны быть примерно: TL ~ -135°, TR ~ -45°, BL ~ 135°
    # Но из-за поворота QR-кода углы могут быть любыми
    # Используем сумму координат для определения TL (минимальная)
    sum_coords = centers_array.sum(axis=1)
    tl_idx = np.argmin(sum_coords)
    
    # Остальные два - по углу относительно TL
    remaining = [i for i in range(len(centers_array)) if i != tl_idx]
    if len(remaining) == 2:
        # Вычисляем углы относительно TL
        tl_center = centers_array[tl_idx]
        vec1 = centers_array[remaining[0]] - tl_center
        vec2 = centers_array[remaining[1]] - tl_center
        
        # Определяем, какой из них TR (более горизонтальный) и BL (более вертикальный)
        angle1 = np.arctan2(vec1[1], vec1[0])
        angle2 = np.arctan2(vec2[1], vec2[0])
        
        # TR должен быть справа от TL (угол ближе к 0°)
        # BL должен быть снизу от TL (угол ближе к 90°)
        if abs(angle1) < abs(angle2):
            tr_idx = remaining[0]
            bl_idx = remaining[1]
        else:
            tr_idx = remaining[1]
            bl_idx = remaining[0]
    else:
        # Fallback на старый метод
        diff_coords = np.diff(centers_array, axis=1)
        tr_idx = np.argmin(diff_coords[:, 0]) if len(diff_coords) > 0 else 1
        bl_idx = np.argmax(diff_coords[:, 0]) if len(diff_coords) > 0 else 2
    
    # Получаем углы finder patterns
    tl_finder = corners_list[tl_idx]
    tr_finder = corners_list[tr_idx]
    bl_finder = corners_list[bl_idx]
    
    # Используем внешние углы finder patterns
    # Для TL finder pattern используем TL угол
    # Для TR finder pattern используем TR угол
    # Для BL finder pattern используем BL угол
    tl = tl_finder['tl']
    tr = tr_finder['tr']
    bl = bl_finder['bl']
    
    # Вычисляем четвертый угол (BR) более точно
    # Используем параллелограмм: BR = TL + (TR - TL) + (BL - TL)
    # Или: BR = TR + (BL - TL)
    br = tr + (bl - tl)
    
    # Дополнительная проверка: если BR слишком далеко, используем альтернативный метод
    # Вычисляем расстояние от TL до TR и от TL до BL
    dist_tr = np.linalg.norm(tr - tl)
    dist_bl = np.linalg.norm(bl - tl)
    
    # Если BR слишком близко к другим углам, пересчитываем
    dist_br_tr = np.linalg.norm(br - tr)
    dist_br_bl = np.linalg.norm(br - bl)
    
    # Если расстояния не соответствуют ожидаемым, используем альтернативный расчет
    if dist_br_tr < dist_bl * 0.5 or dist_br_bl < dist_tr * 0.5:
        # Альтернативный метод: используем геометрию прямоугольника
        # Находим вектор от TL к TR и от TL к BL
        vec_horizontal = tr - tl
        vec_vertical = bl - tl
        # BR = TL + vec_horizontal + vec_vertical
        br = tl + vec_horizontal + vec_vertical

    # Расширяем углы наружу, чтобы захватить весь QR-код
    # Вычисляем средний размер finder patterns и расстояние между ними
    avg_size = np.mean([f['size'] for f in selected_finders])
    
    # Вычисляем расстояния между finder patterns для оценки размера QR-кода
    distances_between = []
    for i in range(len(centers_array)):
        for j in range(i + 1, len(centers_array)):
            dist = np.linalg.norm(centers_array[i] - centers_array[j])
            distances_between.append(dist)
    avg_distance = np.mean(distances_between) if distances_between else avg_size * 10
    max_distance = max(distances_between) if distances_between else avg_size * 10
    
    # Расширяем на основе размера finder pattern и расстояния между ними
    # Finder pattern занимает примерно 7x7 модулей, а весь QR-код может быть 21-40 модулей
    # Расширяем достаточно сильно, чтобы захватить весь QR-код
    # Используем максимальное расстояние между finder patterns как ориентир
    expansion_factor = 0.15  # 15% от максимального расстояния
    expansion = max(avg_size * 0.8, max_distance * expansion_factor)  # Минимум 80% размера finder pattern

    # Расширяем каждый угол от центра QR-кода
    center = np.mean([tl, tr, bl, br], axis=0)
    
    def expand_point(pt, center, expansion):
        """Расширяет точку от центра на заданное расстояние"""
        direction = pt - center
        dist = np.linalg.norm(direction)
        if dist > 0:
            return pt + direction * (expansion / dist)
        return pt
    
    tl_expanded = expand_point(tl, center, expansion)
    tr_expanded = expand_point(tr, center, expansion)
    bl_expanded = expand_point(bl, center, expansion)
    br_expanded = expand_point(br, center, expansion)
    
    # Дополнительно: убеждаемся, что углы не выходят за границы изображения
    if img_shape is not None:
        h, w = img_shape
        tl_expanded = np.clip(tl_expanded, [0, 0], [w-1, h-1])
        tr_expanded = np.clip(tr_expanded, [0, 0], [w-1, h-1])
        bl_expanded = np.clip(bl_expanded, [0, 0], [w-1, h-1])
        br_expanded = np.clip(br_expanded, [0, 0], [w-1, h-1])

    return np.array([tl_expanded, tr_expanded, br_expanded, bl_expanded], dtype='float32')


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
        output_size = dimension * 10  # 10 пикселей на модуль по умолчанию
    
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
