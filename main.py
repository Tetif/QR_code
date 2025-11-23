# === FILE: main.py (Использует встроенный детектор OpenCV) ===
import cv2
import numpy as np
import sys


def run_image(path: str, debug_out: str = 'debug.png'):
    """
    Обнаруживает и декодирует QR-код с изображения используя встроенный детектор OpenCV.
    """
    img = cv2.imread(path)
    if img is None:
        print('Failed to open', path)
        return

    print("Initializing QR code detector...")
    detector = cv2.QRCodeDetector()
    
    print("Detecting and decoding QR code...")
    data, bbox, straight_qrcode = detector.detectAndDecode(img)
    
    # Создаем копию изображения для отрисовки
    dbg = img.copy()
    
    if bbox is not None:
        print(f"✅ QR Code detected!")
        
        # Рисуем границы QR-кода
        n_points = len(bbox[0])
        print(f"Bounding box has {n_points} points")
        
        for i in range(n_points):
            # Преобразуем точки в целые числа
            point1 = (int(bbox[0][i][0]), int(bbox[0][i][1]))
            point2 = (int(bbox[0][(i + 1) % n_points][0]), int(bbox[0][(i + 1) % n_points][1]))
            cv2.line(dbg, point1, point2, color=(0, 255, 0), thickness=3)
        
        # Подписываем углы
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        corner_colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
        
        for i in range(min(n_points, 4)):
            point = (int(bbox[0][i][0]), int(bbox[0][i][1]))
            cv2.circle(dbg, point, 10, corner_colors[i], -1)
            cv2.putText(dbg, f"{i}:{corner_labels[i]}", (point[0] + 15, point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, corner_colors[i], 2)
        
        if data:
            print('✅ Decoded:', data)
            
            # Добавляем текст на изображение
            text_position = (10, 30)
            cv2.putText(dbg, f"Data: {data}", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print('⚠️ QR Code detected but failed to decode')
            
        # Сохраняем выровненный QR-код если он есть
        if straight_qrcode is not None and straight_qrcode.size > 0:
            print(f"Straight QR code shape: {straight_qrcode.shape}")
            cv2.imwrite('warped.png', straight_qrcode)
        else:
            print("No straight QR code available")
            
    else:
        print('❌ No QR code detected')
        # Попробуем с предобработкой изображения
        print("Trying with image preprocessing...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Применяем несколько методов предобработки
        # Метод 1: Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        data, bbox, straight_qrcode = detector.detectAndDecode(binary)
        
        if bbox is None:
            # Метод 2: Otsu бинаризация
            _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            data, bbox, straight_qrcode = detector.detectAndDecode(binary2)
        
        if bbox is not None:
            print("✅ QR Code detected after preprocessing!")
            if data:
                print('✅ Decoded:', data)
            else:
                print('⚠️ QR Code detected but failed to decode')
        else:
            print('❌ Still no QR code detected after preprocessing')
    
    # Сохраняем результат
    cv2.imwrite(debug_out, dbg)
    print(f"Debug image saved to {debug_out}")
    
    return data


def run_image_custom(path: str, debug_out: str = 'debug_custom.png'):
    """
    Обнаруживает и декодирует QR-код с изображения используя кастомную реализацию из src/.
    """
    try:
        from src.detector import preprocess, find_finder_patterns, find_qr_corners, perspective_transform, create_clean_qr_image
        from src.sampler import sample_modules
        from src.decoder import UniversalQRDecoder
        from src.version_info import detect_version_from_dimension, QR_VERSION_TABLE
    except ImportError as e:
        print(f"❌ Ошибка импорта модулей из src/: {e}")
        print("Убедитесь, что все зависимости установлены (reedsolo)")
        return None
    
    img = cv2.imread(path)
    if img is None:
        print('Failed to open', path)
        return None

    print("=" * 60)
    print("Используется кастомная реализация из src/")
    print("=" * 60)
    
    # Предобработка
    print("[1/7] Предобработка изображения...")
    bin_img, gray = preprocess(img)
    cv2.imwrite('debug_01_preprocessed_binary.png', bin_img)
    cv2.imwrite('debug_01_preprocessed_gray.png', gray)
    print("  ✅ Сохранено: debug_01_preprocessed_binary.png, debug_01_preprocessed_gray.png")
    
    # Поиск finder patterns
    print("[2/7] Поиск finder patterns...")
    finders = find_finder_patterns(bin_img)
    
    if len(finders) < 3:
        print(f"❌ Найдено только {len(finders)} finder patterns (нужно минимум 3)")
        print("Попытка использовать OpenCV детектор как fallback...")
        return run_image(path, debug_out)
    
    print(f"✅ Найдено {len(finders)} finder patterns")
    
    # Визуализация найденных finder patterns
    dbg_finders = img.copy()
    for i, finder in enumerate(finders):
        cv2.drawContours(dbg_finders, [finder], -1, (0, 255, 0), 3)
        center = cv2.moments(finder)
        if center["m00"] != 0:
            cx = int(center["m10"] / center["m00"])
            cy = int(center["m01"] / center["m00"])
            cv2.circle(dbg_finders, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(dbg_finders, f"FP{i+1}", (cx + 10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imwrite('debug_02_finder_patterns.png', dbg_finders)
    print("  ✅ Сохранено: debug_02_finder_patterns.png")
    
    # Нахождение углов
    print("[3/7] Нахождение углов QR-кода...")
    try:
        corners = find_qr_corners(finders, img.shape[:2], bin_img)
        print(f"✅ Углы найдены: {corners.shape}")
        print(f"  TL: ({corners[0][0]:.1f}, {corners[0][1]:.1f})")
        print(f"  TR: ({corners[1][0]:.1f}, {corners[1][1]:.1f})")
        print(f"  BR: ({corners[2][0]:.1f}, {corners[2][1]:.1f})")
        print(f"  BL: ({corners[3][0]:.1f}, {corners[3][1]:.1f})")
    except Exception as e:
        print(f"❌ Ошибка при нахождении углов: {e}")
        return None
    
    # Визуализация углов
    dbg_corners = img.copy()
    corner_labels = ['TL', 'TR', 'BR', 'BL']
    corner_colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
    for i, corner in enumerate(corners):
        pt = (int(corner[0]), int(corner[1]))
        cv2.circle(dbg_corners, pt, 15, corner_colors[i], -1)
        cv2.putText(dbg_corners, f"{i}:{corner_labels[i]}", (pt[0] + 20, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, corner_colors[i], 2)
    # Рисуем линии между углами
    for i in range(4):
        pt1 = (int(corners[i][0]), int(corners[i][1]))
        pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
        cv2.line(dbg_corners, pt1, pt2, (0, 255, 0), 3)
    cv2.imwrite('debug_03_corners.png', dbg_corners)
    print("  ✅ Сохранено: debug_03_corners.png")
    
    # Выравнивание
    print("[4/7] Выравнивание QR-кода...")
    warped_size = 210  # Базовый размер для выравнивания
    warped = perspective_transform(gray, corners, output_size=warped_size)
    cv2.imwrite('debug_04_warped_gray.png', warped)
    print(f"  ✅ Выровненное изображение сохранено: debug_04_warped_gray.png")
    
    # Определение версии и создание четкого QR-кода
    print("[5/7] Определение версии и создание четкого QR-кода...")
    decoder = UniversalQRDecoder()
    
    # Пробуем разные версии QR-кодов
    results = []
    for version, info in QR_VERSION_TABLE.items():
        dimension = info['dimension']
        print(f"  Пробуем версию {version} (dimension={dimension})...")
        
        # Создаем четкий бинаризованный QR-код
        clean_qr = create_clean_qr_image(warped, dimension, output_size=dimension * 10)
        cv2.imwrite(f'debug_05_clean_qr_v{version}_{dimension}x{dimension}.png', clean_qr)
        print(f"    ✅ Сохранено: debug_05_clean_qr_v{version}_{dimension}x{dimension}.png")
        
        # Семплирование модулей из четкого QR-кода
        # Преобразуем clean_qr обратно в матрицу модулей
        modules = np.zeros((dimension, dimension), dtype=np.uint8)
        pixel_per_module = clean_qr.shape[0] // dimension
        for r in range(dimension):
            for c in range(dimension):
                y0 = r * pixel_per_module
                y1 = (r + 1) * pixel_per_module
                x0 = c * pixel_per_module
                x1 = (c + 1) * pixel_per_module
                # Majority vote в области модуля
                module_patch = clean_qr[y0:y1, x0:x1]
                black_pixels = np.sum(module_patch == 0)
                modules[r, c] = 0 if black_pixels > module_patch.size / 2 else 1
        
        # Альтернативно: используем sampler для более точного семплирования
        modules_sampled = sample_modules(warped, dimension)
        
        # Декодирование с обоими вариантами
        success1, text1, metadata1 = decoder.decode(modules)
        success2, text2, metadata2 = decoder.decode(modules_sampled)
        
        if success1:
            print(f"    ✅ Успешно декодировано (clean_qr) версией {version}!")
            results.append((version, text1, metadata1, modules, 'clean'))
        elif success2:
            print(f"    ✅ Успешно декодировано (sampled) версией {version}!")
            results.append((version, text2, metadata2, modules_sampled, 'sampled'))
        else:
            print(f"    ❌ Не удалось декодировать версией {version}")
    
    if not results:
        print("❌ Не удалось декодировать ни одной версией")
        return None
    
    # Выбираем лучший результат
    best_version, best_text, best_metadata, best_modules, method = results[0]
    
    print(f"[6/7] Результат декодирования:")
    print(f"  Версия: {best_version}")
    print(f"  Тип: {best_metadata.get('type', 'QR')}")
    print(f"  Уровень коррекции: {best_metadata.get('ec_level', '?')}")
    print(f"  Маска: {best_metadata.get('mask_pattern', '?')}")
    print(f"  Метод: {method}")
    print(f"  ✅ Декодированный текст: {best_text}")
    
    # Визуализация финального результата
    print("[7/7] Создание финальной визуализации...")
    dbg = img.copy()
    
    # Рисуем найденные finder patterns
    for finder in finders:
        cv2.drawContours(dbg, [finder], -1, (0, 255, 0), 2)
    
    # Рисуем углы
    for i, corner in enumerate(corners):
        pt = (int(corner[0]), int(corner[1]))
        cv2.circle(dbg, pt, 10, corner_colors[i], -1)
        cv2.putText(dbg, f"{i}:{corner_labels[i]}", (pt[0] + 15, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, corner_colors[i], 2)
    
    # Рисуем линии между углами
    for i in range(4):
        pt1 = (int(corners[i][0]), int(corners[i][1]))
        pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
        cv2.line(dbg, pt1, pt2, (0, 255, 0), 3)
    
    # Добавляем текст
    text_position = (10, 30)
    display_text = best_text[:40] + "..." if len(best_text) > 40 else best_text
    cv2.putText(dbg, f"Custom Decoder - Ver:{best_version} - {display_text}", text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite(debug_out, dbg)
    print(f"  ✅ Финальное отладочное изображение сохранено: {debug_out}")
    
    # Сохраняем финальную матрицу модулей как изображение
    final_qr_img = np.zeros((best_modules.shape[0] * 10, best_modules.shape[1] * 10), dtype=np.uint8)
    for r in range(best_modules.shape[0]):
        for c in range(best_modules.shape[1]):
            val = 0 if best_modules[r, c] == 0 else 255
            final_qr_img[r*10:(r+1)*10, c*10:(c+1)*10] = val
    cv2.imwrite('debug_06_final_modules.png', final_qr_img)
    print(f"  ✅ Финальная матрица модулей сохранена: debug_06_final_modules.png")
    
    return best_text


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # python -m main images/qr_image.png
        # python -m main images/qr_image_copy.png
        # python -m main images/rickroll_qr_code.png
        # python -m main amir/qrr.jpg
        # python -m main amir/myqrrr.png
        # python -m main amir/2koda_test.png
        # python -m main amir/pmqr.png
        print('Usage: python -m main path/to/image [--custom]')
        print('  --custom  : использовать кастомную реализацию из src/')
        print('  (по умолчанию используется OpenCV детектор)')
        sys.exit(1)
    
    image_path = sys.argv[1]
    use_custom = '--custom' in sys.argv or '-c' in sys.argv
    
    if use_custom:
        run_image_custom(image_path)
    else:
        run_image(image_path)