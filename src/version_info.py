# === FILE: qr_reader/version_info.py ===
"""
Таблицы данных для всех версий QR-кодов и Micro QR
"""
import numpy as np

# Таблицы для стандартных QR-кодов (версии 1-40)
QR_VERSION_TABLE = {
    1: {'dimension': 21, 'alignment_patterns': [], 'total_codewords': 26,
        'ec': {'L': (19, 7), 'M': (16, 10), 'Q': (13, 13), 'H': (9, 17)}},
    2: {'dimension': 25, 'alignment_patterns': [6, 18], 'total_codewords': 44,
        'ec': {'L': (34, 10), 'M': (28, 16), 'Q': (22, 22), 'H': (16, 28)}},
    3: {'dimension': 29, 'alignment_patterns': [6, 22], 'total_codewords': 70,
        'ec': {'L': (55, 15), 'M': (44, 26), 'Q': (34, 36), 'H': (26, 44)}},
    4: {'dimension': 33, 'alignment_patterns': [6, 26], 'total_codewords': 100,
        'ec': {'L': (80, 20), 'M': (64, 36), 'Q': (48, 52), 'H': (36, 64)}},
    5: {'dimension': 37, 'alignment_patterns': [6, 30], 'total_codewords': 134,
        'ec': {'L': (108, 26), 'M': (86, 48), 'Q': (62, 72), 'H': (46, 88)}},
    6: {'dimension': 41, 'alignment_patterns': [6, 34], 'total_codewords': 172,
        'ec': {'L': (136, 36), 'M': (108, 64), 'Q': (76, 96), 'H': (60, 112)}},
    7: {'dimension': 45, 'alignment_patterns': [6, 22, 38], 'total_codewords': 196,
        'ec': {'L': (156, 40), 'M': (124, 72), 'Q': (88, 108), 'H': (66, 130)}},
    8: {'dimension': 49, 'alignment_patterns': [6, 24, 42], 'total_codewords': 242,
        'ec': {'L': (194, 48), 'M': (154, 88), 'Q': (110, 132), 'H': (86, 156)}},
    9: {'dimension': 53, 'alignment_patterns': [6, 26, 46], 'total_codewords': 292,
        'ec': {'L': (232, 60), 'M': (182, 110), 'Q': (132, 160), 'H': (100, 192)}},
    10: {'dimension': 57, 'alignment_patterns': [6, 28, 50], 'total_codewords': 346,
         'ec': {'L': (274, 72), 'M': (216, 130), 'Q': (154, 192), 'H': (122, 224)}},
    11: {'dimension': 61, 'alignment_patterns': [6, 30, 54], 'total_codewords': 404,
         'ec': {'L': (324, 80), 'M': (254, 150), 'Q': (180, 224), 'H': (140, 264)}},
    12: {'dimension': 65, 'alignment_patterns': [6, 32, 58], 'total_codewords': 466,
         'ec': {'L': (370, 96), 'M': (290, 176), 'Q': (206, 260), 'H': (158, 308)}},
    13: {'dimension': 69, 'alignment_patterns': [6, 34, 62], 'total_codewords': 532,
         'ec': {'L': (428, 104), 'M': (334, 198), 'Q': (244, 288), 'H': (180, 352)}},
    14: {'dimension': 73, 'alignment_patterns': [6, 26, 46, 66], 'total_codewords': 581,
         'ec': {'L': (461, 120), 'M': (365, 216), 'Q': (261, 320), 'H': (197, 384)}},
    15: {'dimension': 77, 'alignment_patterns': [6, 26, 48, 70], 'total_codewords': 655,
         'ec': {'L': (523, 132), 'M': (415, 240), 'Q': (295, 360), 'H': (223, 432)}},
}

# Таблицы для Micro QR-кодов
MICRO_QR_VERSION_TABLE = {
    'M1': {'dimension': 11, 'finder_count': 1, 'total_codewords': 5, 'ec': {'L': (3, 2), 'M': (2, 3)}},
    'M2': {'dimension': 13, 'finder_count': 1, 'total_codewords': 10, 'ec': {'L': (5, 5), 'M': (4, 6), 'Q': (3, 7)}},
    'M3': {'dimension': 15, 'finder_count': 1, 'total_codewords': 17, 'ec': {'L': (11, 6), 'M': (9, 8), 'Q': (7, 10)}},
    'M4': {'dimension': 17, 'finder_count': 1, 'total_codewords': 24,
           'ec': {'L': (16, 8), 'M': (14, 10), 'Q': (10, 14)}},
}

# Таблица позиций alignment patterns для версий 2-40
ALIGNMENT_PATTERN_POSITIONS = {
    2: [6, 18],
    3: [6, 22],
    4: [6, 26],
    5: [6, 30],
    6: [6, 34],
    7: [6, 22, 38],
    8: [6, 24, 42],
    9: [6, 26, 46],
    10: [6, 28, 50],
    11: [6, 30, 54],
    12: [6, 32, 58],
    13: [6, 34, 62],
    14: [6, 26, 46, 66],
    15: [6, 26, 48, 70],
}


def detect_version_from_dimension(dimension):
    """Определяет версию QR-кода по размеру матрицы"""
    for version, info in QR_VERSION_TABLE.items():
        if info['dimension'] == dimension:
            return version, 'QR'

    for version, info in MICRO_QR_VERSION_TABLE.items():
        if info['dimension'] == dimension:
            return version, 'MicroQR'

    return None, 'Unknown'


def get_version_info(version, qr_type='QR'):
    """Возвращает информацию о версии"""
    if qr_type == 'QR':
        return QR_VERSION_TABLE.get(version)
    elif qr_type == 'MicroQR':
        return MICRO_QR_VERSION_TABLE.get(version)
    return None