# === FILE: qr_reader/decoder.py ===
"""
Универсальный декодер для всех версий QR и Micro QR
"""
from typing import Tuple, List, Optional, Dict
import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from src.version_info import QR_VERSION_TABLE, ALIGNMENT_PATTERN_POSITIONS, MICRO_QR_VERSION_TABLE, detect_version_from_dimension

class UniversalQRDecoder:
    def __init__(self):
        self.supported_versions = list(QR_VERSION_TABLE.keys()) + list(MICRO_QR_VERSION_TABLE.keys())

    def decode(self, modules: np.ndarray) -> Tuple[bool, str, Dict]:
        """
        Пытается декодировать матрицу модулей
        Возвращает: (успех, текст, метаданные)
        """
        dimension = modules.shape[0]

        # Определяем тип и версию QR-кода
        version, qr_type = detect_version_from_dimension(dimension)
        metadata = {'version': version, 'type': qr_type, 'dimension': dimension}

        if version is None:
            return False, "Unknown QR version", metadata

        # Пробуем декодировать в зависимости от типа
        if qr_type == 'QR':
            return self._decode_standard_qr(modules, version, metadata)
        elif qr_type == 'MicroQR':
            return self._decode_micro_qr(modules, version, metadata)
        else:
            return False, f"Unsupported QR type: {qr_type}", metadata

    def _decode_standard_qr(self, modules: np.ndarray, version: int, metadata: Dict) -> Tuple[bool, str, Dict]:
        """Декодирование стандартного QR-кода"""
        if version not in QR_VERSION_TABLE:
            return False, f"Unsupported QR version: {version}", metadata

        info = QR_VERSION_TABLE[version]
        total_codewords = info['total_codewords']
        total_bits = total_codewords * 8

        # Читаем информацию о формате для определения уровня коррекции и маски
        format_info = self._read_format_info(modules)
        if format_info:
            metadata.update(format_info)
            ec_level = format_info['ec_level']
            mask_pattern = format_info['mask_pattern']

            # Пробуем декодировать с известными параметрами
            result = self._decode_with_parameters(modules, version, ec_level, mask_pattern, 'QR')
            if result[0]:
                return result[0], result[1], {**metadata, **result[2]}

        # Если не удалось прочитать формат, перебираем все комбинации
        for ec_level in ['L', 'M', 'Q', 'H']:
            for mask_pattern in range(8):
                result = self._decode_with_parameters(modules, version, ec_level, mask_pattern, 'QR')
                if result[0]:
                    return result[0], result[1], {**metadata, **result[2]}

        # Если все попытки не удались, пробуем с инвертированными модулями
        # (возможно, черный и белый перепутаны)
        inverted_modules = 1 - modules
        for ec_level in ['L', 'M', 'Q', 'H']:
            for mask_pattern in range(8):
                result = self._decode_with_parameters(inverted_modules, version, ec_level, mask_pattern, 'QR')
                if result[0]:
                    return result[0], result[1], {**metadata, **result[2], 'inverted': True}

        return False, "Failed to decode standard QR", metadata

    def _decode_micro_qr(self, modules: np.ndarray, version: str, metadata: Dict) -> Tuple[bool, str, Dict]:
        """Декодирование Micro QR-кода"""
        if version not in MICRO_QR_VERSION_TABLE:
            return False, f"Unsupported Micro QR version: {version}", metadata

        # Micro QR имеет другую структуру - нужно реализовать отдельную логику
        # Здесь упрощенная версия
        info = MICRO_QR_VERSION_TABLE[version]
        total_codewords = info['total_codewords']

        # Для Micro QR перебираем возможные комбинации
        for ec_level in info['ec'].keys():
            for mask_pattern in range(4):  # Micro QR имеет меньше масок
                result = self._decode_with_parameters(modules, version, ec_level, mask_pattern, 'MicroQR')
                if result[0]:
                    return result[0], result[1], {**metadata, **result[2]}

        return False, "Failed to decode Micro QR", metadata

    def _read_format_info(self, modules: np.ndarray) -> Optional[Dict]:
        """Чтение информации о формате из QR-кода"""
        # Реализация чтения format information (15 бит)
        # Для упрощения возвращаем None - в реальной реализации нужно декодировать
        return None

    def _decode_with_parameters(self, modules: np.ndarray, version, ec_level, mask_pattern, qr_type) -> Tuple[
        bool, str, Dict]:
        """Декодирование с заданными параметрами"""
        try:
            dim = modules.shape[0]

            # Снимаем маску
            unmasked = self._unmask_modules(modules, mask_pattern, qr_type)

            # Читаем биты в правильном порядке
            if qr_type == 'QR':
                bits = self._read_bits_zigzag_standard(unmasked, version)
            else:
                bits = self._read_bits_zigzag_micro(unmasked, version)

            total_codewords = None
            if qr_type == 'QR':
                total_codewords = QR_VERSION_TABLE[version]['total_codewords']
            else:
                total_codewords = MICRO_QR_VERSION_TABLE[version]['total_codewords']
            total_bits_expected = total_codewords * 8

            # Проверяем, что прочитано достаточно битов
            if len(bits) < total_bits_expected * 0.8:  # Минимум 80% от ожидаемого
                return False, "", {}

            # Преобразуем в байты
            data_bytes = self._bits_to_bytes(bits)

            # Применяем коррекцию ошибок
            corrected_data = self._apply_error_correction(data_bytes, version, ec_level, qr_type)
            if corrected_data is None:
                return False, "", {}

            # Парсим полезные данные
            if corrected_data:
                text = self._parse_payload(corrected_data, qr_type)
                if text and not text.startswith("<"):
                    return True, text, {'ec_level': ec_level, 'mask_pattern': mask_pattern}

        except Exception as e:
            # Только критичные ошибки
            pass

        return False, "", {}

    def _unmask_modules(self, modules: np.ndarray, mask_pattern: int, qr_type: str) -> np.ndarray:
        """Снятие маски с модулей"""
        # Реализация для разных типов QR
        if qr_type == 'QR':
            return self._unmask_standard_qr(modules, mask_pattern)
        else:
            return self._unmask_micro_qr(modules, mask_pattern)

    def _unmask_standard_qr(self, modules: np.ndarray, mask_pattern: int) -> np.ndarray:
        """Снятие маски для стандартного QR"""
        # Используем существующую реализацию mask_func
        out = modules.copy()
        dim = modules.shape[0]

        for r in range(dim):
            for c in range(dim):
                if self._is_function_module_standard(r, c, dim):
                    continue
                if self._mask_func_standard(mask_pattern, r, c):
                    out[r, c] = 1 - out[r, c]
        return out

    def _unmask_micro_qr(self, modules: np.ndarray, mask_pattern: int) -> np.ndarray:
        """Снятие маски для Micro QR"""
        # Micro QR использует другие mask functions
        out = modules.copy()
        dim = modules.shape[0]

        for r in range(dim):
            for c in range(dim):
                if self._is_function_module_micro(r, c, dim):
                    continue
                if self._mask_func_micro(mask_pattern, r, c):
                    out[r, c] = 1 - out[r, c]
        return out


    def _mask_func_standard(self, mask: int, i: int, j: int) -> bool:
        """Функции маски для стандартного QR"""
        # Существующая реализация
        if mask == 0: return (i + j) % 2 == 0
        if mask == 1: return i % 2 == 0
        if mask == 2: return j % 3 == 0
        if mask == 3: return (i + j) % 3 == 0
        if mask == 4: return ((i // 2) + (j // 3)) % 2 == 0
        if mask == 5: return (i * j) % 2 + (i * j) % 3 == 0
        if mask == 6: return ((i * j) % 2 + (i * j) % 3) % 2 == 0
        if mask == 7: return ((i + j) % 2 + (i * j) % 3) % 2 == 0
        return False

    def _mask_func_micro(self, mask: int, i: int, j: int) -> bool:
        """Функции маски для Micro QR"""
        # Упрощенная реализация для Micro QR
        if mask == 0: return (i // 2 + j // 3) % 2 == 0
        if mask == 1: return (i * j) % 2 + (i * j) % 3 == 0
        if mask == 2: return ((i * j) % 3 + i + j) % 2 == 0
        if mask == 3: return (i + j) % 2 == 0
        return False

    def _is_function_module_standard(self, r: int, c: int, dim: int) -> bool:
        """Определение функциональных модулей для стандартного QR (точно помечаем только необходимые области)."""
        # Finder + separator + format blocks: occupy indices 0..8 (включая 8) в трёх уголках
        if (r <= 8 and c <= 8) or (r <= 8 and c >= dim - 9) or (r >= dim - 9 and c <= 8):
            return True

        # timing patterns (row 6 and col 6, но только вне finder patterns)
        # Горизонтальный timing pattern (строка 6) - только в центральной области
        if r == 6:
            if not ((c <= 8) or (c >= dim - 9)):
                return True
        # Вертикальный timing pattern (колонка 6) - только в центральной области
        if c == 6:
            if not ((r <= 8) or (r >= dim - 9)):
                return True

        # format information: только в маленьких диапазонах у углов (не вся строка/колонка)
        # верхняя/левая области: (8, 0..8) и (0..8, 8)
        if (r == 8 and (0 <= c <= 8)) or (c == 8 and (0 <= r <= 8)):
            return True
        # верхняя правая / нижняя левая области: (8, dim-8..dim-1) and (dim-8..dim-1, 8)
        if (r == 8 and (dim - 8 <= c <= dim - 1)) or (c == 8 and (dim - 8 <= r <= dim - 1)):
            return True

        # dark module (fixed) at (8, dim-8)
        if r == 8 and c == dim - 8:
            return True

        # alignment patterns (для версий > 1) — пометим их как функциональные области (центр ±2)
        try:
            version, _ = detect_version_from_dimension(dim)
            if version and version > 1:
                positions = ALIGNMENT_PATTERN_POSITIONS.get(version, [])
                for pr in positions:
                    for pc in positions:
                        # skip finder corners (they are already handled above)
                        if (pr <= 8 and pc <= 8) or (pr <= 8 and pc >= dim - 9) or (pr >= dim - 9 and pc <= 8):
                            continue
                        if abs(r - pr) <= 2 and abs(c - pc) <= 2:
                            return True
        except Exception:
            # безопасный fallback — ничего не делаем
            pass

        return False

    def _is_function_module_micro(self, r: int, c: int, dim: int) -> bool:
        """Определение функциональных модулей для Micro QR"""
        # Micro QR имеет только один finder pattern в левом верхнем углу
        if r <= 7 and c <= 7:
            return True
        # Timing patterns
        if (dim == 11 and (r == 8 or c == 8)) or (dim > 11 and (r == 10 or c == 10)):
            return True
        return False

    def _read_bits_zigzag_standard(self, modules: np.ndarray, version: int) -> List[int]:
        """Чтение битов для стандартного QR в правильном порядке (справа налево, зигзагом).
        
        QR-коды читаются начиная с правого нижнего угла, движутся вверх парами колонок,
        затем вниз в следующей паре. Колонка 6 (timing pattern) пропускается.
        """
        dim = modules.shape[0]
        total_codewords = QR_VERSION_TABLE[version]['total_codewords']
        total_bits = total_codewords * 8

        # Подсчитываем общее количество модулей данных для отладки
        total_data_modules = 0
        for r in range(dim):
            for c in range(dim):
                if not self._is_function_module_standard(r, c, dim):
                    total_data_modules += 1

        bits = []
        col = dim - 1
        upwards = True

        # Читаем колонки справа налево, пропуская колонку 6
        while col >= 0 and len(bits) < total_bits:
            # Пропускаем timing pattern (колонка 6)
            if col == 6:
                col -= 1
                if col < 0:
                    break
                continue

            # Обрабатываем пару колонок: сначала col, потом col-1 (если не timing pattern)
            cols_to_process = [col]
            if col - 1 >= 0 and col - 1 != 6:
                cols_to_process.append(col - 1)

            # Обрабатываем каждую колонку в паре в правильном порядке
            for c in cols_to_process:
                # Определяем направление чтения
                if upwards:
                    rows = range(dim - 1, -1, -1)  # Снизу вверх
                else:
                    rows = range(0, dim)  # Сверху вниз
                
                # Читаем все модули в колонке
                for r in rows:
                    # Пропускаем функциональные модули
                    if self._is_function_module_standard(r, c, dim):
                        continue
                    bits.append(int(modules[r, c]))
                    if len(bits) >= total_bits:
                        break
                if len(bits) >= total_bits:
                    break
            
            # Меняем направление после обработки пары колонок
            upwards = not upwards
            
            # Переходим к следующей паре колонок
            if col - 1 == 6:
                # Если следующая колонка - timing pattern, пропускаем ее
                col -= 3
            else:
                col -= 2

        # Проверка на достаточное количество битов
        if len(bits) == 0:
            pass  # Без логирования

        return bits

    def _read_bits_zigzag_micro(self, modules: np.ndarray, version: str) -> List[int]:
        """Чтение битов для Micro QR"""
        # Специфичная для Micro QR логика чтения
        dim = modules.shape[0]
        total_codewords = MICRO_QR_VERSION_TABLE[version]['total_codewords']
        total_bits = total_codewords * 8

        bits = []
        # Упрощенная реализация для демонстрации
        for r in range(dim):
            for c in range(dim):
                if not self._is_function_module_micro(r, c, dim):
                    bits.append(int(modules[r, c]))
                if len(bits) >= total_bits:
                    break
            if len(bits) >= total_bits:
                break

        return bits

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Преобразование битов в байты"""
        out = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i+j]
            out.append(byte)
        return bytes(out)

    def _apply_error_correction(self, data_bytes: bytes, version, ec_level: str, qr_type: str) -> Optional[bytes]:
        """Применение коррекции ошибок Рида-Соломона с логированием"""
        try:
            if qr_type == 'QR':
                info = QR_VERSION_TABLE[version]
                data_cw, ec_cw = info['ec'][ec_level]
            else:
                info = MICRO_QR_VERSION_TABLE[version]
                data_cw, ec_cw = info['ec'][ec_level]

            total_cw = data_cw + ec_cw

            # Подготавливаем блок нужной длины (trim/zero-pad)
            if len(data_bytes) < total_cw:
                padded = data_bytes + bytes(total_cw - len(data_bytes))
            elif len(data_bytes) > total_cw:
                padded = data_bytes[:total_cw]
            else:
                padded = data_bytes

            rsc = RSCodec(ec_cw)
            # reedsolo.decode возвращает кортеж (decoded, ecc, syndromes) в зависимости от версии
            decoded = rsc.decode(padded)
            # reedsolo API может возвращать либо bytes либо tuple, пытемся извлечь "decoded bytes"
            if isinstance(decoded, tuple) or isinstance(decoded, list):
                # Обычно первым элементом — decoded bytes
                decoded_bytes = decoded[0] if len(decoded) > 0 else None
            else:
                decoded_bytes = decoded

            if decoded_bytes is None:
                return None

            # Убедимся, что длина data_cw
            result = bytes(decoded_bytes)[:data_cw]
            return result

        except (ReedSolomonError, Exception):
            return None

    def _parse_payload(self, data: bytes, qr_type: str) -> str:
        """Парсинг полезных данных"""
        if not data:
            return ""

        # Преобразуем в биты для анализа
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        if len(bits) < 4:
            return "<No data>"

        # Определяем режим кодирования
        mode_bits = bits[:4]
        mode = 0
        for bit in mode_bits:
            mode = (mode << 1) | bit

        # Парсим в зависимости от режима
        if mode == 0b0100:  # Byte mode
            return self._parse_byte_mode(bits[4:])
        elif mode == 0b0010:  # Alphanumeric mode
            return self._parse_alphanumeric_mode(bits[4:])
        elif mode == 0b0001:  # Numeric mode
            return self._parse_numeric_mode(bits[4:])
        else:
            return f"<Mode {bin(mode)} not implemented>"

    def _parse_byte_mode(self, bits: List[int]) -> str:
        """Парсинг байтового режима"""
        if len(bits) < 8:
            return "<Incomplete data>"

        # Читаем количество символов
        char_count = 0
        for i in range(8):
            char_count = (char_count << 1) | bits[i]

        # Читаем символы
        chars = []
        idx = 8
        for _ in range(char_count):
            if idx + 8 > len(bits):
                break
            val = 0
            for k in range(8):
                val = (val << 1) | bits[idx + k]
            chars.append(val)
            idx += 8

        # Пробуем декодировать
        try:
            return bytes(chars).decode('utf-8')
        except UnicodeDecodeError:
            try:
                return bytes(chars).decode('iso-8859-1')
            except:
                return bytes(chars).hex()

    def _parse_alphanumeric_mode(self, bits: List[int]) -> str:
        """Парсинг алфавитно-цифрового режима"""
        return "<Alphanumeric mode - basic implementation>"

    def _parse_numeric_mode(self, bits: List[int]) -> str:
        """Парсинг числового режима"""
        return "<Numeric mode - basic implementation>"

# Функция для обратной совместимости
def try_decode_version1(modules: np.ndarray) -> Tuple[bool, str]:
    decoder = UniversalQRDecoder()
    success, text, metadata = decoder.decode(modules)
    return success, text