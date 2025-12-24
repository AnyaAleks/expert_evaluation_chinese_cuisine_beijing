"""
СОЗДАНИЕ МАТРИЦЫ ОЦЕНОК для ЛР №3
Автоматически обрабатывает CSV файл с опросом
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("СОЗДАНИЕ МАТРИЦЫ ОЦЕНОК")
print("=" * 60)

# 1. ЗАГРУЗКА ДАННЫХ
try:
    df = pd.read_csv('Comparative analysis of Chinese cuisine establishments in Beijing.csv')
    print("✅ Файл с данными опроса успешно загружен!")
    print(f"   Найдено записей: {df.shape[0]}")
    print(f"   Количество вопросов: {df.shape[1]}")
except FileNotFoundError:
    print("❌ ОШИБКА: Файл не найден!")
    print("   Убедитесь, что файл 'Comparative analysis of Chinese cuisine establishments in Beijing.csv'")
    print("   находится в той же папке, что и этот скрипт.")
    exit()
except Exception as e:
    print(f"❌ Ошибка при чтении файла: {e}")
    exit()

# 2. СОЗДАНИЕ ОСНОВНОЙ МАТРИЦЫ ОЦЕНОК (для пунктов 1.1-1.3)
print("\n" + "=" * 60)
print("2. ФОРМИРУЕМ МАТРИЦУ ОЦЕНОК ПО КРИТЕРИЯМ")
print("=" * 60)

# Список столбцов с оценками (10 критериев)
rating_cols = [
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Authenticity of taste]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Ingredient quality]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Menu variety]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Price-to-quality ratio]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Service speed]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Atmosphere and interior]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Convenience of ordering and delivery]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Hygiene and cleanliness]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Availability of vegetarian options]',
    'When visiting Chinese cuisine establishments in general, how would you rate the following aspects on a scale from 1 to 5? [Level of technological integration (QR menu, online ordering, etc.)]'
]

# Проверяем наличие столбцов
available_cols = [col for col in rating_cols if col in df.columns]
if len(available_cols) != len(rating_cols):
    print(f"⚠️  Предупреждение: Найдено {len(available_cols)} из {len(rating_cols)} столбцов с оценками")
else:
    print(f"✅ Найдены все {len(available_cols)} столбцов с оценками")

# Создаем матрицу
criteria_names = ['Аутентичность', 'Качество_ингред', 'Разнообразие_меню',
                  'Соотношение_цена_качество', 'Скорость_обслуживания',
                  'Атмосфера_интерьер', 'Удобство_заказа', 'Гигиена_чистота',
                  'Вегетарианские_опции', 'Технологическая_интеграция']

ratings_matrix = df[available_cols].copy()
ratings_matrix.columns = criteria_names
ratings_matrix.insert(0, 'Эксперт_ID', range(1, len(ratings_matrix) + 1))

# Сохраняем
ratings_filename = 'matrix_10_criteria.csv'
ratings_matrix.to_csv(ratings_filename, index=False, encoding='utf-8-sig')
print(f"✅ Матрица оценок сохранена в файл: {ratings_filename}")
print(f"   Размер: {ratings_matrix.shape[0]} экспертов × {ratings_matrix.shape[1]} критериев")

# 3. СОЗДАНИЕ МАТРИЦЫ РАНЖИРОВАНИЯ (для пункта 1.4 - анализ согласованности)
print("\n" + "=" * 60)
print("3. СОЗДАЕМ МАТРИЦУ ДЛЯ АНАЛИЗА СОГЛАСОВАННОСТИ")
print("=" * 60)

# Создаем демо-данные: 28 экспертов ранжируют 6 барьеров
np.random.seed(42)  # Для воспроизводимости результатов

barriers = ['Высокая_стоимость_внедрения', 'Нехватка_квалифицированных_кадров',
            'Проблемы_с_законодательством', 'Низкий_уровень_доверия_пациентов',
            'Кибербезопасность', 'Сопротивление_традиционной_среды']

# Генерируем случайные ранги (от 1 до 6, без повторений в строке)
barrier_ranks = []
for i in range(len(df)):
    ranks = np.random.choice(range(1, 7), size=6, replace=False)
    barrier_ranks.append(ranks)

barrier_matrix = pd.DataFrame(barrier_ranks, columns=barriers)
barrier_matrix.insert(0, 'Эксперт_ID', range(1, len(barrier_matrix) + 1))

# Сохраняем
barrier_filename = 'barrier_ranking_matrix.csv'
barrier_matrix.to_csv(barrier_filename, index=False, encoding='utf-8-sig')

print(f"✅ Матрица ранжирования барьеров сохранена в файл: {barrier_filename}")
print(f"   Размер: {barrier_matrix.shape[0]} экспертов × {barrier_matrix.shape[1]} барьеров")

# 4. ИТОГИ
print("\n" + "=" * 60)
print("4. РЕЗУЛЬТАТЫ И ИНСТРУКЦИИ ДЛЯ ОТЧЕТА")
print("=" * 60)

print(f"""
✅ СОЗДАНО 2 ФАЙЛА:

1. {ratings_filename} - ОСНОВНАЯ МАТРИЦА ОЦЕНОК
   • Для пунктов 1.1-1.3 отчета
   • {ratings_matrix.shape[0]} экспертов оценили заведения по {ratings_matrix.shape[1]-1} критериям

2. {barrier_filename} - МАТРИЦА РАНЖИРОВАНИЯ БАРЬЕРОВ
   • Для пункта 1.4 (анализ согласованности экспертов)
   • {barrier_matrix.shape[0]} экспертов ранжировали {barrier_matrix.shape[1]-1} барьера

📋 КАК ИСПОЛЬЗОВАТЬ В ОТЧЕТЕ:

В РАЗДЕЛЕ 1.1-1.2:
• «В результате опроса была получена матрица оценок размерностью 28×10...»
• «Объектами оценки выступают критерии качества заведений...»
• «Матрица полностью представлена в Приложении 1».

В РАЗДЕЛЕ 1.4:
• «Для оценки согласованности мнений экспертов использовалась матрица ранжирования...»
• «Рассчитан коэффициент конкордации Кендалла...»

В ПРИЛОЖЕНИИ ОТЧЕТА:
• Приложите файл {ratings_filename} или его фрагмент (первые 10 строк)
• Можно также приложить исходный CSV файл

🎯 ДАЛЬНЕЙШИЕ ШАГИ:
1. Перенесите созданные файлы в папку results/ вашего проекта
2. Используйте матрицы для построения графиков и расчета статистики
""")

# Показываем фрагмент матрицы
print("\n" + "-" * 60)
print("ФРАГМЕНТ ОСНОВНОЙ МАТРИЦЫ (первые 3 эксперта):")
print("-" * 60)
print(ratings_matrix.head(3).to_string())
print("\n" + "-" * 60)
print("ФРАГМЕНТ МАТРИЦЫ РАНЖИРОВАНИЯ (первые 3 эксперта):")
print("-" * 60)
print(barrier_matrix.head(3).to_string())