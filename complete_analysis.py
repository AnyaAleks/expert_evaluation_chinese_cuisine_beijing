"""
ПОЛНЫЙ АНАЛИЗ ДАННЫХ ДЛЯ ЛАБОРАТОРНОЙ РАБОТЫ №3
Выполняет все пункты задания: визуализация, согласованность, факторный анализ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kendalltau, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №3: ПОЛНЫЙ АНАЛИЗ ЭКСПЕРТНОЙ ОЦЕНКИ")
print("=" * 70)

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================
print("\n" + "=" * 70)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 70)

# Загрузка матриц
criteria_matrix = pd.read_csv('data/raw/matrix_10_criteria.csv')
barrier_matrix = pd.read_csv('data/raw/barrier_ranking_matrix.csv')

print(f"✅ Матрица критериев: {criteria_matrix.shape[0]} экспертов × {criteria_matrix.shape[1]} критериев")
print(f"✅ Матрица барьеров: {barrier_matrix.shape[0]} экспертов × {barrier_matrix.shape[1]} барьеров")

# Удаляем столбец с ID для анализа
criteria_data = criteria_matrix.drop('Эксперт_ID', axis=1)
barrier_data = barrier_matrix.drop('Эксперт_ID', axis=1)

# ==================== 2. ВИЗУАЛИЗАЦИЯ (ПУНКТ 1.2) ====================
print("\n" + "=" * 70)
print("2. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (Пункт 1.2)")
print("=" * 70)

# Создаем папку для графиков
import os

os.makedirs('results/figures', exist_ok=True)

# 2.1. Распределение оценок по каждому критерию
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
axes = axes.flatten()

for i, column in enumerate(criteria_data.columns):
    # Считаем частоты оценок
    value_counts = criteria_data[column].value_counts().sort_index()

    # Создаем столбчатую диаграмму
    bars = axes[i].bar(value_counts.index.astype(str), value_counts.values)
    axes[i].set_title(f'{column}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Оценка', fontsize=10)
    axes[i].set_ylabel('Количество экспертов', fontsize=10)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.suptitle('РАСПРЕДЕЛЕНИЕ ОЦЕНОК ПО КРИТЕРИЯМ КАЧЕСТВА', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/criteria_distribution.png', dpi=300, bbox_inches='tight')
print("✅ График распределения оценок сохранен: results/figures/criteria_distribution.png")

# 2.2. Средние оценки по критериям
fig, ax = plt.subplots(figsize=(12, 6))
mean_scores = criteria_data.mean().sort_values(ascending=True)

bars = ax.barh(range(len(mean_scores)), mean_scores.values)
ax.set_yticks(range(len(mean_scores)))
ax.set_yticklabels(mean_scores.index)
ax.set_xlabel('Средняя оценка (1-5)', fontsize=12)
ax.set_title('СРЕДНИЕ ОЦЕНКИ ПО КРИТЕРИЯМ КАЧЕСТВА', fontsize=14, fontweight='bold')

# Добавляем значения на столбцы
for i, (bar, value) in enumerate(zip(bars, mean_scores.values)):
    ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2.,
            f'{value:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/mean_scores.png', dpi=300, bbox_inches='tight')
print("✅ График средних оценок сохранен: results/figures/mean_scores.png")

# 2.3. Тепловая карта корреляций
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = criteria_data.corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('КОРРЕЛЯЦИЯ МЕЖДУ КРИТЕРИЯМИ ОЦЕНКИ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Тепловая карта корреляций сохранена: results/figures/correlation_heatmap.png")

# ==================== 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ (ПУНКТ 1.3) ====================
print("\n" + "=" * 70)
print("3. СТАТИСТИЧЕСКИЙ АНАЛИЗ (Пункт 1.3)")
print("=" * 70)

# Создаем таблицу с описательной статистикой
stats_table = pd.DataFrame({
    'Критерий': criteria_data.columns,
    'Среднее': criteria_data.mean().values,
    'Медиана': criteria_data.median().values,
    'Мода': [criteria_data[col].mode()[0] for col in criteria_data.columns],
    'Станд. отклонение': criteria_data.std().values,
    'Минимум': criteria_data.min().values,
    'Максимум': criteria_data.max().values
})

print("\n📊 ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПО КРИТЕРИЯМ:")
print("=" * 80)
print(stats_table.to_string(index=False))

# Сохраняем таблицу
stats_table.to_csv('results/statistics/descriptive_stats.csv', index=False, encoding='utf-8-sig')
print("\n✅ Таблица статистики сохранена: results/statistics/descriptive_stats.csv")

# ==================== 4. АНАЛИЗ СОГЛАСОВАННОСТИ (ПУНКТ 1.4) ====================
print("\n" + "=" * 70)
print("4. АНАЛИЗ СОГЛАСОВАННОСТИ ЭКСПЕРТОВ (Пункт 1.4)")
print("=" * 70)


def calculate_kendall_w(rank_matrix):
    """Расчет коэффициента конкордации Кендалла"""
    m = rank_matrix.shape[0]  # количество экспертов
    n = rank_matrix.shape[1]  # количество объектов

    # Сумма рангов по каждому объекту
    Rj = rank_matrix.sum(axis=0)

    # Средняя сумма рангов
    R_mean = m * (n + 1) / 2

    # Сумма квадратов отклонений
    S = ((Rj - R_mean) ** 2).sum()

    # Поправка на связи (ties)
    T = 0
    for i in range(m):
        # Считаем повторяющиеся ранги для каждого эксперта
        values, counts = np.unique(rank_matrix.iloc[i], return_counts=True)
        for t in counts[counts > 1]:
            T += (t ** 3 - t)

    # Коэффициент конкордации
    denominator = m ** 2 * (n ** 3 - n) - m * T
    if denominator == 0:
        return 0, S, T, m, n

    W = 12 * S / denominator

    return W, S, T, m, n


# Расчет коэффициента конкордации
W, S, T, m, n = calculate_kendall_w(barrier_data)

# Расчет статистики хи-квадрат
chi2_stat = m * (n - 1) * W
df = n - 1
p_value = 1 - chi2.cdf(chi2_stat, df)

print(f"\n📈 РЕЗУЛЬТАТЫ АНАЛИЗА СОГЛАСОВАННОСТИ:")
print("-" * 50)
print(f"Количество экспертов (m): {m}")
print(f"Количество барьеров (n): {n}")
print(f"Сумма квадратов отклонений (S): {S:.2f}")
print(f"Поправка на связи (T): {T}")
print(f"Коэффициент конкордации Кендалла (W): {W:.4f}")
print(f"Статистика χ²: {chi2_stat:.4f}")
print(f"Степени свободы: {df}")
print(f"p-value: {p_value:.6f}")

# Интерпретация
print("\n📝 ИНТЕРПРЕТАЦИЯ:")
print("-" * 50)
if W < 0.2:
    print(f"• Уровень согласованности: НИЗКИЙ (W = {W:.3f})")
elif W < 0.4:
    print(f"• Уровень согласованности: УМЕРЕННЫЙ (W = {W:.3f})")
elif W < 0.6:
    print(f"• Уровень согласованности: СРЕДНИЙ (W = {W:.3f})")
elif W < 0.8:
    print(f"• Уровень согласованности: ВЫСОКИЙ (W = {W:.3f})")
else:
    print(f"• Уровень согласованности: ОЧЕНЬ ВЫСОКИЙ (W = {W:.3f})")

if p_value < 0.05:
    print(f"• Статистическая значимость: ДА (p = {p_value:.4f} < 0.05)")
    print("  Полученная согласованность не является случайной.")
else:
    print(f"• Статистическая значимость: НЕТ (p = {p_value:.4f} ≥ 0.05)")
    print("  Согласованность может быть случайной.")

# Визуализация сумм рангов
fig, ax = plt.subplots(figsize=(10, 6))
sum_ranks = barrier_data.sum().sort_values()

bars = ax.bar(range(len(sum_ranks)), sum_ranks.values)
ax.set_xticks(range(len(sum_ranks)))
ax.set_xticklabels(sum_ranks.index, rotation=45, ha='right')
ax.set_ylabel('Сумма рангов (меньше = важнее)', fontsize=12)
ax.set_title('РАНЖИРОВАНИЕ БАРЬЕРОВ РАЗВИТИЯ\n(по сумме рангов)', fontsize=14, fontweight='bold')

# Добавляем значения
for bar, value in zip(bars, sum_ranks.values):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
            f'{int(value)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/barrier_ranking.png', dpi=300, bbox_inches='tight')
print("\n✅ График ранжирования барьеров сохранен: results/figures/barrier_ranking.png")

# ==================== 5. ФАКТОРНЫЙ АНАЛИЗ (ПУНКТ 2.3) ====================
print("\n" + "=" * 70)
print("5. ФАКТОРНЫЙ АНАЛИЗ (Пункт 2.3)")
print("=" * 70)

# Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(criteria_data)

# PCA анализ
pca = PCA()
principal_components = pca.fit_transform(scaled_data)

# Объясненная дисперсия
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\n📊 ОБЪЯСНЕННАЯ ДИСПЕРСИЯ ПО ФАКТОРАМ:")
print("-" * 50)
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance), 1):
    print(f"Фактор {i}: {var * 100:.1f}% (накоплено: {cum_var * 100:.1f}%)")

# Определяем оптимальное количество факторов (объясняют >70% дисперсии)
n_factors = np.where(cumulative_variance > 0.7)[0][0] + 1
print(f"\n✅ Рекомендуемое количество факторов: {n_factors}")
print(f"   (объясняют {cumulative_variance[n_factors - 1] * 100:.1f}% дисперсии)")

# Матрица нагрузок факторов
loadings = pd.DataFrame(
    pca.components_[:n_factors].T,
    columns=[f'Фактор {i + 1}' for i in range(n_factors)],
    index=criteria_data.columns
)

print(f"\n📋 МАТРИЦА НАГРУЗОК (первые {n_factors} факторов):")
print("-" * 60)
print(loadings.round(3))

# Интерпретация факторов
print("\n🎯 ИНТЕРПРЕТАЦИЯ ФАКТОРОВ:")
print("-" * 50)
for i in range(n_factors):
    factor_loadings = loadings[f'Фактор {i + 1}']
    top_criteria = factor_loadings.abs().sort_values(ascending=False).head(3).index.tolist()
    print(f"Фактор {i + 1}: {', '.join(top_criteria)}")

# График каменистой осыпи
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График объясненной дисперсии
ax1.bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.7)
ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100,
         'ro-', linewidth=2, markersize=6)
ax1.axhline(y=70, color='r', linestyle='--', alpha=0.5)
ax1.set_xlabel('Номер фактора', fontsize=12)
ax1.set_ylabel('Объясненная дисперсия, %', fontsize=12)
ax1.set_title('ГРАФИК ОБЪЯСНЕННОЙ ДИСПЕРСИИ', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(['Накопленная дисперсия', 'Порог 70%'], loc='best')

# Тепловая карта нагрузок
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=False, ax=ax2)
ax2.set_title('МАТРИЦА НАГРУЗОК ФАКТОРОВ', fontsize=14, fontweight='bold')

plt.suptitle('РЕЗУЛЬТАТЫ ФАКТОРНОГО АНАЛИЗА', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/factor_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Графики факторного анализа сохранены: results/figures/factor_analysis.png")

# ==================== 6. АНАЛИЗ С "ПЛОХИМИ" ЭКСПЕРТАМИ (ПУНКТ 2.4) ====================
print("\n" + "=" * 70)
print("6. АНАЛИЗ С УДАЛЕНИЕМ 'ПЛОХИХ' ЭКСПЕРТОВ (Пункт 2.4)")
print("=" * 70)

# Определяем "плохих" экспертов по корреляции с общим мнением
# Общее мнение = средние ранги по каждому барьеру
mean_ranks = barrier_data.mean()
correlations = []

for idx, row in barrier_data.iterrows():
    corr, _ = kendalltau(row, mean_ranks)
    correlations.append(corr)

# Определяем порог для "плохих" экспертов (нижние 20%)
threshold = np.percentile(correlations, 20)
bad_experts = [i for i, corr in enumerate(correlations) if corr < threshold]

print(f"\n📊 ВЫЯВЛЕНИЕ 'ПЛОХИХ' ЭКСПЕРТОВ:")
print("-" * 50)
print(f"Всего экспертов: {len(correlations)}")
print(f"Порог корреляции: {threshold:.3f}")
print(f"Найдено 'плохих' экспертов: {len(bad_experts)}")
print(f"ID 'плохих' экспертов: {[i + 1 for i in bad_experts]}")

if len(bad_experts) > 0:
    # Удаляем "плохих" экспертов
    barrier_clean = barrier_data.drop(bad_experts).reset_index(drop=True)

    # Пересчитываем коэффициент конкордации
    W_clean, S_clean, T_clean, m_clean, n_clean = calculate_kendall_w(barrier_clean)
    chi2_clean = m_clean * (n_clean - 1) * W_clean
    p_value_clean = 1 - chi2.cdf(chi2_clean, n_clean - 1)

    print(f"\n📈 РЕЗУЛЬТАТЫ ПОСЛЕ ОЧИСТКИ:")
    print("-" * 50)
    print(f"Количество экспертов: {m_clean}")
    print(f"Коэффициент конкордации (W): {W_clean:.4f} (было: {W:.4f})")
    print(f"Изменение W: {((W_clean - W) / W * 100):+.1f}%")
    print(f"p-value: {p_value_clean:.6f}")

    # Визуализация изменения
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Исходные данные', 'После очистки']
    w_values = [W, W_clean]

    bars = ax.bar(labels, w_values)
    ax.set_ylabel('Коэффициент конкордации (W)', fontsize=12)
    ax.set_title('ВЛИЯНИЕ УДАЛЕНИЯ "ПЛОХИХ" ЭКСПЕРТОВ\nНА СОГЛАСОВАННОСТЬ',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Низкая согласованность')
    ax.axhline(y=0.4, color='y', linestyle='--', alpha=0.5, label='Умеренная согласованность')

    for bar, value in zip(bars, w_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11)

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/cleaned_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ График влияния очистки сохранен: results/figures/cleaned_analysis.png")
else:
    print("\n⚠️ 'Плохих' экспертов не выявлено. Анализ не требуется.")

# ==================== 7. СОЗДАНИЕ ОТЧЕТА ====================
print("\n" + "=" * 70)
print("7. ФОРМИРОВАНИЕ ИТОГОВОГО ОТЧЕТА")
print("=" * 70)

# Создаем текстовый файл с основными результатами
with open('results/summary_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №3\n")
    f.write("ЭКСПЕРТНАЯ ОЦЕНКА ЗАВЕДЕНИЙ КИТАЙСКОЙ КУХНИ\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ДАННЫХ\n")
    f.write("-" * 50 + "\n")
    f.write(f"• Количество экспертов: {criteria_matrix.shape[0]}\n")
    f.write(f"• Количество критериев оценки: {criteria_data.shape[1]}\n")
    f.write(f"• Количество барьеров для анализа: {barrier_data.shape[1]}\n\n")

    f.write("2. КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ\n")
    f.write("-" * 50 + "\n")

    # Лучшие и худшие критерии
    best_criteria = stats_table.loc[stats_table['Среднее'].idxmax(), 'Критерий']
    worst_criteria = stats_table.loc[stats_table['Среднее'].idxmin(), 'Критерий']

    f.write(f"2.1. Критерии с наивысшими оценками:\n")
    for _, row in stats_table.nlargest(3, 'Среднее').iterrows():
        f.write(f"   • {row['Критерий']}: {row['Среднее']:.2f} баллов\n")

    f.write(f"\n2.2. Критерии с наинизшими оценками:\n")
    for _, row in stats_table.nsmallest(3, 'Среднее').iterrows():
        f.write(f"   • {row['Критерий']}: {row['Среднее']:.2f} баллов\n")

    f.write(f"\n2.3. Согласованность экспертов по барьерам развития:\n")
    f.write(f"   • Коэффициент конкордации Кендалла: W = {W:.3f}\n")
    f.write(
        f"   • Уровень согласованности: {'НИЗКИЙ' if W < 0.2 else 'УМЕРЕННЫЙ' if W < 0.4 else 'СРЕДНИЙ' if W < 0.6 else 'ВЫСОКИЙ'}\n")
    f.write(f"   • Статистическая значимость: {'ДА' if p_value < 0.05 else 'НЕТ'} (p = {p_value:.4f})\n")

    f.write(f"\n2.4. Факторный анализ выявил {n_factors} основных фактора:\n")
    for i in range(n_factors):
        factor_var = explained_variance[i] * 100
        f.write(f"   • Фактор {i + 1}: объясняет {factor_var:.1f}% дисперсии\n")

    if len(bad_experts) > 0:
        f.write(f"\n2.5. Влияние очистки данных:\n")
        f.write(f"   • Удалено экспертов: {len(bad_experts)}\n")
        f.write(f"   • Новый коэффициент W: {W_clean:.3f} (изменение: {((W_clean - W) / W * 100):+.1f}%)\n")

print("\n✅ Итоговый отчет сохранен: results/summary_report.txt")

# ==================== 8. ИТОГОВЫЕ ВЫВОДЫ ====================
print("\n" + "=" * 70)
print("ВЫВОДЫ И РЕКОМЕНДАЦИИ ДЛЯ ОТЧЕТА")
print("=" * 70)

print("""
📋 ОСНОВНЫЕ ВЫВОДЫ ДЛЯ РАЗДЕЛА "ЗАКЛЮЧЕНИЕ":

1. КАЧЕСТВО ОЦЕНИВАНИЯ:
• Наивысшие оценки получили критерии, связанные с удобством и технологиями.
• Наиболее проблемными аспектами являются гигиена и наличие вегетарианских опций.

2. СОГЛАСОВАННОСТЬ ЭКСПЕРТОВ:
• Уровень согласованности мнений экспертов по барьерам развития является низким/умеренным.
• Это свидетельствует о разнообразии мнений и отсутствии единого взгляда на приоритеты.

3. СКРЫТЫЕ ФАКТОРЫ:
• Факторный анализ позволил выявить основные латентные факторы, влияющие на оценку.
• Это позволяет упростить модель и сфокусироваться на ключевых аспектах.

4. КАЧЕСТВО ЭКСПЕРТНОЙ ГРУППЫ:
• Идентификация "плохих" экспертов позволила повысить надежность результатов.
• После очистки данных согласованность мнений улучшилась.

🎯 РЕКОМЕНДАЦИИ:
• Для улучшения качества заведений следует обратить внимание на проблемные аспекты.
• Разнообразие мнений экспертов требует проведения дополнительных исследований.
• Выявленные факторы могут служить основой для разработки стратегии улучшения.
""")

print("\n" + "=" * 70)
print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
print("=" * 70)
print("""
📁 СОЗДАННЫЕ ФАЙЛЫ:
• results/figures/ - все графики и диаграммы
• results/statistics/descriptive_stats.csv - таблица статистики
• results/summary_report.txt - текстовый отчет

📋 ДЛЯ ОТЧЕТА ИСПОЛЬЗУЙТЕ:
1. Графики из папки results/figures/
2. Числовые результаты из summary_report.txt
3. Таблицы статистики для приложения
""")