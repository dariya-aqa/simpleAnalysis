import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMBER_COUNT = 1000
MIN_VALUE = -10000
MAX_VALUE = 10000
RANDOM_SEED = 42

OUTPUT_DIR = Path("results")


# Математическое округление чисел до сотен
def round_to_hundreds_math(values: pd.Series) -> pd.Series:
    rounded = np.where(
        values >= 0,
        np.floor(values / 100 + 0.5) * 100,
        np.ceil(values / 100 - 0.5) * 100,
    )

    return pd.Series(
        rounded.astype(int),
        index=values.index,
        name="Округленные до сотен значения"
    )


# Генерация Dataset из 1000 случайных целых чисел в диапазоне от -10000 до 10000
def create_dataset() -> pd.Series:
    generator = np.random.default_rng(RANDOM_SEED)
    numbers = generator.integers(MIN_VALUE, MAX_VALUE + 1, size=NUMBER_COUNT)

    return pd.Series(numbers, name="Исходные значения")


# Очистка данных от цифрового мусора
def clean_dataset(series: pd.Series) -> pd.Series:
    cleaned_series = pd.to_numeric(series, errors="coerce")
    cleaned_series = cleaned_series.dropna()

    # Проверка целочисленности значений
    cleaned_series = cleaned_series[cleaned_series % 1 == 0]

    # Проверка попадания значений в допустимый диапазон
    cleaned_series = cleaned_series[
        (cleaned_series >= MIN_VALUE) & (cleaned_series <= MAX_VALUE)
    ]

    return cleaned_series.astype(int).reset_index(drop=True)


# Вычисление среднеквадратического отклонения для Series
def calculate_root_mean_square_deviation(series: pd.Series) -> float:
    return math.sqrt(((series - series.mean()) ** 2).mean())


# Формирование текстового описания стандартных числовых характеристик
def get_numeric_characteristics_text(series: pd.Series) -> str:
    characteristics = [
        "Стандартные числовые характеристики для набора данных Series",
        f"Количество значений: {len(series)}",
        f"Минимальное значение: {series.min()}",
        f"Количество повторяющихся значений: {len(series) - series.nunique()}",
        f"Максимальное значение: {series.max()}",
        f"Сумма чисел: {series.sum()}",
        (
            "Среднеквадратическое отклонение: "
            f"{calculate_root_mean_square_deviation(series):.4f}"
        ),
    ]

    return "\n".join(characteristics)


# Вывод стандартных числовых характеристик в консоль
def print_numeric_characteristics(series: pd.Series) -> None:
    print(get_numeric_characteristics_text(series))


# Сохранение стандартных числовых характеристик в текстовый файл
def save_numeric_characteristics_to_txt(
    series: pd.Series,
    filename: str = "numeric_characteristics.txt"
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    file_path = OUTPUT_DIR / filename

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(get_numeric_characteristics_text(series))

    print(f"\nФайл со статистическими характеристиками сохранен: {file_path}")


# Создание DataFrame с исходными значениями и сортировкой по возрастанию и убыванию
def create_dataframe(series: pd.Series) -> pd.DataFrame:
    sorted_ascending = series.sort_values().reset_index(drop=True)
    sorted_descending = series.sort_values(ascending=False).reset_index(drop=True)

    dataframe = pd.DataFrame(
        {
            "Исходные значения": series.reset_index(drop=True),
            "Отсортированные по возрастанию": sorted_ascending,
            "Отсортированные по убыванию": sorted_descending,
        }
    )

    return dataframe


# Сохранение исходного Dataset в CSV-файл
def save_dataset_to_csv(series: pd.Series, filename: str = "dataset.csv") -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    file_path = OUTPUT_DIR / filename
    series.to_csv(file_path, index=False, encoding="utf-8-sig", sep=";")

    print(f"Исходный Dataset сохранен: {file_path}")


# Сохранение DataFrame в CSV-файл
def save_dataframe_to_csv(
    dataframe: pd.DataFrame,
    filename: str = "dataframe.csv"
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    file_path = OUTPUT_DIR / filename
    dataframe.to_csv(file_path, index=False, encoding="utf-8-sig", sep=";")

    print(f"DataFrame сохранен: {file_path}")
# Визуализация исходного Series в виде линейного графика
def plot_original_series(series: pd.Series) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, color="steelblue", linewidth=1)

    plt.title("Линейный график исходного Series")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.grid(True, alpha=0.3)

    file_path = OUTPUT_DIR / "linear_plot_original_series.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    print(f"Линейный график исходного Series сохранен: {file_path}")


# Визуализация гистограммы для значений, округленных до сотен
def plot_rounded_histogram(series: pd.Series) -> None:
    rounded_values = round_to_hundreds_math(series)
    bins = np.arange(rounded_values.min() - 50, rounded_values.max() + 150, 100)

    plt.figure(figsize=(12, 5))
    plt.hist(
        rounded_values,
        bins=bins,
        color="seagreen",
        edgecolor="black",
        rwidth=0.9
    )

    plt.title("Гистограмма значений, округленных до сотен")
    plt.xlabel("Значение, округленное до сотен")
    plt.ylabel("Количество")
    plt.grid(axis="y", alpha=0.3)

    file_path = OUTPUT_DIR / "histogram_rounded_values.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    print(f"Гистограмма округленных значений сохранена: {file_path}")


# Визуализация отсортированных значений по возрастанию и убыванию на одном графике
def plot_sorted_values(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))

    plt.plot(
        dataframe.index,
        dataframe["Отсортированные по возрастанию"],
        label="По возрастанию",
        color="darkorange",
        linewidth=1.5,
    )

    plt.plot(
        dataframe.index,
        dataframe["Отсортированные по убыванию"],
        label="По убыванию",
        color="purple",
        linewidth=1.5,
    )

    plt.title("Сравнение отсортированных значений")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True, alpha=0.3)

    file_path = OUTPUT_DIR / "sorted_values_comparison.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    print(f"График сравнения отсортированных значений сохранен: {file_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    series = create_dataset()
    cleaned_series = clean_dataset(series)
    dataframe = create_dataframe(cleaned_series)

    save_dataset_to_csv(cleaned_series)
    save_dataframe_to_csv(dataframe)

    print_numeric_characteristics(cleaned_series)
    save_numeric_characteristics_to_txt(cleaned_series)

    print("\nПервые 10 строк DataFrame:")
    print(dataframe.head(10))

    plot_original_series(cleaned_series)
    plot_rounded_histogram(cleaned_series)
    plot_sorted_values(dataframe)

    plt.show()


if __name__ == "__main__":
    main()