import math
import unittest

import pandas as pd

import main


class TestSimpleAnalysis(unittest.TestCase):

    def test_create_dataset_count_and_range(self):
        series = main.create_dataset()

        self.assertEqual(len(series), main.NUMBER_COUNT)
        self.assertGreaterEqual(series.min(), main.MIN_VALUE)
        self.assertLessEqual(series.max(), main.MAX_VALUE)

    def test_create_dataset_is_reproducible(self):
        first = main.create_dataset()
        second = main.create_dataset()

        pd.testing.assert_series_equal(first, second)

    def test_clean_dataset(self):
        source = pd.Series([
            10,
            "20",
            30.0,
            30.5,
            None,
            "abc",
            -10000,
            10000,
            -10001,
            10001,
        ])

        result = main.clean_dataset(source)

        expected = pd.Series(
            [10, 20, 30, -10000, 10000],
            dtype=int,
        )

        pd.testing.assert_series_equal(result, expected)

    def test_round_to_hundreds_math(self):
        source = pd.Series([
            149,
            150,
            151,
            -149,
            -150,
            -151,
            0,
            50,
            -50,
        ])

        result = main.round_to_hundreds_math(source)

        expected = pd.Series(
            [100, 200, 200, -100, -200, -200, 0, 100, -100],
            name="Округленные до сотен значения",
            dtype=int,
        )

        pd.testing.assert_series_equal(result, expected)

    def test_create_dataframe(self):
        source = pd.Series([3, 1, 2])

        dataframe = main.create_dataframe(source)

        self.assertEqual(
            list(dataframe.columns),
            [
                "Исходные значения",
                "Отсортированные по возрастанию",
                "Отсортированные по убыванию",
            ],
        )

        self.assertEqual(
            dataframe["Исходные значения"].tolist(),
            [3, 1, 2],
        )

        self.assertEqual(
            dataframe["Отсортированные по возрастанию"].tolist(),
            [1, 2, 3],
        )

        self.assertEqual(
            dataframe["Отсортированные по убыванию"].tolist(),
            [3, 2, 1],
        )

    def test_root_mean_square_deviation(self):
        source = pd.Series([1, 2, 3])

        result = main.calculate_root_mean_square_deviation(source)
        expected = math.sqrt(2 / 3)

        self.assertAlmostEqual(result, expected, places=7)


if __name__ == "__main__":
    unittest.main()