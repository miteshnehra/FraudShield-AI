"""Tests for secure CSV intake."""

import unittest

from fraudshield.data.loader import DatasetLoadError, load_csv_bytes


class CsvLoaderTests(unittest.TestCase):
    def test_loads_valid_csv_and_records_metadata(self) -> None:
        content = b"amount,merchant,Class\n10,Alpha,0\n20,Beta,1\n"

        loaded = load_csv_bytes(content, "transactions.csv")

        self.assertEqual(loaded.frame.shape, (2, 3))
        self.assertEqual(loaded.metadata.filename, "transactions.csv")
        self.assertEqual(loaded.metadata.delimiter, ",")
        self.assertEqual(len(loaded.metadata.fingerprint), 64)

    def test_detects_semicolon_delimiter(self) -> None:
        content = b"amount;merchant;Class\n10;Alpha;0\n20;Beta;1\n"

        loaded = load_csv_bytes(content, "transactions.csv")

        self.assertEqual(loaded.frame.shape, (2, 3))
        self.assertEqual(loaded.metadata.delimiter, ";")

    def test_rejects_wrong_extension(self) -> None:
        with self.assertRaises(DatasetLoadError):
            load_csv_bytes(b"amount\n10\n", "transactions.xlsx")

    def test_rejects_header_without_rows(self) -> None:
        with self.assertRaises(DatasetLoadError):
            load_csv_bytes(b"amount,merchant\n", "transactions.csv")


if __name__ == "__main__":
    unittest.main()

