"""Tests for application configuration."""

import unittest

from fraudshield.config import ensure_runtime_directories, load_settings


class SettingsTests(unittest.TestCase):
    """Verify that the Phase 1 configuration is usable."""

    def test_settings_load_required_sections(self) -> None:
        settings = load_settings()

        self.assertEqual(settings.section("app")["name"], "FraudShield AI")
        self.assertEqual(settings.section("model")["random_state"], 42)
        self.assertEqual(settings.section("model")["cv_folds"], 5)
        self.assertEqual(settings.section("risk")["critical_max"], 100)

    def test_runtime_directories_exist_after_initialization(self) -> None:
        settings = load_settings()
        ensure_runtime_directories(settings)

        for path_name in ("raw_data", "processed_data", "models", "reports", "logs"):
            self.assertTrue(settings.path(path_name).is_dir())


if __name__ == "__main__":
    unittest.main()
