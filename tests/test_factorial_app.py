import subprocess
import sys
import unittest

from factorial_app import factorial


class FactorialAppTests(unittest.TestCase):
    def test_factorial_zero(self):
        self.assertEqual(factorial(0), 1)

    def test_factorial_positive(self):
        self.assertEqual(factorial(5), 120)

    def test_factorial_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            factorial(-1)

    def test_cli_prints_factorial(self):
        completed = subprocess.run(
            [sys.executable, "factorial_app.py", "5"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("120", completed.stdout)


if __name__ == "__main__":
    unittest.main()
