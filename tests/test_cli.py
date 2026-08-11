import contextlib
import io
import unittest

from ot_micromr.cli import main


class CliTests(unittest.TestCase):
    def test_no_arguments_prints_help_and_succeeds(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            return_code = main([])
        self.assertEqual(return_code, 0)
        self.assertIn("validate-config", output.getvalue())
        self.assertIn("run", output.getvalue())


if __name__ == "__main__":
    unittest.main()
