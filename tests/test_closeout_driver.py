"""Phase dispatch must call the tested operation, never execute real data work."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import unittest

spec=importlib.util.spec_from_file_location('closeout_fixture',Path(__file__).resolve().parents[1]/'scripts/closeout_m2.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class CloseoutDriverTests(unittest.TestCase):
    def test_each_cli_phase_dispatches_to_its_exact_operation(self):
        for phase,function in (('prepare','prepare'),('tests','run_tests'),('boundary','boundary'),('seal','seal'),('launch','launch')):
            with patch.object(driver,function,return_value='INVENTED') as called:
                self.assertEqual(driver.run_phase(phase),'INVENTED');called.assert_called_once_with()
        with self.assertRaises(ValueError):driver.run_phase('__builtins__')
