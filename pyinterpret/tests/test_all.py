import unittest
from test_data import TestData
from test_lime import TestLime

dataset_tests = unittest.TestLoader().loadTestsFromTestCase(TestData)
lime_tests = unittest.TestLoader().loadTestsFromTestCase(TestLime)

for test in [dataset_tests, lime_tests]:
    unittest.TextTestRunner(verbosity=2).run(test)
