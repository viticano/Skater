"""Run all tests"""

import unittest

def run_tests():
    """Run all tests in current directory with signature test*.py"""
    testsuite = unittest.TestLoader().discover('', pattern="test*.py")
    unittest.TextTestRunner(verbosity=2).run(testsuite)

if __name__ == '__main__':
    run_tests()

