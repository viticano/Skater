import unittest


def run_tests():
    testsuite = unittest.TestLoader().discover('./pyinterpret/tests/', pattern="test*.py")
    unittest.TextTestRunner(verbosity=2).run(testsuite)


if __name__ == '__main__':
    run_tests()
