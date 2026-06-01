import unittest
import numpy as np
import pandas as pd
from src.backtesting import (
    perform_kupiec_pof_test,
    perform_christoffersen_independence_test,
    perform_conditional_coverage_test,
    determine_basel_zone
)

class TestVaRBacktesting(unittest.TestCase):
    
    def test_kupiec_pof_basic(self):
        # N = 250, cl = 0.99, exceptions = 2 (expected is 2.5)
        # This should definitely pass (no rejection)
        exceptions = np.zeros(250)
        exceptions[10] = 1
        exceptions[100] = 1
        
        result = perform_kupiec_pof_test(exceptions, 0.99)
        self.assertFalse(result['reject'])
        self.assertGreater(result['p_value'], 0.05)
        self.assertGreater(result['lr_stat'], 0.0)

    def test_kupiec_pof_rejection(self):
        # N = 250, cl = 0.99, exceptions = 12 (expected is 2.5)
        # This should fail (rejection)
        exceptions = np.zeros(250)
        exceptions[::20] = 1 # 13 exceptions
        
        result = perform_kupiec_pof_test(exceptions, 0.99)
        self.assertTrue(result['reject'])
        self.assertLess(result['p_value'], 0.05)

    def test_kupiec_pof_edge_cases(self):
        # 0 exceptions
        exceptions_zero = np.zeros(100)
        result_zero = perform_kupiec_pof_test(exceptions_zero, 0.95)
        self.assertFalse(np.isnan(result_zero['lr_stat']))
        self.assertFalse(np.isnan(result_zero['p_value']))
        
        # All exceptions
        exceptions_all = np.ones(100)
        result_all = perform_kupiec_pof_test(exceptions_all, 0.95)
        self.assertFalse(np.isnan(result_all['lr_stat']))
        self.assertFalse(np.isnan(result_all['p_value']))
        self.assertTrue(result_all['reject'])

    def test_christoffersen_independence_no_clustering(self):
        # Generate a truly independent random sequence of exceptions
        np.random.seed(42)
        exceptions = np.random.binomial(1, 0.05, 250)
        
        result = perform_christoffersen_independence_test(exceptions)
        # Should pass independence test
        self.assertFalse(result['reject'])
        self.assertGreater(result['p_value'], 0.05)

    def test_christoffersen_independence_with_clustering(self):
        # Exceptions are clustered: e.g., consecutive ones
        # sequence: 0, 0, 0, 1, 1, 1, 1, 0, 0, 0
        exceptions = np.zeros(100)
        exceptions[20:25] = 1
        exceptions[50:55] = 1
        
        result = perform_christoffersen_independence_test(exceptions)
        # Should fail (reject) independence test because of clustering
        self.assertTrue(result['reject'])
        self.assertLess(result['p_value'], 0.05)

    def test_christoffersen_independence_edge_cases(self):
        # No exceptions
        exceptions_zero = np.zeros(100)
        result_zero = perform_christoffersen_independence_test(exceptions_zero)
        self.assertEqual(result_zero['lr_stat'], 0.0)
        self.assertEqual(result_zero['p_value'], 1.0)
        self.assertFalse(result_zero['reject'])
        
        # 1 element array (too short for transitions)
        result_short = perform_christoffersen_independence_test(np.array([1]))
        self.assertEqual(result_short['lr_stat'], 0.0)
        self.assertEqual(result_short['p_value'], 1.0)

    def test_conditional_coverage(self):
        # Normal correct model
        exceptions = np.zeros(250)
        exceptions[::100] = 1 # 3 exceptions
        
        result = perform_conditional_coverage_test(exceptions, 0.99)
        self.assertFalse(result['reject'])
        self.assertGreater(result['p_value'], 0.05)

    def test_basel_zones_99cl(self):
        # Basel zones for 99% VaR, N=250
        # Green zone: x <= 4
        exc_green = np.zeros(250)
        exc_green[10:13] = 1 # 3 exceptions
        res_green = determine_basel_zone(exc_green, 0.99)
        self.assertEqual(res_green['zone'], 'Green')
        self.assertEqual(res_green['penalty'], 0.00)
        
        # Yellow zone: x = 5
        exc_yellow = np.zeros(250)
        exc_yellow[10:15] = 1 # 5 exceptions
        res_yellow = determine_basel_zone(exc_yellow, 0.99)
        self.assertEqual(res_yellow['zone'], 'Yellow')
        self.assertEqual(res_yellow['penalty'], 0.40)

        # Red zone: x >= 10
        exc_red = np.zeros(250)
        exc_red[::20] = 1 # 13 exceptions
        res_red = determine_basel_zone(exc_red, 0.99)
        self.assertEqual(res_red['zone'], 'Red')
        self.assertEqual(res_red['penalty'], 1.00)

if __name__ == '__main__':
    unittest.main()
