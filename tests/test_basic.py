"""
Basic tests for the Dendritic Cyber Sentinel package.
"""

import unittest
import numpy as np
import pandas as pd

from dendritic_cyber_sentinel import (
    MRA_SdDCA, 
    FeatureSelector, 
    mutual_information
)


class TestMutualInformation(unittest.TestCase):
    """Test mutual information calculation."""
    
    def test_mutual_information_basic(self):
        """Test basic mutual information calculation."""
        # Create simple test data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        
        # Perfect correlation should have positive MI
        mi = mutual_information(x, y)
        self.assertGreater(mi, 0)
        
        # Independent variables should have low MI
        y_random = np.random.randint(1, 100, size=5)
        mi_random = mutual_information(x, y_random)
        self.assertLess(mi_random, mi)


class TestFeatureSelector(unittest.TestCase):
    """Test feature selection functionality."""
    
    def test_normalize_signal(self):
        """Test signal normalization."""
        selector = FeatureSelector(verbose=False)
        
        # Test normalization with regular signal
        signal = np.array([1, 2, 3, 4, 5])
        normalized = selector._normalize_signal(signal)
        self.assertEqual(normalized.min(), 0)
        self.assertEqual(normalized.max(), 1)
        
        # Test normalization with constant signal
        constant_signal = np.array([3, 3, 3, 3, 3])
        normalized_constant = selector._normalize_signal(constant_signal)
        self.assertTrue(np.all(normalized_constant == 0))


class TestMRA_SdDCA(unittest.TestCase):
    """Test MRA_SdDCA functionality."""
    
    def test_normalize_range(self):
        """Test range normalization."""
        mra_sdca = MRA_SdDCA(verbose=False)
        
        # Test 2D array normalization
        data = np.array([
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50]
        ])
        
        normalized = mra_sdca._normalize_range(data)
        
        # Check shape is preserved
        self.assertEqual(normalized.shape, data.shape)
        
        # Check each row is normalized to [0, 1]
        for row in normalized:
            self.assertAlmostEqual(row.min(), 0)
            self.assertAlmostEqual(row.max(), 1)


if __name__ == "__main__":
    unittest.main() 