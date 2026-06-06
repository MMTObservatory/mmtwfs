# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import numpy as np

from mmtwfs.photometry import make_spot_mask


class TestMakeSpotMask:
    """Test suite for make_spot_mask function"""

    def test_make_spot_mask_with_sources(self):
        """Test make_spot_mask with detectable sources"""
        # Create image with bright spots
        data = np.zeros((100, 100))
        # Add some bright spots
        data[25, 25] = 1000
        data[75, 75] = 1000
        data[50, 50] = 1000
        # Add some background noise
        data += np.random.normal(0, 1, data.shape)

        mask = make_spot_mask(data, nsigma=3.0, npixels=1)
        assert mask.shape == data.shape
        assert mask.dtype == bool
        # Should have detected sources
        assert mask.any()

    def test_make_spot_mask_no_sources(self):
        """Test make_spot_mask with no detectable sources (returns zeros)"""
        import warnings
        from photutils.utils.exceptions import NoDetectionsWarning
        # Create uniform low-level noise image with no bright spots
        np.random.seed(42)
        data = np.random.normal(0, 0.1, (100, 100))

        # Use very high threshold to ensure no sources detected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            mask = make_spot_mask(data, nsigma=100.0, npixels=100)
        assert mask.shape == data.shape
        assert mask.dtype == bool
        # Should be all zeros when no sources detected
        assert not mask.any()

    def test_make_spot_mask_with_mask(self):
        """Test make_spot_mask with input mask"""
        data = np.zeros((100, 100))
        data[50, 50] = 1000
        data += np.random.normal(0, 1, data.shape)

        # Create a mask to exclude some regions
        input_mask = np.zeros((100, 100), dtype=bool)
        input_mask[45:55, 45:55] = True  # Mask the bright spot region

        mask = make_spot_mask(data, nsigma=3.0, npixels=1, mask=input_mask)
        assert mask.shape == data.shape
