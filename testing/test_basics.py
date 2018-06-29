# For testing the code

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose
from numpy.random import rand
import pickle

from ..k2sff_wallace.decorrelate_position_bsplineiter import (one_object, one_object_one_aperture)


one_test_lc = pickle.load(open("test_data.p","rb"))

"""
def test_ones():
    # Build the object
    num_ones = 1000
    ones_lc = one_object("ones",np.arange(num_ones),
                         np.linspace(0,300.,num_ones),
                         rand(num_ones),
                         np.zeros(num_ones))
    ones_lc.calc_cadence_window_indices([0,500,1001])
    ones_lc.calc_arclength()


    ones_lc_one_aperture = one_object_one_aperture(ones_lc,
                                 np.ones(num_ones),np.array([.1]*num_ones),
                                 np.zeros(num_ones),np.zeros(num_ones),
                                 [0,500,1001])
    ones_lc_one_aperture.pre_decorrelation_filtering()
    assert_allclose(ones_lc_one_aperture,np.ones(num_ones))
"""

def test_realdata():
    cadence_number_divisions = [1,49+1, 570, 1084, 1600, 2154, 2570, 2995, 3419,3856]

    lc = one_object(604544444,one_test_lc['cadence_nose'],
                         one_test_lc['BJD'],one_test_lc['x'],
                         one_test_lc['y'])
    lc.calc_cadence_window_indices(cadence_number_divisions)
    lc.calc_arclength()

    lc_one_aperture = one_object_one_aperture(lc,
                                 one_test_lc['mags'],one_test_lc['errs'],
                                 cadence_number_divisions)
    
    



# Then the same as above, but with random x and y
