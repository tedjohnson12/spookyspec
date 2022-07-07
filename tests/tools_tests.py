import pytest
from astropy import units as u,constants as c

from spooky import tools

def test_redshift():

    #speed of light blueshift
    assert 0 == tools.redshift(4000*u.AA,-c.c)