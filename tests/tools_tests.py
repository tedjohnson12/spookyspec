import pytest
from astropy import units as u,constants as c

from spooky import tools

def test_redshift():

    #speed of light blueshift
    assert 0 == tools.redshift(4000*u.AA,-c.c)
    
    #no radial velocity
    l = 4000*u.AA
    assert l == tools.redshift(l,0*u.km/u.s)

    #Ca K
    l = 3933.667 * u.AA
    assert 3934.0213 * u.AA == pytest.approx(tools.redshift(l,27*u.km/u.s),abs=1e-3*u.AA)

    # array
    l = np.array(1100,1200)*u.AA
    assert (l == tools.redshift(l,0*u.km/s)).all()