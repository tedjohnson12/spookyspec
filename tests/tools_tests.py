import pytest
from astropy import units as u,constants as c
import numpy as np

from spooky import tools

def test_redshift():
    """Test redshift

    Unit tests for the tools.redshift() function

    Args:
        None
    
    Returns:
        None
    """
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
    assert (l == tools.redshift(l,0*u.km/u.s)).all()

def test_to_air():
    """Test to_air
    
    Unit tests for the tools.to_air() function

    Args:
        None
    
    Returns:
        None
    """
    # Ca K
    assert 3933.663*u.AA == pytest.approx(tools.to_air(3934.777*u.AA),abs=1e-3*u.AA)
    
    #array
    assert isinstance(tools.to_air(np.array([4000,4100])*u.AA),u.quantity.Quantity)

def test_poly_x():
    """Test poly_x
    
    Unit tests for the tools.poly_x() function

    Args:
        None
    
    Returns:
        None
    """
    #zero
    assert tools.poly_x(1,[0]) == 0

    #quadratic
    x = np.linspace(-10,10,21)
    assert (tools.poly_x(x,[2,1,0]) == 2*x**2 + x).all()

def test_ddx():
    """Test ddx

    Unit tests for the tools.ddx() function

    Args:
        None
    
    Returns:
        None
    """
    #quadratic
    x = np.linspace(-10,10,21)
    y = tools.poly_x(x,[1,0,0])
    assert (tools.ddx(x,y)[1] == pytest.approx(tools.poly_x(x,[2,0]),abs=1e-3)).all()

    #sine
    x = np.linspace(-10,10,2001)
    y = np.sin(x)
    yy = dydx(x,y)[1]
    dydx = np.cos(x)
    assert (yy == pytest.approx(dydx,abs=1e-3)).all()

def test_trap_rule():
    """Test trap_rule

    Unit tests for the tools.trap_rule() function

    Args:
        None
    
    Returns:
        None
    """
    #sine
    x = np.linspace(0,2*np.pi,10)
    y = np.sin(x)
    assert 0 == pytest.approx(tools.trap_rule(x,y),abs=1e-6)

    #quadratic
    x = np.linspace(0,4,11)
    y = tools.poly_x(x,[2,0])
    assert 16 == pytest.approx(tools.trap_rule(x,y),abs=1e-6)

def test_reject():
    """Test reject
    
    Unit test for the tools.reject() function

    Args:
        None

    