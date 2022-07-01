import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const

def redshift(l,v):
    """Redshift

    Get the doppler shifted wavelength of a line given the linecenter
    wavelength and radial velocity

    Args:
        l (astropy.Quantity): wavelength to be shifted
        v (astropy.Quantity): radial velocity
    
    Returns:
        (astropy.Quantity): red/blue shifted wavelength
    """
    dl = v*l/const.c
    return l+dl

def to_air(vac):
    """To air
    
    Convert from vacuum to air wavelengths

    Args:
        vac (astropy.Quantity): wavelength(s) to be shifted

    Returns:
        (astropy.Quantity): wavelength(s) in air
    """
    s = 1e4 / (vac)
    n = (1 + 0.0000834254)/(s.unit ** 2) + 0.02406147 / (130*(s.unit ** 2) - s**2) + 0.00015998 / (38.9*(s.unit ** 2) - s**2)
    
    #air = vac/(1+2.735182E-4+131.4182/vac**2+2.76249E8/vac**4)
    return vac/n.value


def poly_x(x, coeffs):
    """Polynomial

    Transform an array of x values given coefficients of a polynomial function

    Args:
        x (np.array): values to be transformed
        coeffs (array-like): coefficents of the polynomical function starting with the highest order

    Returns:
        (np.array): y values of the polynomial function
    """
    y = x*0
    l = len(coeffs)
    for i in range(l):
        y = y + x**i * coeffs[l-i-1]
    return y

def ddx(x,y):
    """Derivative

    Numerical derivitive with `np.gradient`
    
    Args:
        x (array-like): x values
        y (array-like): y values
    
    Returns:
        (np.array): x values
        (np.array): derivative of `x` and `y`
    """
    return x, np.gradient(y,x,edge_order=2)

def trap_rule(x,f):
    """Trapazoid rule

    Integrate using Trapizoid rule.

    Args:
        x (array-like): x values to integrate over
        f (array-like): y values to integrate over

    Returns:
        (float) integral of f over the domain of x
    """
    total = 0
    for i in range(0,len(x)-1):
        total += (f[i] + f[i+1])/2 * (x[i+1]-x[i])
    return total

def reject(x,y,**kwargs):
    """Reject
    Rejection procedure for finding the continuum of a spectrum
    works best if y has some noise

    Args:
        x (np.array): wavelength values of the spectrum
        y (np.array): flux values of the spectrum
    
    Keyword Args:
        pct (float in range [0,1]): residual cutoff percentile. Default `0.85`
        box (int): radius of surrounding points to check. Default `5`
        degree (int): fitting degree. Default `1`
    
    Returns:
        (np.array of bool): points that part of the spectral continuum
    """
    pct = kwargs.get('pct',0.85)
    box = kwargs.get('box',5)
    degree = kwargs.get('degree',1)
    
    # get initial continuum
    x0 = x[0]
    coeffs = np.polyfit(x-x0,y,degree)
    cont = poly_x(x-x0,coeffs)
    res = abs(cont-y)
    ny = len(y)
    cutoff = np.sort(res)[int(pct*ny)]
    is_cont = res < cutoff
    
    #second pass
    
    coeffs = np.polyfit(x[is_cont] - x0, y[is_cont],degree)
    cont = poly_x(x-x0,coeffs)
    res = abs(cont-y)
    ny = len(y)
    cutoff = np.sort(res)[int(pct*ny)]
    is_cont = res < cutoff
    
    #check surrounding points
    is_cont = np.convolve(is_cont,np.ones(box)/box,mode='same').astype('bool')
    
    return is_cont


def get_continuum_points(x,y,**kwargs):
    """Get continuum points
    
    Get a boolean array indicating which points are part of the continuum
    
    Args:
        x (np.array): wavelength values of the spectrum
        y (np.array): flux values of the spectrum
    
    Keyword Args:
        pct (float in range [0,1]): residual cutoff percentile. Default `0.85`
        box (int): radius of surrounding points to check. Default `5`
        degree (int): fitting degree. Default `1`
        cutoff (float): second derivitive cutoff for fitting a model continuum. Default 1e-12
        stype (str): `data` or `model`. Type of spectrum to fit continuum. Default `data`

    Returns:
        (np.array of bool): points that part of the spectral continuum
    """
    cutoff = kwargs.get('cutoff',1e-12)
    stype = kwargs.get('stype','data')
    
    #first constrain to region
    
    if stype == 'data':
        is_cont = reject(x,y,**kwargs)
        return is_cont
    elif stype == 'model':
        is_cont = abs(ddx(*ddx(x,y))[1]) < cutoff
        return is_cont
    
def get_continuum(x,y,w1,w2,**kwargs):
    """Get continuum

    Fit the continuum of a spectrum

    Args:
        x
    fit a polynomial f to the continuum and return the array f(x) between the wavelengths w1,w2
    """
    degree = kwargs.get('degree',1)
    
    reg = (x>=w1) & (x<=w2)
    xx = x[reg]
    yy = y[reg]
    
    is_cont = get_continuum_points(xx,yy,**kwargs)
    
    xx = xx[is_cont]
    yy = yy[is_cont]
    x0 = xx[0]
    
    coeffs = np.polyfit(xx-x0,yy,degree)
    
    cont = poly_x(x-x0,coeffs)
    
    return cont

def convert_line(x,y,ycont,lam):
    """
    change coordinate system so line center is at zero, continuum is zero, and absorption is positive
    this is to make fitting easier
    """
    return x-lam,(y-ycont)*-1

def equivalent_width(x,y,ycont,w1,w2,unit,**kwargs):
    """
    Get the EW of a region
    only unit passed as Quantity
    """
    
    plot = kwargs.get('plot',False)
    
    reg = (x >= w1) & (x <= w2)
    xx = x[reg]
    yy = (1-(y/ycont))[reg]
    
    ew = trap_rule(xx,yy)
    
    
    if plot:
        fig, ax = plt.subplots(1,1)
        dw = w2-w1
        ax.plot(x,y,c='k')
        ax.plot(x,ycont, c='b')
        ax.set_xlim(w1-dw,w2+dw)
        ymin = min(y[reg]) * 0.9
        ymax = max(y[reg]) * 1.1
        ax.set_ylim(ymin,ymax)
        ax.axvline(w1,c='k',alpha=0.5)
        ax.axvline(w2,c='k',alpha=0.5)
        ax.set_xlabel('Lambda (' + str(unit) + ')')
        ax.set_ylabel('Flux')
        ax.set_title('EW = %.4f ' % ew + str(unit))
        fig.show()
        plt.pause(2)
        plt.close(fig)
    
    
    return ew*unit


def equivalent_width_error(x,y,W1,W2,linecenter,unit,**kwargs):
    """
    estimate the error of the equivalent width of a region of the spectrum
    use with caution
    """
    
    plot = kwargs.get('plot',False)
    
    reg = (x>W1) & (x < W2)
    xx = x[reg]
    yy = y[reg]
    is_cont = get_continuum_points(xx,yy,**kwargs)
    
    # reduce to linecenter region
    indicies = []
    for i in range(1,len(is_cont)):
        if is_cont[i-1] != is_cont[i]:
            indicies.append(i)
    indicies = np.array(indicies)
    
    line_index = np.argwhere((xx-linecenter) > 0 )[0][0]
    i1 = indicies[indicies < line_index][-1]
    i2=indicies[indicies > line_index][0]
    is_cont[:] = True
    is_cont[i1:i2] = False
    
    #now compute error
    df = np.std(yy[is_cont])
    lx=xx[np.logical_not(is_cont)]
    ly=yy[np.logical_not(is_cont)]
    
    error = trap_rule(lx,df/ly)
    
    if plot:
        fig, ax = plt.subplots(1,1)
        ax.plot(xx[is_cont],yy[is_cont],c='k')
        ax.plot(lx,ly, c='b')
        ax.set_xlim(W1,W2)
        ymin = min(yy) * 0.9
        ymax = max(yy) * 1.1
        ax.set_ylim(ymin,ymax)
        ax.axvline(linecenter,c='k',alpha=0.5)
        ax.set_xlabel('Lambda (' + str(unit) + ')')
        ax.set_ylabel('Flux')
        ax.set_title('EW_error = %.4f ' % error + str(unit))
        fig.show()
        plt.pause(2)
        plt.close(fig)
    
    return error*unit


def Voigt(x,linecenter,AG,sigma,AL,gamma):
    """
    Pseudo-Voigt function
    
    x: array like
    linecenter: center of profile
    AG: Gaussian amplitude
    sigma: gaussian sigma
    AL: Lorentz amplitude
    gamma: Lorentzian gamma
    
    Note: The Lorentzian portion of the curve is not normalized for purposes of easier fitting using scipy.optimize.curve_fit.
    The area under the curve it therefore int(-inf-->inf) Voigt dx = AG + AL*pi*gamma
    
    """
    y = (G_amp*(1/(G_sig*(np.sqrt(2*np.pi))))*(np.exp(-((x-linecenter)**2)/(2*(G_sig)**2)))) +((L_amp*L_wid**2/((x-linecenter)**2+L_wid**2)))
    return y

def multiVoigt(x,*args):
    """
    use with caution
    """
    if len(args) % 5 != 0:
        raise ValueError('must have 5 arguments for each profile')
    nlines = int(len(args)/5)
    y = np.zeros(len(x))
    for i in range(nlines):
        y = y + Voigt(x,args[i],args[i+1],args[i+2],args[i+3],args[i+4])
    return y


class Voigt_Profile:
    """
    use with caution
    """
    def __init__(self,popt,pconv=(0,0,0,0,0)):
        self.center = (popt[0],pconv[0])
        self.AG = (popt[1],pconv[1])
        self.sigma = (popt[2],pconv[2])
        self.AL = (popt[3],pconv[3])
        self.gamma = (popt[4],pconv[4])
    def Vofx(self,x):
        return Voigt(x,self.center[0],self.AG[0],self.sigma[0],self.AL[0],self.gamma[0])
    def area(self):
        return self.AG + self.AL*np.pi*self.gamma
    
