import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from copy import deepcopy
import matplotlib.pyplot as plt
from spookyspec.tools import get_continuum, equivalent_width, to_air#,equivalent_width_error
from spookyspec.fits import fits_to_np
from spookyspec.fits import read_two_column
from spookyspec.fits import read_psg
from scipy.interpolate import interp1d

class Spec:
    """Store 1D spectral data

    This class was designed to be easy to use, especially to create figures
    spectra are transformed by appending class methods to the variable. For example,
    to redshift a spectrum spec by 10 km/s, call spec.dopshift(10)
    self.show() returns the axes as astropy.Quantity objects.

    Args:
        l (np.array): wavelength points
        f (np.array): flux points

    Keyword Args:
        u_l (str): unit of wavelength axis. Should be readable by astropy.units.Unit(). Default `'Angstrom'`
        u_f (str): unit of flux axis. Should be readable by astropy.units.Unit(). Default `'erg cm-2 s-1 Angstrom-1'`
        stype (str): type of information stored. Either `'data'` or `'model'`. This is important for finding the continuum
        and normalization. Default `'data'`
        hdr (astropy.io.fits.header.Header): Fits header to be stored with the spectrum. Default `None`
    """
    def __init__(self,l,f,**kwargs):
        self.l = l
        self.f = f
        self.u_l = u.Unit(kwargs.get('u_l','Angstrom'))
        self.u_f = u.Unit(kwargs.get('u_f','erg cm-2 s-1 Angstrom-1'))
        self.stype = kwargs.get('stype','data')
        self.hdr = kwargs.get('hdr',None)
        if self.stype not in ['model','data']:
            raise ValueError('stype must be \'model\' or \'data\'')
    def show(self):
        """Show spectrum

        Return wavelength and flux so spectrum can be plotted or otherwise used

        Args:
            None
        
        Returns:
            (astropy.Quantity array-like): wavelengths
            (astropy.Quantity array-like): fluxes
        """
        return self.l * self.u_l, self.f * self.u_f
    def __str__(self):
        """String descriptor
        
        Get a string describing the spectrum

        Args:
            None
        
        Returns:
            (str): short description of spectrum
        """
        s = 'Spec %s from %.3f to %.3f %s' % (self.stype,self.l.min(),self.l.max(),str(self.u_l))
        return s
    def scale(self,k):
        """Scale spectrum

        Multiply flux by a scaler, preserving equivalent width.

        Args:
            k (float): scale factor
        
        Returns:
            (Spec): new spectrum
        """
        new_spec = deepcopy(self)
        new_spec.f = self.f * k
        return new_spec
    def new_units(self,**kwargs):
        """Change unit keyword
        Change the units stored in `self.u_l` and/or `self.u_f`. Does not change the arrays that store data.
        This is for fixing mistakes and other data management actions. To cast your data to another unit,
        please use `astropy.units.to()`.
        
        Args:
            None
        
        Keyword Args:
            u_l (str): new unit of wavelength axis. Should be readable by astropy.units.Unit().
            u_f (str): new unit of flux axis. Should be readable by astropy.units.Unit().
        """
        new_spec = deepcopy(self)
        if 'u_l' in kwargs:
            new_spec.u_l = u.Unit(kwargs['u_l'])
        if 'u_f' in kwargs:
            new_spec.u_f = u.Unit(kwargs['u_f'])
        return new_spec
    def quick_plot(self,w1,w2):
        """Plot a simple view

        Show the spectrum in a simple non-interactive window. This is good to quickly check your work.

        Args:
            w1 (float): starting wavelength in units of `self.u_l`
            w2 (float): ending wavelength in units of `self.u_l`

        Returns:
            None
        """
        plt.plot(*self.show(),c='k')
        # get y limit
        reg = (w1<=self.l) & (w2>=self.l)
        ymin = self.f[reg].min()
        ymax = self.f[reg].max()
        plt.ylim(ymin*0.9,ymax*1.1)
        plt.xlim(w1,w2)
        plt.xlabel(str(self.u_l))
        plt.ylabel(str(self.u_f))
        plt.show()
    def normalize(self,w1,w2,**kwargs):
        """Normalize

        Normalize a region of the spectrum. Great if comparing a model to uncalibrated echelle spectra.

        Args:
            w1 (float): starting wavelength to normalize in units of `self.u_l`
            w2 (float): ending wavelength to normalize in units of `self.u_l`
        
        Keyword Args:
            degree (int): degree of polynomial to fit

        Returns:
            (Spec): normalized spectrum
        """
        new_spec = deepcopy(self)
        continuum = self.continuum(w1,w2,**kwargs)
        new_spec.f = self.f / continuum.value
        new_spec.u_f = u.Unit('')
        return new_spec
    def continuum(self,w1,w2,**kwargs):
        """Continuum
        
        Get a polynomial fit of the continuum.

        Args:
            w1 (float): starting wavelength to fit in units of `self.u_l`
            w2 (float): ending wavelength to fit in units of `self.u_l`
        
        Keyword Args:
            degree (int): degree of polynomial to fit

        Returns:
            (np.array): flux of the continuum. Shape is same as self.l
        """
        continuum = get_continuum(self.l,self.f,w1,w2, **kwargs) * self.u_f
        return continuum
    def dopshift(self,v):
        """Doppler Shift

        Shift the spectrum by some radial velocity

        Args:
            v (float): velocity in km/s
        
        Returns:
            (Spec): Doppler shifted spectrum
        """
        ckms = 299792.458 # c in km/s
        l = self.l * (1+(v/ckms))
        new_spec = deepcopy(self)
        new_spec.l = l
        return new_spec
    def air(self):
        """Vacuum to Air
        
        Convert a spectrum from vacuum to air.

        Args:
            None

        Returns:
            (Spec): Converted spectrum
        """
        new_spec = deepcopy(self)
        new_spec.l = to_air(self.l*self.u_l).value
        return new_spec
    def smooth(self,shape):
        """Smooth
        
        Smooth with a boxcar function

        Args:
            shape (int): size of boxcar in pixels
        
        Returns:
            (Spec): Smoothed spectrum
        """
        new_spec = deepcopy(self)
        new_spec.l = np.convolve(self.l,np.ones(shape)/shape,mode='same')
        return new_spec
    def ew(self,w1,w2,W1,W2,**kwargs):
        """Equivalent Width

        Get the equivalent width of a region of spectrum.

        Args:
            w1 (float): starting wavelength of EW region in units of `self.u_l`
            w2 (float): ending wavelength of EW region in units of `self.u_l`
            W1 (float): starting wavelength of continuum region in units of `self.u_l`
            W2 (float): ending wavelength of continuum region in units of `self.u_l`

        Keyword Args:
            plot (bool): whether or not to plot the line and region. Default `False`
        
        Returns:
            (astropy.Quantity): Equivalent width of the region

        """
        ew = equivalent_width(self.l,self.f,self.continuum(W1,W2,**kwargs).value,w1,w2,self.u_l,**kwargs)
        #error = equivalent_width_error()
        
        return ew
    def regions(self,*wavelengths):
        """Select Regions

        Choose regions of the spectrum to keep, discarding the rest.

        Args:
            wavelength (array-like of type astropy.Quantity): start-stop points for each region. Must have even length.

        Returns:
            (Spec): Spectrum containing only selected regions
        """
        if len(wavelengths) % 2 != 0:
            raise ValueError('Spec.regions must have an even number of arguments')
        x = self.l * self.u_l
        n_regions = int(len(wavelengths)/2)
        reg = np.zeros(len(x)).astype('bool')
        for i in range(n_regions):
            reg = ((x >= wavelengths[2*i]) & (x <= wavelengths[2*i+1])) | (reg)
        return Spec(self.l[reg],self.f[reg],u_l = self.u_l,u_f = self.u_f,stype = self.stype,hdr=self.hdr)
    def set_snr(self,l0,snr0,thrux=None,thruy=None,snr_max=np.inf):
        """Set SNR

        Add noise to the spectrum to simulate an observation. The signal-to-noise ratio (SNR) is set at a reference wavelength and from there
        scales with the square root of the flux according to Poisson satistics.
        Optionally, a description of the instrument throughput can be given and applied to the SNR calculation.
        This is best when used in conjunction with an exposure time calculator.

        Args:
            l0 (float): wavelength of the reference SNR in units of `self.u_l`
            snr0 (float): reference SNR in SNR per pixel. For SNR per resolution element, smooth first (e.g. self.smooth(5) for COS)
        
        Keyword Args:
            thrux (np.array): wavelength values of throughput data in `self.u_l`. Default `None`
            thruy (np.array): thoughput at each wavelength point in `thrux`. Only the relative value is important.
            snr_max (float): the maximum SNR achievable with this instrument. e.g. for COS use 45
        
        Returns:
            (Spec): spectrum with noise added
        """
        f0 = interp1d(self.l,self.f)(l0)
        frac = self.f/f0
        if (thrux is not None) and (thruy is not None):
            throughput = interp1d(thrux,thruy,bounds_error=False,fill_value=0)
            frac = frac * throughput(self.l) / throughput(l0)
        snr = snr0 * np.sqrt(frac)
        snr[snr > snr_max] = snr_max
        sigma = self.f/snr
        noise = np.random.normal(scale=sigma)
        new_spec = deepcopy(self)
        new_spec.f = new_spec.f + noise
        return new_spec
    def fnu(self):
        """Fnu

        Convert from Flambda to Fnu.
        This function should not allow a spectrum that is already in Fnu to be converted.

        Args:
            None

        Returns:
            (Spec): new spectrum in Fnu
        """
        l,f = self.show()
        fnu = (f * l**2 / const.c).to(u.mJy)
        new_spec = deepcopy(self)
        new_spec.f = fnu.value
        new_spec.u_f = fnu.unit
        return new_spec
    def yoffset(self,dy):
        """y-offset

        Add a constant offset to the spectra. For plotting only. This does not preserve equivalent width.

        Args:
            dy (float): constant to be added to the flux in units of `self.u_f`

        Returns:
            (Spec): new spectrum with offset applied
        """
        new_spec = deepcopy(self)
        new_spec.f = new_spec.f + dy
        return new_spec
    def resample(self,lnew):
        """Resample
        
        Resample the flux at new wavelengths using `scipy.interp1d`

        Args:
            lnew(np.array): new wavelength values to sample at in units of `self.u_l`
        
        Returns:
            (Spec): new resampled spectrum
        """
        new_spec = deepcopy(self)
        func = interp1d(self.l,self.f,bounds_error=False,fill_value=np.nan)
        new_spec.l = lnew
        new_spec.f = func(lnew)
        return new_spec
    def combine(self,*others):
        """Combine

        Combine multiple Spec objects to increase snr in a wavelength range that they share

        Args:
            others (list of type Spec): Spec objects to be combined with `self`

        Returns:
            (Spec): new combined spectrum
        """
        new_spec = deepcopy(self)
        n_points = np.ones(len(new_spec.l))
        for spec in others:
            otherspec = interp1d(spec.l,spec.f,bounds_error=False,fill_value=0)(new_spec.l)
            new_spec.f = new_spec.f + otherspec
            n_points = n_points + (otherspec != 0).astype('int32')
            
        new_spec.f = new_spec.f/n_points
        return new_spec
    def gauss(self,R=None,sigma=None):
        """Gauss
        Convolve to instrument resolutuion with gaussian.
        May have to convert to evenly spaced x axis for this to work
        not implemented yet

        Args:
            None
    
        Keyword Args:
            R (float): Resolving power of convolution.
            sigma (float): resolution of convolution in units of `self.u_l`
        
        Returns:
            None -- Raises NotImplementedError
        """
        raise NotImplementedError
        if (R==None) and (sigma==None):
            raise ValueError('Please specify either the resolving power R or the resolution sigma')
        if (sigma == None):
            sigma = self.l * self.u_l / R /2.385
        else:
            if not isinstance(sigma,u.quantity.Quantity):
                sigma = sigma * self.u_l
        

def readfits(filename,u_l = u.Unit('Angstrom'),u_f = u.Unit('erg cm-2 s-1 Angstrom-1'),stype='data'):
    """Read fits

    Read a spectrum from a fits file

    Args:
        filename (str): path to the file

    Keyword Args:
        u_l (str): unit of wavelength axis. Should be readable by astropy.units.Unit(). Default `'Angstrom'`
        u_f (str): unit of flux axis. Should be readable by astropy.units.Unit(). Default `'erg cm-2 s-1 Angstrom-1'`
        stype (str): type of information stored. Either `'data'` or `'model'`. This is important for finding the continuum
        and normalization. Default `'data'`
    
    Returns:
        (Spec): Spectrum contained in fits file
    """
    im = fits.open(filename)
    hdr = im['Primary'].header
    ctype = hdr['CTYPE1']
    return Spec(*fits_to_np(filename),u_l=u_l,u_f=u_f,stype=stype,hdr=fits.getheader(filename))

def read(filename,**kwargs):
    """Read file

    Read a spectrum from a fits file or a two-column file

    Args:
        filename (str): path to the file

    Keyword Args:
        u_l (str): unit of wavelength axis. Should be readable by astropy.units.Unit(). Default `'Angstrom'`
        u_f (str): unit of flux axis. Should be readable by astropy.units.Unit(). Default `'erg cm-2 s-1 Angstrom-1'`
        stype (str): type of information stored. Either `'data'` or `'model'`. This is important for finding the continuum
        and normalization. Default `'data'`
        col (str): column of psg file to read. Default `'Total'`
    
    Returns:
        (Spec): Spectrum contained in fits file
    """
    try:
        if 'fits' in filename:
            return readfits(filename,**kwargs)
        elif ('rad' in filename.lower()) or ('psg' in filename.lower()):
            a,k = read_psg(filename,kwargs.get('col','Total'))
            return Spec(*a,**k)
        else:
            a,k = read_two_column(filename)
            return Spec(*a,**k)
    except ValueError:
        a,k = read_two_column(filename)
        return Spec(*a,**k)






