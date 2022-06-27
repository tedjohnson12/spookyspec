import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from copy import deepcopy
import matplotlib.pyplot as plt
from .tools import get_continuum, equivalent_width, to_air#,equivalent_width_error
from .fits import fits_to_np
from .fits import read_two_column
from scipy.interpolate import interp1d

class Spec:
    """
    class containing 1D spectral data
    This class was designed to be easy to use, especially to create figures
    spectra are transformed by appending class methods to the variable. For example,
    to redshift a spectrum spec by 10 km/s, call spec.dopshift(10)
    self.show() returns the axes as astropy.Quantity objects
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
        """
        return axes as quantity objects
        """
        return self.l * self.u_l, self.f * self.u_f
    def __str__(self):
        s = 'Spec %s from %.3f to %.3f %s' % (self.stype,self.l.min(),self.l.max(),str(self.u_l))
        return s
    def scale(self,k):
        """
        scale spectrum, preserving EW
        """
        new_spec = deepcopy(self)
        new_spec.f = self.f * k
        return new_spec
    def new_units(**kwargs):
        """
        change units in either axis
        """
        new_spec = deepcopy(self)
        if 'u_l' in kwargs:
            new_spec.u_l = u.Unit(kwargs['u_l'])
        if 'u_f' in kwargs:
            new_spec.u_f = u.Unit(kwargs['u_f'])
        return new_spec
    def quick_plot(self,w1,w2):
        """
        a quick view of the spectrum for informational purposes
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
        """
        Normalize a region of the spectrum. Great of comparing a model to uncalibrate echelle spectra.
        """
        new_spec = deepcopy(self)
        continuum = self.continuum(w1,w2,**kwargs)
        new_spec.f = self.f / continuum.value
        new_spec.u_f = u.Unit('')
        return new_spec
    def continuum(self,w1,w2,**kwargs):
        """
        get a plynomial fit of the continuum
        """
        continuum = get_continuum(self.l,self.f,w1,w2, **kwargs) * self.u_f
        return continuum
    def dopshift(self,v):
        """
        float v in km/s
        """
        ckms = 299792.458 # c in km/s
        l = self.l * (1+(v/ckms))
        f = self.f
        return Spec(l,f,u_l = self.u_l,u_f = self.u_f)
    def air(self):
        """
        convert spec from vacuum to air.
        """
        new_spec = deepcopy(self)
        new_spec.l = to_air(self.l*self.u_l).value
        return new_spec
    def smooth(self,shape):
        """
        smooth with boxcar of size shape
        """
        new_spec = deepcopy(self)
        new_spec.l = np.convolve(self.l,np.ones(shape)/shape,mode='same')
        return new_spec
    def ew(self,w1,w2,W1,W2,**kwargs):
        """
        get the equivalent width of a region of spectrum.
        """
        ew = equivalent_width(self.l,self.f,self.continuum(W1,W2,**kwargs).value,w1,w2,self.u_l,**kwargs)
        #error = equivalent_width_error()
        
        return ew
    def regions(self,*args):
        """
        *args is a list of length quantities in top,start order. Purposes of removing bad regions from a spectrum.
        """
        if len(args) % 2 != 0:
            raise ValueError('Spec.regions must have an even number of arguments')
        x = self.l * self.u_l
        n_regions = int(len(args)/2)
        reg = np.zeros(len(x)).astype('bool')
        for i in range(n_regions):
            reg = ((x >= args[2*i]) & (x <= args[2*i+1])) | (reg)
        return Spec(self.l[reg],self.f[reg],u_l = self.u_l,u_f = self.u_f,stype = self.stype)
    def set_snr(self,l0,snr0,thrux=None,thruy=None,snr_max=np.inf):
        """
        add noise to a spectrum to simulate an observation given SNR=snr0 and wavelenght l0
        if instrument throughput is known, set thrux and thruy to arrays of throughput
        this can easily be done for HST with the ETC
        
        If there is a certain maximum snr achievable on a certain instrument, then set that value to snr_max
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
        """
        convert to fnu
        """
        l,f = self.show()
        fnu = (f * l**2 / const.c).to(u.mJy)
        new_spec = deepcopy(self)
        new_spec.f = fnu.value
        new_spec.u_f = fnu.unit
        return new_spec
    def yoffset(self,dy):
        """
        add a linear offset to the spectra. For plotting only. This does not preserve equivalent width
        """
        new_spec = deepcopy(self)
        new_spec.f = new_spec.f + dy
        return new_spec
    def resample(self,lnew):
        """
        resample the flux at new wavelengths
        """
        new_spec = deepcopy(self)
        func = interp1d(self.l,self.f,bounds_error=False,fill_value=np.nan)
        new_spec.l = lnew
        new_spec.f = func(lnew)
        return new_spec
    def combine(self,*others):
        """
        combine multiple Spec objects to increase snr
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
        """
        convolve to instrument resolutuion with gaussian
        may have to convert to evenly spaced x axis for this to work
        not implemented yet
        """
        raise NotImplementedError
        if (R==None) and (sigma==None):
            raise ValueError('Please specify either the resolving power R or the resolution sigma')
        if (sigma == None):
            sigma = self.l * self.u_l / R /2.385
        else:
            if not isinstance(sigma,u.quantity.Quantity):
                sigma = sigma * self.u_l
        
    
class MultiSpec:
    """
    similar to list of Spec objects
    Not as complete, not as functional
    """
    def __init__(self,ls,fs,u_l = u.Unit('Angstrom'),u_f = u.Unit('erg cm-2 s-1 Angstrom-1'),hdr=None,stype='data'):
        """
        ls is list of l
        fs is list of f
        """
        assert ls.shape == fs.shape
        self.ls = ls
        self.fs = fs
        self.u_l = u_l
        self.u_f = u_f
        self.stype = stype
        self.hdr = hdr
        self.n_orders = ls.shape[0]
        self.n_pix = ls.shape[1]
    def show(self,order):
        return self.ls[order] * self.u_l, self.fs[order] * self.u_f
    def __str__(self):
        s = 'MultiSpec %s from %.3f to %.3f %s' % (self.stype,self.ls.min(),self.ls.max(),str(self.u_l))
    def scale(self,k):
        new_spec = deepcopy(self)
        new_spec.fs = self.fs * k
        return new_spec
    def new_units(**kwargs):
        new_spec = deepcopy(self)
        if 'u_l' in kwargs:
            new_spec.u_l = u.Unit(kwargs['u_l'])
        if 'u_f' in kwargs:
            new_spec.u_f = u.Unit(kwargs['u_f'])
        return new_spec
    def quick_plot(self,order,w1,w2):
        plt.plot(*self.show(order),c='k')
        # get y limit
        reg = (w1<=self.ls) & (w2>=self.ls)
        ymin = self.fs[reg].min()
        ymax = self.fs[reg].max()
        plt.ylim(ymin*0.9,ymax*1.1)
        plt.xlim(w1,w2)
        plt.xlabel(str(self.u_l))
        plt.ylabel(str(self.u_f))
        plt.show()
    def continuum(self,order,w1,w2,**kwargs):
        continuum = get_continuum(self.ls[order],self.fs[order],w1,w2, **kwargs) * self.u_f
        return continuum
    def normalize(self,order,w1,w2,**kwargs):
        new_spec = Spec(self.ls[order],self.fs[order],u_l=self.u_l,u_f=self.u_f,hdr=self.hdr,stype=self.stype)
        continuum = self.continuum(order,w1,w2,**kwargs)
        new_spec.f = self.fs[order] / continuum.value
        new_spec.u_f = u.Unit('')
        return new_spec
    def dopshift(self,v):
        """
        int v in km/s
        """
        new_spec = deepcopy(self)
        ckms = 299792.458 # c in km/s
        new_spec.ls = self.ls * (1+(v/ckms))
        new_spec.fs = self.fs
        #new_spec.hdr['dopshift'] = v
        return newspec
    def air(self):
        new_spec = deepcopy(self)
        new_spec.ls = to_air(self.ls * self.u_l).value
        return new_spec
    def smooth(self,shape):
        new_spec = deepcopy(self)
        new_spec.ls = np.convolve(self.ls,np.array([1]*shape)/shape,mode='same')
        return new_spec
    def specshift(self,x):
        """
        int v in km/s
        """
        new_spec = deepcopy(self)
        ckms = 299792.458 # c in km/s
        new_spec.ls = self.ls + x
        #new_spec.hdr['dopshift'] = v
        return new_spec
    def ew(self,order,w1,w2,W1,W2,**kwargs):
        """
        get the equivalent width of a region of spectrum.
        """
        ew = equivalent_width(self.ls[order],self.fs[order],self.continuum(order,W1,W2,**kwargs).value,w1,w2,self.u_l,**kwargs)
        #error = equivalent_width_error()
        return ew
    def regions(self,order,*args):
        """
        *args is a list of length quantities in top,start order. Purposes of removing bad regions from a spectrum.
        """
        if len(args) % 2 != 0:
            raise ValueError('Spec.regions must have an even number of arguments after order')
        x = self.ls[order] * self.u_l
        n_regions = int(len(args)/2)
        reg = np.zeros(len(x)).astype('bool')
        for i in range(n_regions):
            reg = ((x >= args[2*i]) & (x <= args[2*i+1])) | (reg)
        return Spec(self.ls[order][reg],self.fs[order][reg],u_l = self.u_l,u_f = self.u_f,stype = self.stype,hdr=self.hdr)

#The below function is not mine, but I have written similar functions for the same purpose.
# I have doubts about this function being accurate, and I think it tends to mess up the dispersion function
# when reading multispec fits files. This issue is persitent, and I have found no solution (or real cause).
# currently, it is best to break multispec files up into onedspec images or combine them all in PyRAF
def nonlinearwave(nwave, specstr, verbose=False):
    """Compute non-linear wavelengths from multispec string
    
    Returns wavelength array and dispersion fields.
    Raises a ValueError if it can't understand the dispersion string.
    
    COPIED from https://github.com/kgullikson88/General/blob/master/readmultispec.py
    
    """

    fields = specstr.split()
    if int(fields[2]) != 2:
        raise ValueError('Not nonlinear dispersion: dtype=' + fields[2])
    if len(fields) < 12:
        raise ValueError('Bad spectrum format (only %d fields)' % len(fields))
    wt = float(fields[9])
    w0 = float(fields[10])
    ftype = int(fields[11])
    if ftype == 3:

        # cubic spline

        if len(fields) < 15:
            raise ValueError('Bad spline format (only %d fields)' % len(fields))
        npieces = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            print('Dispersion is order-%d cubic spline' % npieces)
        if len(fields) != 15 + npieces + 3:
            raise ValueError('Bad order-%d spline format (%d fields)' % (npieces, len(fields)))
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        s = (np.arange(nwave, dtype=float) + 1 - pmin) / (pmax - pmin) * npieces
        j = s.astype(int).clip(0, npieces - 1)
        a = (j + 1) - s
        b = s - j
        x0 = a ** 3
        x1 = 1 + 3 * a * (1 + a * b)
        x2 = 1 + 3 * b * (1 + a * b)
        x3 = b ** 3
        wave = coeff[j] * x0 + coeff[j + 1] * x1 + coeff[j + 2] * x2 + coeff[j + 3] * x3

    elif ftype == 1 or ftype == 2:

        # chebyshev or legendre polynomial
        # legendre not tested yet

        if len(fields) < 15:
            raise ValueError('Bad polynomial format (only %d fields)' % len(fields))
        order = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            if ftype == 1:
                print( 'Dispersion is order-%d Chebyshev polynomial' % order)
            else:
                print( 'Dispersion is order-%d Legendre polynomial (NEEDS TEST)' % order)
        if len(fields) != 15 + order:
            # raise ValueError('Bad order-%d polynomial format (%d fields)' % (order, len(fields)))
            if verbose:
                print( 'Bad order-%d polynomial format (%d fields)' % (order, len(fields)))
                print( "Changing order from %i to %i" % (order, len(fields) - 15))
            order = len(fields) - 15
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        pmiddle = (pmax + pmin) / 2
        prange = pmax - pmin
        x = (np.arange(nwave, dtype=float) + 1 - pmiddle) / (prange / 2)
        p0 = np.ones(nwave, dtype=float)
        p1 = x
        wave = p0 * coeff[0] + p1 * coeff[1]
        for i in range(2, order):
            if ftype == 1:
                # chebyshev
                p2 = 2 * x * p1 - p0
            else:
                # legendre
                p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
            wave = wave + p2 * coeff[i]
            p0 = p1
            p1 = p2

    else:
        raise ValueError('Cannot handle dispersion function of type %d' % ftype)

    return wave, fields

# The below function has the same problems as the above
def readmultispec(fitsfile, reform=True, quiet=False,u_l = u.Unit('Angstrom'),u_f = u.Unit('erg cm-2 s-1 Angstrom-1'),stype='data'):
    """Read IRAF echelle spectrum in multispec format from a FITS file
    
    Can read most multispec formats including linear, log, cubic spline,
    Chebyshev or Legendre dispersion spectra
    
    If reform is true, a single spectrum dimensioned 4,1,NWAVE is returned
    as 4,NWAVE (this is the default.)  If reform is false, it is returned as
    a 3-D array.
    
    COPIED AND EDITED from https://github.com/kgullikson88/General/blob/master/readmultispec.py
    """

    fh = fits.open(fitsfile)
    try:
        header = fh[0].header
        flux = fh[0].data
    finally:
        fh.close()
    temp = flux.shape
    nwave = temp[-1]
    if len(temp) == 1:
        nspec = 1
    else:
        nspec = temp[-2]

    # first try linear dispersion
    try:
        crval1 = header['crval1']
        crpix1 = header['crpix1']
        cd1_1 = header['cd1_1']
        ctype1 = header['ctype1']
        if ctype1.strip() == 'LINEAR':
            wavelen = np.zeros((nspec, nwave), dtype=float)
            ww = (np.arange(nwave, dtype=float) + 1 - crpix1) * cd1_1 + crval1
            for i in range(nspec):
                wavelen[i, :] = ww
            # handle log spacing too
            dcflag = header.get('dc-flag', 0)
            if dcflag == 1:
                wavelen = 10.0 ** wavelen
                if not quiet:
                    print( 'Dispersion is linear in log wavelength')
            elif dcflag == 0:
                if not quiet:
                    print( 'Dispersion is linear')
            else:
                raise ValueError('Dispersion not linear or log (DC-FLAG=%s)' % dcflag)

            if nspec == 1 and reform:
                # get rid of unity dimensions
                flux = np.squeeze(flux)
                wavelen.shape = (nwave,)
            return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': None}
    except KeyError:
        pass

    # get wavelength parameters from multispec keywords
    try:
        wat2 = header['wat2_*']
        count = len(wat2)
    except KeyError:
        raise ValueError('Cannot decipher header, need either WAT2_ or CRVAL keywords')

    # concatenate them all together into one big string
    watstr = []
    for i in range(len(wat2)):
        # hack to fix the fact that older pyfits versions (< 3.1)
        # strip trailing blanks from string values in an apparently
        # irrecoverable way
        # v = wat2[i].value
        v = wat2[i]
        v = v + (" " * (68 - len(v)))  # restore trailing blanks
        watstr.append(v)
    watstr = ''.join(watstr)

    # find all the spec#="..." strings
    specstr = [''] * nspec
    for i in range(nspec):
        sname = 'spec' + str(i + 1)
        p1 = watstr.find(sname)
        p2 = watstr.find('"', p1)
        p3 = watstr.find('"', p2 + 1)
        if p1 < 0 or p1 < 0 or p3 < 0:
            raise ValueError('Cannot find ' + sname + ' in WAT2_* keyword')
        specstr[i] = watstr[p2 + 1:p3]

    wparms = np.zeros((nspec, 9), dtype=float)
    w1 = np.zeros(9, dtype=float)
    for i in range(nspec):
        w1 = np.asarray(specstr[i].split(), dtype=float)
        wparms[i, :] = w1[:9]
        if w1[2] == -1:
            raise ValueError('Spectrum %d has no wavelength calibration (type=%d)' %
                             (i + 1, w1[2]))
            # elif w1[6] != 0:
            #    raise ValueError('Spectrum %d has non-zero redshift (z=%f)' % (i+1,w1[6]))

    wavelen = np.zeros((nspec, nwave), dtype=float)
    wavefields = [None] * nspec
    for i in range(nspec):
        # if i in skipped_orders:
        #    continue
        verbose = (not quiet) and (i == 0)
        if wparms[i, 2] == 0 or wparms[i, 2] == 1:
            # simple linear or log spacing
            wavelen[i, :] = np.arange(nwave, dtype=float) * wparms[i, 4] + wparms[i, 3]
            if wparms[i, 2] == 1:
                wavelen[i, :] = 10.0 ** wavelen[i, :]
                if verbose:
                    print( 'Dispersion is linear in log wavelength')
            elif verbose:
                print( 'Dispersion is linear')
        else:
            # non-linear wavelengths
            wavelen[i, :], wavefields[i] = nonlinearwave(nwave, specstr[i],
                                                         verbose=verbose)
        wavelen *= 1.0 + wparms[i, 6]
        if verbose:
            print( "Correcting for redshift: z=%f" % wparms[i, 6])
    if nspec == 1 and reform:
        # get rid of unity dimensions
        flux = np.squeeze(flux)
        wavelen.shape = (nwave,)
    return MultiSpec(wavelen,flux[0],u_l=u_l,u_f=u_f,hdr=header,stype=stype)
    #return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': wavefields}


def readfits(filename,u_l = u.Unit('Angstrom'),u_f = u.Unit('erg cm-2 s-1 Angstrom-1'),stype='data'):
    """
    read a fits file, return a Spec or MultiSpec objet
    """
    im = fits.open(filename)
    hdr = im['Primary'].header
    ctype = hdr['CTYPE1']
    if 'MULTI' in ctype:
        return readmultispec(filename,u_l=u_l,u_f=u_f,stype=stype)
    else:
        return Spec(*fits_to_np(filename),u_l=u_l,u_f=u_f,stype=stype,hdr=fits.getheader(filename))
def read(filename,**kwargs):
    """
    decide whether a file is fits or two column, and read it
    """
    try:
        name,ext = filename.split('.')
        if (ext.lower()=='fits') or (ext.lower()=='fit'):
            return readfits(filename,**kwargs)
        else:
            a,k = read_two_column(filename)
            return Spec(*a,**k)
    except ValueError:
        a,k = read_two_column(filename)
        return Spec(*a,**k)






