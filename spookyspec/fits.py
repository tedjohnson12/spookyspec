import numpy as np
from astropy.io import fits
from pandas import read_csv
from astropy import units as u

def fits_to_np(spec):
    """Convert from fits to numpy

    Convert a file from fits format to two numpy arrays.
    
    Args:
        spec (str): path to .fits file containing data

    Returns:
        lambda (np.array): wavelength points
        flux (np.array): flux points
    """
    f = fits.getdata(spec)
    h = fits.getheader(spec)
    l = np.arange(len(f), dtype='float64')
    l *= float(h['cdelt1']) #scale to delta lambda
    l += float(h['crval1'] - h['crpix1'] * h['cdelt1'])
    return (l,f)


def get_image_data(filename,order):
    """Get an order from a multispec fits file

    Get the information from a single order in a fits file containing multiple orders
    of data (e.g. from an echelle spectrograph)

    Args:
        filename (str): Path to the .fits file containing the data
        order (int): echelle order to pull data from
    
    Returns
        l (np.array): wavelength points
        f (np.array): flux points
        """
    hdu = fits.open(filename)[0] #first from hdulist object
    
    #get parameters from hdr
    i=1
    s = ''
    while True:
        try:
            if i < 10:
                s+=hdu.header['WAT2_00%i' % i]
            else:
                s+=hdu.header['WAT2_0%i' % i]
        except KeyError:
            break
        i+=1
    s = s.split('spec')
    params = s[order+2].split('"')[1].split()
    l0 = float(params[3])
    dl = float(params[4])
    N = int(float(params[5]))
    
    l = np.arange(N) * dl + l0
    f = hdu.data[0,order,:]
    
    return l,f

def read_two_column(filename):
    """Read a two column spectrum

    Read a two column file. pandas with the python regex parsing engine is used so that the read in is 'smart'
    i.e. you don't have to worry about whitespace so long as the file is whitespace delimited.
    This function also stores the fits header. That portion of the code was written by Beth Klein.

    Args
        filename (str): path to file containing data

    Returns
        args (2-tuple of type np.array): arguments of the Spec class
        kwargs (dict): keyword arguments of the Spec class    
    
    Example
        Use the `lambda    flux` format as follows::

            Header
            ...
            END
            920.00003    6.58511E+09
            920.00472    6.58510E+09
            920.00942    6.58509E+09
            920.01411    6.58507E+09
            ...
    """
    hdr = fits.header.Header()
    header = True
    skip=0
    for line in open(filename,'r'):
        line = line.strip()
        if line[0:3] == 'END':
            skip+=1
            break
        if header:
            skip+=1
            key = line[0:8]
            value = line[10:len(line)].strip()
            if value.startswith("'"):    # GET RID OF EXTRA STRING QUOTES
                value = value[1:-1]
            if key.find('COMMENT') == 0:  # zero for the position of the found string
                hdr['COMMENT'] = value
            if key.find('TEFF') == 0:
                hdr['TEFF'] = int(value)
            if key.find('LOG_G') == 0:
                hdr['LOG_G'] = float(value)
            if key.find('TTYPE1') == 0:
                hdr['AXIS1TYP'] = str(value)
            if key.find('TUNIT1') == 0:
                hdr['UNIT1'] = str(value)
            if key.find('TTYPE2') == 0:
                hdr['AXIS2TYP'] = str(value)
            if key.find('TUNIT2') == 0:
                hdr['UNIT2'] =  value
    df = read_csv(filename,engine='python',skiprows=skip,sep=' +',names=['lam','flux'])
    args = (np.array(df['lam']),np.array(df['flux']))
    kwargs = {'stype':'model','u_l' : u.Unit(hdr['UNIT1']),'u_f' : u.Unit(hdr['UNIT2']),'hdr' : hdr}
    return args,kwargs


def read_psg(filename,col):
    """ Read PSG-style output

    Read a spectrum output from the Planetary Spectrum Generator (psg.gsfc.nasa.gov)

    Args:
        filename (str): path to the file containing the data
        col (str): column the rad file to read
    """
    u_l = ''
    u_f = ''
    names = []
    with open(filename,'r') as file:
        for line in file:
            if 'Spectral unit:' in line:
                u_l = line.replace(']','[').split('[')[1]
            elif 'Radiance unit:' in line:
                u_f = line.replace(']','[').split('[')[1]
            elif 'Wave/freq' in line:
                names = line[2:].split(' ')
    df = read_csv(filename,sep = ' +', engine='python',comment='#',names = names)
    if u_l == 'ppm':
        u_l = ''
    args = (np.array(df['Wave/freq']),np.array(df[col]))
    kwargs = {'stype':'model','u_l' : u.Unit(u_l),'u_f' : u.Unit(u_f),'hdr' : None}