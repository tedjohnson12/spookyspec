__version__ = '1.0.0'

from spooky.fits import fits_to_np
from spooky.fits import read_two_column
from spooky.fits import get_image_data
from spooky.Spec import Spec
from spooky.Spec import readfits
from spooky.Spec import read
from spooky.tools import redshift
from spooky.tools import to_air
from spooky.tools import get_continuum_points
#from spooky.tools import equivalent_width_error
from spooky.tools import equivalent_width

print('Loading spooky')
