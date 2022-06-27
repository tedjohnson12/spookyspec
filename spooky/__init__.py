__version__ = '1.0.0'

from .fits import fits_to_np
from .fits import read_two_column
from .fits import get_image_data
from .Spec import Spec
from .Spec import MultiSpec
from .Spec import readfits
from .Spec import read
from .tools import redshift
from .tools import to_air
from .tools import get_continuum_points
#from .tools import equivalent_width_error
from .tools import equivalent_width

print('Loading Specplot')
