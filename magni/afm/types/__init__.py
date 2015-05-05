"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Subpackage providing data container classes for .mi files.

Routine listings
----------------
image
    Module providing data container classes for .mi image files.
spectroscopy
    Module providing data container classes for .mi spectroscopy files.
BaseClass(object)
    Base class of every `magni.afm.types` data class.
File(BaseClass)
    Base class of the `magni.afm.types` file classes.
FileCollection(BaseClass)
    Data class for collections of File instances with identical settings.

"""

# the BaseClass, File, and FileCollection classes need to be imported first to
# avoid recursive imports
from magni.afm.types._util import BaseClass
from magni.afm.types._util import File
from magni.afm.types._util import FileCollection

from magni.afm.types import image
from magni.afm.types import spectroscopy
