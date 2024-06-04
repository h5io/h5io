"""Python Objects Onto HDF5"""

from ._h5io import (
    read_hdf5,
    write_hdf5,
    object_diff,
    list_file_contents,
)  # noqa, analysis:ignore
from ._version import __version__
