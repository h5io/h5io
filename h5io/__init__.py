"""Python Objects Onto HDF5
"""

__version__ = '0.1.0'

from ._h5io import (read_hdf5, write_hdf5, _TempDir,  # noqa, analysis:ignore
                    object_diff, list_file_contents)  # noqa, analysis:ignore
