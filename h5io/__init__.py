"""Python Objects Onto HDF5
"""


from ._h5io import (read_hdf5, write_hdf5, _TempDir,  # noqa, analysis:ignore
                    object_diff, list_file_contents)  # noqa, analysis:ignore

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
