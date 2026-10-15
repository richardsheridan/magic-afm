# Copyright (C) Richard J. Sheridan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os


def get_int_env_var(var_name):
    """
    Retrieves an integer environment variable, returning None on missing values and
    conversion errors.
    """

    try:
        return int(os.environ[var_name])
    except (ValueError, KeyError):
        return None


def max_workers():
    """
    Worker count from MAFM_MAX_CPUS, else physical cores if psutil is
    installed, else os.cpu_count().
    """
    if n := get_int_env_var("MAFM_MAX_CPUS"):
        return n
    try:
        import psutil
    except ImportError:
        pass
    else:
        if n := psutil.cpu_count(logical=False):
            return n
    return os.cpu_count() or 1
