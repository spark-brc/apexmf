apexmf
------

.. image:: https://img.shields.io/pypi/v/apexmf?color=blue
   :target: https://pypi.python.org/pypi/apexmf
   :alt: PyPI Version
.. image:: https://img.shields.io/pypi/l/apexmf?color=green
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: PyPI - License
.. image:: https://zenodo.org/badge/304147230.svg?color=orange
   :target: https://zenodo.org/badge/latestdoi/304147230

`apexmf` is a set of python modules for APEX-MODFLOW model (Bailey et al., 2021) parameter estimation and uncertainty analysis with the open-source suite PEST (Doherty 2010a and 2010b, and Doherty and other, 2010).

.. rubric:: Installation

.. code-block:: python
   
   >>> pip install apexmf


.. rubric:: Brief overview of the API

.. code-block:: python

   from apexmf import apexmf_pst_utils

   >>> wd = "User-APEX-MODFLOW working directory"
   >>> APEX_wd = "User-APEX working directory"
   >>> apexmf_pst_utils.init_setup(wd, APEX_wd)

   'apex.parm.xlsx' file copied ... passed
   'beopest64.exe' file copied ... passed
   'i64pest.exe' file copied ... passed
   'i64pwtadj1.exe' file copied ... passed
   'forward_run.py' file copied ... passed

