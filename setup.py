from setuptools import setup, find_packages
import codecs
import os

# here = os.path.abspath(os.path.dirname(__file__))

# with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()


with open("README.rst", "r") as fd:
    long_desc = fd.read()


VERSION = '0.0.1'
DESCRIPTION = 'apexmf is a set of python modules for APEX-MODFLOW model evaluation and parameter estimation.'
# LONG_DESCRIPTION = 'A package that allows to work with apex-MODFLOW model'

license = "BSD-3-Clause"
# Setting up
setup(
    name="apexmf",
    version=VERSION,
    author="Seonggyu Park",
    author_email="<envpsg@gmail.com>",
    description=DESCRIPTION,
    long_description=long_desc,
    long_description_content_type="text/x-rst",
    download_url="https://pypi.org/project/apexmf",
    project_urls={
        "Bug Tracker": "https://github.com/spark-brc/apexmf/issues",
        "Source Code": "https://github.com/spark-brc/apexmf",
        "Documentation": "https://github.com/spark-brc/apexmf",
    },
    # include_package_data=True,
    package_data = {
        'opt_files': ['*'],
    },
    packages=find_packages(),
    install_requires=[
        'pandas', 'numpy', 'pyemu', 'flopy', 'scipy', 'matplotlib',
        'hydroeval', 'tqdm', 'termcolor'],
    keywords=['python', 'APEX-MODFLOW', 'PEST'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: BSD License"
    ]
)