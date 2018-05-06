#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2015 by Brian Horn, trycatchhorn@gmail.com.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The configuration of the PyAlgDat package
"""

from setuptools import setup
from setuptools import find_packages

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

# Get suff in README.rst
with open('README.rst') as readme_file:
    README = readme_file.read()

# Get stuff in CHANGES.rst
with open('CHANGES.rst') as changes_file:
    CHANGES = changes_file.read().replace('.. :changelog:', '')

# Get stuff in LICENSE.txt
with open('LICENSE.txt') as license_file:
    MIT_LICENSE = license_file.read()

AUTHOR = __author__
AUTHOR_EMAIL = __email__
CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Developers',
               'Intended Audience :: Education',
               'Intended Audience :: Information Technology',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Telecommunications Industry',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'License :: OSI Approved :: MIT License',
               'Natural Language :: English',
               'Operating System :: OS Independent',
               'Topic :: Scientific/Engineering',
               'Topic :: Software Development :: Libraries :: Python Modules']
DESCRIPTION = "Various data structures and algorithms."
DOWNLOAD_URL = "http://www.brianhorn.dk/projects/pyalgdat/release/PyAlgDat.tar.gz"
KEYWORDS = ['algorithms', 'data structures', 'graph', 'heap', 'linked list', 'partition']
LICENSE = MIT_LICENSE
LONG_DESCRIPTION = README + '\n\n' + CHANGES
PACKAGE_NAME = "PyAlgDat"
URL = "http://www.brianhorn.dk"
VERSION = __version__

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    include_package_data=True,
    zip_safe=False,
    download_url=DOWNLOAD_URL,
    keywords=KEYWORDS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=['build']),
    install_requires=["setuptools"],
    platforms=["any"],
    test_suite='testsuite'
)
