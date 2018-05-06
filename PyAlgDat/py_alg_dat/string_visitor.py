#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (C) 2015 by Brian Horn, trycatchhorn@gmail.com.
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
Provides a visitor which turns the objects visited into a string.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

from py_alg_dat.visitor import Visitor

class StringVisitor(Visitor):
    """
    Visitor that accumulates visited items in a string.
    """

    def __init__(self):
        """
        (Container.StrVisitor) -> None
        Constructor.
        """
        super(StringVisitor, self).__init__()
        self.string = ""
        self.comma = False

    def visit(self, obj):
        """
        (Container.StrVisitor, Object) -> None
        Appends the given object to the string of visited items.
        """
        if self.comma:
            self.string = self.string + ", "
        self.string = self.string + str(obj)
        self.comma = True

    def __str__(self):
        """
        (Container.StrVisitor) -> string
        Returns the string of visited items.
        """
        return self.string
