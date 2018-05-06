#!/usr/bin/env python

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
Provides a data structure for an association - a key/value pair.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class Association(object):

    """
    The interface of an association.
    """

    def __init__(self, *args):
        """
        Constructs an association as key/value pair in a tuple.

        @param arg1: The key of this association.
        @type arg1: C{object}
        @param arg2: The value of this association.
        @type arg2: C{object}
        @raises: ValueError if the number of arguments is not 1 or 2.
        @type: C{ValueError}
        """
        if len(args) == 1:
            self.tuple = (args[0], None)
        elif len(args) == 2:
            self.tuple = args
        else:
            raise ValueError

    def __repr__(self):
        """
        Returns the canonical representation of this association

        @return: Canonical string representation of the association.
        @rtype: C{str}
        """
        return repr((self.tuple[0], self.tuple[1]))

    def __str__(self):
        """
        Returns a string representation of this association

        @return: String representation of the association.
        @rtype: C{str}
        """

        class_name = str(self.__class__.__name__) + ": ("
        association_str = class_name + str(self.tuple[0]) + ", " + str(self.tuple[1]) + ")"
        return association_str

    def __hash__(self):
        """
        Returns the hash value of this aassociation.
        Note: both the key -and the value of the
        association are hashed.

        @return: Hash value of the association.
        @rtype: C{int}
        """
        return hash(self.get_key()) + hash(self.get_value())

    def __eq__(self, other):
        """
        Compares two associations for equality. The comparison
        is done by comparing both the key -and the value fields
        in the two associations.

        @param other: The other association.
        @type other: L{Association}
        @return: True if the associations are equal, false otherwise.
        @rtype: C{bool}
        """
        return (self.tuple[0], self.tuple[1]) == (other.tuple[0], other.tuple[1])

    def __ne__(self, other):
        """
        Compares two associations for inequality. The comparison
        is done by comparing both the key -and the value fields
        in the two associations.

        @param other: The other association.
        @type other: L{Association}
        @return: True if the associations are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this association.
        Note: the comparison is done by comparing only the keys
        in the two associations.

        @param other: The other association.
        @type other: L{Association}
        @return: True if this association is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.get_key() < other.get_key()

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for this association.
        Note: the comparison is done by comparing only the keys
        in the two associations.

        @param other: The other association.
        @type other: L{Association}
        @return: True if this association is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.get_key() <= other.get_key()

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this association.
        Note: the comparison is done by comparing only the keys
        in the two associations.

        @param other: The other association.
        @type other: L{Association}
        @return: True if this association is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.get_key() > other.get_key()

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for this association.
        Note: the comparison is done by comparing only the keys
        in the two associations.

        @param other: The other association.
        @type other: L{Association}
        @return: True if this association is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.get_key() >= other.get_key()

    def get_key(self):
        """
        Returns the key of this association.

        @return: The key of the association.
        @rtype: C{object}
        """
        return self.tuple[0]

    def get_value(self):
        """
        Returns the value of this association.

        @return: The value of the association.
        @rtype: C{object}
        """
        return self.tuple[1]



