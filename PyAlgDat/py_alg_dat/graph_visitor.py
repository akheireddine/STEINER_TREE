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
Graph visitor that visits all vertices in a graph.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

from py_alg_dat.string_visitor import StringVisitor

class GraphVisitor(StringVisitor):

    """
    Visitor that accumulates visited items in a string.
    """

    def __init__(self):
        """
        Constructs a graph visitor that visits
        all the vertices in the graph.
        """
        super(GraphVisitor, self).__init__()

    def visit(self, v):
        """
        Appends the given vertex and all the edges emanating
        from that vertex to a string.

        @param v: The graph vertex.
        @type: L{GraphVertex}
        """
        self.string = self.string + str(v) + "\n"
        for edge in v.get_emanating_edges():
            self.string = self.string + "    " + str(edge) + "\n"


