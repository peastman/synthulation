# Copyright (c) 2016 Peter Eastman
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class Connector(object):
    """Adapter for reading the output of modules in fixed sized increments.

    A module is free to produce output in arbitrary sized blocks.  In some cases that can be inconvenient to work with,
    and it is easier if the receiver can dictate how much data it wants to receive.  A Connector acts as an adapter to
    allow that.
    """
    def __init__(self, input):
        """Construct a Connector for reading data from a module.

        Parameters
        ----------
        input : module
            the input module from which data will be read
        """
        self._input = input
        self._inputBuffer = None
        self._inputPos = 0

    def fillBuffer(self, buffer):
        """Read data from the input module.

        Parameters
        ----------
        buffer : NumPy array
            a buffer to hold the data.  Exactly enough data is read to fill the buffer.
        """
        pos = 0
        while pos < len(buffer):
            if self._inputBuffer is None or self._inputPos == len(self._inputBuffer):
                self._inputBuffer = self._input.getData()
                self._inputPos = 0
            elementsToCopy = min(len(self._inputBuffer)-self._inputPos, len(buffer)-pos)
            buffer[pos:pos+elementsToCopy] = self._inputBuffer[self._inputPos:self._inputPos+elementsToCopy]
            pos += elementsToCopy
            self._inputPos += elementsToCopy
