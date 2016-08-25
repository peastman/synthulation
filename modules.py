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


from connector import Connector
import numpy as np
import scipy.signal
import math


class Adder(object):
    """A module whose output equals the sum of the data from all its inputs."""
    def __init__(self, *inputs):
        """ Construct an Adder module.

        Parameters
        ----------
        inputs : modules
            the input modules
        """
        self._connectors = [Connector(i) for i in inputs]
        bufferSize = 512
        self._buffer1 = np.empty((bufferSize,), np.float32)
        self._buffer2 = np.empty((bufferSize,), np.float32)

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        if len(self._connectors) == 0:
            return np.zeros((512,), np.float32)
        self._connectors[0].fillBuffer(self._buffer1)
        for i in range(1, len(self._connectors)):
            self._connectors[i].fillBuffer(self._buffer2)
            self._buffer1 += self._buffer2
        return self._buffer1


class Multiplier(object):
    """A module whose output equals the product of the data from all its inputs."""
    def __init__(self, *inputs):
        """ Construct a Multiplier module.

        Parameters
        ----------
        inputs : modules
            the input modules
        """
        self._connectors = [Connector(i) for i in inputs]
        bufferSize = 512
        self._buffer1 = np.empty((bufferSize,), np.float32)
        self._buffer2 = np.empty((bufferSize,), np.float32)

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        if len(self._connectors) == 0:
            return np.zeros((512,), np.float32)
        self._connectors[0].fillBuffer(self._buffer1)
        for i in range(1, len(self._connectors)):
            self._connectors[i].fillBuffer(self._buffer2)
            self._buffer1 *= self._buffer2
        return self._buffer1


class Sine(object):
    """A module that outputs a sine wave.

    Attributes
    ----------
    frequency : float
        the frequency of the sine wave in Hz
    amplitude : float
        the amplitude of the sine wave
    """
    def __init__(self, frequency=440.0, amplitude=1.0, sampleRate=44100):
        """Construct a Sine module

        Parameters
        ----------
        frequency : float
            the frequency of the sine wave in Hz
        amplitude : float
            the amplitude of the sine wave
        sampleRate : float
            the sample rate at which data should be generated in samples/second
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self._sampleRate = sampleRate
        self._offset = 0

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        bufferSize = 512
        period = self._sampleRate/self.frequency
        data = self.amplitude*np.sin((2*math.pi/period)*np.linspace(self._offset, self._offset+bufferSize, bufferSize, False))
        self._offset = (self._offset+bufferSize)%period
        return data


class Noise(object):
    """A module that outputs white noise.

    Attributes
    ----------
    amplitude : float
        the amplitude of the noise
    """
    def __init__(self, amplitude=1.0):
        """Construct a Noise module.

        Parameters
        ----------
        amplitude : float
            the amplitude of the noise
        """
        self.amplitude = amplitude

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        return self.amplitude*(np.random.uniform(-1.0, 1.0, 512))


class Sample(object):
    """A module that plays back a fixed block of sampled data.

    Attributes
    ----------
    data : NumPy array
        the sampled data to output
    """
    def __init__(self, data):
        """Construct a Sample module

        Parameters
        ----------
        data : NumPy array
            the sampled data to output
        """
        self.data = data
        self._offset = 0
        self._zeros = np.zeros((512,), np.float32)

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        if self._offset < len(self.data):
            bufferSize = 512;
            start = self._offset
            end = min(self._offset+bufferSize, len(self.data))
            self._offset = end
            return self.data[start:end]
        return self._zeros


class Filter(object):
    """A module that applies an IIR or FIR filter to its input.

    The filter is defined by the coefficients of the difference equation.  scipy.signal.iirfilter() and
    scipy.signal.iirdesign() are useful routines for selecting these coefficients.

    Attributes
    ----------
    a : array-like
        the feedback filter coefficients
    b : array-like
        the feedforward filter coefficients
    """
    def __init__(self, input, a, b):
        """Construct a Filter module

        Parameters
        ----------
        input : module
            the input module whose output should be filtered
        a : array-like
            the feedback filter coefficients
        b : array-like
            the feedforward filter coefficients
        """
        self._input = input
        self._a = a
        self._b = b
        self._lastInputs = np.array([])
        self._lastOutputs = np.array([])
        self._computeZ()

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value
        self._computeZ()

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._a = value
        self._computeZ()

    def _computeZ(self):
        self._z = scipy.signal.lfiltic(self._b, self._a, self._lastOutputs[::-1], self._lastInputs[::-1])

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        inputData = self._input.getData()
        (filtered, self._z) = scipy.signal.lfilter(self._b, self._a, inputData, zi=self._z)
        self._lastInputs = np.concatenate((self._lastInputs, inputData))[-len(self._b)-1:]
        self._lastOutputs = np.concatenate((self._lastOutputs, filtered))[-len(self._a)-1:]
        return filtered


class LowpassFilter(Filter):
    """A module that applies a lowpass Butterworth filter to its input.

    Attributes
    ----------
    cutoff : float
        the cutoff frequency in Hz
    order : int
        the order of the filter
    """
    def __init__(self, input, cutoff=1000.0, order=2, sampleRate=44100):
        """Construct a LowpassFilter module

        Parameters
        ----------
        input : module
            the input module whose output should be filtered
        cutoff : float
            the cutoff frequency in Hz
        order : int
            the order of the filter
        sampleRate : float
            the sample rate at which data is being generated in samples/second
        """
        self._cutoff = cutoff
        self._order = order
        self._sampleRate = sampleRate
        self._computeCoefficients()
        Filter.__init__(self, input, self._a, self._b)

    def _computeCoefficients(self):
        f = 2*self._cutoff/self._sampleRate
        (self._b, self._a) = scipy.signal.iirfilter(self._order, f, btype='lowpass')

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value
        self._computeCoefficients()
        self._computeZ()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._computeCoefficients()
        self._computeZ()


class HighpassFilter(Filter):
    """A module that applies a highpass Butterworth filter to its input.

    Attributes
    ----------
    cutoff : float
        the cutoff frequency in Hz
    order : int
        the order of the filter
    """
    def __init__(self, input, cutoff=1000.0, order=2, sampleRate=44100):
        """Construct a HighpassFilter module

        Parameters
        ----------
        input : module
            the input module whose output should be filtered
        cutoff : float
            the cutoff frequency in Hz
        order : int
            the order of the filter
        sampleRate : float
            the sample rate at which data is being generated in samples/second
        """
        self._cutoff = cutoff
        self._order = order
        self._sampleRate = sampleRate
        self._computeCoefficients()
        Filter.__init__(self, input, self._a, self._b)

    def _computeCoefficients(self):
        f = 2*self._cutoff/self._sampleRate
        (self._b, self._a) = scipy.signal.iirfilter(self._order, f, btype='highpass')

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value
        self._computeCoefficients()
        self._computeZ()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._computeCoefficients()
        self._computeZ()


class BandpassFilter(Filter):
    """A module that applies a bandpass Butterworth filter to its input.

    Attributes
    ----------
    cutoff1 : float
        the lower cutoff frequency in Hz
    cutoff2 : float
        the upper cutoff frequency in Hz
    order : int
        the order of the filter
    """
    def __init__(self, input, cutoff1=500.0, cutoff2=1000.0, order=2, sampleRate=44100):
        """Construct a BandpassFilter module

        Parameters
        ----------
        input : module
            the input module whose output should be filtered
        cutoff1 : float
            the lower cutoff frequency in Hz
        cutoff2 : float
            the upper cutoff frequency in Hz
        order : int
            the order of the filter
        sampleRate : float
            the sample rate at which data is being generated in samples/second
        """
        self._cutoff1 = cutoff1
        self._cutoff2 = cutoff2
        self._order = order
        self._sampleRate = sampleRate
        self._computeCoefficients()
        Filter.__init__(self, input, self._a, self._b)

    def _computeCoefficients(self):
        f1 = 2*self._cutoff1/self._sampleRate
        f2 = 2*self._cutoff2/self._sampleRate
        (self._b, self._a) = scipy.signal.iirfilter(self._order, (f1, f2), btype='bandpass')

    @property
    def cutoff1(self):
        return self._cutoff1

    @cutoff1.setter
    def cutoff1(self, value):
        self._cutoff1 = value
        self._computeCoefficients()
        self._computeZ()

    @property
    def cutoff2(self):
        return self._cutoff2

    @cutoff2.setter
    def cutoff2(self, value):
        self._cutoff2 = value
        self._computeCoefficients()
        self._computeZ()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._computeCoefficients()
        self._computeZ()


class BandstopFilter(Filter):
    """A module that applies a bandstop Butterworth filter to its input.

    Attributes
    ----------
    cutoff1 : float
        the lower cutoff frequency in Hz
    cutoff2 : float
        the upper cutoff frequency in Hz
    order : int
        the order of the filter
    """
    def __init__(self, input, cutoff1=500.0, cutoff2=1000.0, order=2, sampleRate=44100):
        """Construct a BandstopFilter module

        Parameters
        ----------
        input : module
            the input module whose output should be filtered
        cutoff1 : float
            the lower cutoff frequency in Hz
        cutoff2 : float
            the upper cutoff frequency in Hz
        order : int
            the order of the filter
        sampleRate : float
            the sample rate at which data is being generated in samples/second
        """
        self._cutoff1 = cutoff1
        self._cutoff2 = cutoff2
        self._order = order
        self._sampleRate = sampleRate
        self._computeCoefficients()
        Filter.__init__(self, input, self._a, self._b)

    def _computeCoefficients(self):
        f1 = 2*self._cutoff1/self._sampleRate
        f2 = 2*self._cutoff2/self._sampleRate
        (self._b, self._a) = scipy.signal.iirfilter(self._order, (f1, f2), btype='bandstop')

    @property
    def cutoff1(self):
        return self._cutoff1

    @cutoff1.setter
    def cutoff1(self, value):
        self._cutoff1 = value
        self._computeCoefficients()
        self._computeZ()

    @property
    def cutoff2(self):
        return self._cutoff2

    @cutoff2.setter
    def cutoff2(self, value):
        self._cutoff2 = value
        self._computeCoefficients()
        self._computeZ()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._computeCoefficients()
        self._computeZ()


class Resonator(Filter):
    """A module that applies a resonant filter to its input.

    More precisely, this is a driven, damped harmonic oscillator.  The input signal acts as the driving force.  It is
    characterized by the frequency at which it resonates and the friction coefficient.  Small friction values produce
    strong resonance over a narrow range of frequencies, while larger friction values produce weaker resonance over a
    wider range of frequencies.

    Attributes
    ----------
    frequency : float
        the resonant frequency in Hz
    friction : float
        the friction coefficient in 1/seconds
    """
    def __init__(self, input, frequency=500.0, friction=10.0, sampleRate=44100):
        """Construct a Resonator module

        Parameters
        ----------
        input : module
            the input module whose output should be filtered
        frequency : float
            the resonant frequency in Hz
        friction : float
            the friction coefficient in 1/seconds
        sampleRate : float
            the sample rate at which data is being generated in samples/second
        """
        self._frequency = frequency
        self._friction = friction
        self._sampleRate = sampleRate
        self._computeCoefficients()
        Filter.__init__(self, input, self._a, self._b)

    def _computeCoefficients(self):
        c = 2*self._friction
        k = (4*math.pi*math.pi)*(self._frequency**2 + self._friction**2)
        dt = 1.0/self._sampleRate
        self._a = [1.0, -2.0+c*dt+k*dt*dt, 1.0-c*dt]
        self._b = [1.0]

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self._computeCoefficients()
        self._computeZ()

    @property
    def friction(self):
        return self._friction

    @friction.setter
    def friction(self, value):
        self._friction = value
        self._computeCoefficients()
        self._computeZ()


class Envelope(object):
    """A module implementing an attack-hold-decay-sustain-release envelope.

    The output of this module is initially 0.  When beginAttack() is called, it rises linearly to 1 over the attack
    time; remains there for the hold time; decreases linearly over the decay time until it reaches the sustain level;
    remains constant until beginRelease() is called; and then decreases linearly to 0.

    Attributes
    ----------
    attack : float
        the attack time in seconds
    hold : float
        the hold time in seconds
    decay : float
        the decay time in seconds
    sustain : float
        the sustain level
    release : float
        the release time in seconds
    """
    def __init__(self, attack=0.1, hold=0.1, decay=0.8, sustain=0.5, release=0.1, sampleRate=44100):
        """Construct an Envelope module

        Parameters
        ----------
        attack : float
            the attack time in seconds
        hold : float
            the hold time in seconds
        decay : float
            the decay time in seconds
        sustain : float
            the sustain level
        release : float
            the release time in seconds
        sampleRate : float
            the sample rate at which data is being generated in samples/second
        """
        self.attack = attack
        self.hold = hold
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self._sampleRate = sampleRate
        self._stage = 0
        self._phase = 0
        self._initialLevel = 0.0
        self._currentLevel = 0.0

    def beginAttack(self):
        """Begin the attack stage."""
        self._stage = 1
        self._phase = 0
        self._initialLevel = self._currentLevel

    def beginRelease(self):
        """Begin the release stage."""
        self._stage = 5
        self._phase = 0
        self._initialLevel = self._currentLevel

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        if self._stage == 0: # no note playing
            self._currentLevel = 0.0
            return np.zeros((512,), np.float32)
        if self._stage == 4: # sustain
            self._currentLevel = self.sustain
            return self.sustain*np.ones((512,), np.float32)
        stageTime = (0, self.attack, self.hold, self.decay, 0, self.release)[self._stage]
        finalLevel = (0, 1, 1, self.sustain, self.sustain, 0)[self._stage]
        blockSize = min(self._sampleRate*stageTime-self._phase, 512)
        if blockSize == 0:
            self._stage = (self._stage+1)%6
            self._phase = 0
            self._initialLevel = self._currentLevel
            return np.zeros((0,), np.float32)
        start = (self._phase/self._sampleRate)/stageTime
        end = ((self._phase+blockSize)/self._sampleRate)/stageTime
        self._phase += blockSize
        self._currentLevel = finalLevel*end+self._initialLevel*(1-end)
        return np.linspace(finalLevel*start+self._initialLevel*(1-start), self._currentLevel, blockSize, endpoint=False)


class FunctionModule(object):
    """A module that applies an arbitrary function to each element of its input."""

    def __init__(self, input, function):
        """Construct a Function module.

        Parameters
        ----------
        input : module
            the input module whose output should have the function applied to it
        function : function
            the function to apply to the input
        """
        self._input = input
        self._function = function

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        return self._function(self._input.getData())


class String(object):
    """A module implementing a physically inspired model of a vibrating string.

    This module contains a buffer whose contents are repeatedly output.  The length of the buffer determines the
    fundamental frequency of the output signal.  An input signal representing an external excitation of the string is
    continuously added to the buffer.  After each repetition, a FIR filter is applied to the content of the buffer,
    representing frequency dependent dissipation.

    Attributes
    ----------
    frequency : float
        the fundamental frequency of the string
    filter : array-like
        the coefficients of the FIR filter.  scipy.signal.firwin() and scipy.signal.firwin2() are useful functions
        for selecting them.
    """
    def __init__(self, input, frequency=440.0, filter=[0.25, 0.49, 0.25], sampleRate=44100):
        """Construct a String module

        Parameters
        ----------
        input : module
            the input module used to excite the string
        frequency : float
            the fundamental frequency of the string
        filter : array-like
            the coefficients of the FIR filter.  scipy.signal.firwin() and scipy.signal.firwin2() are useful functions
            for selecting them.
        sampleRate : float
            the sample rate at which data is being generated in samples/second
        """
        self._frequency = frequency
        self._sampleRate = sampleRate
        self._period = sampleRate/frequency
        self._buffer = np.zeros((math.ceil(self._period),), np.float32)
        self._inputConnector = Connector(input)
        self._offset = 0.0
        self.filter = filter
        self._inputBuffer = np.empty(self._buffer.shape, np.float32)

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self._period = self._sampleRate/value
        newSize = math.ceil(self._period)
        if newSize != len(self._buffer):
            if newSize < len(self._buffer):
                self._buffer = self._buffer[:newSize]
            else:
                newBuffer = np.zeros((newSize,), np.float32)
                newBuffer[:len(self._buffer)] = self._buffer
                self._buffer = newBuffer
            self._inputBuffer = np.empty(self._buffer.shape, np.float32)

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        self._inputConnector.fillBuffer(self._inputBuffer)
        filtered = np.convolve(self._buffer, self.filter)
        start = len(self.filter)//2
        end = start+len(self._buffer)
        self._buffer = filtered[start:end]
        if start > 0:
            self._buffer[-start:] += filtered[:start]
        if end < len(filtered):
            numAtEnd = len(filtered)-end
            self._buffer[:numAtEnd] += filtered[-numAtEnd:]
        self._buffer[:] += self._inputBuffer
        self._offset += self._period
        elementsToReturn = math.floor(self._offset)
        self._offset -= elementsToReturn
        if elementsToReturn < len(self._buffer):
            return self._buffer[:elementsToReturn]
        return self._buffer


class Bow(object):
    """A module that outputs irregularly spaced bursts of noise, approximating the excitation a bow applies to a string.

    Attributes
    ----------
    amplitude : float
        the amplitude of the bursts of noise
    width : float
        the duration of each burst of noise, in seconds
    minGap : float
        the minimum gap between bursts of noise, in seconds
    maxGap : float
        the maximum gap between bursts of noise, in seconds
    """
    def __init__(self, amplitude=0.01, width=0.0005, minGap=0.000, maxGap=0.001, sampleRate=44100):
        """Construct a Bow module.

        Parameters
        ----------
        amplitude : float
            the amplitude of the bursts of noise
        width : float
            the duration of each burst of noise, in seconds
        minGap : float
            the minimum gap between bursts of noise, in seconds
        maxGap : float
            the maximum gap between bursts of noise, in seconds
        sampleRate : float
            the sample rate at which data should be generated in samples/second
        """
        self.amplitude = amplitude
        self.width = width
        self.minGap = minGap
        self.maxGap = maxGap
        self._sampleRate = sampleRate
        self._gap = False

    def getData(self):
        """Get output data from the module.

        Returns
        -------
        output : NumPy array
            an array containing the next block of output data
        """
        gap = self._gap
        self._gap = not gap
        if gap:
            samples = int(self._sampleRate*np.random.uniform(self.minGap, self.maxGap))
            return np.zeros((samples,), np.float32)
        return self.amplitude*np.random.uniform(-1.0, 1.0, int(self.width*self._sampleRate))

