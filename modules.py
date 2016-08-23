from connector import Connector
import numpy as np
import scipy.signal
import math


class Adder(object):
    def __init__(self, *inputs):
        self.connectors = [Connector(i) for i in inputs] 
        bufferSize = 512
        self.buffer1 = np.empty((bufferSize,), np.float32)
        self.buffer2 = np.empty((bufferSize,), np.float32)
    
    def getData(self):
        if len(self.connectors) == 0:
            return np.zeros((512,), np.float32)
        self.connectors[0].fillBuffer(self.buffer1)
        for i in range(1, len(self.connectors)):
            self.connectors[i].fillBuffer(self.buffer2)
            self.buffer1 += self.buffer2
        return self.buffer1


class Multiplier(object):
    def __init__(self, *inputs):
        self.connectors = [Connector(i) for i in inputs] 
        bufferSize = 512
        self.buffer1 = np.empty((bufferSize,), np.float32)
        self.buffer2 = np.empty((bufferSize,), np.float32)
    
    def getData(self):
        if len(self.connectors) == 0:
            return np.zeros((512,), np.float32)
        self.connectors[0].fillBuffer(self.buffer1)
        for i in range(1, len(self.connectors)):
            self.connectors[i].fillBuffer(self.buffer2)
            self.buffer1 *= self.buffer2
        return self.buffer1


class Sine(object):
    def __init__(self, frequency=440.0, amplitude=1.0, sampleRate=44100):
        self.offset = 0
        self.period = sampleRate/frequency
        self.amplitude = amplitude
    
    def getData(self):
        bufferSize = 512
        data = self.amplitude*np.sin((2*math.pi/self.period)*np.linspace(self.offset, self.offset+bufferSize, bufferSize, False))
        self.offset = (self.offset+bufferSize)%self.period
        return data


class Noise(object):
    def __init__(self, amplitude=1.0):
        self.amplitude = amplitude
    
    def getData(self):
        return self.amplitude*(np.random.uniform(-1.0, 1.0, 512))


class Sample(object):
    def __init__(self, data):
        self.data = data
        self.offset = 0
        self.zeros = np.zeros((512,), np.float32)
    
    def getData(self):
        if self.offset < len(self.data):
            bufferSize = 512;
            start = self.offset
            end = min(self.offset+bufferSize, len(self.data))
            self.offset += bufferSize
            return self.data[start:end]
        return self.zeros


class Filter(object):
    def __init__(self, input, a, b):
        self.input = input
        self.a = a
        self.b = b
        self.z = scipy.signal.lfiltic(b, a, [], [])
    
    def getData(self):
        inputData = self.input.getData()
        (filtered, self.z) = scipy.signal.lfilter(self.b, self.a, inputData, zi=self.z)
        return filtered


class LowpassFilter(Filter):
        def __init__(self, input, cutoff=1000.0, order=2, sampleRate=44100):
            f = 2*cutoff/sampleRate
            (b, a) = scipy.signal.iirfilter(order, f, btype='lowpass')
            Filter.__init__(self, input, a, b)


class HighpassFilter(Filter):
        def __init__(self, input, cutoff=1000.0, order=2, sampleRate=44100):
            f = 2*cutoff/sampleRate
            (b, a) = scipy.signal.iirfilter(order, f, btype='highpass')
            Filter.__init__(self, input, a, b)


class BandpassFilter(Filter):
        def __init__(self, input, cutoff1=500.0, cutoff2=1000.0, order=2, sampleRate=44100):
            f1 = 2*cutoff1/sampleRate
            f2 = 2*cutoff2/sampleRate
            (b, a) = scipy.signal.iirfilter(order, (f1, f2), btype='bandpass')
            Filter.__init__(self, input, a, b)


class BandstopFilter(Filter):
        def __init__(self, input, cutoff1=500.0, cutoff2=1000.0, order=2, sampleRate=44100):
            f1 = 2*cutoff1/sampleRate
            f2 = 2*cutoff2/sampleRate
            (b, a) = scipy.signal.iirfilter(order, (f1, f2), btype='bandstop')
            Filter.__init__(self, input, a, b)


class String(object):
    def __init__(self, input, frequency=440.0, filter=[0.25, 0.49, 0.25], sampleRate=44100):
        self._frequency = frequency
        self.sampleRate = sampleRate
        self.period = sampleRate/frequency
        self.buffer = np.zeros((math.ceil(self.period),), np.float32)
        self.inputConnector = Connector(input)
        self.offset = 0.0
        self.filter = filter
        self.inputBuffer = np.empty(self.buffer.shape, np.float32)
    
    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self.period = self.sampleRate/value
        newSize = math.ceil(self.period)
        if newSize != len(self.buffer):
            if newSize < len(self.buffer):
                self.buffer = self.buffer[:newSize]
            else:
                newBuffer = np.zeros((newSize,), np.float32)
                newBuffer[:len(self.buffer)] = self.buffer
                self.buffer = newBuffer
            self.inputBuffer = np.empty(self.buffer.shape, np.float32)
    
    def getData(self):
        self.inputConnector.fillBuffer(self.inputBuffer)
        filtered = np.convolve(self.buffer, self.filter)
        start = len(self.filter)//2
        end = start+len(self.buffer)
        self.buffer = filtered[start:end]
        if start > 0:
            self.buffer[-start:] += filtered[:start]
        if end < len(filtered):
            numAtEnd = len(filtered)-end
            self.buffer[:numAtEnd] += filtered[-numAtEnd:]
        self.buffer[:] += self.inputBuffer
        self.offset += self.period
        elementsToReturn = math.floor(self.offset)
        self.offset -= elementsToReturn
        if elementsToReturn < len(self.buffer):
            return self.buffer[:elementsToReturn]
        return self.buffer


class Bow(object):
    def __init__(self, amplitude=0.01, width=0.0005, minGap=0.000, maxGap=0.001, sampleRate=44100):
        self.amplitude = amplitude
        self.width = width
        self.minGap = minGap
        self.maxGap = maxGap
        self.sampleRate = sampleRate
        self.gap = False
    
    def getData(self):
        gap = self.gap
        self.gap = not gap
        if gap:
            samples = int(self.sampleRate*np.random.uniform(self.minGap, self.maxGap))
            return np.zeros((samples,), np.float32)
        return self.amplitude*np.random.uniform(-1.0, 1.0, int(self.width*self.sampleRate))

