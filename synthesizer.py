import sounddevice as sd
import mido
import threading
from connector import Connector

class Synthesizer(object):
    def __init__(self, instrument):
        self.instrument = instrument
        self.note = None
        self.frequency = 0
        self.velocity = 0
        self.pitchbend = 0
        self._controls = {}
        self.samplerate = sd.default.samplerate
        self.lock = threading.Lock()
        try:
            self.pitchbendRange = instrument.pitchbendRange
        except:
            self.pitchbendRange = 2
    
    def play(self):
        c = Connector(self.instrument.getOutputModule())
        hasSoundPlaying = ('soundPlaying' in dir(self.instrument))
        hasNoteOn = ('noteOn' in dir(self.instrument))
        hasNoteOff = ('noteOff' in dir(self.instrument))
        hasPitchbendChanged = ('pitchbendChanged' in dir(self.instrument))
        hasControlChanged = ('controlChanged' in dir(self.instrument))
        
        def callback(indata, outdata, frames, time, status):
            self.lock.acquire()
            c.fillBuffer(outdata[:,0])
            volume = self.getControl(7, default=127)
            if hasSoundPlaying:
                self.instrument.soundPlaying(self, outdata.shape[0]/self.samplerate)
            self.lock.release()
            outdata *= volume/127.0
        
        with sd.Stream(channels=1, callback=callback) as stream:
            self.samplerate = stream.samplerate
            inport = mido.open_input()
            for message in inport:
                self.lock.acquire()
                if message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                    if self.note == message.note:
                        self.note = None
                        if hasNoteOff:
                            self.instrument.noteOff(self)
                elif message.type == 'note_on':
                    self.note = message.note
                    self._computeFrequency()
                    self.velocity = message.velocity
                    if hasNoteOn:
                        self.instrument.noteOn(self)
                elif message.type == 'pitchwheel':
                    self.pitchbend = message.pitch
                    self._computeFrequency()
                    if hasPitchbendChanged:
                        self.instrument.pitchbendChanged(self)
                elif message.type == 'control_change':
                    self._controls[message.control] = message.value
                    if hasControlChanged:
                        self.instrument.controlChanged(self, message.control)
                self.lock.release()

    def getControl(self, id, default=None):
        if id not in self._controls:
            return default
        return self._controls[id]

    def _computeFrequency(self):
        if self.note is not None:
            bentNote = self.note + self.pitchbendRange*self.pitchbend/8192.0
            self.frequency = 440.0*(2.0**((bentNote-69)/12.0))

