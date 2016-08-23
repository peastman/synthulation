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


import sounddevice as sd
import mido
import threading
from connector import Connector

class Synthesizer(object):
    """A Synthesizer listens to MIDI events and plays sound in respons to them.

    The actual synthesis is done by an instrument.  The Synthesizer is responsible for receiving MIDI data, translating
    them into a more convenient form for use by the instrument, requesting output data from the instrument, and sending
    it to the speakers.

    Attributes
    ----------
    instrument : object
        the object defining the instrument
    note : int
        the ID of the note currently being played, or None if no note is currently on
    frequency : float
        the frequency in Hz of the note currently being played (taking pitch bend into account), or None if no note is
        currently on
    velocity : float
        the attack velocity of the note currently being played, or None if no note is currently on
    pitchbend : int
        the current pitch bend value, between -8192 and 8191
    pitchbendRange : float
        the maximum distance by which the pitch can be bent, measured in semitones
    samplerate : the sample being used for output, measured in samples/second

    """
    def __init__(self, instrument):
        """Create a Synthesizer.

        Parameters
        ----------
        instrument : object
            the object defining the instrument
        """
        self.instrument = instrument
        self.note = None
        self.frequency = None
        self.velocity = None
        self.pitchbend = 0
        try:
            self.pitchbendRange = instrument.pitchbendRange
        except:
            self.pitchbendRange = 2
        self.samplerate = sd.default.samplerate
        self._controls = {}
        self._lock = threading.Lock()

    def play(self):
        """Begin listening for MIDI events and playing sound in response to them."""
        c = Connector(self.instrument.getOutputModule())
        hasSoundPlaying = ('soundPlaying' in dir(self.instrument))
        hasNoteOn = ('noteOn' in dir(self.instrument))
        hasNoteOff = ('noteOff' in dir(self.instrument))
        hasPitchbendChanged = ('pitchbendChanged' in dir(self.instrument))
        hasControlChanged = ('controlChanged' in dir(self.instrument))

        def callback(indata, outdata, frames, time, status):
            self._lock.acquire()
            c.fillBuffer(outdata[:,0])
            volume = self.getControl(7, default=127)
            if hasSoundPlaying:
                self.instrument.soundPlaying(self, outdata.shape[0]/self.samplerate)
            self._lock.release()
            outdata *= volume/127.0

        with sd.Stream(channels=1, callback=callback) as stream:
            self.samplerate = stream.samplerate
            inport = mido.open_input()
            for message in inport:
                self._lock.acquire()
                if message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                    if self.note == message.note:
                        self.note = None
                        self.frequency = None
                        self.velocity = None
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
                self._lock.release()

    def getControl(self, id, default=None):
        """Get the current value of a MIDI control.

        Parameters
        ----------
        id : int
            the ID of the control to get
        default : int
            the value to return if the current value is unknown.  This happens if no control change message for the
            specified control has yet been received since play() was called.

        Returns
        -------
        value : int
            the current value of the control, betwee 0 and 127
        """
        if id not in self._controls:
            return default
        return self._controls[id]

    def _computeFrequency(self):
        if self.note is not None:
            bentNote = self.note + self.pitchbendRange*self.pitchbend/8192.0
            self.frequency = 440.0*(2.0**((bentNote-69)/12.0))

