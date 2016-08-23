class Connector(object):
    def __init__(self, input):
        self.input = input
        self.inputBuffer = None
        self.inputPos = 0
    
    def fillBuffer(self, buffer):
        pos = 0
        while pos < len(buffer):
            if self.inputBuffer is None or self.inputPos == len(self.inputBuffer):
                self.inputBuffer = self.input.getData()
                self.inputPos = 0
            elementsToCopy = min(len(self.inputBuffer)-self.inputPos, len(buffer)-pos)
            buffer[pos:pos+elementsToCopy] = self.inputBuffer[self.inputPos:self.inputPos+elementsToCopy]
            pos += elementsToCopy
            self.inputPos += elementsToCopy
