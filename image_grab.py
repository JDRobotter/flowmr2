#

import PIL.ImageGrab

class ImageGrabber:
    def __init__(self,x,y,w,h):
        self.bbox = (x,y,w,h)

    def grab(self):
        return PIL.ImageGrab.grab(self.bbox)

