# 
# only works for win32 right now

import time
import win32gui, win32api, win32con, win32ui, pywintypes

class GameControlException(Exception):
    pass

class GameControl:

    KUP = win32con.VK_UP
    KDOWN = win32con.VK_DOWN
    KLEFT = win32con.VK_LEFT
    KRIGHT = win32con.VK_RIGHT

    def __init__(self, name, keys_vector):

        # search for window handle by title
        matches = []
        def callback(hndl, extra):
            s = win32gui.GetWindowText(hndl)
            if s.startswith(name):
                matches.append((hndl,s))
        win32gui.EnumWindows(callback,None)

        if len(matches) > 1:
            raise GameControlException(
                "More than one matching hndl found")
        
        if len(matches) == 0:
            raise GameControlException(
                "No window found matching title %s"%(name))

        (ihndl, self.window_name), = matches

        self.pycwnd = win32ui.CreateWindowFromHandle(ihndl)

        self.keys_vector = keys_vector
        self.keys_state = [None for _ in keys_vector]

    def window_rect(self):
        return self.pycwnd.GetWindowRect()

    def get_keyboard_state(self):
        return [(win32api.GetKeyState(key) & 0xFE > 0)
                    for key in self.keys_vector]


    def set_keyboard_state(self, keys):
        # keybd_event ref
        # https://msdn.microsoft.com/fr-fr/library/windows/desktop/ms646304(v=vs.85).aspx
        for code,pk,k in zip(self.keys_vector, self.keys_state, keys):
            print(code,pk,k)
            if pk != k:
                # key as changed
                if k:
                    win32api.keybd_event(code, 0, 0x00)
                else:
                    win32api.keybd_event(code, 0, 0x02)

        self.keys_state[:] = keys

