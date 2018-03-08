#

from control import GameControl as GC
from image_grab import ImageGrabber
import pygame
import PIL
import PIL.ImageOps
import time
import os,sys
import random

RED = (255,0,0)
BLACK = (0,0,0)
GRAY = (150,150,150)
WHITE = (255,255,255)


def main():

    GAME_WINDOW_TITLE_PREFIX = "DOSBox"

    print("[+] initializing")
    # prepare debug display
    pygame.init()
    win = pygame.display.set_mode((640,480))
    clock = pygame.time.Clock()

    # find game 
    print("[+] looking for game window handle")
    game = GC(GAME_WINDOW_TITLE_PREFIX, (GC.KUP,GC.KDOWN,GC.KLEFT,GC.KRIGHT))
    print("    game found: \"%s\""%(game.window_name))

    # prepare image grabber
    x,y,w,h = game.window_rect()
    # shift to avoid window title bar
    ig = ImageGrabber(x,y+30,w,h)


    # create a directory to store session data
    logdir = time.strftime("session_%Y_%m_%d__%H_%M_%S")
    print("[+] storing session in",logdir)
    os.makedirs(logdir)

    # let's roll
    running = True
    t = 0
    while running:
        # check events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # grab image
        sshot = ig.grab()

        nw = int(sshot.width/2)
        nh = int(sshot.height/2)
        sshot = sshot.resize((nw,nh), PIL.Image.NEAREST)

        # convert image to tf friendly format
        nw = 40
        nh = 30
        view = sshot.resize((nw,nh))
        view = PIL.ImageOps.equalize(view)
        view = view.convert('RGB')

        #print(view.size, len(view.tobytes('raw')))
        open(os.path.join(logdir,"img_%d"%t), "wb").write(view.tobytes('raw'))

        kbstate = game.get_keyboard_state()
        v = b''.join(b'1' if b else b'0' for b in kbstate)
        open(os.path.join(logdir,"kb_%d"%t), "wb").write(v)

        # draw interface
        win.fill(BLACK)

        # show (resized) grabbed image on debug display
        img = pygame.image.frombuffer(
            sshot.tobytes('raw'),
            sshot.size,
            sshot.mode)
        win.blit(img,(0,0))
        # show input (view) image on debug display
        sview = view.resize(sshot.size)
        img = pygame.image.frombuffer(
            sview.tobytes('raw'),
            sview.size,
            sview.mode)
        win.blit(img,(sshot.width,0))

        # show gray

        # show keyboard state on debug display
        bx,by = sshot.size
        for i,b in enumerate(kbstate):
            win.fill(WHITE if b else GRAY, (5 + 30*i,by+5,20,20))

        pygame.display.flip()

        # DEBUG
        # DEBUG

        clock.tick(10)

        t += 1

if __name__ == '__main__':
    main()

