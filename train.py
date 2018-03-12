#

import tensorflow as tf
import numpy as np
import os,sys,glob
import logging
logging.getLogger().setLevel(logging.INFO)

import loader

def main():
    
    vimgs,vkbs = loader.load_session("session_2018_03_06__21_03_54")
    vimgs = np.reshape(vimgs,(-1,30*40,3))

    print(vimgs)



if __name__ == '__main__':
    main()
