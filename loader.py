#!/usr/bin/env python3

import os,glob
import numpy as np

def load_file(filename):
    return np.frombuffer(
        open(filename,"rb").read(),
        dtype=np.float16)

def load_session(path):
    print("[+] loading",path)
    return (load_file(path + ".imgs"),
            load_file(path + ".keys"))

if __name__ == '__main__':
    vimgs,vkbs = load_session("session_2018_03_06__21_03_54")
    print(vimgs)
    print(vkbs)
