#!/usr/bin/env python3
import numpy as np
import os
import sys
import scipy.io as io


args = sys.argv
if len(args) == 1:
    no_args = """Program rpjbStats Error(s)
    - 0 arguments passed: 1 expected!
rpjbStats takes 1 argument, the path to a given .rpjb file.
It then returns statistics assosiated with said file in the following format:
    File name: Filename
    Minimum radius: mnrad
    Maximum radius: mxrad
    Minimum longitude: mnlon
    Maximum Longitude: mxlon
    rrpi dimensions: (x,y)
    Percent non-zero: %
To execute file, follow the follwing template:
    'python3 rpjbStats.py [filename]'"""
    print(no_args)
elif len(args) == 2:
    if not os.path.exists(args[1]):
        print("It appears that .rpjb file does not exist. Please enter a propper path.")
        exit()


    filepath = args[1]
    idl = io.readsav(filepath)

    stats = f"""File name: {filepath.split('/')[-1]}
                    Minimum radius: {idl.mnrad}
                    Maximum radius: {idl.mxrad}
                    Minimum longitude: {idl.mnlon}
                    Maximum Longitude: {idl.mxlon}
                    rrpi dimensions: {idl.rrpi.shape}
                    Percent non-zero: {np.count_nonzero(idl.rrpi)/(idl.rrpi.shape[0]*idl.rrpi.shape[1])*100}"""
    print(stats)
else:
    print("Wrong number of arguments detected. Please type 'rpjbStats.py' for instructions.")






