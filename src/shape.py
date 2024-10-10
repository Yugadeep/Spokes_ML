#!/usr/bin/env -S python3

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:06:34 2024

@author: yugad
"""

import sys
from PIL import Image
import numpy as np



filepath = sys.argv[1]


img = Image.open(filepath).convert('L')


image_array = np.array(img)


print(f'{filepath} shape:', image_array.shape)

