# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:15:05 2023

@author: fbrev
"""

import PIL
from pathlib import Path

path = Path("./data/screenshots").rglob("*.png")
for img_p in path:
    try:
        img = PIL.Image.open(img_p)
    except PIL.UnidentifiedImageError:
            print(img_p)