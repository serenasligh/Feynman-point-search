#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import os

os.system("") #initialize terminal for ANSI colors

ANSI_ESC = u'\x1b'
ANSI_RESET = ANSI_ESC + "[0m"
ANSI_ERASE = ANSI_ESC + "[F" + "\n" + ANSI_ESC + "[K"

basic_colors = {
    "black": (0,0,0)
    ,"red": (255,0,0)
    ,"green": (0,255,0)
    ,"blue": (0,0,255)
    ,"yellow": (255,255,0)
    ,"cyan": (0,255,255)
    ,"magenta": (255,0,255)
    ,"white": (255,255,255)
}

# colors = {
#     "RED": "\033[1;31m"
#     ,"BLUE": "\033[1;34m"
#     ,"CYAN": "\033[1;36m"
#     ,"GREEN": "\033[0;32m"
#     ,"RESET": "\033[0;0m"
#     ,"BOLD": "\033[;1m" #makes things yellow in powershell
#     ,"REVERSE": "\033[;7m"
# }

def hex_to_rgb(h):
    """
    input: "#FFFF00" -> output: (0.9,1.0,0.1)
    """
    h = h[1:]
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_ANSICtrl(rgb, bg=False):
    """
    input: (243,99,20) -> output: \x1b[38;2;243;99;20m
    """
    fg_or_bg_code = 48 if bg else 38
    r, g, b = rgb
    return ANSI_ESC + f'[{fg_or_bg_code};2;{r};{g};{b}m'

def color_text( text, fg=None, bg=None ):
    """
    Colors the foreground and background of text printed to the terminal.

    fg and bg -
      1) One of the color keywords "red", "green", "blue", "cyan", "yellow", "magenta", "white", or
         "black".
      2) 24-bit rgb color values represented as 3-tuples with either
        a) integers values ranging from 0-255 (e.g. (120,0,255) ),
        b) float values ranging from 0-1.0 (e.g. (0.1,0.5,1.0) ), or
        c) hexcolor strings (e.g. "#FF0000").

    text    - (string) Text to colorize.
    fg      - (string, or 3-tuple of integers or floats) foreground color of text.
    bg      - (string, or 3-tuple of integers or floats) background color of text.

    output  - (string) The input text wrapped with colorizing ANSI control characters.
    """

    prefix = ""
    suffix = ""
    for i, inp in enumerate((fg,bg)):
        if inp is None:
            continue

        if inp in basic_colors.keys():
            rgb = basic_colors[inp]
        elif type(inp) is str:
            # Convert hexcolor string to ANSI control string
            rgb = hex_to_rgb(inp)
        elif type(inp) is tuple:
            rgb = inp
        else:
            raise ValueError

        color_bg = i==1
        prefix += rgb_to_ANSICtrl( rgb, bg=color_bg )
        suffix = ANSI_RESET
    text = prefix + text + suffix
    return text

def set_resolution(rows,cols):
    """
    Sets the width and height of the terminal window.
    'Rows' and 'cols' are integers representing the number of characters along each axis.
    """
    assert type(cols) is int and type(rows) is int, "Terminal rows and cols must be integers."
    import sys
    sys.stdout.write( ANSI_ESC + f"[8;{rows};{cols}t" )
