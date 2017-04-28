#!/usr/bin/env python

import sys
import argparse
import re
import os
import os.path
from PyQt4.QtGui import *
from PyQt4.Qt import *


def saveWrap(dir='.', letter='A', font="Arial", size=40, align=Qt.AlignCenter):
    png_file = dir + "/" + font + "_" + letter + "_" + str(size) + "1" + ".png"
    save(png_file, letter, font, size, align)


def save(png_file, letter='A', font="Arial", size=40, align=Qt.AlignCenter):
    img = QImage(64, 64, QImage.Format_RGB32)
    img.fill(Qt.white)
    p = QPainter(img)
    p.setPen(Qt.black)
    p.setFont(QFont(font, size))
    p.drawText(img.rect(), align, letter)
    p.end()
    img.save(png_file)

def main():
    app = QApplication([])
    p = argparse.ArgumentParser(description='Symbols image generator')
    p.add_argument('-f', '--font', default='Arial', help='Font name, default=Arial')
    p.add_argument('-s', '--size', type=int, default=40, help='Font size, default=40')
    p.add_argument('-d', '--dir', default='.', help='Output directory, default=current')
    p.add_argument('letters', help='Array of letters(abc) or range (a-z)')
    args = p.parse_args()
    path = os.path.abspath(args.dir)
    if not os.path.exists(path):
        print("Directory not exists, created!")
        os.makedirs(path)
    if re.match('^([a-z]-[a-z])|([A-Z]-[A-Z])$', args.letters):
        begin = args.letters[0]
        end = args.letters[2]
        if (ord(end) - ord(begin)) > 26:
            print("Error using letters. Only A-Z or a-z available, not A-z.")
            p.print_help()
            return
        letters = [chr(a) for a in range(ord(begin), ord(end) + 1)]
    else:
        letters = args.letters
    for lett in letters:
        saveWrap(path, lett, args.font, args.size)
    return 0

if __name__ == "__main__":
    sys.exit(main())
