# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:36:40 2016
@author: Viacheslav Klimenko & Serhei Balashev
"""
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__))[:-16])
import PDRviewer

from PyQt5.QtWidgets import (QApplication)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PDRviewer.PDRviewer()
    sys.exit(app.exec_())