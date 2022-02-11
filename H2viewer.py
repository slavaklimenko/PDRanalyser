from astropy.io import ascii
import astropy.constants as ac
from astropy.cosmology import FlatLambdaCDM
from functools import partial
from io import StringIO
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import pickle
from PyQt5.QtCore import (Qt, )
from PyQt5.QtGui import (QFont, )
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow, QSplitter, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QHeaderView, QCheckBox,
                             QRadioButton, QButtonGroup, QComboBox, QTableView, QLineEdit)
import pyqtgraph as pg
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, Rbf
from scipy.interpolate import RBFInterpolator
from scipy.optimize import bisect
import sys
sys.path.append('/home/slava/science/codes/python')
from H2_exc import *
from H2_exc import H2_exc
from spectro.stats import distr2d,distr1d
from spectro.a_unc import a
from spectro.pyratio import pyratio
import copy

class image():
    """
    class for working with images (2d spectra) inside Spectrum plotting
    """
    def __init__(self, x=None, y=None, z=None, err=None, mask=None):
        if any([v is not None for v in [x, y, z, err, mask]]):
            self.set_data(x=x, y=y, z=z, err=err, mask=mask)
        else:
            self.z = None

    def set_data(self, x=None, y=None, z=None, err=None, mask=None):
        for attr, val in zip(['z', 'err', 'mask'], [z, err, mask]):
            if val is not None:
                setattr(self, attr, np.asarray(val))
            else:
                setattr(self, attr, val)
        if x is not None:
            self.x = np.asarray(x)
        else:
            self.x = np.arange(z.shape[0])
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = np.arange(z.shape[1])

        self.pos = [self.x[0] - (self.x[1] - self.x[0]) / 2, self.y[0] - (self.y[1] - self.y[0]) / 2]
        self.scale = [(self.x[-1] - self.x[0]) / (self.x.shape[0]-1), (self.y[-1] - self.y[0]) / (self.y.shape[0]-1)]
        for attr in ['z', 'err']:
            self.getQuantile(attr=attr)
            self.setLevels(attr=attr)


    def getQuantile(self, quantile=0.997, attr='z'):
        if getattr(self, attr) is not None:
            x = np.sort(getattr(self, attr).flatten())
            x = x[~np.isnan(x)]
            setattr(self, attr+'_quantile', [x[int(len(x)*(1-quantile)/2)], x[int(len(x)*(1+quantile)/2)]])
        else:
            setattr(self, attr + '_quantile', [0, 1])

    def setLevels(self, bottom=None, top=None, attr='z'):
        quantile = getattr(self, attr+'_quantile')
        if bottom is None:
            bottom = quantile[0]
        if top is None:
            top = quantile[1]
        top, bottom = np.max([top, bottom]), np.min([top, bottom])
        if top - bottom < (quantile[1] - quantile[0]) / 100:
            top += ((quantile[1] - quantile[0]) / 100 - (top - bottom)) /2
            bottom -= ((quantile[1] - quantile[0]) / 100 - (top - bottom)) / 2
        setattr(self, attr+'_levels', [bottom, top])

    def find_nearest(self, x, y, attr='z'):
        z = getattr(self, attr)
        if len(z.shape) == 2:
            return z[np.min([z.shape[0]-1, (np.abs(self.y - y)).argmin()]), np.min([z.shape[1]-1, (np.abs(self.x - x)).argmin()])]
        else:
            return None

class plotExc(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'log(N/g)', 'bottom': 'Energy, cm-1'})
        self.initstatus()
        self.vb = self.getViewBox()
        self.view = {}
        self.models = {}
        self.legend = pg.LegendItem(offset=(-70, 30))
        self.legend.setParentItem(self.vb)
        self.legend_model = pg.LegendItem(offset=(-70, -30))
        self.legend_model.setParentItem(self.vb)

    def initstatus(self):
        pass

    def getatomic(self, species, levels=[0, 1, 2]):
        if species == 'H2':
            return [H2energy[0, i] for i in levels], [stat_H2[i] for i in levels]
        elif species == 'CI':
            return [CIenergy[i] for i in levels], [stat_CI[i] for i in levels]
        elif species == 'CO':
            return [COenergy[i] for i in levels], [stat_CO[i] for i in levels]

    def add(self, name, add):
        if add:
            species = str(self.parent.grid_pars.species.currentText())
            relative = self.parent.grid_pars.relative.isChecked()
            q = self.parent.H2.comp(name)
            sp = [s for s in q.e.keys() if species+'j' in s]
            j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
            x, stat = self.getatomic(species, levels=j)
            y = [q.e[species+'j' + str(i)].col / stat[i] for i in j]
            if relative:
                y = [y[i] / q.e[species + 'j0'].col for i in j]
            self.view[name] = [pg.ErrorBarItem(x=np.asarray(x), y=column(y, 'v'), top=column(y, 'p'), bottom=column(y, 'm'), beam=2),
                               pg.ScatterPlotItem(x, column(y, 'v'), symbol='o', size=15)]
            self.vb.addItem(self.view[name][0])
            self.vb.addItem(self.view[name][1])
            self.legend.addItem(self.view[name][1], name)
            self.redraw()
        else:
            try:
                self.vb.removeItem(self.view[name][0])
                self.vb.removeItem(self.view[name][1])
                self.legend.removeItem(self.view[name][1])
                del self.view[name]
            except:
                pass


    def add_model(self, name, add=True):
        if add:
            species = str(self.parent.grid_pars.species.currentText())
            relative = self.parent.grid_pars.relative.isChecked()
            for ind, m in enumerate(self.parent.H2.listofmodels(name)):
                j = np.sort([int(s[3:]) for s in m.cols.keys()])
                x, stat = self.getatomic(species, levels=j)
                mod = [m.cols[species+'j'+str(i)] - np.log10(stat[i]) for i in j]
                if relative:
                    mod -=mod[0]
            print('add model', x, mod)
            self.models[name] = pg.PlotCurveItem(x, mod)
            self.vb.addItem(self.models[name])
            self.legend_model.addItem(self.models[name], name)
            self.redraw()
        else:
            try:
                self.vb.removeItem(self.models[name])
                self.legend_model.removeItem(self.models[name])
                del self.models[name]
            except:
                pass


    def add_cmb(self, cols=None, redshift=0, add=True):
        if add:
            species = str(self.parent.grid_pars.species.currentText())
            relative = self.parent.grid_pars.relative.isChecked()
            tcmb = 2.725*(1+redshift)
            j = np.sort([int(s[3:]) for s in cols.keys()])
            x, stat = self.getatomic(species, levels=j)
            cols0 = cols[species + 'j0']
            mod = [cols0  + np.log10(np.exp(-x[i] *1.438/ tcmb )) for i in j]
            if relative:
                mod -= mod[0]
            self.temp_cmb = pg.PlotCurveItem(x, mod, pen='r')
            self.vb.addItem(self.temp_cmb)
            self.redraw()
            if 0:
                tcmb = 2.725/1.438
                print(x,tcmb)
                mod = [cols0 - np.log10(np.exp(1))*x[i]*1.438 / tcmb  for i in j]
                if relative:
                    mod -= mod[0]
                self.temp_cmb_corr = pg.PlotCurveItem(x, mod, pen='g')
                self.vb.addItem(self.temp_cmb_corr)
                self.redraw()
        else:
            try:
                self.vb.removeItem(self.temp_cmb)
                self.legend_model.removeItem(self.temp_cmb)
                self.vb.removeItem(self.temp_cmb_corr)
                self.legend_model.removeItem(self.temp_cmb_corr)
            except:
                pass

    def add_texc(self, cols=None, Texc=2.723, add=True):
        if add:
            species = str(self.parent.grid_pars.species.currentText())
            relative = self.parent.grid_pars.relative.isChecked()
            j = np.sort([int(s[3:]) for s in cols.keys()])
            x, stat = self.getatomic(species, levels=j)
            cols0 = cols[species+'j0']
            mod = [cols0 + np.log10(np.exp(-x[i] * 1.438 / Texc)) for i in j]
            if relative:
                mod -= mod[0]
            self.temp_exc = pg.PlotCurveItem(x, mod, pen='b')
            self.vb.addItem(self.temp_exc)
            self.redraw()
            if 0:
                Texc = Texc / 1.438
                print(x, Texc)
                mod = [cols0 - np.log10(np.exp(1)) * x[i] * 1.438 / Texc for i in j]
                if relative:
                    mod -= mod[0]
                self.temp_exc_corr = pg.PlotCurveItem(x, mod, pen='g')
                self.vb.addItem(self.temp_exc_corr)
                self.redraw()
        else:
            try:
                self.vb.removeItem(self.temp_exc)
                self.legend_model.removeItem(self.temp_exc)
                self.vb.removeItem(self.temp_exc_corr)
                self.legend_model.removeItem(self.temp_exc_corr)
            except:
                pass

    def add_temp(self, cols=None, pars=None, add=True):
        if add:
            species = str(self.parent.grid_pars.species.currentText())
            relative = self.parent.grid_pars.relative.isChecked()
            j = np.sort([int(s[3:]) for s in cols.keys()])
            x, stat = self.getatomic(species, levels=j)
            mod = [cols[species+'j'+str(i)] - np.log10(stat[i]) for i in j]
            if relative:
                mod -=mod[0]
            # save local fit to species excitation
            if 0:
                with open('temp/local_fit.pkl', 'wb') as f:
                    if species == 'H2':
                        pickle.dump([x, mod, spec], f)
                    elif species == 'CI':
                        pickle.dump([x, mod], f)
            print('add temp_interpolation', x, mod)
            self.temp_model = pg.PlotCurveItem(x, mod)
            self.vb.addItem(self.temp_model)
            text = 'selected'
            if pars is not None:
                text += ' {0:.2f} {1:.2f}'.format(pars[0], pars[1])
            self.legend_model.addItem(self.temp_model, text)

            if 0:

                # calc and print lnL
                others = self.parent.grid_pars.othermode.currentText()
                levels = self.parent.grid_pars.H2levels
                syst = float(self.parent.grid_pars.addSyst.text())
                grid = self.parent.H2.grid
                full_keys = grid['cols'][0].keys()
                label = species + 'j{:}'
                keys = [label.format(i) for i in levels if label.format(i) in full_keys]
                spec = {}
                for s in keys:
                    v1 = self.parent.H2.comp(grid['name']).e[s].col.log().copy()
                    v1 *= a(0, syst, syst, 'l')
                    spec[s] = v1
                    if relative:
                        spec[s] -= spec[species + 'j0']
                if others in ['lower', 'upper']:
                    for k in full_keys:
                        if k not in keys:
                            v = self.parent.H2.comp(grid['name']).e[k].col.log().copy()
                            if others == 'lower':
                                spec[k] = a(v.val - v.minus, t=others[0])
                            else:
                                spec[k] = a(v.val + v.plus, t=others[0])

                lnL = 0
                for s, v in spec.items():
                    if v.type in ['m', 'u', 'l']:
                        lnL += v.lnL(mod[int(s[-1])])
                print(lnL)
        else:
            try:
                self.vb.removeItem(self.temp_model)
                self.legend_model.removeItem(self.temp_model)
            except:
                pass



    def redraw(self):
        for i, v in enumerate(self.view.values()):
            v[1].setBrush(pg.mkBrush(cm.rainbow(0.01 + 0.98 * i / len(self.view), bytes=True)[:3] + (255,)))
        for i, v in enumerate(self.models.values()):
            v.setPen(pg.mkPen(cm.rainbow(0.01 + 0.98 * i / len(self.models), bytes=True)[:3] + (255,)))

class textLabel(pg.TextItem):
    def __init__(self, parent, text, x=None, y=None, name=None):
        self.parent = parent
        self.text = text
        self.active = False
        self.name = name
        pg.TextItem.__init__(self, text=text, fill=pg.mkBrush(0, 0, 0, 0), anchor=(0.5,0.5))
        self.setFont(QFont("SansSerif", 16))
        self.setPos(x, y)
        self.redraw()

    def redraw(self):
        if self.active:
            self.setColor((255, 225, 53))
        else:
            self.setColor((255, 255, 255))
        self.parent.parent.plot_exc.add_model(self.name, add=self.active)

    def plot_model(self,limit=None):
        m = self.parent.parent.H2.listofmodels(self.name)[0] # name of the seleceted node in the grid
        if self.parent.parent.H2.grid['NH2tot'] is not None:
            limit = {'NH2': 10 ** self.parent.parent.H2.grid['NH2tot'] / (
                        int(self.parent.parent.grid_pars.sides.currentIndex()))}
        limit=None
        band = {}
        for el in ['NH2', 'NCO']:
            if self.parent.parent.H2.grid[el+'tot'] is not None:
                band[el] = 10 ** self.parent.parent.H2.grid[el+'tot'] / (
                        int(self.parent.parent.grid_pars.sides.currentIndex()))
            else:
                band[el] = None


        if self.parent.parent.grid_pars.plot_model_set.currentText() == 'H2+CI':
            m.plot_model(parx='x', pars=['tgas', 'n', 'pgas'],
                         species=[['H', 'H+', 'H2'], ['H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5'], ['CIj0', 'CIj1', 'CIj2']],
                         logx=True, logy=True, limit=limit,band = band)
        if self.parent.parent.grid_pars.plot_model_set.currentText() == 'OH':
            m.plot_model(parx='h2', pars=['tgas', 'n'],
                         species=[['H', 'H+', 'H2', 'OH'],
                                  ['NH/NH2', 'NOH/NH2']],
                         logx=True, logy=True, limit=limit)
        if self.parent.parent.grid_pars.plot_model_set.currentText() == 'CO':
            limit = None
            m.plot_model(parx='co', pars=['tgas', 'n','pgas'],
                         species=[['H', 'H+', 'H2', 'CI', 'CO'], ['NH2','NCI','NCO'],['COj1/COj0', 'COj2/COj0', 'COj3/COj0', 'COj4/COj0', 'COj5/COj0', 'COj6/COj0']],
                         logx=True, logy=True, limit=limit,band = band
                         )
        plt.show()
        '''
        m.plot_model(parx='av', #pars=[['tgas','n', 'pgas','T01']], #
                     pars=[['tgas','n','pgas']],
                     #species=[['NH2j0/NH2', 'NH2j1/NH2', 'NH2j2/NH2', 'NH2j3/NH2'], ['NH', 'NH2', 'NCI','NCO'],
                     #         ['H/H2', 'H+/H2']],
                     species = [['H', 'H2','CI','CO','C+','H+'],['NH', 'NH2','NCI','NCO'],['cool_cp','cool_c','cool_co', 'cool_tot']],
                     #species=[['H', 'H2','CI','CO','C+','H+'],['H2j0/H2', 'H2j1/H2', 'H2j2/H2','H2j3/H2','H2j4/H2'],['NH', 'NH2','NCI','NCO','NC+','NH+'],
                              #['NH2j0', 'NH2j1', 'NH2j2', 'NH2j3', 'NH2j4'], ['NH', 'NH2','NCI','NCO','NC+','NH+']],
                              #['COj0/CO','COj1/CO','COj2/CO','COj3/CO'],['CIj0/CI','CIj1/CI','CIj2/CI']],
                     #species=[['NH', 'NH2', 'NCI', 'NCO']],
                     logx=True, logy=True, borders=borders) #, 'CI': self.parent.parent.H2.grid['NCOtot']-0.3}) #, limit={'H2':24})
                     #limit={'H2': self.parent.parent.H2.grid['NH2tot'] -0.3}) #, limit={'H2':21} #['OPR','OPR_logNJ1/J02','OPR_logNJ1/J0'],['uv_dens'],['H2_dest_rate'],
        '''

    def plot_fit(self):
        m = self.parent.parent.H2.listofmodels(self.name)[0]
        sp = self.parent.parent.grid_pars.species.currentText()
        if sp == 'CI':
            m.plot_model(parx='co', species=[['CIj0/CI','CIj1/CI','CIj2/CI'],['NCj0/NCI','NCj1/NCI','NCj2/NCI']], pyfit=True,
                         logx=True, logy=True, borders={'H2': self.parent.parent.H2.grid['NH2tot']-0.3, 'CI': self.parent.parent.H2.grid['NCItot']-0.3,})
        elif sp == 'CO':
            m.plot_model(parx='co',  pars=[['tgas','n','pgas']], species=[['COj1/COj0', 'COj2/COj0','COj3/COj0','COj4/COj0'],['NCOj0', 'NCOj1','NCOj2','NCOj3','NCOj4']],
                         pyfit=True, lev = 0,
                         logx=True, logy=True, borders={'H2': self.parent.parent.H2.grid['NH2tot'] - 0.3,
                                                        'CO': self.parent.parent.H2.grid['NCOtot'] - 0.3, })

        plt.show()

    def mouseClickEvent(self, ev):

        if (QApplication.keyboardModifiers() == Qt.ShiftModifier):
            self.active = not self.active
            self.redraw()

        if (QApplication.keyboardModifiers() == Qt.ControlModifier):
            self.plot_model()

        if (QApplication.keyboardModifiers() == Qt.AltModifier):
            self.plot_fit()

    def clicked(self, pts):
        print("clicked: %s" % pts)

class plotGrid(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'log(n)', 'bottom': 'log(UV)'})
        self.initstatus()
        self.vb = self.getViewBox()
        self.image = None
        self.text = None
        self.grid = image()
        cdict = cm.get_cmap('viridis')
        cmap = np.array(cdict.colors)
        cmap[-1] = [1, 0.4, 0]
        map = pg.ColorMap(np.linspace(0, 1, cdict.N), cmap, mode='rgb')
        self.colormap = map.getLookupTable(0.0, 1.0, 256, alpha=False)

    def initstatus(self):
        self.s_status = True
        self.selected_point = None

    def set_data(self, x=None, y=None, z=None, view='text'):
        if view == 'text':
            if self.text is not None:
                for t in self.text:
                    self.vb.removeItem(t)
            self.text = []
            for xi, yi, zi, name in zip(x, y, z, [self.parent.H2.models[m].name for m in self.parent.H2.mask]):
                self.text.append(textLabel(self, '{:.1f}'.format(zi), x=np.log10(xi), y=np.log10(yi), name=name))
                #self.text.append(pg.TextItem(html='<div style="text-align: center"><span style="color: #FF0; font-size: 16pt;">' + '{:.1f}'.format(zi) + '</span></div>'))
                #self.text[-1].setPos(np.log10(xi), np.log10(yi))
                self.vb.addItem(self.text[-1])

        if view == 'image':
            if self.image is not None:
                self.vb.removeItem(self.image)
            self.image = pg.ImageItem()
            self.grid.set_data(x=x, y=y, z=z)
            self.image.translate(self.grid.pos[0], self.grid.pos[1])
            self.image.scale(self.grid.scale[0], self.grid.scale[1])
            self.image.setLookupTable(self.colormap)
            self.image.setLevels(self.grid.levels)
            self.vb.addItem(self.image)

    def updateUVAxis(self):
        self.uv_axis.setGeometry(self.getPlotItem().sceneBoundingRect())
        self.uv_axis.linkedViewChanged(self.getViewBox(), self.uv_axis.YAxis)
        MainPlotXMin, MainPlotXMax = self.viewRange()[1]
        print(self.viewRange())
        scale = {'Mathis': 0, 'Draine': np.log10(1.39), 'Habing': np.log10(0.81)}
        if self.parent.grid_pars.uv_type.currentText() in ['Mathis', 'Draine', 'Habing']:
            AuxPlotXMin, AuxPlotXMax = MainPlotXMin + scale[
                self.parent.grid_pars.uv_type.currentText()], MainPlotXMax + self.parent.grid_pars.uv_type.currentText()

        elif self.parent.grid_pars.uv_type.currentText() == 'AGN':
            AuxPlotXMin, AuxPlotXMax = -0.5 * MainPlotXMin, -0.5 * MainPlotXMax
        print(AuxPlotXMin, AuxPlotXMax)

        self.uv_axis.setYRange(AuxPlotXMax, AuxPlotXMin, padding=0)

    def mousePressEvent(self, event, pos=None):
        print(pos)
        if pos is None:
            super(plotGrid, self).mousePressEvent(event)
            if event.button() == Qt.LeftButton:
                self.mousePoint = self.vb.mapSceneToView(event.pos())
                self.x, self.y = self.mousePoint.x(), self.mousePoint.y()
        else:
            self.x, self.y = pos
        print(self.x, self.y)
        if self.s_status:
            name = self.parent.H2.grid['name']
            spname = str(self.parent.grid_pars.species.currentText())
            relative = self.parent.grid_pars.relative.isChecked()

            if self.parent.grid_pars.uv_type.currentText() == 'AGN':
                grid = self.parent.grid_pars.pars
                pars = [list(grid.keys())[list(grid.values()).index('x')],
                        list(grid.keys())[list(grid.values()).index('y')]]
                ind = pars.index('uv')
                print(self.parent.grid_pars.rescale)
                if ind == 0:
                    self.x = np.log10(self.parent.grid_pars.rescale / 10 ** (self.x * 2))
                else:
                    self.y = np.log10(self.parent.grid_pars.rescale / 10 ** (self.y * 2))
            if self.parent.grid_pars.cols is not None:
                cols = {}
                sp = self.parent.H2.grid['cols'][0].keys()
                lnL = 0
                for s in sp:
                    v = self.parent.H2.comp(name).e[s].col
                    if self.parent.H2.grid['ndims'] == 2:
                        cols[s] = self.parent.grid_pars.cols[s](self.x, self.y)
                        #colsj0 = self.parent.grid_pars.cols[spname + 'j0'](self.x, self.y)
                        #if relative:
                        #    v = v / self.parent.H2.comp(name).e[spname + 'j0'].col.log()  # a(cols0,0,0,'l')
                        #    cols[s] -= colsj0
                    elif self.parent.H2.grid['ndims'] == 3:
                        cols[s] = self.parent.grid_pars.cols[s]([[self.x, self.y, float(self.parent.grid_pars.zval.text())]])[0]
                        #colsj0 = self.parent.grid_pars.cols[spname + 'j0']([[self.x, self.y, float(self.parent.grid_pars.zval.text())]])[0]
                        #if relative:
                        #    print('v',v)
                        #    v = v / self.parent.H2.comp(name).e[spname + 'j0'].col.log()  # a(cols0,0,0,'l')
                        #    cols[s] -= colsj0
                    if v.type == 'm':
                        if relative:
                            v = v / self.parent.H2.comp(name).e[spname + 'j0'].col.log()
                            v1 = v * a(0, 0.2, 0.2, 'l')
                            lnL += v1.lnL(cols[s]-cols[spname + 'j0'])
                        else:
                            v1 = v * a(0, 0.2, 0.2, 'l')
                            lnL += v1.lnL(cols[s])
                print('cols', cols)
                print('lnL',lnL)
                self.parent.plot_exc.add_temp(cols, add=False)
                self.parent.plot_exc.add_temp(cols, pars=[self.x, self.y])
                if 0:
                    sy = self.parent.H2.comp(name)
                    if spname == 'H2' or spname == 'CI':
                        if 'T01' in sy.e.keys():
                            Texc = sy.e['T01'].col.val
                    elif spname == 'CO':
                        if 'T_co' in sy.e.keys():
                            Texc = sy.e['T_co'].col.val
                    self.parent.plot_exc.add_texc(add=False)
                    self.parent.plot_exc.add_texc(cols,Texc=Texc)
                    #self.parent.plot_exc.add_texc(cols, cols0, Texc=Texcp)
                    #self.parent.plot_exc.add_texc(cols, cols0, Texc=Texcm)
                if 0:
                    if spname == 'CO':
                        #self.parent.plot_exc.add_cmb(add=False)
                        sy = self.parent.H2.comp(name)
                        #self.parent.plot_exc.add_cmb(cols,redshift=sy.z)


    def keyPressEvent(self, event):
        super(plotGrid, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_S:
                self.s_status = True

    def keyReleaseEvent(self, event):
        super(plotGrid, self).keyReleaseEvent(event)
        key = event.key()

        if not event.isAutoRepeat():

            if event.key() == Qt.Key_S:
                self.s_status = True
                self.parent.plot_exc.add_temp([], add=False)

class QSOlistTable(pg.TableWidget):
    def __init__(self, parent):
        super().__init__(editable=False, sortable=False)
        self.setStyleSheet(open('styles.ini').read())
        self.parent = parent
        self.format = None

        self.contextMenu.addSeparator()
        self.contextMenu.addAction('results').triggered.connect(self.show_results)

        self.resize(100, 1200)
        self.show()

    def setdata(self, data):
        self.data = data
        self.setData(data)
        if self.format is not None:
            for k, v in self.format.items():
                self.setFormat(v, self.columnIndex(k))
        self.resizeColumnsToContents()
        self.horizontalHeader().setResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setResizeMode(1, QHeaderView.ResizeToContents)
        w = 180 + self.verticalHeader().width() + self.autoScrollMargin()*1.5
        w += sum([self.columnWidth(c) for c in range(self.columnCount())])
        self.resize(w, self.size().height())
        self.setSortingEnabled(False)

    def compare(self, species='H2'):
        grid = self.parent.parent.grid_pars.pars
        syst = float(self.parent.parent.grid_pars.addSyst.text()) if self.parent.parent.grid_pars.addSyst.text().strip() != '' else 0
        syst_factor = float(self.parent.parent.grid_pars.multSyst.text()) if self.parent.parent.grid_pars.multSyst.text().strip() != '' else 1
        pars = [list(grid.keys())[list(grid.values()).index('x')],
                list(grid.keys())[list(grid.values()).index('y')]]
        if 'z' in list(grid.values()):
            pars.append(list(grid.keys())[list(grid.values()).index('z')])
        fix = [k for k in grid.keys() if grid[k] == 'fixed']
        fixed = {}
        for f in fix:
            if getattr(self.parent.parent.grid_pars, f + '_val').currentText() != '':
                fixed[f] = float(getattr(self.parent.parent.grid_pars, f + '_val').currentText())
            else:
                fixed[f] = 'all'
        name = ''
        for idx in self.selectedIndexes():
            if idx.column() == 0:
                name = self.cell_value('name')
        #self.parent.parent.H2.comparegrid(name, species=species, pars=pars, fixed=fixed, syst=syst, syst_factor=syst_factor,plot=False,
        #                                  levels=self.parent.parent.grid_pars.H2levels,
        #                                  others=self.parent.parent.grid_pars.othermode.currentText(),
        #                                  sides=int(self.parent.parent.grid_pars.sides.isChecked())+1,
        #                                  spmode = spmode)
        self.parent.parent.H2.comparegrid(name, species=species, pars=pars, fixed=fixed, syst=syst,
                                          syst_factor=syst_factor, plot=False,
                                          levels=self.parent.parent.grid_pars.H2levels,
                                          others=self.parent.parent.grid_pars.othermode.currentText(),
                                          relative=self.parent.parent.grid_pars.relative.isChecked(),
                                          sides=int(self.parent.parent.grid_pars.sides.currentIndex())
                                          )
        grid = self.parent.parent.H2.grid
        self.parent.parent.H2.grid['name'] = name
        self.parent.parent.H2.grid['ndims'] = len(pars)
        #print('grid', grid['uv'], grid['n0'], grid['lnL'])
        self.parent.parent.plot_reg.set_data(x=grid[pars[0]], y=grid[pars[1]], z=grid['lnL'])
        self.parent.parent.plot_reg.setLabels(bottom='log('+pars[0]+')', left='log('+pars[1]+')')
        #self.pos = [self.x[0] - (self.x[1] - self.x[0]) / 2, self.y[0] - (self.y[1] - self.y[0]) / 2]
        #self.scale = [(self.x[-1] - self.x[0]) / (self.x.shape[0] - 1), (self.y[-1] - self.y[0]) / (self.y.shape[0] - 1)]

    def show_results(self):
        for idx in self.selectedIndexes():
            if idx.column() == 0:
                name = self.cell_value('name')
                str = "".join(['result/all/', name, '.pkl'])
                strnodes = "".join(['result/all/nodes/', name, '.pkl'])
                with open(strnodes, 'rb') as f:
                    x1, y1, z1 = pickle.load(f)
                with open(str, 'rb') as f:
                    x, y, z = pickle.load(f)
                    X, Y = np.meshgrid(x, y)
                    plt.subplot(1, 2, 1)
                    plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
                    plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
                    plt.title('RBF interpolation - likelihood')
                    plt.xlim(-1.2, 3.2)
                    plt.ylim(0.8, 5.2)
                    plt.colorbar()

                    d = distr2d(x=x, y=y, z=np.exp(z))
                    dx, dy = d.marginalize('y'), d.marginalize('x')
                    dx.stats(latex=2, name='log UV')
                    dy.stats(latex=2, name='log n')
                    # d.plot(color=None)
                    ax = plt.subplot(1, 2, 2)
                    d.plot_contour(ax=ax, color='lime', xlabel='$\log n$ [cm$^{-3}$]', ylabel='$\log I_{UV}$ [Draine field]', conf_levels = [0.3, 0.683])
                    d.plot(color='lime', xlabel='$\log n$ [cm$^{-3}$]', ylabel='$\log I_{UV}$ [Draine field]')

                    plt.show()

    def columnIndex(self, columnname):
        return [self.horizontalHeaderItem(x).text() for x in range(self.columnCount())].index(columnname)

    def cell_value(self, columnname, row=None):
        if row is None:
            row = self.currentItem().row()

        cell = self.item(row, self.columnIndex(columnname)).text()  # get cell at row, col

        return cell

class chooseH2SystemWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent, closebutton=True):
        super().__init__()
        self.parent = parent
        #self.resize(700, 900)
        #self.move(400, 100)
        self.setStyleSheet(open('styles.ini').read())

        self.saved = []

        layout = QVBoxLayout()

        self.table = QSOlistTable(self)
        data = self.parent.H2.H2.makelist(pars=['z_dla', 'CO__val'], sys=self.parent.H2.H2.all(species='CO'), view='numpy')
        #data = self.H2.H2.list(pars=['name', 'H2', 'metallicity'])
        self.table.setdata(data)
        self.table.setSelectionBehavior(QTableView.SelectRows);
        self.buttons = {}
        for i, d in enumerate(data):
            wdg = QWidget()
            l = QVBoxLayout()
            l.addSpacing(3)
            button = QPushButton(d[0], self, checkable=True)
            button.setFixedSize(100, 30)
            button.setChecked(False)
            button.clicked[bool].connect(partial(self.click, d[0]))
            self.buttons[d[0]] = button
            l.addWidget(button)
            l.addSpacing(3)
            l.setContentsMargins(0, 0, 0, 0)
            wdg.setLayout(l)
            self.table.setCellWidget(i, 0, wdg)

        layout.addWidget(self.table)

        self.scroll = None

        self.layout = QVBoxLayout()
        layout.addLayout(self.layout)

        if closebutton:
            self.okButton = QPushButton("Close")
            self.okButton.setFixedSize(110, 30)
            self.okButton.clicked[bool].connect(self.ok)
            hbox = QHBoxLayout()
            hbox.addWidget(self.okButton)
            hbox.addStretch()
            layout.addLayout(hbox)

        self.setLayout(layout)

    def click(self, name):
        self.parent.plot_exc.add(name, self.buttons[name].isChecked())
        self.table.setCurrentCell(np.where(self.table.data['name'] == name)[0][0], 0)

    def ok(self):
        self.hide()
        self.parent.chooseFit = None
        self.deleteLater()

    def cancel(self):
        for par in self.parent.fit.list():
            par.fit = self.saved[str(par)]
        self.close()

class gridParsWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        #self.resize(700, 900)
        #self.move(400, 100)
        #self.pars = {'n0': 'x', 'avmax': 'y', 'Z': 'fixed','uv': 'fixed'}
        self.pars = {'n0': 'x', 'uv': 'y', 'Z': 'fixed', 'Av': 'disable', 'NCO': 'z'}
        self.parent.H2.setgrid(pars=list(self.pars.keys()), show=False)
        self.cols, self.x_, self.y_, self.z_, self.lnL_, self.mpars = None, None, None, None, None, None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('grid parameters:'))

        for n in self.pars.keys():
            l = QHBoxLayout(self)
            l.addWidget(QLabel((n + ': ')[:3]))
            self.group = QButtonGroup(self)
            for b in ('x', 'y', 'z', 'disable', 'fixed'):
                setattr(self, b, QCheckBox(b, checkable=True))
                getattr(self, b).clicked[bool].connect(partial(self.setGridView, par=n, b=b))
                l.addWidget(getattr(self, b))
                self.group.addButton(getattr(self, b))
            getattr(self, self.pars[n]).setChecked(True)
            setattr(self, n + '_val', QComboBox(self))
            getattr(self, n + '_val').setFixedSize(80, 25)
            if None not in self.parent.H2.grid[n]:
                getattr(self, n + '_val').addItems(np.asarray(np.sort(np.unique(self.parent.H2.grid[n])), dtype=str))
                if self.pars[n] == 'fixed':
                    getattr(self, n + '_val').setCurrentIndex(0)
                getattr(self, n + '_val').currentTextChanged[str].connect(partial(self.setGridView, par=n, b='val'))
            else:
                self.pars[n] = 'disable'
                getattr(self, self.pars[n]).setChecked(True)
                for b in ('x', 'y', 'disable', 'fixed'):
                    getattr(self, b).setEnabled(False)
            l.addWidget(getattr(self, n + '_val'))
            if n == 'uv':
                setattr(self, n + '_type', QComboBox(self))
                getattr(self, n + '_type').setFixedSize(80, 25)
                getattr(self, n + '_type').addItems(['Habing', 'Draine', 'Mathis', 'AGN'])
                getattr(self, n + '_type').setCurrentIndex(2)
                getattr(self, n + '_type').currentIndexChanged[str].connect(self.changeUV)
                l.addWidget(getattr(self, n + '_type'))
            l.addStretch(1)
            layout.addLayout(l)

        l = QHBoxLayout(self)
        l.addWidget(QLabel('add systematic unc.:'))
        self.addSyst = QLineEdit()
        self.addSyst.setText(str(0.2))
        self.addSyst.setFixedSize(90, 30)
        l.addWidget(self.addSyst)
        l.addWidget(QLabel(' or x by'))
        self.multSyst = QLineEdit()
        self.multSyst.setText('')
        self.multSyst.setFixedSize(40, 30)
        l.addWidget(self.multSyst)
        l.addWidget(QLabel('  geom:'))
        self.sides = QComboBox()
        self.sides.addItems(['total', 'one side', 'both sides'])
        self.sides.setCurrentIndex(2)
        self.sides.setFixedSize(90, 30)
        l.addWidget(self.sides)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.compare = QPushButton('Compare')
        self.compare.clicked[bool].connect(partial(self.compareIt, False))
        self.compare.setFixedSize(90, 30)
        l.addWidget(self.compare)
        self.species = QComboBox(self)
        self.species.setFixedSize(50, 25)
        self.species.addItems(['H2', 'CI', 'CO'])
        self.species.setCurrentIndex(2)
        l.addWidget(self.species)
        self.H2levels = np.arange(6)
        self.levels = QLineEdit(" ".join([str(i) for i in self.H2levels]))
        self.levels.setFixedSize(90, 30)
        self.levels.editingFinished.connect(self.setLevels)
        l.addWidget(self.levels)
        l.addWidget(QLabel('others:'))
        self.othermode = QComboBox(self)
        self.othermode.setFixedSize(70, 25)
        self.othermode.addItems(['ignore', 'lower', 'upper'])
        self.othermode.setCurrentIndex(0)
        l.addWidget(self.othermode)
        self.relative = QCheckBox('relative')
        self.relative.setChecked(False)
        l.addWidget(self.relative)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.refine = QPushButton('Refine:')
        self.refine.clicked[bool].connect(partial(self.regridIt, kind='accurate'))
        self.refine.setFixedSize(90, 30)
        l.addWidget(self.refine)
        self.numPlot = QLineEdit(str(30))
        self.numPlot.setFixedSize(90, 30)
        l.addWidget(self.numPlot)
        l.addWidget(QLabel('x:'))
        self.xmin = QLineEdit('')
        self.xmin.setFixedSize(30, 30)
        l.addWidget(self.xmin)
        l.addWidget(QLabel('..'))
        self.xmax = QLineEdit('')
        self.xmax.setFixedSize(30, 30)
        l.addWidget(self.xmax)
        l.addWidget(QLabel('y:'))
        self.ymin = QLineEdit('')
        self.ymin.setFixedSize(30, 30)
        l.addWidget(self.ymin)
        l.addWidget(QLabel('..'))
        self.ymax = QLineEdit('')
        self.ymax.setFixedSize(30, 30)
        l.addWidget(self.ymax)
        l.addWidget(QLabel('z='))
        self.zval = QLineEdit('')
        self.zval.setFixedSize(60, 30)
        self.zval.setText(str(15.0))
        l.addWidget(self.zval)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.plot = QPushButton('Plot')
        self.plot.clicked[bool].connect(self.plotIt)
        self.plot.setFixedSize(90, 30)
        l.addWidget(self.plot)
        self.export = QPushButton('Export')
        self.export.clicked[bool].connect(self.exportIt)
        self.export.setFixedSize(90, 30)
        l.addWidget(self.export)
        self.plotphys = QPushButton('PhysC')
        self.plotphys.clicked[bool].connect(self.plot_mpars)
        self.plotphys.setFixedSize(90, 30)
        l.addWidget(self.plotphys)
        l.addStretch(1)
        self.best_fit = QPushButton('Best')
        self.best_fit.clicked[bool].connect(self.bestIt)
        self.best_fit.setFixedSize(60, 30)
        l.addWidget(self.best_fit)
        self.export_table = QPushButton('Table')
        self.export_table.clicked[bool].connect(self.tableIt)
        self.export_table.setFixedSize(90, 30)
        l.addWidget(self.export_table)
        self.visual = QPushButton('Vis')
        self.visual.clicked[bool].connect(self.visGrid)
        self.visual.setFixedSize(30, 30)
        l.addWidget(self.visual)
        layout.addLayout(l)

        #l = QHBoxLayout(self)
        #self.plotphys = QPushButton('PhysC')
        #self.plotphys.clicked[bool].connect(self.plot_mpars)
        #self.plotphys.setFixedSize(90, 30)
        #.addWidget(self.plotphys)
        #l.addStretch(1)
        #layout.addLayout(l)

        l = QHBoxLayout(self)
        self.plot_model_set = QComboBox(self)
        self.plot_model_set.setFixedSize(80, 25)
        self.plot_model_set.addItems(['H2+CI', 'H2+CO', 'OH','CO'])
        self.plot_model_set.setCurrentIndex(0)
        l.addWidget(self.plot_model_set)
        self.joint = QPushButton('Joint')
        self.joint.clicked[bool].connect(self.joinIt)
        self.joint.setFixedSize(90, 30)
        l.addWidget(self.joint)
        l.addStretch(1)
        layout.addLayout(l)

        layout.addStretch(1)

        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())

    def setLevels(self):
        try:
            self.H2levels = [int(s) for s in self.levels.text().split()]
        except:
            pass

    def compareIt(self, init=False):
        print('compareIt', init)
        if init:
            self.parent.H2_systems.table.compare(species='H2')
        else:
            self.parent.H2_systems.table.compare(species=str(self.species.currentText()))
            self.interpolateIt()
        grid = self.parent.H2.grid
        x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
        self.xmin.setText(str(np.min(x)))
        self.xmax.setText(str(np.max(x)))
        self.ymin.setText(str(np.min(y)))
        self.ymax.setText(str(np.max(y)))

    def interpolateIt(self):
        print('interpolate')
        grid = self.parent.H2.grid
        for attr in ['x', 'y', 'z']:
            setattr(self, attr, None)
            if attr in list(self.pars.values()):
                setattr(self, attr, np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index(attr)]]))
        sp = grid['cols'][0].keys()
        #mp = grid['mpars'][0].keys()
        self.cols = {}
        self.mpars = {}
        #sp_full = grid['cols'][0].keys()
        for s in sp:
            if grid['ndims'] == 2:
                if 0:
                    self.cols[s] = interp2d(x, y, [c[s] for c in grid['cols']], kind='cubic')
                if 0:
                    xt, yt = np.unique(sorted(x)), np.unique(sorted(y))
                    z = np.zeros([xt.shape[0], yt.shape[0]])
                    for i, xi in enumerate(xt):
                        for k, yk in enumerate(yt):
                            z[i, k] = grid['cols'][np.argmin((xi - x) ** 2 + (yk - y) ** 2)][s]

                    self.cols[s] = RectBivariateSpline(xt, yt, z, kx=2, ky=2)
                if 0:
                    if s=='H2j0' or s=='H2j1':
                        self.cols[s] = Rbf(x, y, np.asarray([c[s] for c in grid['cols']]), function='inverse', epsilon=0.3)
                    else:
                        self.cols[s] = Rbf(x, y, np.asarray([c[s] for c in grid['cols']]), function='multiquadric',smooth=0.2)
                if 0:
                    self.cols[s] = Rbf(x, y, np.asarray([c[s] for c in grid['cols']]), function='multiquadric', smooth=0.1)
                if 1:
                    self.cols[s] = Rbf(self.x, self.y, np.asarray([c[s] for c in grid['cols']]), function='multiquadric',
                                       smooth=0.1)
                # for p in mp:
                #     self.mpars[p] = Rbf(x, y, np.asarray([np.log10(c[p]) for c in grid['mpars']]), function='multiquadric', smooth=0.1)

            if self.x is not None and self.y is not None and self.z is not None:
                xobs = np.c_[self.x, self.y, self.z]
                print('xobs:',xobs)
                yobs = np.asarray([c[s] for c in grid['cols']])
                print('yobs:',yobs)
                self.cols[s] = RBFInterpolator(xobs, yobs, kernel='multiquadric', epsilon=1)  # , smoothing=0.001)
                # rbf = Rbf(x,y,z,function='multiquadric',smooth=0.2)

    def regridIt(self, kind='accurate', save=True):
        grid = self.parent.H2.grid
        syst = float(self.addSyst.text()) if self.addSyst.text().strip() != '' else 0
        syst_factor = float(self.multSyst.text()) if self.multSyst.text().strip() != '' else 1
        spname = str(self.species.currentText())
        relative = self.parent.grid_pars.relative.isChecked()


        sp = grid['cols'][0].keys()
        sp = [s for s in sp if int(s[3:]) in self.H2levels]
        species = {}
        # add calc phys cond
        if 0:
            mp = grid['mpars'][0].keys()
            mparsgrid = {}
            for p in mp:
                mparsgrid[p] = np.zeros_like(X)
        for s in sp:
            self.parent.H2.comp(grid['name']).e[s].col.log()
            v1 = a(self.parent.H2.comp(grid['name']).e[s].col.log().val,
                   self.parent.H2.comp(grid['name']).e[s].col.log().plus * syst_factor,
                   self.parent.H2.comp(grid['name']).e[s].col.log().minus * syst_factor)
            if kind == 'fast':
                v1.minus, v1.plus = np.sqrt(v1.minus ** 2 + syst ** 2), np.sqrt(v1.plus ** 2 + syst ** 2)
            elif kind == 'accurate':
                v1 *= a(0, syst, syst, 'l')
            if relative:
                species[s] = v1 / self.parent.H2.comp(grid['name']).e[sp[0]].col.log()
            else:
                species[s] = v1
            #if spmode == 'abs' or spmode == 'test':
            #    species[s] = v1
            #elif spmode == 'rel':
            #    species[s] = v1/self.parent.H2.comp(grid['name']).e[sp[0]].col.log()

        if 1:
            others = self.parent.grid_pars.othermode.currentText()
            levels = self.parent.grid_pars.H2levels
            full_keys = grid['cols'][0].keys()
            label = spname + 'j{:}'
            keys = [label.format(i) for i in levels if label.format(i) in full_keys]
            print('regrid keys',keys)

            if others in ['lower', 'upper']:
                for k in full_keys:
                    if k not in keys:
                        v = self.parent.H2.comp(grid['name']).e[k].col.log().copy()
                        if others == 'lower':
                            species[k] = a(v.val - v.minus, t=others[0])
                        else:
                            species[k] = a(v.val + v.plus, t=others[0])


        num = int(self.numPlot.text())
        x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(
            grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
        x1, y1 = x[:], y[:]  # copy for save
        if grid['ndims'] == 2:
            x, y = np.linspace(float(self.xmin.text()), float(self.xmax.text()), num), np.linspace(float(self.ymin.text()), float(self.ymax.text()), num)
            X, Y = np.meshgrid(x, y)
            lnL = np.zeros_like(X)
            if save:
                cols = {}
                for s in self.cols.keys():
                    cols[s] = np.zeros_like(lnL)
            for i, xi in enumerate(x):
                for k, yi in enumerate(y):
                    L = 0
                    for s, v in species.items():
                        if v.type in ['m', 'u', 'l']:
                            if relative:
                                cols[s][k, i] = self.cols[s](xi, yi) - self.cols[spname + 'j0'](xi, yi)
                                L += v.lnL(cols[s][k, i])
                            else:
                                cols[s][k, i] = self.cols[s](xi, yi)
                                L += v.lnL(cols[s][k, i])

                            #if spmode == 'abs' or spmode == 'test':
                            #    cols[s][k, i] = self.cols[s](xi, yi)
                            #    L += v.lnL(cols[s][k, i])
                            #elif spmode == 'rel':
                            #    cols[s][k, i] = self.cols[s](xi, yi) - self.cols[spname+'j0'](xi, yi)
                            #    L += v.lnL(cols[s][k, i])
                                #print('v=', v, 'PDR', cols[s][k, i], '+lnL', v.lnL(cols[s][k, i]))
                    lnL[k, i] = L
            self.x_, self.y_, self.lnL_ = x, y, lnL


        elif grid['ndims'] == 3:
            x, y = np.linspace(float(self.xmin.text()), float(self.xmax.text()), num), np.linspace(
                float(self.ymin.text()), float(self.ymax.text()), num)
            z = float(self.zval.text())
            print('selected frame at z-axis:',z)
            X, Y = np.meshgrid(x, y)
            lnL = np.zeros_like(X)

            cols = {}
            for s in self.cols.keys():
                cols[s] = np.zeros_like(lnL)

            for i, xi in enumerate(x):
                for k, yi in enumerate(y):
                    data, L = [[xi, yi, z]], 0
                    for s, v in species.items():
                        if v.type in ['m', 'u', 'l']:
                            #if spmode == 'abs' or spmode == 'test':
                            #    cols[s][k, i] = self.cols[s](data)
                            #elif spmode == 'rel':
                            #    cols[s][k, i] = self.cols[s](data) - self.cols[spname + 'j0'](data)
                            if relative:
                                cols[s][k, i] = self.cols[s](data) - self.cols[spname + 'j0'](data)
                            else:
                                cols[s][k, i] = self.cols[s](data)

                            L += v.lnL(cols[s][k, i])

                    lnL[k, i] = L
            self.x_, self.y_, self.lnL_ = x, y, lnL

        if save:
            for s in cols.keys():
                with open('temp/{:s}'.format(s), 'wb') as f:
                    pickle.dump([self.x_, self.y_, cols[s]], f)

                z1 = np.asarray([c[s] for c in grid['cols']])
                with open('temp/{:s}_nodes'.format(s), 'wb') as f:
                    pickle.dump([x1, y1, z1], f)

            lnL1 = np.asarray(grid['lnL'])
            with open('output/nodes/{:s}_lnL.pkl'.format(self.parent.H2.grid['name']), 'wb') as f:
                pickle.dump([x1, y1, lnL1], f)
            with open('temp/lnL_nodes.pkl', 'wb') as f:
                pickle.dump([x1, y1, lnL1], f)
            with open('temp/lnL.pkl', 'wb') as f:
                pickle.dump([self.x_, self.y_, self.z_], f)

    def plot_mpars(self):
        if self.mpars is not None:
            mp = self.mpars.keys()
            mparsgrid = {}
            for p in mp:
                mparsgrid[p] = np.zeros_like(self.z_)
            #levels = [2.0,2.1]



            if self.x_ is not None:
                for i, xi in enumerate(self.x_):
                    for k, yi in enumerate(self.y_):
                        for p in mp:
                            mparsgrid[p][k,i] = self.mpars[p](xi, yi)

            X, Y = np.meshgrid(self.x_, self.y_)




            fig, ax = plt.subplots(1,len(mp))
            for k,p in enumerate(mp):
                z = mparsgrid[p]
                if p == 'tgas':
                    vmin, vmax = 1,3
                    cmap = 'Greens'
                elif p == 'pgas':
                    vmin, vmax = 2,7
                    cmap = 'Reds'
                elif p == 'pgas_h2':
                    vmin, vmax = 2, 7
                    cmap = 'Reds'
                if vmin is None:
                    ax[k].pcolor(X, Y, z, cmap=cmap)
                else:
                    ax[k].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)

                #ax.contour(self.x_, self.y_, z, levels=levels, colors='black', alpha=1,
                #                 linestyles=['-'], linewidths=1.0)

                # plot text grids
                if 1:
                    grid = self.parent.H2.grid
                    x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(
                        grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
                    z = np.asarray([np.log10(c[p]) for c in grid['mpars']])
                    for v1, v2, l in zip(x, y, z):
                        ax[k].text(v1, v2, '{:.1f}'.format(l), size=10, color='black')
                    #for i, xi in enumerate(self.x_):
                    #    for j, yj in enumerate(self.y_):
                    #        ax[k].text(xi, yj, '{:.1f}'.format(mparsgrid[p][j,i]), size=5, color='red')

            # plot colorbar
            if 0:
                cax = fig.add_axes([0.93, 0.27, 0.01, 0.47])
                fig.colorbar(c, cax=cax, orientation='vertical') #, ticks=[1, 1.5, 1.7, 2, 2.3])

            #plot H2 contour
            if 0:
                d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_))
                dx, dy = d.marginalize('y'), d.marginalize('x')
                dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
                dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
                d.plot_contour(ax=ax,color='blue', color_point=None, cmap=None, alpha=0, lw=1.0, zorder=5)

            # calc boot t_gas
            if 0:
                p = 'tgas'
                pmin = np.min(mparsgrid[p])
                pmax = np.max(mparsgrid[p])
                #if pmax>1.0e+03:
                #    pmax = 1.0e+03
                print('tgas:stat:',pmin, pmax)
                lp = np.linspace(pmin,pmax,1000)
                lnL_p = np.zeros_like(lp)
                lnL = np.exp(self.z_)
                for i, xi in enumerate(self.x_):
                    for k, yi in enumerate(self.y_):
                        pval = mparsgrid[p][k,i]
                        j = np.searchsorted(lp,pval)
                        lnL_p[j] +=lnL[k][i]
                lnL_p = lnL_p/np.sum(lnL_p)
                #print(lp,lnL_p)
                d = distr1d(lp,lnL_p)
                d.stats()
                d.plot()


            plt.show()

    def visGrid(self):
        grid = self.parent.parent.grid_pars.pars
        pars = [list(grid.keys())[list(grid.values()).index('x')],
                list(grid.keys())[list(grid.values()).index('y')]]
        if 'z' in list(grid.values()):
            pars.append(list(grid.keys())[list(grid.values()).index('z')])
        #grid = self.parent.H2.grid
        #pars = self.pars
        if grid['ndims'] == 3:
            x, y, z = np.log10(grid[list(pars.keys())[list(pars.values()).index('x')]]), np.log10(
                grid[list(pars.keys())[list(pars.values()).index('y')]]), np.log10(
                grid[list(pars.keys())[list(pars.values()).index('z')]])
            names = list(pars.keys())
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                               name='a',
                                               mode='markers',
                                               marker=dict(
                                                   size=8,
                                                   color=z,  # set color to an array/list of desired values
                                                   colorscale='Viridis',  # choose a colorscale
                                                   opacity=0.8
                                               )
                                               )])
            fig.update_layout(scene=dict(
                xaxis_title=names[0],
                yaxis_title=names[1],
                zaxis_title=names[2]),
                # width=700,
                margin=dict(r=20, b=10, l=10, t=10)
            )

            fig.show()

    def plotIt(self):
        print('Plot:', str(self.plot_model_set.currentText()))
        if self.x is not None:
            print(self.uv_type.currentText())
            print('self.x_',self.x_)
            data = self.rescaleUV(self.uv_type.currentText(), data={'x': 10**self.x_, 'y': 10**self.y_, 'z': self.lnL_})
            d = distr2d(x=np.log10(data['x']), y=np.log10(data['y']), z=np.exp(data['z']))
            dx, dy = d.marginalize('y'), d.marginalize('x')
            dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
            dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
            d.plot(color=None, xlabel=data['bottom'], ylabel=data['left'])
            plt.show()
        #self.plot_mpars()

    # export likelihood to file: in case of 'H2' in addition plot join fit to 'H2+CI'
    def exportIt(self):
        data = self.rescaleUV(self.uv_type.currentText(), data={'x': 10**self.x_, 'y': 10**self.y_, 'z': self.lnL_})
        with open('output/{0:s}_{1:s}_lnL.pkl'.format(self.parent.H2.grid['name'], str(self.species.currentText())),
                  'wb') as f:
            pickle.dump([data['x'], data['y'], data['z']], f)
        with open('temp/lnL.pkl', 'wb') as f:
            pickle.dump([data['x'], data['y'], data['z']], f)
        data = self.rescaleUV(self.uv_type.currentText(), data={'x': 10**self.x_, 'y': 10**self.y_, 'z': self.cols})

        with open('output/{0:s}_{1:s}_cols.pkl'.format(self.parent.H2.grid['name'], str(self.species.currentText())),
                  'wb') as f:
            pickle.dump([data['x'], data['y'], data['z']], f)

    '''        
    def exportIt(self):
        # export likelihood and cols grid
        if str(self.species.currentText()) == 'CI':
            name = 'output/{:s}_CI_lnL.pkl'
            with open(name.format(self.parent.H2.grid['name']), 'wb') as f:
                pickle.dump([self.x_, self.y_, self.z_], f)
            name = 'output/{:s}_CI_cols.pkl'
            with open(name.format(self.parent.H2.grid['name']), 'wb') as f:
                pickle.dump([self.x_, self.y_, self.cols], f)

        if str(self.species.currentText()) == 'CO':
            name = 'output/{:s}_CO_lnL.pkl'
            with open(name.format(self.parent.H2.grid['name']), 'wb') as f:
                pickle.dump([self.x_, self.y_, self.z_], f)
            name = 'output/{:s}_CO_cols.pkl'
            with open(name.format(self.parent.H2.grid['name']), 'wb') as f:
                pickle.dump([self.x_, self.y_, self.cols], f)

        if str(self.species.currentText()) == 'H2':
            name = 'output/{:s}_H2_lnL.pkl'
            with open(name.format(self.parent.H2.grid['name']), 'wb') as f:
                pickle.dump([self.x_, self.y_, self.z_], f)
            name = 'output/{:s}_H2_cols.pkl'
            with open(name.format(self.parent.H2.grid['name']), 'wb') as f:
                pickle.dump([self.x_, self.y_, self.cols], f)

            # create a join likelihood
            # self.x_, self.y_, self.z_ - likelihood grid from fit to H2 exc
            if 0:
                if str(self.plot_model_set.currentText()) == 'H2+CI':
                    q = self.parent.H2.comp(self.parent.H2.grid['name'])
                    sp = [s for s in q.e.keys()]
                    prior = []
                    if 'CIj1' in sp:
                        prior.append('ci')
                        with open('output/{:s}_CI_lnL.pkl'.format(self.parent.H2.grid['name']), 'rb') as f:
                            x, y, z_ci = pickle.load(f)
                        with open('output/{:s}_CI_cols.pkl'.format(self.parent.H2.grid['name']), 'rb') as f:
                            x, y, cols_ci = pickle.load(f)
                    else:
                        # set uniform prior if CI is not detected
                        prior.append('uni')
                        x = self.x_
                        z_uni = np.zeros([len(self.y_), len(self.x_)])

                    if 'ci' in prior:
                        z_prior = z_ci
                    elif 'uni' in prior:
                        z_prior = z_uni

                    # create a join likelihood (H2 + prior)
                    d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_ + z_prior))
                    d.dopoint()
                    print('min chi2 coords:', d.point[0], d.point[1])
                    name = 'output/{:s}_H2CI_lnL.pkl'
                    # save a join likelihood
                    with open(name.format(self.parent.H2.grid['name']), 'wb') as f:
                        pickle.dump([self.x_, self.y_, self.z_ + z_prior], f)

                    # show join likelihood and excitation plots
                    if 1:
                        fig_ex, ax_export = plt.subplots(1, 3, figsize=(9, 2))

                        # plot contours for the join constraint
                        d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
                        dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
                        d.plot_contour(ax=ax_export[0], color='black', color_point=None, cmap='Purples', alpha=0,
                                       lw=2.0)

                        d = distr2d(x=self.x_, y=self.y_, z=np.exp(z_prior))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
                        dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
                        d.plot_contour(ax=ax_export[0], color='black', color_point=None, cmap='Greens', alpha=0,
                                       lw=2.0)

                        print('CI+H2 likelihood')
                        d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_ + z_prior))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
                        dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
                        d.plot_contour(ax=ax_export[0], color='black', color_point=None, cmap='Reds', alpha=0,
                                       lw=2.0)

                        # plot excitation for H2
                        def getatomic(species, levels=[0, 1, 2]):
                            if species == 'H2':
                                return [H2energy[0, i] for i in levels], [stat_H2[i] for i in levels]
                            elif species == 'CI':
                                return [CIenergy[i] for i in levels], [stat_CI[i] for i in levels]
                            elif species == 'CO':
                                return [COenergy[i] for i in levels], [stat_CO[i] for i in levels]

                        species = 'H2'
                        sp = [s for s in q.e.keys() if species + 'j' in s]
                        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                        x, stat = getatomic(species, levels=j)
                        y = [q.e[species + 'j' + str(i)].col / stat[i] for i in j]
                        ax_export[1].plot(x, [y[i].val for i in j], 'o')
                        # j = [0,1,2,3,4,5,6]
                        x, stat = getatomic(species, levels=j)
                        mod = [self.cols['H2j' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
                        ax_export[1].plot(x, mod)

                        # plot excitation for CI
                        if 'ci' in prior:
                            species = 'CI'
                            sp = [s for s in q.e.keys() if species + 'j' in s]
                            j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                            x, stat = getatomic(species, levels=j)
                            y = [q.e[species + 'j' + str(i)].col / stat[i] for i in j]
                            mod = [cols_ci['CIj' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
                            if str(self.spmode.currentText()) == 'abs':
                                ax_export[2].plot(x, [y[i].val for i in j], 'o')
                                ax_export[2].plot(x, mod)
                            elif str(self.spmode.currentText()) == 'rel':
                                ax_export[2].plot(x, [y[i].val - y[0].val for i in j], 'o')
                                ax_export[2].plot(x, mod - mod[0])

                            #x, stat = getatomic(species, levels=j)
                            #mod = [cols_ci['CIj' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
                            #ax_export[2].plot(x, mod - mod[0])

                if str(self.plot_model_set.currentText()) == 'H2+CO':
                    q = self.parent.H2.comp(self.parent.H2.grid['name'])
                    sp = [s for s in q.e.keys()]
                    prior = []
                    if 'COj2' in sp:
                        prior.append('co')
                        with open('output/{:s}_CO_lnL.pkl'.format(self.parent.H2.grid['name']), 'rb') as f:
                            x, y, z_co = pickle.load(f)
                        with open('output/{:s}_CO_cols.pkl'.format(self.parent.H2.grid['name']), 'rb') as f:
                            x, y, cols_co = pickle.load(f)
                    else:
                        # set uniform prior if CI did not detected
                        prior.append('uni')
                        x = self.x_
                        z_uni = np.zeros([len(self.y_), len(self.x_)])

                    if 'co' in prior:
                        z_prior = z_co
                    elif 'uni' in prior:
                        z_prior = z_uni

                    # create a join likelihood (H2 + prior)
                    print('Join likelihood for H2 + CO levels')
                    d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_ + z_prior))
                    d.dopoint()
                    print('min chi2 coords:', d.point[0], d.point[1])
                    name = 'output/{:s}_H2CO_lnL.pkl'
                    # save a join likelihood
                    with open(name.format(self.parent.H2.grid['name']), 'wb') as f:
                        pickle.dump([self.x_, self.y_, self.z_ + z_prior], f)

                    # show join likelihood and excitation plots
                    if 1:
                        fig_ex, ax_export = plt.subplots(1, 3, figsize=(9, 2))

                        # plot contours for the join constraint
                        d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
                        dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
                        d.plot_contour(ax=ax_export[0], color='black', color_point=None, cmap='Purples', alpha=0,
                                       lw=2.0)

                        d = distr2d(x=self.x_, y=self.y_, z=np.exp(z_prior))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
                        dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
                        d.plot_contour(ax=ax_export[0], color='black', color_point=None, cmap='Greens', alpha=0,
                                       lw=2.0)

                        print('CO+H2 likelihood')
                        d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_ + z_prior))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
                        dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
                        d.plot_contour(ax=ax_export[0], color='black', color_point=None, cmap='Reds', alpha=0,
                                       lw=2.0)

                        # plot excitation for H2
                        def getatomic(species, levels=[0, 1, 2]):
                            if species == 'H2':
                                return [H2energy[0, i] for i in levels], [stat_H2[i] for i in levels]
                            elif species == 'CI':
                                return [CIenergy[i] for i in levels], [stat_CI[i] for i in levels]
                            elif species == 'CO':
                                return [COenergy[i] for i in levels], [stat_CO[i] for i in levels]

                        species = 'H2'
                        sp = [s for s in q.e.keys() if species + 'j' in s]
                        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                        x, stat = getatomic(species, levels=j)
                        y = [q.e[species + 'j' + str(i)].col / stat[i] for i in j]
                        ax_export[1].plot(x, [y[i].val for i in j], 'o')
                        x, stat = getatomic(species, levels=j)
                        mod = [self.cols['H2j' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
                        ax_export[1].plot(x, mod)

                        # plot excitation for CO
                        if 'co' in prior:
                            species = 'CO'
                            sp = [s for s in q.e.keys() if species + 'j' in s]
                            j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                            x, stat = getatomic(species, levels=j)
                            y = [q.e[species + 'j' + str(i)].col / stat[i] for i in j]
                            mod = [cols_co['COj' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
                            if str(self.spmode.currentText()) == 'abs':
                                ax_export[2].plot(x, [y[i].val for i in j], 'o')
                                ax_export[2].plot(x, mod)
                            elif str(self.spmode.currentText()) == 'rel':
                                ax_export[2].plot(x, [y[i].val - y[0].val for i in j], 'o')
                                ax_export[2].plot(x, mod - mod[0])
                            #ax_export[2].plot(x, [y[i].val for i in j], 'o')
                            #x, stat = getatomic(species, levels=j)
                            #mod = [cols_co['COj' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
                            #ax_export[2].plot(x, mod)

                plt.show()
    '''

    def bestIt(self):
        inds = np.where(self.lnL_ == np.max(self.lnL_.flatten()))
        print(self.x_[inds[1][0]], self.y_[inds[0][0]])
        self.parent.plot_reg.mousePressEvent(None, pos=(self.x_[inds[1][0]], self.y_[inds[0][0]]))
        #data = self.rescaleUV(self.uv_type.currentText(), data={'x': 10 ** self.x_, 'y': 10 ** self.y_, 'z': self.lnL_})

# Debug - udapt to 3 parameter's space
    def tableIt(self):
        #print(self.parent.plot_reg.mousePoint.x(), self.parent.plot_reg.mousePoint.y())
        print(self.parent.plot_reg.x, self.parent.plot_reg.y)
        d = [['species', 'observed', 'model']]
        if self.parent.grid_pars.cols is not None:
            cols = {}
            for s in self.parent.H2.grid['cols'][0].keys():
                cols[s] = self.parent.grid_pars.cols[s](self.parent.plot_reg.x, self.parent.plot_reg.y)
                #cols[s] = self.parent.grid_pars.cols[s](self.parent.plot_reg.mousePoint.x(), self.parent.plot_reg.mousePoint.y())

        q = self.parent.H2.H2.getcomp(self.parent.H2.grid['name'])
        for e in ['H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7']:
            d.append([e.replace('H2', 'H$_2$ ').replace('j', 'J='), q.e[e].col.latex(f=2), '{:5.2f}'.format(cols[e])])
        if 0:
            pr = pyratio(z=q.z)
            n = [q.e['CIj0'].col, q.e['CIj1'].col, q.e['CIj2'].col]
            pr.add_spec('CI', n)
            pr.set_pars(['T', 'n', 'f', 'UV'])
            pr.pars['UV'].value = self.parent.plot_reg.mousePoint.y()
            pr.pars['n'].value = self.parent.plot_reg.mousePoint.x() - 0.3
            pr.set_prior('f', a(0, 0, 0))
            pr.set_prior('T', a(q.e['T01'].col.log().val, 0, 0))
            p = pr.predict(name='CI', level=-1, logN=n[0]+n[1]+n[2])
            for i, e in enumerate(['CIj0', 'CIj1', 'CIj2']):
                d.append([e.replace('j0', '').replace('j1', '*').replace('j2', '**'), q.e[e].col.log().latex(f=2), '{:5.2f}$^a$'.format(p[i].val)])

        print(d)
        output = StringIO()
        ascii.write([list(i) for i in zip(*d[1:])], output, names=d[0], format='latex')
        table = output.getvalue()
        print(table)
        output.close()

    def joinIt(self):
        for idx in self.parent.H2_systems.table.selectedIndexes():
            if idx.column() == 0:
                name = self.parent.H2_systems.table.cell_value('name')

        if 'H2+CI' in self.plot_model_set.currentText():
            H2 = {}
            with open('output/{0:s}_{1:s}_lnL.pkl'.format(name, 'H2'), 'rb') as f:
                H2['n'], H2['UV'], H2['lnL'] = pickle.load(f)
                H2['n'], H2['UV'] = np.log10(H2['n']), np.log10(H2['UV'])
            CI = {}
            with open('output/{0:s}_{1:s}_lnL.pkl'.format(name, 'CI'), 'rb') as f:
                CI['n'], CI['UV'], CI['lnL'] = pickle.load(f)
                CI['n'], CI['UV'] = np.log10(CI['n']), np.log10(CI['UV'])

            X, Y = np.meshgrid(np.linspace(np.min(H2['n']) + 0.1, np.max(H2['n']) - 0.1, 100),
                               np.linspace(np.min(H2['UV']) + 0.1, np.max(H2['UV']) - 0.1, 100))
            X, Y = X.flatten(), Y.flatten()
            z = np.zeros_like(X.flatten())

            x, y = np.meshgrid(H2['n'], H2['UV'])
            h2 = Rbf(x.flatten(), y.flatten(), H2['lnL'].flatten(), function='multiquadric', smooth=0.1)

            x, y = np.meshgrid(CI['n'], CI['UV'])
            ci = Rbf(x.flatten(), y.flatten(), CI['lnL'].flatten(), function='multiquadric', smooth=0.1)
            z = h2(X, Y) + ci(X, Y)
            z -= np.max(z.flatten())
            d = distr2d(X, Y, np.exp(z))

            xlabel, ylabel = r'$\log$ n$_{\rm H}$ [cm$^{-3}$]', r'$\log\,I_{\rm UV}$'
            proxy = []
            fig = d.plot(frac=0.15, indent=0.11, xlabel=xlabel, ylabel=ylabel,
                         color='orangered', cmap='Reds', color_marg='orangered', alpha=0, zorder=20)
            for attr, par in zip(['x', 'y'], ['UV', 'n']):
                d1 = d.marginalize(attr)
                d1.stats(latex=2, name=par)

            ax = fig.get_axes()[0]
            d = distr2d(H2['n'], H2['UV'], np.exp(H2['lnL']))
            d.plot_contour(ax=ax, color="violet", cmap='BuPu', color_point=None, xlabel=xlabel, ylabel=ylabel,
                           ls=None, alpha=0, label=r'H$_2$')
            # proxy.append(mpatches.Patch(color='blueviolet', label=r'H$_2$'))

            d = distr2d(CI['n'], CI['UV'], np.exp(CI['lnL']))
            d.plot_contour(ax=ax, color="limegreen", cmap='BuGn', color_point=None, xlabel=xlabel, ylabel=ylabel,
                           ls=None, alpha=0, label=r'C$\,$I')
            proxy.append(mpatches.Patch(color='green', label=r'C$\,$I'))

            proxy.append(mpatches.Patch(color='orangered', label=r'joint'))

            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.legend(handles=proxy, frameon=False, fontsize=12)
            fig.tight_layout()

            plt.show()
        if 'H2+CO' in self.plot_model_set.currentText():
            H2 = {}
            with open('output/{0:s}_{1:s}_lnL.pkl'.format(name, 'H2'), 'rb') as f:
                H2['n'], H2['UV'], H2['lnL'] = pickle.load(f)
                H2['n'], H2['UV'] = np.log10(H2['n']), np.log10(H2['UV'])
            CO = {}
            with open('output/{0:s}_{1:s}_lnL.pkl'.format(name, 'CO'), 'rb') as f:
                CO['n'], CO['UV'], CO['lnL'] = pickle.load(f)
                CO['n'], CO['UV'] = np.log10(CO['n']), np.log10(CO['UV'])

            X, Y = np.meshgrid(np.linspace(np.min(H2['n']) + 0.1, np.max(H2['n']) - 0.1, 100),
                               np.linspace(np.min(H2['UV']) + 0.1, np.max(H2['UV']) - 0.1, 100))
            X, Y = X.flatten(), Y.flatten()
            z = np.zeros_like(X.flatten())

            x, y = np.meshgrid(H2['n'], H2['UV'])
            h2 = Rbf(x.flatten(), y.flatten(), H2['lnL'].flatten(), function='multiquadric', smooth=0.1)

            x, y = np.meshgrid(CO['n'], CO['UV'])
            co = Rbf(x.flatten(), y.flatten(), CO['lnL'].flatten(), function='multiquadric', smooth=0.1)
            z = h2(X, Y) + co(X, Y)
            z -= np.max(z.flatten())
            d = distr2d(X, Y, np.exp(z))

            xlabel, ylabel = r'$\log$ n$_{\rm H}$ [cm$^{-3}$]', r'$\log\,I_{\rm UV}$'
            proxy = []
            fig = d.plot(frac=0.15, indent=0.11, xlabel=xlabel, ylabel=ylabel,
                         color='orangered', cmap='Reds', color_marg='orangered', alpha=0, zorder=20)
            for attr, par in zip(['x', 'y'], ['UV', 'n']):
                d1 = d.marginalize(attr)
                d1.stats(latex=2, name=par)

            ax = fig.get_axes()[0]
            d = distr2d(H2['n'], H2['UV'], np.exp(H2['lnL']))
            d.plot_contour(ax=ax, color="violet", cmap='BuPu', color_point=None, xlabel=xlabel, ylabel=ylabel,
                           ls=None, alpha=0, label=r'H$_2$')
            # proxy.append(mpatches.Patch(color='blueviolet', label=r'H$_2$'))

            d = distr2d(CO['n'], CO['UV'], np.exp(CO['lnL']))
            d.plot_contour(ax=ax, color="limegreen", cmap='BuGn', color_point=None, xlabel=xlabel, ylabel=ylabel,
                           ls=None, alpha=0, label=r'CO')
            proxy.append(mpatches.Patch(color='green', label=r'CO'))

            proxy.append(mpatches.Patch(color='orangered', label=r'joint'))

            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.legend(handles=proxy, frameon=False, fontsize=12)
            fig.tight_layout()

            plt.show()


    def setGridView(self, par, b):
        if b != 'val':
            self.pars[par] = b
        #if b in ['fixed', 'val']:
        #    self.pars[par] = getattr(self, par + '_val').currentText()
        if list(self.pars.values()).count('x') == 1 and list(self.pars.values()).count('y') == 1 and list(self.pars.values()).count('z') == 0:
            print('done')
            self.compareIt(init=True)
        print(self.pars)

    def flux_to_mag_solve(self, c, flux, x, b, inter, mag):
        m = - 2.5 / np.log(10) * (np.arcsinh(flux * 10**c * x**2 / ac.c.to('Angstrom/s').value / 3.631e-20 / 2 / b) + np.log(b))
        return mag - np.trapz(m * inter(x), x=x) / np.trapz(inter(x), x=x)


    def UVrescale(self, init='Mathis', agn=None):

        if init == 'Mathis':
            l = np.linspace(912, 2460, 100)
            uv = np.zeros_like(l)
            mask = np.logical_and(912 <= l, l <= 1100)
            uv[mask] = 1.287e-9 * (l[mask] / 1e4)**4.4172
            mask = np.logical_and(1110 < l, l <= 1340)
            uv[mask] = 6.825e-13 * (l[mask] / 1e4)
            mask = np.logical_and(1340 < l, l <= 2460)
            uv[mask] = 2.373e-14 * (l[mask] / 1e4)**(-0.6678)
            uv = interp1d(l, uv / (ac.c.cgs.value / l / 1e-8), bounds_error=0, fill_value=0)

        return np.trapz(agn(l), x=l) / np.trapz(uv(l), x=l)

    def rescaleUV(self, s, data=None):
        grid = self.parent.grid_pars.pars
        pars = [list(grid.keys())[list(grid.values()).index('x')], list(grid.keys())[list(grid.values()).index('y')]]
        grid = self.parent.H2.grid
        if data == None:
            data = {'x': grid[pars[0]], 'y': grid[pars[1]], 'z': grid['lnL']}
        if 'uv' in pars:
            ind = pars.index('uv')
            ref, labels = ['x', 'y'], ['bottom', 'left']
            print(pars)
            d = {}
            d[ref[np.delete(np.arange(2), ind)[0]]] = data[ref[np.delete(np.arange(2), ind)[0]]]
            d[labels[np.delete(np.arange(2), ind)[0]]] = 'log(' + pars[np.delete(np.arange(2), ind)[0]] + ')'
            if s == 'Habing':
                d[ref[np.delete(np.arange(2), 1 - ind)[0]]] = np.asarray(data[ref[np.delete(np.arange(2), 1 - ind)[0]]]) * 0.81
                d[labels[np.delete(np.arange(2), 1 - ind)[0]]] = 'log(' + pars[np.delete(np.arange(2), 1 - ind)[0]] + '), Habing'
            if s == 'Draine':
                d[ref[np.delete(np.arange(2), 1 - ind)[0]]] = np.asarray(data[ref[np.delete(np.arange(2), 1 - ind)[0]]]) * 1.37
                d[labels[np.delete(np.arange(2), 1 - ind)[0]]] = 'log(' + pars[np.delete(np.arange(2), 1 - ind)[0]] + '), Draine'
            if s == 'Mathis':
                d[ref[np.delete(np.arange(2), 1 - ind)[0]]] = np.asarray(data[ref[np.delete(np.arange(2), 1 - ind)[0]]])
                d[labels[np.delete(np.arange(2), 1 - ind)[0]]] = 'log(' + pars[np.delete(np.arange(2), 1 - ind)[0]] + '), Mathis'
            if s == 'AGN':
                b = {'u': 1.4e-10, 'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
                fil = np.genfromtxt('sdss_filter_r.dat', skip_header=6, usecols=(0, 1), unpack=True)
                filter = interp1d(fil[0], fil[1], bounds_error=False, fill_value=0, assume_sorted=True)
                for idx in self.parent.H2_systems.table.selectedIndexes():
                    if idx.column() == 0:
                        name = self.parent.H2_systems.table.cell_value('name')
                        print(name)
                        qso = self.parent.H2.H2.get(name.split('_')[0])
                        print(qso.z_em, qso.m['r'])
                        DL = FlatLambdaCDM(70, 0.3, Tcmb0=2.725, Neff=0).luminosity_distance(qso.z_em).to('cm').value
                        from astroquery.sdss import SDSS
                        q = SDSS.get_spectral_template('qso')
                        x, flux = 10 ** (np.arange(len(q[0][0].data[0])) * 0.0001 + q[0][0].header['CRVAL1']), q[0][0].data[0] * 1e-17
                        mask = (x * (1 + qso.z_em) > fil[0][0]) * (x * (1 + qso.z_em) < fil[0][-1])
                        scale = 10 ** bisect(self.flux_to_mag_solve, -25, 25,
                                             args=(flux[mask], x[mask] * (1 + qso.z_em), b['r'], filter, qso.m['r']))
                        print(scale)
                        agn = interp1d(x, scale * flux * (DL / ac.kpc.cgs.value) ** 2 * x ** 2 / 1e8 / ac.c.cgs.value ** 2 * (1 + qso.z_em), bounds_error=0, fill_value='extrapolate')
                        self.rescale = self.UVrescale(agn=agn)

                d[ref[np.delete(np.arange(2), 1 - ind)[0]]] = np.sqrt(self.rescale / np.asarray(data[ref[np.delete(np.arange(2), 1 - ind)[0]]]))
                np.savetxt('output/UVrescale.dat', np.c_[data[ref[np.delete(np.arange(2), 1 - ind)[0]]], np.sqrt(self.rescale / np.asarray(data[ref[np.delete(np.arange(2), 1 - ind)[0]]]))], fmt='%.5f', delimiter=' ')
                d[labels[np.delete(np.arange(2), 1 - ind)[0]]] = 'log(d), kpc'

            d['z'] = data['z']
            print(d)
            return d
        else:
            ind = 0
            ref, labels = ['x', 'y'], ['bottom', 'left']
            print(pars)
            d = {}
            d[ref[np.delete(np.arange(2), ind)[0]]] = data[ref[np.delete(np.arange(2), ind)[0]]]
            d[labels[np.delete(np.arange(2), ind)[0]]] = 'log(' + pars[np.delete(np.arange(2), ind)[0]] + ')'
            d[ref[np.delete(np.arange(2), 1- ind)[0]]] = data[ref[np.delete(np.arange(2), 1-ind)[0]]]
            d[labels[np.delete(np.arange(2), 1- ind)[0]]] = 'log(' + pars[np.delete(np.arange(2), 1- ind)[0]] + ')'
            d['z'] = data['z']

            return d

    def changeUV(self, s):
        d = self.rescaleUV(s)
        print(d)
        self.parent.plot_reg.set_data(x=d['x'], y=d['y'], z=d['z'])
        self.parent.plot_reg.setLabels(bottom=d['bottom'], left=d['left'])
            #self.parent.plot_reg.getAxis('left').setScale(10)



class H2viewer(QMainWindow):

    def __init__(self):
        super().__init__()
        #self.H2 = H2_exc(folder='data_z0.3') #h2_uv177_n_17_7_z0_31_s_25.hdf5', 'h2_uv0_1_n_1_z0_31_s_25.hdf5', 'h2_uv5_62_n_5_62_z0_31_s_23.hdf5
        #self.H2 = H2_exc(folder='data/sample/1_5_4/av2_0_cmb0_0_z1_0_n_uv/', H2database='CO') #z0_1
        #self.H2 = H2_exc(folder='data/sample/1_5_4/av0_5_cmb2_5_z0_1_n_uv', H2database='MW')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/co_grid_n_uv_av10_cmb0_0_me1e0_cr15', H2database='MW')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/co_grid_n_uv_av10_cmb1_7_me1e0_cr1e1', H2database='CO')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/co_grid_n_uv_av10_cmb0_0_me1e0', H2database='CO')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/av2_0_cmb2_5_z1_0_n_uv/', H2database='CO')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/ver2083/z=0.0/av02_cr1e1_me1e0', H2database='CO')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/ver2053/av10_cmb0_0_me1e0_n_uv', H2database='CO')
        self.H2 = H2_exc(folder='data/sample/1_5_4/ver2083/grid-n-av/', H2database='CO')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/ver2083/test/', H2database='CO')
        self.H2.readfolder()
        self.initStyles()
        self.initUI()

    def initStyles(self):
        self.setStyleSheet(open('styles.ini').read())

    def initUI(self):
        dbg = pg.dbg()
        # self.specview sets the type of plot representation

        # >>> create panel for plotting spectra
        self.plot_exc = plotExc(self)
        self.plot_reg = plotGrid(self)
        self.H2_systems = chooseH2SystemWidget(self, closebutton=False)
        self.grid_pars = gridParsWidget(self)
        # self.plot.setFrameShape(QFrame.StyledPanel)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter_plot = QSplitter(Qt.Vertical)
        self.splitter_plot.addWidget(self.plot_exc)
        self.splitter_plot.addWidget(self.plot_reg)
        self.splitter.addWidget(self.splitter_plot)
        self.splitter_pars = QSplitter(Qt.Vertical)
        self.splitter_pars.addWidget(self.H2_systems)
        self.splitter_pars.addWidget(self.grid_pars)
        self.splitter_pars.setSizes([1000, 150])
        self.splitter.addWidget(self.splitter_pars)
        self.splitter.setSizes([1500, 250])

        self.setCentralWidget(self.splitter)

        # >>> create Menu
        #self.initMenu()

        # create toolbar
        # self.toolbar = self.addToolBar('B-spline')
        # self.toolbar.addAction(Bspline)

        #self.draw()
        self.showMaximized()
        self.show()