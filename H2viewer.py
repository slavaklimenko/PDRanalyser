from astropy.io import ascii
from functools import partial
from matplotlib import cm
from io import StringIO
import matplotlib.pyplot as plt
import pickle
from PyQt5.QtCore import (Qt, )
from PyQt5.QtGui import (QFont, )
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow, QSplitter, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QHeaderView, QCheckBox,
                             QRadioButton, QButtonGroup, QComboBox, QTableView, QLineEdit)
import pyqtgraph as pg
from scipy.interpolate import interp2d, RectBivariateSpline, Rbf
import sys
sys.path.append('C:/science/python')
from H2_exc import *
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
            spmode = str(self.parent.grid_pars.spmode.currentText())
            q = self.parent.H2.comp(name)
            sp = [s for s in q.e.keys() if species+'j' in s]
            j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
            x, stat = self.getatomic(species, levels=j)
            y = [q.e[species+'j' + str(i)].col / stat[i] for i in j]
            typ = [q.e[species+'j' + str(i)].col.type for i in j]
            if spmode == 'rel':
                print('def add: rel:',y)
                y = [y[i]/q.e[species + 'j0'].col for i in j]
                print(y)
            #if species == 'CI':
            #    y = [(q.e[species + 'j' + str(i)].col/q.e['CIj0'].col)/ stat[i] for i in j]
            #    y = [(q.e[species + 'j' + str(i)].col) / stat[i] for i in j]
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
            spmode = str(self.parent.grid_pars.spmode.currentText())
            for ind, m in enumerate(self.parent.H2.listofmodels(name)):
                j = np.sort([int(s[3:]) for s in m.cols.keys()])
                x, stat = self.getatomic(species, levels=j)
                mod = [m.cols[species+'j'+str(i)] - np.log10(stat[i]) for i in j]
                if spmode == 'rel':
                    mod -=mod[0]
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


    def add_cmb(self, cols=None, cols0=None, redshift=0, add=True):
        if add:
            species = str(self.parent.grid_pars.species.currentText())
            spmode = str(self.parent.grid_pars.spmode.currentText())
            tcmb = 2.725*(1+redshift)
            j = np.sort([int(s[3:]) for s in cols.keys()])
            x, stat = self.getatomic(species, levels=j)
            mod = [cols0 -np.log10(stat[i]) + np.log10(np.exp(-x[i] / (tcmb / 1.428))) for i in j]
            if spmode == 'rel':
                mod -= mod[0]
            self.temp_cmb = pg.PlotCurveItem(x, mod)
            self.vb.addItem(self.temp_cmb)
            self.redraw()
        else:
            try:
                self.vb.removeItem(self.temp_cmb)
                self.legend_model.removeItem(self.temp_cmb)
            except:
                pass

    def add_temp(self, cols=None, pars=None, add=True):
        if add:
            species = str(self.parent.grid_pars.species.currentText())
            spmode =  str(self.parent.grid_pars.spmode.currentText())
            j = np.sort([int(s[3:]) for s in cols.keys()])
            x, stat = self.getatomic(species, levels=j)
            mod = [cols[species+'j'+str(i)] - np.log10(stat[i]) for i in j]
            if spmode == 'rel':
                mod -=mod[0]


            # save local fit to species excitation
            if 0:
                with open('temp/local_fit.pkl', 'wb') as f:
                    if species == 'H2':
                        pickle.dump([x, mod, spec], f)
                    elif species == 'CI':
                        pickle.dump([x, mod], f)


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
                    if spmode == 'rel':
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

    def plot_model(self):
        #print(self.parent.parent.H2.grid['NH2tot'])
        m = self.parent.parent.H2.listofmodels(self.name)[0]
        borders = {}
        for el in ['H2', 'CI', 'CO']:
            if self.parent.parent.H2.grid['N' + el + 'tot'] is not None:
                borders[el] = self.parent.parent.H2.grid['N' + el + 'tot']-0.3
            else:
                borders[el] = None
        print('borders:',borders)
        #m.plot_model(parx='h2', pars=[['tgas','n','pgas'],['heat_phel','heat_h2','heat_tot'],['cool_h2','cool_o','cool_elrec','cool_tot','cool_cp']],
        #             species=[['H2j0/H2', 'H2j1/H2', 'H2j2/H2','H2j3/H2', 'H2j4/H2'],['H','H2','H+','el','C']],
        #             logx=True, logy=True, limit={'H2': self.parent.parent.H2.grid['NH2tot'] -0.3 }) #, limit={'H2':21} #['OPR','OPR_logNJ1/J02','OPR_logNJ1/J0'],['uv_dens'],['H2_dest_rate'],
        m.plot_model(parx='x', pars=[['tgas','n', 'pgas']], #pars=[['tgas','Nh2t01','Nh2t02','tgas_m']],
                     species=[['NH2j0/NH2', 'NH2j1/NH2', 'NH2j2/NH2','NH2j3/NH2'],['NH', 'NH2','NCI','NCO'],['H', 'H2','CI','CO','C+'],
                              ['COj0','COj1','COj2','COj3'], ['NCj0/NCI','NCj1/NCI','NCj2/NCI']],
                     logx=True, logy=True, borders=borders) #, 'CI': self.parent.parent.H2.grid['NCOtot']-0.3}) #, limit={'H2':24})
                     #limit={'H2': self.parent.parent.H2.grid['NH2tot'] -0.3}) #, limit={'H2':21} #['OPR','OPR_logNJ1/J02','OPR_logNJ1/J0'],['uv_dens'],['H2_dest_rate'],

        # ['cool_cp','cool_o','cool_elrec','cool_tot','heat_phel','heat_phot','heat_tot']],
        #m.calc_Hp()
        plt.show()

    def plot_fit(self):
        m = self.parent.parent.H2.listofmodels(self.name)[0]
        sp = self.parent.parent.grid_pars.species.currentText()
        if sp == 'CI':
            m.plot_model(parx='x', species=[['CIj0/CI','CIj1/CI','CIj2/CI'],['NCj0/NCI','NCj1/NCI','NCj2/NCI']], pyfit=True,
                         logx=True, logy=True, borders={'H2': self.parent.parent.H2.grid['NH2tot']-0.3, 'CI': self.parent.parent.H2.grid['NCItot']-0.3,})
        elif sp == 'CO':
            m.plot_model(parx='x', species=[['COj0/CO', 'COj1/CO', 'COj2/CO'], ['NCOj0/NCO', 'NCOj1/NCO', 'NCOj2/NCO']],
                         pyfit=True,
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

    def mousePressEvent(self, event):
        super(plotGrid, self).mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            if self.s_status:
                name = self.parent.H2.grid['name']
                self.mousePoint = self.vb.mapSceneToView(event.pos())
                spmode = str(self.parent.grid_pars.spmode.currentText())
                spname = str(self.parent.grid_pars.species.currentText())
                print(self.mousePoint.x(), self.mousePoint.y())
                if self.parent.grid_pars.cols is not None:
                    cols = {}
                    sp = self.parent.H2.grid['cols'][0].keys()
                    lnL = 0
                    cols0 = self.parent.grid_pars.cols[spname + 'j0'](self.mousePoint.x(), self.mousePoint.y())
                    for s in sp:
                        v = self.parent.H2.comp(name).e[s].col
                        cols[s] = self.parent.grid_pars.cols[s](self.mousePoint.x(), self.mousePoint.y())
                        if spmode == 'rel':
                            v = v/ self.parent.H2.comp(name).e[spname + 'j0'].col.log() #a(cols0,0,0,'l')
                            cols[s] -=cols0
                        v1 = v * a(0, 0.2, 0.2, 'l')
                        if v.type == 'm':
                            lnL += v1.lnL(cols[s])
                        print(s, cols[s], v1.val, lnL)
                    self.parent.plot_exc.add_temp(cols, add=False)
                    self.parent.plot_exc.add_temp(cols, pars=[self.mousePoint.x(), self.mousePoint.y()])
                    if spname == 'CO':
                        self.parent.plot_exc.add_cmb(add=False)
                        self.parent.plot_exc.add_cmb(cols,cols0)


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

    def compare(self, species='H2', spmode ='abs'):
        grid = self.parent.parent.grid_pars.pars
        syst = float(self.parent.parent.grid_pars.addSyst.text()) if self.parent.parent.grid_pars.addSyst.text().strip() != '' else 0
        pars = [list(grid.keys())[list(grid.values()).index('x')],
                list(grid.keys())[list(grid.values()).index('y')]]
        fixed = list(grid.keys())[list(grid.values()).index('fixed')]
        if getattr(self.parent.parent.grid_pars, fixed + '_val').currentText() != '':
            fixed = {fixed: float(getattr(self.parent.parent.grid_pars, fixed + '_val').currentText())}
        else:
            fixed = {fixed: 'all'}
        for idx in self.selectedIndexes():
            if idx.column() == 0:
                name = self.cell_value('name')
                self.parent.parent.H2.comparegrid(name, species=species, pars=pars, fixed=fixed, syst=syst, plot=False,
                                                  levels=self.parent.parent.grid_pars.H2levels,
                                                  others=self.parent.parent.grid_pars.othermode.currentText(), spmode = spmode)
                grid = self.parent.parent.H2.grid
                self.parent.parent.H2.grid['name'] = name
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
        data = self.parent.H2.H2.makelist(pars=['z_dla', 'Me__val', 'H2__val'], sys=self.parent.H2.H2.all(), view='numpy')
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
        self.pars = {'n0': 'x', 'uv': 'y', 'me': 'fixed'}
        #self.pars = {'n0': 'x', 'me': 'y', 'uv': 'fixed'}
        self.parent.H2.setgrid(pars=list(self.pars.keys()), show=False)
        self.cols, self.x_, self.y_, self.z_, self.mpars = None, None, None, None, None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('grid parameters:'))

        for n in self.pars.keys():
            l = QHBoxLayout(self)
            l.addWidget(QLabel((n + ': ')[:3]))
            self.group = QButtonGroup(self)
            for b in ('x', 'y', 'z', 'fixed'):
                setattr(self, b, QCheckBox(b, checkable=True))
                getattr(self, b).clicked[bool].connect(partial(self.setGridView, par=n, b=b))
                l.addWidget(getattr(self, b))
                self.group.addButton(getattr(self, b))
            getattr(self, self.pars[n]).setChecked(True)
            setattr(self, n + '_val', QComboBox(self))
            getattr(self, n + '_val').setFixedSize(80, 25)
            getattr(self, n + '_val').addItems(np.append([''], np.asarray(np.sort(np.unique(self.parent.H2.grid[n])), dtype=str)))
            if self.pars[n] is 'fixed':
                getattr(self, n + '_val').setCurrentIndex(1)
            l.addWidget(getattr(self, n + '_val'))
            l.addStretch(1)
            layout.addLayout(l)

        l = QHBoxLayout(self)
        l.addWidget(QLabel('add systematic unc.:'))
        self.addSyst = QLineEdit()
        self.addSyst.setText(str(0.2))
        self.addSyst.setFixedSize(90, 30)
        l.addWidget(self.addSyst)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.compare = QPushButton('Compare')
        self.compare.clicked[bool].connect(self.compareIt)
        self.compare.setFixedSize(90, 30)
        l.addWidget(self.compare)
        self.species = QComboBox(self)
        self.species.setFixedSize(50, 25)
        self.species.addItems(['H2', 'CI','CO'])
        self.species.setCurrentIndex(0)
        l.addWidget(self.species)
        self.spmode = QComboBox(self)
        self.spmode.setFixedSize(60, 25)
        self.spmode.addItems(['abs', 'rel'])
        self.spmode.setCurrentIndex(0)
        l.addWidget(self.spmode)
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
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.refine = QPushButton('Refine:')
        self.refine.clicked[bool].connect(partial(self.regridIt, kind='accurate'))
        self.refine.setFixedSize(90, 30)
        l.addWidget(self.refine)
        self.numPlot = QLineEdit(str(100))
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
        #l.addStretch(1)
        self.export_table = QPushButton('Table')
        self.export_table.clicked[bool].connect(self.tableIt)
        self.export_table.setFixedSize(90, 30)
        l.addWidget(self.export_table)
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

    def compareIt(self):
        self.parent.H2_systems.table.compare(species=str(self.species.currentText()), spmode = str(self.spmode.currentText()))
        self.interpolateIt()
        grid = self.parent.H2.grid
        x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
        self.xmin.setText(str(np.min(x)))
        self.xmax.setText(str(np.max(x)))
        self.ymin.setText(str(np.min(y)))
        self.ymax.setText(str(np.max(y)))

    def interpolateIt(self):
        grid = self.parent.H2.grid
        x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
        sp = grid['cols'][0].keys()
        mp = grid['mpars'][0].keys()
        self.cols = {}
        self.mpars = {}
        #sp_full = grid['cols'][0].keys()
        for s in sp:
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
            if 1:
                self.cols[s] = Rbf(x, y, np.asarray([c[s] for c in grid['cols']]), function='multiquadric', smooth=0.1)
        for p in mp:
            self.mpars[p] = Rbf(x, y, np.asarray([np.log10(c[p]) for c in grid['mpars']]), function='multiquadric', smooth=0.1)

            #rbf = Rbf(x,y,z,function='multiquadric',smooth=0.2)

    def regridIt(self, kind='accurate', save=True):
        grid = self.parent.H2.grid
        num = int(self.numPlot.text())
        x, y = np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('x')]]), np.log10(grid[list(self.pars.keys())[list(self.pars.values()).index('y')]])
        x1, y1 = x, y # copy for save
        x, y = np.linspace(float(self.xmin.text()), float(self.xmax.text()), num), np.linspace(float(self.ymin.text()), float(self.ymax.text()), num)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X)
        sp = grid['cols'][0].keys()
        sp = [s for s in sp if int(s[3:]) in self.H2levels]
        species = {}
        spmode = str(self.spmode.currentText())
        spname = str(self.species.currentText())
        # add calc phys cond
        #mp = grid['mpars'][0].keys()
        #mparsgrid = {}
        #for p in mp:
        #    mparsgrid[p] = np.zeros_like(X)

        for s in sp:
            v1 = self.parent.H2.comp(grid['name']).e[s].col.log().copy()
            if kind == 'fast':
                v1.minus, v1.plus = np.sqrt(v1.minus ** 2 + float(self.addSyst.text()) ** 2), np.sqrt(v1.plus ** 2 + float(self.addSyst.text()) ** 2)
            elif kind == 'accurate':
                v1 *= a(0, float(self.addSyst.text()), float(self.addSyst.text()), 'l')
            if spmode == 'abs':
                species[s] = v1
            elif spmode == 'rel':
                species[s] = v1/self.parent.H2.comp(grid['name']).e[sp[0]].col.log()

#            elif str(self.species.currentText()) == 'CI':
#                species[s] = v1
#            elif str(self.species.currentText()) == 'CO':
#                species[s] = v1


        if 1:
            others = self.parent.grid_pars.othermode.currentText()
            levels = self.parent.grid_pars.H2levels
            full_keys = grid['cols'][0].keys()
            label = spname + 'j{:}'
            keys = [label.format(i) for i in levels if label.format(i) in full_keys]
            print('regrid keys',keys)
            #if str(self.species.currentText()) == 'H2':
            #    full_keys = grid['cols'][0].keys()
            #    keys = ['H2j{:}'.format(i) for i in levels if 'H2j{:}'.format(i) in full_keys]

            #if str(self.species.currentText()) == 'CI':
            #    full_keys = grid['cols'][0].keys()
            #    keys = ['CIj{:}'.format(i) for i in levels if 'CIj{:}'.format(i) in full_keys]

            #if str(self.species.currentText()) == 'CO':
            #    full_keys = grid['cols'][0].keys()
            #    keys = ['COj{:}'.format(i) for i in levels if 'COj{:}'.format(i) in full_keys]

            if others in ['lower', 'upper']:
                for k in full_keys:
                    if k not in keys:
                        v = self.parent.H2.comp(grid['name']).e[k].col.log().copy()
                        if others == 'lower':
                            species[k] = a(v.val - v.minus, t=others[0])
                        else:
                            species[k] = a(v.val + v.plus, t=others[0])

        if save:
            cols = {}
            for s in self.cols.keys():
                cols[s] = np.zeros_like(z)
        for i, xi in enumerate(x):
            for k, yi in enumerate(y):
                lnL = 0
                for s, v in species.items():
                    if v.type in ['m', 'u', 'l']:
                        if spmode == 'abs':
                        #if str(self.species.currentText()) == 'H2':
                            cols[s][k, i] = self.cols[s](xi, yi)
                            lnL += v.lnL(cols[s][k, i])
                        elif spmode == 'rel':
                            cols[s][k, i] = self.cols[s](xi, yi) - self.cols[spname+'j0'](xi, yi)
                            lnL += v.lnL(cols[s][k, i])
                            print('v=', v, 'PDR', cols[s][k, i], '+lnL', v.lnL(cols[s][k, i]))
                print(xi, yi, lnL)
                        #elif str(self.species.currentText()) == 'CI':
                        #    #cols[s][k, i] = self.cols[s](xi, yi) - self.cols['CIj0'](xi, yi)
                        #    cols[s][k, i] = self.cols[s](xi, yi)
                        #    lnL += v.lnL(cols[s][k, i])
                        #
                        #elif str(self.species.currentText()) == 'CO':
                        #    cols[s][k, i] = self.cols[s](xi, yi)
                        #    lnL += v.lnL(cols[s][k, i])

                z[k, i] = lnL



        self.x_, self.y_, self.z_ = x, y, z

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


    def plotIt(self):
        if str(self.plot_model_set.currentText()) == 'H2+CI':
            print('H2+CI')
        if self.x is not None:
            d = distr2d(x=self.x_, y=self.y_, z=np.exp(self.z_))
            dx, dy = d.marginalize('y'), d.marginalize('x')
            dx.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('x')])
            dy.stats(latex=2, name=list(self.pars.keys())[list(self.pars.values()).index('y')])
            d.plot(color=None)

            plt.show()
        #self.plot_mpars()

    # export likelihood to file: in case of 'H2' in addition plot join fit to 'H2+CI'
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
            if 1:
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
                    name = 'output/{:s}_join_lnL.pkl'
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
                    name = 'output/{:s}_join_lnL.pkl'
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



    def tableIt(self):
        print(self.parent.plot_reg.mousePoint.x(), self.parent.plot_reg.mousePoint.y())
        d = [['species', 'observed', 'model']]
        if self.parent.grid_pars.cols is not None:
            cols = {}
            for s in self.parent.H2.grid['cols'][0].keys():
                cols[s] = self.parent.grid_pars.cols[s](self.parent.plot_reg.mousePoint.x(), self.parent.plot_reg.mousePoint.y())

        q = self.parent.H2.H2.getcomp(self.parent.H2.grid['name'])
        for e in ['H2j0', 'H2j1', 'H2j2']:
            d.append([e.replace('H2', 'H$_2$ ').replace('j', 'J='), q.e[e].col.latex(f=2), '{:5.2f}'.format(cols[e])])
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


    def setGridView(self, par, b):
        self.pars[par] = b
        if b is 'fixed':
            print(getattr(self, par + '_val').text())
            print(getattr(self, par + '_val').text())
            self.pars[par] = getattr(self, par + '_val').currentText()
        #print(self.pars)

class H2viewer(QMainWindow):

    def __init__(self):
        super().__init__()
        #self.H2 = H2_exc(folder='data_z0.3') #h2_uv177_n_17_7_z0_31_s_25.hdf5', 'h2_uv0_1_n_1_z0_31_s_25.hdf5', 'h2_uv5_62_n_5_62_z0_31_s_23.hdf5
        #self.H2 = H2_exc(folder='data/sample/1_5_4/av2_0_cmb0_0_z0_3_n_uv/', H2database='MW') #z0_1
        #self.H2 = H2_exc(folder='data/sample/1_5_4/av0_5_cmb2_5_z0_1_n_uv', H2database='MW')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/co_grid_n_uv_av10_cmb0_0_me1e0_cr15', H2database='MW')
        self.H2 = H2_exc(folder='data/sample/1_5_4/co_grid_n_uv_av10_cmb0_0_me1e0', H2database='MW')
        #self.H2 = H2_exc(folder='data/sample/1_5_4/test', H2database='MW')
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