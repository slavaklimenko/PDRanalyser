from collections import OrderedDict
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from scipy import integrate
from scipy.optimize import root
from scipy.interpolate import interp1d
import sys
sys.path.append('/home/toksovogo/science/codes/python')
sys.path.append('/science/python')
from spectro.a_unc import a
from spectro.sviewer.utils import Timer
import warnings
from scipy.interpolate import interp2d, RectBivariateSpline, Rbf
import pickle




def polyJ01(x, OPR):
    # equation for OPR of H2 rotational levels
    # x = exp(-170.5/Tkin)
    err = OPR * (1 ) - 3 * (3 * x ) # + 7 * x ** 6.
    return err

def polyJ012(x, OPR):
    # equation for OPR of H2 rotational levels
    # x = exp(-170.5/Tkin)
    err = OPR * (1 +5*x**3.) - 3 * (3*x + 7*x**6.)
    return err




def column(matrix, i):
    if i == 0 or (isinstance(i, str) and i[0] == 'v'):
        return np.asarray([row.val for row in matrix])
    if i == 1 or (isinstance(i, str) and i[0] == 'p'):
        return np.asarray([row.plus for row in matrix])
    if i == 2 or (isinstance(i, str) and i[0] == 'm'):
        return np.asarray([row.minus for row in matrix])

H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/energy_X_H2.dat', dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                          unpack=True, skip_header=3, comments='#')
H2energy = np.zeros([max(H2_energy['nu']) + 1, max(H2_energy['j']) + 1])
for e in H2_energy:
    H2energy[e[0], e[1]] = e[2]
CIenergy = [0, 16.42, 43.41]
COenergy = [2.766 * ((i+1) * (i) )/1.428 for i in range(10)] # in cm-1

stat_H2 = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]
stat_CI = [(2 * i + 1) for i in range(3)]
stat_CO = [(2 * i + 1) for i in range(10)]

spcode = {'H': 'n_h', 'H2': 'n_h2', 'H+': 'n_hp', 'He': 'n_he',
          'H2j0': 'pop_h2_v0_j0', 'H2j1': 'pop_h2_v0_j1', 'H2j2': 'pop_h2_v0_j2', 'H2j3': 'pop_h2_v0_j3',
          'H2j4': 'pop_h2_v0_j4', 'H2j5': 'pop_h2_v0_j5', 'H2j6': 'pop_h2_v0_j6', 'H2j7': 'pop_h2_v0_j7',
          'H2j8': 'pop_h2_v0_j8', 'H2j9': 'pop_h2_v0_j9', 'H2j10': 'pop_h2_v0_j10',
          'D': 'n_d', 'D+': 'n_dp', 'HD': 'n_hd', 'HDj0': 'pop_hd_v0_j0', 'HDj1': 'pop_hd_v0_j1', 'HDj2': 'pop_hd_v0_j2',
          'CI': 'n_c', 'C+': 'n_cp', 'CO': 'n_co','C13O': 'n_13co', 'C2': 'n_c2',
          'COj0': 'pop_co_v0_j0','COj1': 'pop_co_v0_j1','COj2': 'pop_co_v0_j2','COj3': 'pop_co_v0_j3',
          'COj4': 'pop_co_v0_j4', 'COj5': 'pop_co_v0_j5', 'COj6': 'pop_co_v0_j6', 'COj7': 'pop_co_v0_j7',
          'el': 'n_electr', 'O': 'n_o', 'O+': 'n_op',
          'CIj0': 'pop_c_el3p_j0', 'CIj1': 'pop_c_el3p_j1', 'CIj2': 'pop_c_el3p_j2',
          'He': 'n_he','He+': 'n_hep',
          'NH': 'cd_prof_h','NH2': 'cd_prof_h2',  'NH2j0': 'cd_lev_prof_h2_v0_j0', 'NH2j1': 'cd_lev_prof_h2_v0_j1','NH2j2': 'cd_lev_prof_h2_v0_j2','NH2j3': 'cd_lev_prof_h2_v0_j3','NH2j4': 'cd_lev_prof_h2_v0_j4','NH2j5': 'cd_lev_prof_h2_v0_j5',
          'NH2j6': 'cd_lev_prof_h2_v0_j6', 'NH2j7': 'cd_lev_prof_h2_v0_j7','NH2j8': 'cd_lev_prof_h2_v0_j8','NH2j9': 'cd_lev_prof_h2_v0_j9','NH2j10': 'cd_lev_prof_h2_v0_j10','NH2j11': 'cd_lev_prof_h2_v0_j11',
          'NHD': 'cd_prof_hd', 'NCI': 'cd_prof_c',  'NCj0': 'cd_lev_prof_c_el3p_j0', 'NCj1': 'cd_lev_prof_c_el3p_j1','NCj2': 'cd_lev_prof_c_el3p_j2',
          'NCO': 'cd_prof_co', 'NCOj0': 'cd_lev_prof_co_v0_j0','NCOj1': 'cd_lev_prof_co_v0_j1','NCOj2': 'cd_lev_prof_co_v0_j2',
          'H2_dest_rate': 'h2_dest_rate_ph', 'H2_form_rate_er': 'h2_form_rate_er','H2_form_rate_lh': 'h2_form_rate_lh','H2_photo_dest_prob': 'photo_prob___h2_photon_gives_h_h',
          'cool_tot': 'coolrate_tot', 'cool_o': 'coolrate_o','cool_cp': 'coolrate_cp', 'cool_c': 'coolrate_c', 'cool_elrec': 'coolrate_elrecomb', 'cool_free': 'coolrate_freefree', 'cool_h': 'coolrate_h',
          'cool_h2': 'coolrate_h2',
          'heat_tot': 'heatrate_tot', 'heat_phel': 'heatrate_pe','heat_secp': 'heatrate_secp', 'heat_phot': 'heatrate_ph','heat_chem': 'heatrate_chem', 'heat_h2': 'heatrate_h2', 'heat_cr': 'heatrate_cr',
          'H2_diss': 'h2_dest_rate_ph', 'OPR': 't01_ab', 'metal': 'metal', 'radm_ini': 'radm_ini'
         }

class plot(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29))
        self.initstatus()
        self.vb = self.getViewBox()

    def initstatus(self):
        self.s_status = False
        self.selected_point = None

    def set_data(self, data=None):
        if data is None:
            try:
                self.vb.removeItem(self.d)
            except:
                pass
        else:
            self.data = data
            self.points = pg.ScatterPlotItem(self.data[0], self.data[1], symbol='o', pen={'color': 0.8, 'width': 1}, brush=pg.mkBrush(100, 100, 200))
            self.vb.addItem(self.points)

    def mousePressEvent(self, event):
        super(plot, self).mousePressEvent(event)
        print('KEY PRESSED')
        if event.button() == Qt.LeftButton:
            if self.s_status:
                self.mousePoint = self.vb.mapSceneToView(event.pos())
                r = self.vb.viewRange()
                self.ind = np.argmin(((self.mousePoint.x() - self.data[0]) / (r[0][1] - r[0][0]))**2   + ((self.mousePoint.y() - self.data[1]) / (r[1][1] - r[1][0]))**2)
                if self.selected_point is not None:
                    self.vb.removeItem(self.selected_point)
                self.selected_point = pg.ScatterPlotItem(x=[self.data[0][self.ind]], y=[self.data[1][self.ind]], symbol='o', size=15,
                                                        pen={'color': 0.8, 'width': 1}, brush=pg.mkBrush(230, 100, 10))
                self.vb.addItem(self.selected_point)


    def keyPressEvent(self, event):
        super(plot, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_S:
                self.s_status = True

    def keyReleaseEvent(self, event):
        super(plot, self).keyReleaseEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_S:
                self.s_status = False


class model():
    def __init__(self, folder='', name=None, filename=None, species=[], show_summary=True, show_meta=False, fastread=True):
        self.folder = folder
        self.sp = {}
        self.species = species
        self
        self.filename = filename
        self.initpardict()
        self.fastread = fastread
        if filename is not None:
            if name is None:
                name = filename.replace('.hdf5', '')
            self.name = name
            self.read(show_summary=show_summary, show_meta=show_meta)

    def initpardict(self):
        self.pardict = {}
        #self.pardict['metal'] = ('Parameters/Parameters', 13, float)
        #self.pardict['radm_ini'] = ('Parameters/Parameters', 5, float)
        self.pardict['proton_density_input'] = ('Parameters/Parameters', 3, float)
        self.pardict['distance'] = ('Local quantities/Positions', 2, float)
        self.pardict['av'] = ('Local quantities/Positions', 1, float)
        self.pardict['tgas'] = ('Local quantities/Gas state', 2, float)
        self.pardict['tcmb'] = ('Parameters/Informations/', 18, float)
        self.pardict['pgas'] = ('Local quantities/Gas state', 3, float)
        self.pardict['ntot'] = ('Local quantities/Gas state', 1, float)
        self.pardict['h2t01'] =('Local quantities/Auxiliary/Excitation/H2 T01 abundances/', 0, float)
        #self.pardict['h2t01_cd'] = ('Integrated quantities/Excitation/T01 column densities/', 0, float)
        self.pardict['OPR'] = ('Local quantities/Auxiliary/Excitation/H2 ortho-para ratio/',0, float)
        self.pardict['h2fr'] = ('Local quantities/Auxiliary/Molecular fraction/', 0, float)
        #self.pardict['OPR_cd'] = ('Integrated quantities/Excitation/H2 ortho-para all levels/', 0, float)

       # for n, ind in zip(['n_h', 'n_h2', 'n_c', 'n_cp', 'n_co'], [0, 2, 5, 93, 44]):
       #     self.pardict[n] = ('Local quantities/Densities/Densities', ind, float)
        for i in range(11):
            self.pardict['pop_h2_v0_j'+str(i)] = ('Local quantities/Auxiliary/Excitation/Level densities', 9+i, float)

    def read(self, show_meta=True, show_summary=True, fast=None):
        """
        Read model data from hdf5 file

        :param
            -  show_meta           :  if True, show Metadata table
            -  show_summary        :  if True, show summary

        :return: None
        """
        self.file = h5py.File(self.folder + self.filename, 'r')

        if fast is not None:
            self.fastread = fast


        # >>> model input parameters
        self.me = self.par('metal')
        self.P = self.par('gas_pressure_input')
        self.n0 = self.par('proton_density_input')
        self.uv = self.par('radm_ini')
        self.zeta = self.par('zeta')

        # >>> profile of physical quantities
        self.x = self.par('distance')

        self.h2 = self.par('cd_prof_h2')
        self.hi = self.par('cd_prof_h')
        self.hd = self.par('cd_prof_hd')
        self.av = self.par('av')
        self.tgas = self.par('tgas')
        self.pgas = self.par('pgas')
        self.n = self.par('ntot')
        self.nH = self.par('protdens')
        self.uv_flux = self.par('uv_flux')
        self.uv_dens = self.par('uv_dens')
        self.h2t01= self.par('h2t01')
        self.h2t01_cd = self.par('h2t01_cd')
        self.OPR = self.par('OPR')
        self.OPR_cd = self.par('OPR_cd')
        self.tcmb = self.par('tcmb')
        self.h2fr = self.par('h2fr')
       # self.t01 = self.par('t01')

        self.readspecies()

        if show_meta:
            self.showMetadata()

        self.file.close()
        self.file = None

        if show_summary:
            self.show_summary()

    def par(self, par=None):
        """
        Read parameter or array from the hdf5 file, specified by Metadata name

        :param:
            -  par          :  the name of the parameter to read

        :return: x
            - x             :  corresponding data (string, number, array) correspond to the parameter
        """
        if self.file == None:
            self.file = h5py.File(self.folder + self.filename, 'r')

        meta = self.file['Metadata/Metadata']
        if par is not None:
            if self.fastread and par in self.pardict:
                attr, ind, typ = self.pardict[par]
            else:
                ind = np.where(meta[:, 3] == par.encode())[0]
                if len(ind) > 0:
                    attr = meta[ind, 0][0].decode() + '/' + meta[ind, 1][0].decode()
                    typ = {'string': str, 'real': float, 'integer': int}[meta[ind, 5][0].decode()]
                    ind = int(meta[ind, 2][0].decode())
                else:
                    return None
        x = self.file[attr][:, ind]
        if len(x) == 1:
            return typ(x[0].decode())
        else:
            return x

    def showMetadata(self):
        """
        Show Metadata information in the table
        """

        self.w = pg.TableWidget()
        self.w.show()
        self.w.resize(500, 900)
        self.w.setData(self.file['Metadata/Metadata'][:])

    def readspecies(self, species=None):
        """
        Read the profiles of the species

        :param
            -  species       : the list of the names of the species to read from the file

        :return: None

        """
        if species is None:
            species = self.species

        for s in species:
            self.sp[s] = self.par(spcode[s])

        self.species = species

    def show_summary(self, pars=['me', 'n', 'uv'], output=False):
        print('model: ' + self.name)
        if output:
            f = open(self.name+'dat', 'w')
        if 'all' in pars:
            self.file = h5py.File(self.folder + self.filename, 'r')
            meta = self.file['Metadata/Metadata']
            for ind in np.where(meta[:, 0] == b'/Parameters')[0]:
                attr = meta[ind, 0].decode() + '/' + meta[ind, 1].decode()
                typ = {'string': str, 'real': float, 'integer': int}[meta[ind, 5].decode()]
                #if len(x) == 1:
                print(meta[ind, 4].decode(), ' : ', typ(self.file[attr][:, int(meta[ind, 2].decode())][0].decode()))
                if output:
                    f.write(meta[ind, 4].decode() + ': ' + self.file[attr][:, int(meta[ind, 2].decode())][0].decode() + '\n')
            self.file.close()
        else:
            for p in pars:
                if hasattr(self, p):
                    print(p, ' : ', getattr(self, p))
                else:
                    print(p, ' : ', self.par(p))
                #print("{0:s} : {1:.2f}".format(p, getattr(self, p)))
        if output:
            f.close()

    def plot_model(self,  parx='av', pars=['tgas', 'n', 'av','h2t01'], excs = None, species=None, logx=False, logy=True,
                   legend=True, limit=None,borders=None):
        """
        Plot the model quantities

        :parameters:
            -  pars          :  list of the parameters to plot
            -  species       :  list of the lists of species to plot,
                                    e.g. [['H', 'H+', 'H2'], ['NH2j0', 'NH2j1', 'NH2j2']]
                                    each list will be plotted in independently
            -  logx          :  if True x in log10
            -  ax            :  axes object to plot
            -  legend        :  show Legend
            -  parx          :  what is on x axis
            -  limit         :  if not None plot only part of the cloud specified by limit
                                e.g. {'NH2': 19.5}

        :return: ax
            -  ax            :  axes object
        """

        m = 0
        if pars is not None:
            if sum([isinstance(s, list) for s in pars]) == 0:
                pars = [pars]
            m += len(pars)

        n = 0
        if species is not None:
            if sum([isinstance(s, list) for s in species]) == 0:
                species = [species]
            n += len(species)

        k = 0
        if excs is not None:
           k+=1

        print('m=',m, 'n=',n, 'k=', k)
        fig, ax = plt.subplots(nrows=1, ncols=n + m+k, figsize=(2 + 4*(n+m+k), 6))

        if pars is not None:
            for m, p in enumerate(pars):
                self.plot_phys_cond(pars=p, logx=logx, ax=ax[m], legend=legend, parx=parx, limit=limit)

        if species is not None:
            for n, sp in enumerate(species):
                self.plot_profiles(species=sp, logx=logx, logy=logy, ax=ax[n+1+m], label=None, legend=legend, parx=parx, limit=limit,normed=False)

        if excs is not None:
            self.plot_exciation(ax=ax[n+1+m+k],logN=excs)

        if 1:
            #plot h2fr border
            for axs in ax:
                mask_fr = self.h2<self.h2[-1]/2
                mask_fr *= self.h2fr < 0.1
                v = np.log10(self.h2[mask_fr][-1])
                print('h2fr border:', v)
                axs.axvline(x=v, ls='--', color='purple', alpha=0.7)

                mask_fr = self.h2 < self.h2[-1] / 2
                mask_fr *= 2*self.h2/(2*self.h2+self.hi) < 0.1
                v = np.log10(self.h2[mask_fr][-1])
                print('h2fr border:', v)
                axs.axvline(x=v, ls='-.', color='purple', alpha=0.7)


        if borders is not None:
            print(borders)
            if v is not None:
                for k in borders.keys():
                    self.set_mask(logN={k:borders[k]}, sides=2)
                    v = np.log10(self.h2[self.mask][-1])
                    print(k, v)
                    if k == 'H2': color = 'black'
                    if k == 'CO': color = 'magenta'
                    if k == 'CI': color = 'red'
                    for axs in ax:
                        axs.axvline(x=v, ls='--', color=color,alpha=0.7)

        #fig.suptitle(str(('n0=', self.n0, 'uv=', self.uv, 'me=',self.me, 'zeta=', self.zeta)), fontsize=14)
        fig.suptitle(self.name, fontsize=14)



        return fig
        #fig.tight_layout()

    def plot_phys_cond(self, pars=['tgas', 'n', 'av','h2t01','uv_dens'], logx=True, ax=None, legend=True, parx='x', limit=None,yscale='linear',bot=None, ls='-'):
        """
        Plot the physical parameters in the model

        :parameters:
            -  pars          :  list of the parameters to plot
            -  logx          :  if True x in log10
            -  ax            :  axes object to plot
            -  legend        :  show Legend
            -  parx          :  what is on x axis
            -  limit         :  if not None plot only part of the cloud specified by limit
                                e.g. {'NH2': 19.5}

        :return: ax
            -  ax            :  axes object
        """

        if parx == 'av':
            xlabel = 'Av'
        elif parx == 'x':
            xlabel = '$\log$(Distance), cm'
        elif parx == 'h2':
            xlabel = 'log(NH2), cm-2'
        elif parx == 'hi':
            xlabel = 'log(NHI), cm-2'


        fsize=10

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if logx:
            mask = getattr(self, parx) > 0
        else:
            mask = getattr(self, parx) > -1

        if limit is not None:
            if hasattr(self, list(limit.keys())[0]):
                v = getattr(self, list(limit.keys())[0])
                mask *= v < list(limit.values())[0]
            elif list(limit.keys())[0] in self.sp.keys():
                #v = self.sp[list(limit.keys())[0]]
                self.set_mask(logN=limit)
                mask *= self.mask
            else:
                warnings.warn('limit key {:s} is not found in the model. Mask did not applied'.format(list(limit.keys())[0]))

        if logx:
            x = np.log10(getattr(self, parx)[mask])
        else:
            x = getattr(self, parx)[mask]

        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel(xlabel,fontsize=fsize)

        lines = []
        thb_names = ['cool_tot', 'cool_o','cool_cp', 'cool_c', 'cool_elrec', 'cool_free', 'cool_h', 'cool_h2',
          'heat_tot', 'heat_phel','heat_secp', 'heat_phot','heat_chem', 'heat_h2', 'heat_cr']
        for i, p in enumerate(pars):
            if i == 0:
                axi = ax
            else:
                axi = ax.twinx()
            ylabel=''
            if p == 'tgas':
                ylabel = 'Temperature, K'
                yscale = 'dec'
                bot,top = 10,1e3
            elif p == 'n':
                ylabel = 'Density, cm'
                yscale = 'dec'
            elif p in ['OPR']:
                ylabel = 'OPR'
                yscale = 'dec'
                bot, top = 1, 15
            elif p in thb_names:
                ylabel = 'thb_rates'
                yscale = 'dec'
            elif p == 'heat_tot':
                ls = '--'

            if p in ['tgas', 'n', 'av', 'pgas','h2t01','uv_dens','uv_flux','H2_photo_dest_prob','OPR','h2fr']:
                y = getattr(self, p)[mask]
            elif p == 'tgas_m':
                y = []
                for x0 in self.x[mask]:
                    mask_loc=self.x<x0
                    mask_loc*=self.h2fr>0.001
                    if np.size(self.x[mask_loc])>1:
                        y.append(np.trapz(self.tgas[mask_loc], x=self.x[mask_loc])/(self.x[mask_loc][-1]-self.x[mask_loc][0]))
                    else:
                        y.append(0)
                y = np.array(y)
                ls = '--'
            elif p == 'Nh2t01':
                y = (self.sp['NH2j1'][mask]) / \
                    (self.sp['NH2j0'][mask])
                z = []
                for e in y:
                    z1 = root(polyJ01, x0=0.2, args=(e))
                    print('root_T01', e, -170.5 / np.log(z1.x), z1.x, polyJ01(z1.x, e))
                    z.append(z1.x)
                y = -170.5 / np.log(z)
                bot, top = 10, 1e3
            elif p == 'Nh2topr':
                y = (self.sp['NH2j1'][mask] + self.sp['NH2j3'][mask]) / \
                    (self.sp['NH2j0'][mask] + self.sp['NH2j2'][mask])
                z = []
                for k,e in enumerate(y):
                    z1 = root(polyJ012, x0=0.2, args=(e))
                    print('root_T012', x[k], e, -170.5 / np.log(z1.x), z1.x, polyJ012(z1.x, e))
                    z.append(z1.x)
                z = np.array(z)
                y = -170.5 / np.log(z)
                bot, top = 10, 1e3
            elif p == 'Nh2t02':
                y = 85.3 * 6 / np.log(5. / (self.sp['NH2j2'][mask] / self.sp['NH2j0'][mask]))
                bot, top = 10, 1e3
            elif p == 'Nh2t13':
                y = 85.3 * 9 / np.log(7. / 3. /(self.sp['NH2j3'][mask] / self.sp['NH2j1'][mask]))
                bot, top = 10, 1e3
            elif p == 'T02':
                y = 85.3*6/np.log(5./(self.sp['H2j2'][mask]/self.sp['H2j0'][mask]))
            elif p == 'T01':
                y = 85.3 * 2 / np.log(9. / (self.sp['H2j1'][mask] / self.sp['H2j0'][mask]))
            elif p == 'T13':
                y = 85.3 * 9 / np.log(7. / 3. /(self.sp['H2j3'][mask] / self.sp['H2j1'][mask]))
            elif p == 'T24':
                y = 85.3 * 14 / np.log(9. / 5. /(self.sp['H2j4'][mask] / self.sp['H2j2'][mask]))
            elif p == 'OPR_logNJ1/J02':
                y = (self.sp['NH2j1'][mask]+self.sp['NH2j3'][mask]) / \
                    (self.sp['NH2j0'][mask] + self.sp['NH2j2'][mask])
                print(p,y)
                bot, top = 1, 15
            elif p == 'OPR_logNJ1/J0':
                y = (self.sp['NH2j1'][mask]) / \
                    (self.sp['NH2j0'][mask])
                print(p, y)
                bot, top = 1, 15
            elif p in thb_names:
                if p in ['heat_tot', 'heat_phel','heat_secp', 'heat_phot','heat_chem', 'heat_h2', 'heat_cr']:
                    y = self.sp[p][mask]/(self.sp['H'][mask]+self.sp['H2'][mask])**2.0
                else:
                    y = self.sp[p][mask]/(self.sp['H'][mask]+self.sp['H2'][mask])**2.0
                if p in ['heat_tot','cool_tot']:
                    bot, top = -0.5*np.max(y),2*np.max(y)
            elif p in ['H2_diss']:
                y = np.log10(self.sp[p][mask])
            else:
                if 'N_' in p:
                    y = np.log10(integrate.cumtrapz(self.sp[p.replace('N_', '')][mask], 10**x, initial=0))
                else:
                    y = self.sp[p][mask]

            color = plt.cm.tab10(i / 10)
            if i==0 :
                axi.set_ylabel(ylabel, color=color,fontsize=14)
            line, = axi.plot(x, y, color=color, label=p, ls=ls)
            lines.append(line)

            if 1:
                axi.tick_params(which='both', width=1, direction='in', right='True', top='True')
                axi.tick_params(which='major', length=5, direction='in', right='True', top='True')
                axi.tick_params(which='minor', length=4, direction='in')
                axi.tick_params(axis='both', which='major', labelsize=14, direction='in')

            if i > 0:
                axi.spines['right'].set_position(('outward', 60*(i-1)))
                axi.set_frame_on(True)
                axi.patch.set_visible(False)
                axi.axis('off')
                axi.set_ylim(ax.get_ylim())


            for t in axi.get_yticklabels():
                t.set_color(color)

            if bot is not None:
                axi.set_ylim(bot, top)
                ax.set_ylim(bot, top)

            if yscale == 'log':
                ax.set_yscale('log')
                axi.set_yscale('log')



        if legend:
            ax.legend(handles=lines, loc='best',fontsize=fsize)

        return ax

    def plot_profiles(self, species=None, logx=False, logy=False, label=None, ylabel=True, ax=None, legend=True, ls='-', lw=1, parx='av', limit=None,normed=False):
        """
        Plot the profiles of the species

        :param:
            -  species       :  list of the species to plot
            -  ax            :  axis object to plot in, if None, then figure is created
            -  legend        :  show legend
            -  ls            :  linestyles
            -  lw            :  linewidth
            -  logx          :  log of x axis
            -  label         :  set label of x axis
            -  parx          :  what is on x axis
            -  limit         :  if not None plot only part of the cloud specified by limit
                                e.g. {'NH2': 19.5}

        :return: ax
            -  ax            :  axis object
        """
        if species is None:
            species = self.species

        if parx == 'av':
            xlabel = 'Av'
        elif parx == 'x':
            xlabel = 'log(Distance), cm'
        elif parx == 'h2':
            xlabel = 'log(NH2), cm-2'
        elif parx == 'hi':
            xlabel = 'log(NHI), cm-2'

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if logx:
            mask = getattr(self, parx) > 0
        else:
            mask = getattr(self, parx) > -1

        if limit is not None:
            if hasattr(self, list(limit.keys())[0]):
                v = getattr(self, list(limit.keys())[0])
                mask *= v < list(limit.values())[0]
            elif list(limit.keys())[0] in self.sp.keys():
                self.set_mask(logN=limit)
                mask *= self.mask
                #v = self.sp[list(limit.keys())[0]]
                #mask *= v < list(limit.values())[0]
            else:
                warnings.warn('limit key {:s} is not found in the model. Mask did not applied'.format(list(limit.keys())[0]))

        if logx:
            x = np.log10(getattr(self, parx)[mask])
        else:
            x = getattr(self, parx)[mask]

        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel(xlabel,fontsize=14)


        if ax is None:
            fig, ax = plt.subplots()

        lab = label
        print(x)
        for s in species:
            if '/' in s:
                s1, s2 = s.split('/')[:]
                y = self.sp[s1][mask] / self.sp[s2][mask]
                if s in ['NH2j1/NH2j0']:
                    logy = False
            elif s == 'H2exc':
                y = (self.sp['NH2j1'][mask]) / \
                    (self.sp['NH2j0'][mask] + self.sp['NH2j2'][mask])
                logy = False
            elif s == 'Texc':
                y = (self.sp['NH2j1'][mask])/\
                (self.sp['NH2j0'][mask])
                logy = True # + self.sp['H2j2'][mask] + self.sp['H2j3'][mask]
                ax.set_ylim(1,3)
                z = []
                for e in y:
                    z1 = root(poly, x0=0.2, args=(e))
                    print('root',z1.x, poly(z1.x,e))
                    z.append(z1.x)
                y = -170.5/np.log(z)
            else:
                y = self.sp[s][mask]
            print(s, y)
            if label is None:
                lab = s
            if normed:
                y = y/self.n0
            if logy:
                ax.plot(x, np.log10(y), ls=ls, label=s, lw=lw, linewidth=2.0)
            else:
                ax.plot(x, y, ls=ls, label=s, lw=lw, linewidth=2.0)
            #if s in ['H2j1/H2']:
            #    ax.axvline(self.calc_T01_limit(limith2={'H2': 21.5}, case=1), ls='--', color='green')

        if ylabel:
            ax.set_ylabel(label, fontsize=12)

        if legend:
            ax.legend(fontsize=12)

        return ax

    def plot_exciation(self, ax=None, H2levels=['H2j0','H2j1','H2j2','H2j3','H2j4'], logN=[17,18],legend=True):

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['red','blue','black', 'green','magenta']
        ind =-1
        if logN is not None:
            for N in logN:
                ind+=1
                self.calc_cols(species=H2levels, logN={'H2': N})
                j = np.sort([int(s[3:]) for s in H2levels])
                x = [H2energy[0, i] for i in j]
                mod = [self.cols['H2j' + str(i)] - np.log10(stat_H2[i]) for i in j]
                if len(mod) > 0:
                    ax.plot(x, mod-mod[0], marker='o', ls='--', lw=1, markersize=5, color=colors[ind], zorder=0,
                            linewidth=2.0,label='LogN='+str(N))

        if 0:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_major_locator(MultipleLocator(500))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            ax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            ax.tick_params(which='minor', length=3, direction='in')
            ax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')

            ax.set_xlabel('Energy of H$_2$ levels, cm$^{-1}$', fontsize=labelsize)
            ax.set_ylabel('$\log N(H_2,J)/g(J)$', fontsize=labelsize)

        if legend:
            ax.legend(fontsize=12)

    def calc_cols(self, species=[], logN=None, sides=2):
        """
        Calculate column densities for given species

        :param:
            -  species       :  list of the species to plot
            -  logN          :  column density threshold, dictionary with species and logN value
            -  side          :  make calculation to be one or two sided

        :return: sp
            -  sp            :  dictionary of the column densities by species
        """

        cols = OrderedDict()

        if logN is not None:
            logN[list(logN.keys())[0]] -= np.log10(sides)
            self.set_mask(logN=logN, sides=sides)

            for s in species:
                cols[s] = np.log10(np.trapz(self.sp[s][self.mask], x=self.x[self.mask])) + np.log10(sides)

        else:
            for s in species:
                cols[s] = np.log10(integrate.cumtrapz(self.sp[s], x=self.x))

        self.cols = cols

        return self.cols

    def set_mask(self, logN={'H': None}, sides=2):
        """
        Calculate mask for a given threshold

        :param:
            -  logN          :  column density threshold

        :return: None
        """
        cols = np.insert(np.log10(integrate.cumtrapz(self.sp[list(logN.keys())[0]], x=self.x)), 0, 0)

        l = int(len(self.x) / sides) + 1 if sides > 1 else len(self.x)
        if logN[list(logN.keys())[0]] > cols[l-1]:
            logN[list(logN.keys())[0]] = cols[l-1]

        if logN is not None:
            self.mask = cols < logN[list(logN.keys())[0]]
        else:
            self.mask = cols > -1

    def lnLike(self, species={}, syst=0, verbose=False, relative=None):
        print(species, relative)
        lnL = 0
        if 0:
            #verbose:
            self.showSummary()
        for k, v in species.items():
            if relative is None:
                v1 = v
                s = self.cols[k]
            else:
                v1 = v / species[relative]
                s = self.cols[k] - self.cols[relative]
                #print('v1:', v1)
                #print('s:', s)

            if syst > 0:
                v1 *= a(0, syst, syst, 'l')
            if verbose:
                print(np.log10(self.uv), np.log10(self.n0), self.cols[k], v1.log(), v1.lnL(self.cols[k]))
            if v.type in ['m', 'u', 'l']:
                lnL += v1.lnL(s)

        self.lnL = lnL
        return self.lnL

    def calc_mean_pars(self, pars=['tgas'], logN=None, sides=2,logscale=True):
        """
        Calculate mean values of phys parameters for given column density

        :param:
            -  pars          :  list of the pars to plot
            -  logN          :  column density threshold, dictionary with species and logN value
            -  side          :  make calculation to be one or two sided

        :return: meanpars
            -  mpars            :  dictionary of the mean values by phys parameters
        """

        mpars = OrderedDict()

        if logN is not None:
            logN[list(logN.keys())[0]] -= np.log10(sides)
            self.set_mask(logN=logN, sides=sides)

            for p in pars:
                if p in ['T01']:
                    mpars[p] = 0
                else:
                    #self.mask *= 2 * self.h2 / (2 * self.h2 + self.hi) < 0.01
                    #mpars[p] = np.trapz(getattr(self, p)[self.mask], x=self.x[self.mask])/(self.x[self.mask][-1]-self.x[self.mask][0])
                    if logscale:
                        mpars[p] = np.log10(getattr(self, p)[self.mask][-1])
                    else:
                        mpars[p] = getattr(self, p)[self.mask][-1]
                    #print(p,mpars[p])
        else:
            for p in pars:
                if p in ['av']:
                    mpars[p] = 0
                else:
                    #mpars[p] = np.array(integrate.cumtrapz(getattr(self, p), x=self.x))/self.x[-1]
                    mpars[p] = np.array(integrate.cumtrapz(getattr(self, p), x=self.x)) / self.x[-1]

        self.mpars = mpars
        return self.mpars


    def calc_T01_limit(self, parx='h2', limith2={'H2': 21.5}, logx=True,case=1):
        """calculate the H2 column density, where
        H2 rotational levels become thermal populated
        i.e. T01 ~ Tkin
        :return: logNH2_th
        """

        if logx:
            mask = getattr(self, parx) > 0
        else:
            mask = getattr(self, parx) > -1

        if limith2 is not None:
            if hasattr(self, list(limith2.keys())[0]):
                v = getattr(self, list(limith2.keys())[0])
                mask *= v < list(limith2.values())[0]
            elif list(limith2.keys())[0] in self.sp.keys():
                # v = self.sp[list(limit.keys())[0]]
                self.set_mask(logN=limith2)
                mask *= self.mask
            else:
                warnings.warn(
                    'limit key {:s} is not found in the model. Mask did not applied'.format(list(limit.keys())[0]))

        if logx:
            x = np.log10(getattr(self, parx)[mask])
        else:
            x = getattr(self, parx)[mask]

        if case == 1:
            # compare tkin and t01h2
            # mask *=  self.sp['NH2j2'] < self.sp['NH2j0'] #*2
            mask *= np.log10(getattr(self, 'h2'))>16
            y = (self.sp['NH2j2']) / \
                (self.sp['NH2j0'])
            y1 = self.tgas
            #mask *=  85.3 * 6 /np.log(5./y)  > y1
            tkin = self.tgas[mask]
            y = (self.sp['NH2j2'][mask]) / \
                (self.sp['NH2j0'][mask])
            t01 = 170.5 / np.log(9./(self.sp['NH2j1'][mask] / self.sp['NH2j0'][mask]))
            t02 = 85.3 * 6 /np.log(5./y)
            t13 = 85.3 * 9 / np.log(7. / 3. /(self.sp['NH2j3'][mask] / self.sp['NH2j1'][mask]))
            x = np.log10(getattr(self, 'h2'))[mask]

            tfit = t01
            #tgit = t02
            mask1 = np.abs(tkin - tfit)  < 100
            delta = np.zeros_like(tkin) + 20
            for k,e in enumerate(delta):
                if delta[k]<0.5*(tkin[k]+tfit[k])/2:
                    delta[k] = 0.5*(tkin[k]+tfit[k])/2
            mask1 *= np.abs(tkin - tfit) < delta
            xx = mask1[::-1]
            ind = 0
            for k,e in enumerate(mask1):
                if e:
                    ind = k
                elif ind >0:
                    mask1[0:k] = False
                    ind = 0

            #        e = 0
            #    else:
            #        e =1
            if x[mask1].size <10:
                logNH2_th = 21
            else:
                logNH2_th = x[mask1][0]
            #print('logNH2_th:', logNH2_th)
            return logNH2_th
        elif case == 2:
            # compare tkin and t01h2
            #mask *= self.sp['H2j2']<self.sp['H2j0']
            #mask *= np.log10(getattr(self, 'h2'))>17
            tkin = self.tgas[mask]
            y = (self.sp['NH2j1'][mask]) / \
                (self.sp['NH2j0'][mask])
            t01 = -170.5/np.log(y/9.)
            x = np.log10(getattr(self, 'h2'))[mask]

            mask = np.abs(tkin-t01)/tkin < 0.5
            xx = x[mask]
            if x[mask].size == 0:
                logNH2_th = 21
            else:
                logNH2_th = x[mask][0]
            #print('logNH2_th:',logNH2_th)
            return logNH2_th
        elif case == 3:
            # compare tkin and t01h2
            mask *= np.log10(getattr(self, 'h2'))>15
            #mask *= self.sp['NH2j2'] < self.sp['NH2j0']
            opr_logN = (self.sp['NH2j1'] + self.sp['NH2j3']) / (
                    self.sp['NH2j0'] + self.sp['NH2j2'])
            opr_logN[0]=1
            imax = np.where(opr_logN == np.amax(opr_logN))
            mask *= getattr(self, 'h2') > 2*getattr(self, 'h2')[imax]

            opr = getattr(self, 'OPR')[mask]
            opr_logN = (self.sp['NH2j1'][mask] + self.sp['NH2j3'][mask]) / (
                        self.sp['NH2j0'][mask] + self.sp['NH2j2'][mask])
            x = np.log10(getattr(self, 'h2'))[mask]

            #mask = np.abs(opr-opr_logN)/opr < 0.07
            mask = np.abs(opr - opr_logN) < 0.1*opr

            if x[mask].size == 0:
                logNH2_th = 21
            else:
                logNH2_th = x[mask][1]
            #print('logNH2_th:',logNH2_th)
            return logNH2_th

class H2_exc():
    def __init__(self, folder='', H2database='all'):
        self.folder = folder if folder.endswith('/') else folder + '/'
        self.models = {}
        self.species = ['H', 'H+', 'H2', 'CI', 'C+', 'CO', 'C2', 'C13O', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7', 'H2j8', 'H2j9', 'H2j10',
                        'COj0','COj1','COj2','COj3','COj4','COj5','COj6','COj7',
                        'NH','NH2', 'NH2j0', 'NH2j1', 'NH2j2', 'NH2j3', 'NH2j4', 'NH2j5', 'NH2j6', 'NH2j7', 'NH2j8', 'NH2j9', 'NH2j10',
                        'NCI','NCj0', 'NCj1', 'NCj2',
                        'HD', 'HDj0', 'HDj1', 'CIj0', 'CIj1', 'CIj2',
                        'NCO', 'NCOj0','NCOj1','NCOj2', 'COj0', 'COj1', 'COj2','COj3',
                        'O','O+','el','He','He+',
                        'H2_dest_rate', 'H2_form_rate_er', 'H2_form_rate_lh', 'H2_photo_dest_prob','H2_diss',
                        'cool_tot', 'cool_o', 'cool_cp', 'cool_c', 'cool_elrec', 'cool_free', 'cool_h','cool_h2',
                        'heat_tot', 'heat_phel','heat_secp','OPR', 'heat_phot','heat_chem', 'heat_h2','heat_cr']
        self.readH2database(H2database)

    def readH2database(self, data='all'):
        import sys
        sys.path.append('/home/toksovogo/science/codes/python/3.5/H2_excitation')
        import H2_summary

        self.H2 = H2_summary.load_empty()
        if data == 'all':
            self.H2.append(H2_summary.load_QSO())
        if data in ['all', 'P94']:
            self.H2.append(H2_summary.load_P94())
        if data == 'z=-1':
            self.H2.append(H2_summary.load_ex1())
        if data == 'z=-0.5':
            self.H2.append(H2_summary.load_ex2())
        if data == 'z=0':
            self.H2.append(H2_summary.load_ex3())
        if data == 'z=0.5':
            self.H2.append(H2_summary.load_ex4())
        if data == 'H2UV':
            self.H2.append(H2_summary.load_total())
        if data == 'Magellan':
            self.H2.append(H2_summary.load_Magellan())
        if data == 'Galaxy':
            self.H2.append(H2_summary.load_MWH2CI())
        if data == 'lowzDLAs':
            self.H2.append(H2_summary.load_lowzDLAs())
        if data == 'LMC':
            self.H2.append(H2_summary.load_LMC())
        if data == 'SMC':
            self.H2.append(H2_summary.load_SMC())
        if data == 'MW':
            self.H2.append(H2_summary.load_MV())
        if data == 'local':
            self.H2.append(H2_summary.load_Magellan())
            self.H2.append(H2_summary.load_MWH2CI())

    def readmodel(self, filename=None, show_summary=False, folder=None, printname=True):
        """
        Read one model by filename
        :param:
            -  filename             :  filename contains the model
            -  print_summary        :  if True, print summary for each model
        """
        if folder == None:
            folder = self.folder
        if filename is not None:
            print('read name:', filename)
            m = model(folder=folder, filename=filename, species=self.species, show_summary=False)
            self.models[m.name] = m
            self.current = m.name

            if printname:
                print(m.name, 'n0=', m.n0, 'uv=', m.uv, 'me=',m.me, 'zeta=', m.zeta, 'av=', m.av[-1],'tcmb=',m.tcmb)
                if m.zeta != 2e-016*m.uv and m.zeta != 1.0e-016:
                    print('err in CR rate', 2e-016*m.uv)

            if show_summary:
                m.showSummary()

    def readfolder(self, verbose=False):
        """
        Read list of models from the folder
        """
        if 1:
            for (dirpath, dirname, filenames) in os.walk(self.folder):
                print(dirpath, dirname, filenames)
                for f in filenames:
                    if f.endswith('.hdf5'):
                        self.readmodel(filename=f, folder=dirpath + '/')
        else:
            for f in os.listdir(self.folder):
                if f.endswith('.hdf5'):
                    self.readmodel(f, show_summary=verbose)

    def setgrid(self, pars=[], fixed={}, show=True):
        """
        Show and mask models for grid of specified parameters
        :param:
            -  pars           :  list of parameters in the grid, e.g. pars=['uv', 'P']
            -  fixed          :  dict of parameters to be fixed, e.g. fixed={'z': 0.1}
            -  show           :  if true show the plot for the grid
        :return: mask
            -  mask           :  list of names of the models in the grid
        """
        self.grid = {p: [] for p in pars}
        self.mask = []

        #print(self.models)
        for name, model in self.models.items():
            for k, v in fixed.items():
                if getattr(model, k) != v and v != 'all':
                    break
            else:
                for p in pars:
                    self.grid[p].append(getattr(model, p))
                self.mask.append(name)

        #print(self.grid)
        if show and len(pars) == 2:
            fig, ax = plt.subplots()
            for v1, v2 in zip(self.grid[pars[0]], self.grid[pars[1]]):
                ax.scatter(v1, v2, 100, c='orangered')
            ax.set_xscale("log", nonposy='clip')
            ax.set_yscale("log", nonposy='clip')
            ax.set_xlabel(pars[0])
            ax.set_ylabel(pars[1])

        if show and len(pars) == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for v1, v2, v3 in zip(self.grid[pars[0]], self.grid[pars[1]], self.grid[pars[2]]):
                ax.scatter(v1, v2, v3, c='orangered')
            ax.set_xlabel(pars[0])
            ax.set_ylabel(pars[1])
            ax.set_zlabel(pars[2])

        return self.mask

    def comp(self, object):
        """
        Return componet object from self.H2
        :param:
            -  object         :  object name.
                                    Examples: '0643' - will search for the 0643 im quasar names. Return first component.
                                              '0643_1' - will search for the 0643 im quasar names. Return second component
        :return: q
            -  q              :  qso.comp object (see file H2_summary.py how to retrieve data (e.g. column densities) from it
        """
        qso = self.H2.get(object.split('_')[0])
        if len(object.split('_')) > 1:
            q = qso.comp[int(object.split('_')[1])]
        else:
            q = qso.comp[0]

        return q

    def listofPDR(self):
        list = []
        for q in self.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                list.append(name)
        return list

    def listofmodels(self, models=[]):
        """
        Return list of models

        :param:
            -  models         :  names of the models, can be list or string for individual model

        :return: models
            -  models         :  list of models

        """
        if isinstance(models, str):
            if models == 'current':
                models = [self.models[self.current]]
            elif models == 'all':
                models = list(self.models.values())
            else:
                models = [self.models[models]]
        elif isinstance(models, list):
            if len(models) == 0:
                models = list(self.models.values())
            else:
                models = [self.models[m] for m in models]

        return models

    def compare(self, object='', species='H2', spmode = 'abs', mpars = ['tgas'], models='current', syst=0.0, levels=[], others='ignore'):
        """
        Calculate the column densities of H2 rotational levels for the list of models given the total H2 column density.
        and also log of likelihood

        :param:
            -  object            :  object name
            - species            : which species to use. Can be 'H2' or 'CI'¶
            -  models            :  names of the models, can be list or string
            -  syst              :  add systematic uncertainty to the calculation of the likelihood
            -  levels            :  levels that used to constraint, if empty list used all avaliable

        :return: None
            column densities are stored in the dictionary <cols> attribute for each model
            log of likelihood value is stored in <lnL> attribute
        """

        q = self.comp(object)
        if species in ['H2', 'CI', 'CO']:
            if len(levels) > 0:
                label = species + 'j'
                full_keys = [s for s in q.e.keys() if (label in s) and ('v' not in s)]
                label = species + 'j{:}'
                keys = [label.format(i) for i in levels if label.format(i) in full_keys]
                spec = OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in keys])
                if others in ['lower', 'upper']:
                    for k in full_keys:
                        if k not in keys:
                            v = q.e[k].col * a(0.0, syst, syst)
                            if others == 'lower':
                                spec[k] = a(v.val - v.minus, t=others[0])
                            else:
                                spec[k] = a(v.val + v.plus, t=others[0])
        #if species == 'H2':
        #    if len(levels) > 0:
        #        full_keys = [s for s in q.e.keys() if ('H2j' in s) and ('v' not in s)]
        #        keys = ['H2j{:}'.format(i) for i in levels if 'H2j{:}'.format(i) in full_keys]
        #        spec = OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in keys])
        #        if others in ['lower', 'upper']:
        #            for k in full_keys:
        #                if k not in keys:
        #                    v = q.e[k].col * a(0.0, syst, syst)
        #                    if others == 'lower':
        #                        spec[k] = a(v.val - v.minus, t=others[0])
        #                    else:
        #                        spec[k] = a(v.val + v.plus, t=others[0])
        #    #print(spec)
        #elif species == 'CI':
        #    if len(levels) > 0:
        #        full_keys = [s for s in q.e.keys() if ('CIj' in s) and ('v' not in s)]
        #        keys = ['CIj{:}'.format(i) for i in list(set(levels) & set([0, 1, 2]))  if 'CIj{:}'.format(i) in full_keys]
        #        print(keys)
        #    spec = OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in full_keys])

        #elif species == 'CO':
        #    if len(levels) > 0:
        #        full_keys = [s for s in q.e.keys() if ('COj' in s) and ('v' not in s)]
        #        keys = ['COj{:}'.format(i) for i in levels if 'COj{:}'.format(i) in full_keys]
        #        spec = OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in keys])
        #        if others in ['lower', 'upper']:
        #            for k in full_keys:
        #                if k not in keys:
        #                    v = q.e[k].col * a(0.0, syst, syst)
        #                    if others == 'lower':
        #                        spec[k] = a(v.val - v.minus, t=others[0])
        #                    else:
        #                        spec[k] = a(v.val + v.plus, t=others[0])


        for model in self.listofmodels(models):
            if spmode == 'abs':
                relative = None
                logN = {species: q.e[species].col.val}
            elif spmode == 'rel':
                relative = species+'j0'
                logN = {'H2': q.e['H2'].col.val}
#            if species == 'CO':
#                logN = {'CO': q.e['CO'].col.val}
#            elif species == 'H2':
#                logN = {'H2': q.e['H2'].col.val}
#            elif species == 'CI':
 #               logN = {'CI': q.e['Ci'].col.val}
            model.calc_cols(spec.keys(), logN=logN)
            #model.lnLike(OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in keys]), relative=relative)
            model.lnLike(spec, relative=relative)
            if mpars is not None:
                model.calc_mean_pars(pars=mpars, logN={'H2': q.e['H2'].col.val})

    def comparegrid(self, object='0643', species='H2', spmode = 'abs', pars=[], fixed={}, syst=0.0, plot=True, show_best=True, levels='all', others='ignore'):
        self.setgrid(pars=pars, fixed=fixed, show=False)
        for el in ['H2', 'CI', 'CO']:
            if el in self.comp(object).e.keys():
                self.grid['N'+el+'tot'] = self.comp(object).e[el].col.val
            else:
                self.grid['N' + el + 'tot'] = None
                #if 'CO' in self.comp(object).e.keys():
        #    self.grid['NCOtot'] = self.comp(object).e['CO'].col.val
        #else:
        #    self.grid['NCOtot'] = None
        #if 'CI' in self.comp(object).e.keys():
        #    self.grid['NCItot'] = self.comp(object).e['CI'].col.val
        #else:
        #    self.grid['NCItot'] = None
        #print(others)
        self.compare(object, species=species, spmode = spmode, models=self.mask, syst=syst, levels=levels, others=others)
        self.grid['lnL'] = np.asarray([self.models[m].lnL for m in self.mask])
        self.grid['cols'] = np.asarray([self.models[m].cols for m in self.mask])
        self.grid['mpars'] = np.asarray([self.models[m].mpars for m in self.mask])
        #print(self.grid)

        if plot:
            if len(pars) == 1:
                x = np.asarray(self.grid[list(self.grid.keys())[0]])
                inds = np.argsort(x)
                fig, ax = plt.subplots()
                ax.scatter(x[inds], self.grid['lnL'][inds], 100, c='orangered')
                self.plot = plot(self)
                self.plot.set_data([x[inds], self.grid['lnL'][inds]])
                self.plot.show()

            if len(pars) == 2:
                fig, ax = plt.subplots()
                for v1, v2, l in zip(self.grid[pars[0]], self.grid[pars[1]], self.grid['lnL']):
                    ax.scatter(v1, v2, 0)
                    ax.text(v1, v2, '{:.1f}'.format(l), size=20)

            if show_best:
                imax = np.argmax(lnL)
                ax = self.plot_objects(objects=object)
                self.plot_models(ax=ax, models=self.mask[imax])

    def plot_objects(self, objects=[], species=[], ax=None, plotstyle='scatter', legend=False,syst=None,label=None,msize=3,color='black'):
        """
        Plot object from the data

        :param:
            -  objects              :  names of the object to plot
            -  species              :  names of the species to plot
            -  ax                   :  axes object, where to plot. If None, then it will be created
            -  plotstyle            :  style of plotting. Can be 'scatter' or 'lines'
            -  legend               :  show legend

        :return: ax
            -  ax                   :  axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        if not isinstance(objects, list):
            objects = [objects]
        for o in objects:
            q = self.comp(o)
            if species is None or len(species) == 0:
                sp = [s for s in q.e.keys() if 'H2j' in s]
                j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                x = [H2energy[0, i] for i in j]
                sort='H2j'
                stat = stat_H2
            elif species == 'CI':
                sp = [s for s in q.e.keys() if 'CIj' in s]
                j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                x = [CIenergy[i] for i in j]
                sort = 'CIj'
                stat = stat_CI
            elif species == 'CO':
                sp = [s for s in q.e.keys() if 'COj' in s]
                j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                x = [COenergy[i] for i in j]
                sort = 'COj'
                stat = stat_CO
            if syst is not None:
                y = [q.e[sort + str(i)].col*a(0.0, syst, syst) / stat[i] for i in j]
                #if species == 'CI':
                #    y = [(q.e[sort + str(i)].col/q.e[sort + '0'].col)*a(0.0, syst, syst)  / stat[i] for i in j]
            else:
                y = [q.e[sort + str(i)].col/ stat[i] for i in j]
            typ = [q.e[sort + str(i)].col.type for i in j]
            #if label==None:
                #label = o

            if len(y) > 0:
                #color = 'black'
                #color = plt.cm.tab10(objects.index(o) / 10)
                color= color
                if plotstyle == 'line':
                    ax.plot(x, column(y, 'v'), marker='o', ls='-', lw=2, color=color, label=label)
                    label = "_nolegend_"
                for k in range(len(y)):
                    if typ[k] == 'm':
                        print('err_H2_',k,column(y, 2)[k], column(y, 1)[k])
                        #ax.errorbar([x[k]], [column(y, 0)[k]], yerr=[[column(y, 1)[k]], [column(y, 2)[k]]],
                        if 1:
                            ax.errorbar([x[k]], [column(y, 0)[k]], yerr=[[column(y, 2)[k]], [column(y, 1)[k]]],
                                    fmt='o', lw=1, elinewidth=1, color=color, label=label,markersize=msize,capsize=2)
                        if 0:
                            ax.errorbar([x[k]], [column(y, 0)[k]-column(y, 0)[0]], yerr=[[column(y, 1)[k]], [column(y, 2)[k]]],
                                    fmt='o', lw=1, elinewidth=1, color=color, label=label,markersize=msize,capsize=2)

                    label = "_nolegend_"
                    if typ[k] == 'u':
                        ax.errorbar(x=[x[k]], y=[column(y, 0)[k]],
                                    fmt='o', lw=1, elinewidth=1, color=color, uplims=True, yerr=0.4,markersize=msize)

        if legend:
            handles, labs = ax.get_legend_handles_labels()
            labels = np.unique(labs)
            handles = [handles[np.where(np.asarray(labs) == l)[0][0]] for l in labels]
            ax.legend(handles, labels, loc='best')

        return ax

    def plot_models(self, ax=None, models='current', speciesname=['H'], logN=None, species='<7', legend=True, logy=True,color='royalblue',labelsize = 12,msize=5,label=None):
        """
        Plot excitation for specified models

        :param:
            -  ax                :  axes object, where to plot. If None, it will be created
            -  models            :  names of the models to plot
            -  logN              :  total H2 column density
            -  species           :  list of rotational levels to plot, can be string as '<N', where N is the number of limited rotational level
            -  legend            :  if True, then plot legend

        :return: ax
            -  ax                :  axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        if isinstance(species, str):
            species = ['H2j' + str(i) for i in range(int(species[1:]))]

        if logN is not None:
            for m in self.listofmodels(models):
                m.calc_cols(species, logN={'H2': logN})

        for ind, m in enumerate(self.listofmodels(models)):
            j = np.sort([int(s[3:]) for s in m.cols.keys()])
            x = [H2energy[0, i] for i in j]
            mod = [m.cols['H2j'+str(i)] - np.log10(stat_H2[i]) for i in j]
            #mod = [m.cols['H2j' + str(i)] for i in j]
            print('T01:', -170.5/np.log(10**(mod[1]-mod[0])))

            if len(mod) > 0:
                #color = plt.cm.tab10(ind/10)
                if label is None:
                    label = 'H$_2$(J)'
                ax.plot(x, mod, marker='o', ls='--', lw=1, markersize=5, color=color, label=label, zorder=0, linewidth=2.0)

        if 1:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_major_locator(MultipleLocator(500))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            ax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            ax.tick_params(which='minor', length=3, direction='in')
            ax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')

            ax.set_xlabel('Energy of H$_2$ levels, cm$^{-1}$', fontsize=labelsize)
            ax.set_ylabel('$\log N(H_2,J)/g(J)$', fontsize=labelsize)
            #ax.set_ylabel('$\log N({\\rm{H_2}})_{\\rm J}/g_{\\rm{J}} - \log N({\\rm{H_2}})_{\\rm 0}$', fontsize=labelsize)

        if legend:
            ax.legend(loc='best',fontsize=labelsize-2)

        return ax

#    def compare_models(self, speciesname=['H'], ax=None, models='current', physcondname=False, logy=False, parx='av'):
#        """
#        Plot comparison of certain quantities for specified models
#
#        :param:
#            -  ax                :  axes object, where to plot. If None, it will be created
#            -  models            :  names of the models to plot
#            -  speciesname       :  name of plotted from list ['H', 'H2', 'C', 'C+', 'CO', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7']
#
#        :return: ax
#            -  ax                :  axes object
#        """
#
#        if ax is None:
#            fig, ax = plt.subplots(figsize=(12, 8))
#
#        if 1:
#            legend_m = []
#            for ind, m in enumerate(self.listofmodels(models)):
#                m.plot_profiles(species=speciesname, ax=ax, logy=logy,parx=parx)
#                legend_m.append(m.name[0:40])
#            ax.legend(labels=legend_m, fontsize=20)
#            ax.set_title(speciesname)
#
#        if physcondname:
#            fig2, ax2 = plt.subplots(figsize=(12, 8))
#            legend_m = []
#            for ind, m in enumerate(self.listofmodels(models)):
#                m.plot_phys_cond(pars=physcondname, logx=False, ax=ax2, parx=parx)
#                legend_m.append(m.name[0:40])
#            ax2.legend(labels=legend_m, fontsize=20)
#
#        if physcondname:
#            return ax, ax2
#        else:
#            return ax

#    def show_model(self, models='current',  ax=None, speciesname=['H'],  physcond_show=True, logy=False):
#        """
#        Plot excitation for specified models
#
#        :param:
#            -  ax                :  axes object, where to plot. If None, it will be created
#            -  models            :  names of the models to plot
#            -  speciesname       :  name of plotted from list ['H', 'H2', 'C', 'C+', 'CO', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7']
#
#        :return: ax
#            -  ax                :  axes object
#        """
#
#        if ax is None:
#            fig, ax = plt.subplots(figsize=(12, 8))
#
#        if 1:
#            #legend_m = []
#            for ind, m in enumerate(self.listofmodels(models)):
#                for s in speciesname:
#                    m.plot_profiles(species=[s], ax=ax, logy=logy)
#
#            ax.set_title(m.name[0:30])
#
#        if physcond_show:
#            if 1:
#                fig2, ax2 = plt.subplots(figsize=(12, 8))
#                legend_m = []
#                for ind, m in enumerate(self.listofmodels(models)):
#                    m.plot_phys_cond(pars=['n', 'tgas'], parx='av', logx=False, ax=ax2)
#                    legend_m.append(m.name[8:19])
#                ax2.legend(fontsize=20)
#                ax2.set_title(m.name[0:30])
#
#        if physcond_show is False:
#            return ax
#        else:
#            return ax,  ax2

    def red_hdf_file(self, filename=None):
        self.file = h5py.File(self.folder + filename, 'r+')
        meta = self.file['Metadata/Metadata']
        ind = np.where(meta[:, 3] == 'metal'.encode())[0]
        attr = meta[ind, 0][0].decode() + '/' + meta[ind, 1][0].decode()
        data = self.file[attr]
        d = data[0, int(meta[ind, 2][0].decode())]
        data[0, int(meta[ind, 2][0].decode())] = d[0:3] + b'160' + d[6:]
        self.file.close()

    def best(self, object='', models='all', syst=0.0):
        models = self.listofmodels(models)
        self.compare(object=object, models=[m.name for m in models], syst=syst)
        return models[np.argmax([m.lnL for m in models])].name

    def calculate_T01_grid(self,species='H2', pars=[], fixed={},plot=False,case=1):
        self.setgrid(pars=pars, fixed=fixed, show=False)
        self.grid['NH2th'] = np.asarray([self.models[m].calc_T01_limit(limith2={'H2': 21.5},case=case) for m in self.mask])
        if plot:
            if len(pars) == 2:
                fig, ax = plt.subplots()
                for v1, v2, l in zip(self.grid[pars[0]],self.grid[pars[1]], self.grid['NH2th']):
                    v1,v2 = np.log10(v1),np.log10(v2)
                    ax.scatter(v1, v2, 0)
                    ax.text(v1, v2, '{:.1f}'.format(l), size=20)

    def calc_pars_grid(self,pars = [], models='current', logN = {'H2': 20}):
        self.setgrid(pars=pars, show=False)
        for m in self.mask:
            self.models[m].calc_mean_pars(pars=['tgas','pgas','T01'], logN=logN)
        self.grid['mpars'] = np.asarray([self.models[m].mpars for m in self.mask])



if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])

    labelsize=12

    if 0:
        fig, ax = plt.subplots()
        H2 = H2_exc(folder='./data/sample/1_5_4/COsample/')
        H2.readfolder()
        for k,m in enumerate(H2.listofmodels()):
            if 1:
                m.plot_profiles(species=['NH','NH2','NCO','NCI'], ax=ax, logx=True, logy=True,
                                 parx='h2', normed=False)

        if 0:
            H2.plot_models(ax=ax, models='all', logN=19, species='<5', color='orange', labelsize=labelsize,
                           label='$\log~N=21$', legend=False)

    if 1:
#        for s in ['z0_1','z0_3','z1_0','z3_0']:
        for s in ['z0_1']:
            if 1:
                #H2 = H2_exc(folder='data/sample/1_5_4/av0_5_cmb2_5_{:s}_n_uv'.format(s))
                #H2 = H2_exc(folder='data/sample/1_5_4/av2_0_cmb0_0_z1_0_n_uv')
                H2 = H2_exc(folder='data/sample/1_5_4/test')
                H2.readfolder()
                pars = {'n0': 'x', 'uv': 'y'}
                H2.calc_pars_grid(pars=list(pars.keys()))
                #H2.calculate_T01_grid(pars=list(pars.keys()),plot=True,case=1)
                grid = H2.grid
                x, y = np.log10(grid[list(pars.keys())[list(pars.values()).index('x')]]), np.log10(
                    grid[list(pars.keys())[list(pars.values()).index('y')]])
                print('stop')
                mp = H2.grid['mpars'][0].keys()
                H2.mpars = {}
                for p in mp:
                    H2.mpars[p] = Rbf(x, y, np.asarray([c[p] for c in grid['mpars']]), function='multiquadric',
                                        smooth=0.1)

                delta = 0.2
                num = 100
                xmin, xmax, ymin, ymax = np.min(x) - delta, np.max(x) + delta, np.min(y) - delta, np.max(y) + delta
                x1, y1 = x, y
                z1 = np.zeros_like(x1)
                for i, xi in enumerate(x):
                    z1[i] = H2.mpars['tgas'](xi,y[i])
                x, y = np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num)
                X, Y = np.meshgrid(x, y)
                z = np.zeros_like(X)
                for i, xi in enumerate(x):
                    for k, yi in enumerate(y):
                        z[k, i] = H2.mpars['tgas'](xi, yi)
                if 1:
                    fig, ax = plt.subplots()
                    c = ax.pcolor(X, Y, z, cmap='Purples') #, vmin=1.0, vmax=4.0)
                    cax = fig.add_axes([0.93, 0.27, 0.01, 0.47])
                    fig.colorbar(c, cax=cax, orientation='vertical') #, ticks=[-3, -2.5, -2, -1.5])
                    for v1, v2, l in zip(x1, y1, z1):
                        ax.text(v1, v2, '{:.1f}'.format(l), size=10, color='black')
                    ax.set_xlabel('log nH')
                    ax.set_ylabel('log Iuv')

                print('stop')
                #z=np.asarray([c for c in grid['NH2th']])
                #with open('temp/thermal_limit_{:s}_c1_fr50_t01.pkl'.format(s), 'wb') as f:
                #    pickle.dump([x, y, z], f)
            if 0:
                #interpolate
                z_rbf = Rbf(x, y, np.asarray([c for c in z]), function='multiquadric', smooth=0.1)

                num = 50
                x, y = np.log10(grid[list(pars.keys())[list(pars.values()).index('x')]]), np.log10(
                    grid[list(pars.keys())[list(pars.values()).index('y')]])
                x1, y1,z1 = x, y,z  # copy for save
                xmin,xmax,ymin,ymax = 1,4,0,1
                x, y = np.linspace(xmin, xmax, num), np.linspace(ymin,ymax,num)
                X, Y = np.meshgrid(x, y)
                z = np.zeros_like(X)
                for i, xi in enumerate(x):
                    for k, yi in enumerate(y):
                        z[k, i] = z_rbf(xi,yi)
                if 1:
                    fig, ax = plt.subplots()
                    ax.pcolor(X, Y, z, cmap='hot_r', vmin=15, vmax=18)
                    ax.scatter(x1, y1, 100, z1, cmap='hot_r', vmin=15, vmax=18)  # ,edgecolors='black')

    #plt.tight_layout()
    plt.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()