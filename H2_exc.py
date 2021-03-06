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
sys.path.append('home/slava/science/codes/python/')
from spectro.a_unc import a
from spectro.sviewer.utils import Timer
from spectro.pyratio import *
import warnings
from scipy.interpolate import interp2d, RectBivariateSpline, Rbf
import pickle

#pathH2exc = '/home/slava/science/codes/python/H2_excitation'
pathH2exc = '/home/toksovogo/science/codes/python/3.5/H2_excitation'




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

#H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/energy_X_H2.dat', dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
#                          unpack=True, skip_header=3, comments='#')
#H2energy = np.zeros([max(H2_energy['nu']) + 1, max(H2_energy['j']) + 1])
#for e in H2_energy:
#    H2energy[e[0], e[1]] = e[2]
H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/energy_X_H2.dat', dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                          unpack=True, skip_header=3, comments='#')
H2energy = np.zeros([max(H2_energy[0]) + 1, max(H2_energy[1]) + 1])
for k,e in enumerate(H2_energy[0]):
    H2energy[e, H2_energy[1][k]] = H2_energy[2][k]
CIenergy = [0, 16.42, 43.41]
COenergy = [0.0, 3.84, 11.53, 23.06, 38.44, 57.67, 80.73, 107.64, 138.39, 172.97] # in cm-1
#COenergy = [2.766 * ((i+1) * (i) )/1.428 for i in range(10)] # in cm-1

stat_H2 = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]
stat_CI = [(2 * i + 1) for i in range(3)]
stat_CO = [(2 * i + 1) for i in range(10)]

spcode = {'H': 'n_h', 'H2': 'n_h2', 'H+': 'n_hp', 'He': 'n_he',
          'H2j0': 'pop_h2_v0_j0', 'H2j1': 'pop_h2_v0_j1', 'H2j2': 'pop_h2_v0_j2', 'H2j3': 'pop_h2_v0_j3',
          'H2j4': 'pop_h2_v0_j4', 'H2j5': 'pop_h2_v0_j5', 'H2j6': 'pop_h2_v0_j6', 'H2j7': 'pop_h2_v0_j7',
          'H2j8': 'pop_h2_v0_j8', 'H2j9': 'pop_h2_v0_j9', 'H2j10': 'pop_h2_v0_j10',
          'D': 'n_d', 'D+': 'n_dp', 'HD': 'n_hd', 'HDj0': 'pop_hd_v0_j0', 'HDj1': 'pop_hd_v0_j1', 'HDj2': 'pop_hd_v0_j2',
          'CI': 'n_c', 'C+': 'n_cp', 'CO': 'n_co','C13O': 'n_13co',
          'COj0': 'pop_co_v0_j0','COj1': 'pop_co_v0_j1','COj2': 'pop_co_v0_j2','COj3': 'pop_co_v0_j3',
          'COj4': 'pop_co_v0_j4', 'COj5': 'pop_co_v0_j5', 'COj6': 'pop_co_v0_j6', 'COj7': 'pop_co_v0_j7',
          'el': 'n_electr', 'O': 'n_o', 'O+': 'n_op',
          'CIj0': 'pop_c_el3p_j0', 'CIj1': 'pop_c_el3p_j1', 'CIj2': 'pop_c_el3p_j2',
          'He': 'n_he','He+': 'n_hep',
          'NH': 'cd_prof_h','NH2': 'cd_prof_h2',  'NH+': 'cd_prof_hp',
          'NH2j0': 'cd_lev_prof_h2_v0_j0', 'NH2j1': 'cd_lev_prof_h2_v0_j1','NH2j2': 'cd_lev_prof_h2_v0_j2','NH2j3': 'cd_lev_prof_h2_v0_j3','NH2j4': 'cd_lev_prof_h2_v0_j4','NH2j5': 'cd_lev_prof_h2_v0_j5',
          'NH2j6': 'cd_lev_prof_h2_v0_j6', 'NH2j7': 'cd_lev_prof_h2_v0_j7','NH2j8': 'cd_lev_prof_h2_v0_j8','NH2j9': 'cd_lev_prof_h2_v0_j9','NH2j10': 'cd_lev_prof_h2_v0_j10','NH2j11': 'cd_lev_prof_h2_v0_j11',
          'NHD': 'cd_prof_hd', 'NCI': 'cd_prof_c',  'NCj0': 'cd_lev_prof_c_el3p_j0', 'NCj1': 'cd_lev_prof_c_el3p_j1','NCj2': 'cd_lev_prof_c_el3p_j2',
          'NCO': 'cd_prof_co', 'NCOj0': 'cd_lev_prof_co_v0_j0','NCOj1': 'cd_lev_prof_co_v0_j1','NCOj2': 'cd_lev_prof_co_v0_j2',
          'NCOj3': 'cd_lev_prof_co_v0_j3','NCOj4': 'cd_lev_prof_co_v0_j4','NCOj5': 'cd_lev_prof_co_v0_j5','NCOj6': 'cd_lev_prof_co_v0_j6','NCOj7': 'cd_lev_prof_co_v0_j7',
          'NC+': 'cd_prof_cp',
          'H2_dest_rate': 'h2_dest_rate_ph', 'H2_form_rate_er': 'h2_form_rate_er','H2_form_rate_lh': 'h2_form_rate_lh','H2_photo_dest_prob': 'photo_prob___h2_photon_gives_h_h',
          'cool_tot': 'coolrate_tot', 'cool_o': 'coolrate_o','cool_cp': 'coolrate_cp', 'cool_c': 'coolrate_c','cool_co': 'coolrate_co', 'cool_elrec': 'coolrate_elrecomb', 'cool_free': 'coolrate_freefree', 'cool_h': 'coolrate_h',
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
        self.pardict['codeversion'] = ('Parameters/Informations/', 0, str)
        self.pardict['metal'] = ('Parameters/Parameters', 13, float)
        self.pardict['zeta'] = ('Parameters/Parameters', 9, float)
        self.pardict['radm_ini'] = ('Parameters/Parameters', 5, float)
        self.pardict['proton_density_input'] = ('Parameters/Parameters', 3, float)
        self.pardict['avmax'] = ('Parameters/Parameters', 1, float)
        self.pardict['distance'] = ('Local quantities/Positions', 2, float)
        self.pardict['av'] = ('Local quantities/Positions', 1, float)
        self.pardict['tgas'] = ('Local quantities/Gas state', 2, float)
        self.pardict['pgas'] = ('Local quantities/Gas state', 3, float)
        self.pardict['ntot'] = ('Local quantities/Gas state', 1, float)
        self.pardict['protdens'] = ('Local quantities/Gas state', 0, float)
        self.pardict['t_cmb'] = ('Parameters/Informations/', 18, float)
        self.pardict['h2t01'] =('Local quantities/Auxiliary/Excitation/H2 T01 abundances/', 0, float)
        self.pardict['OPR'] = ('Local quantities/Auxiliary/Excitation/H2 ortho-para ratio/',0, float)
        self.pardict['h2fr'] = ('Local quantities/Auxiliary/Molecular fraction/', 0, float)
        self.pardict['uv_flux']=('Local quantities/Auxiliary/Radiation/', 0, float)
        self.pardict['uv_dens']=('Local quantities/Auxiliary/Radiation/', 1, float)

        #for n, ind in zip(['n_h', 'n_hp', 'n_h2', 'n_c', 'n_cp', 'n_co'], [0, 92, 2, 5, 100, 46]):
        #    self.pardict[n] = ('Local quantities/Densities/Densities', ind, float)
        # 'H', 'H+', 'H2',  'HD','CI', 'C+', 'CO','O','el'
        for el, ind in zip(['h', 'h2', 'hd', 'c', 'co', 'oh', 'o', 'hp', 'cp', 'electr'],
                           [0, 2, 3, 5, 46, 36, 34, 92, 100, 238]):
            self.pardict[f'n_{el}'] = ('Local quantities/Densities/Densities', ind, float)
        #'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7', 'H2j8', 'H2j9', 'H2j10',
        for i in range(11):
            self.pardict['pop_h2_v0_j'+str(i)] = ('Local quantities/Auxiliary/Excitation/Level densities', 9+i, float)
        # 'CIj0', 'CIj1', 'CIj2',
        for i in range(3):
            self.pardict[f'pop_c_el3p_j' + str(i)] = (
            'Local quantities/Auxiliary/Excitation/Level densities', 427 + i, float)
        #'HDj0', 'HDj1',
        for i in range(2):
            self.pardict[f'pop_hd_v0_j' + str(i)] = ('Local quantities/Auxiliary/Excitation/Level densities', 259 + i, float)
        #'COj0','COj1','COj2','COj3','COj4','COj5','COj6','COj7',
        for i in range(8):
            self.pardict[f'pop_co_v0_j' + str(i)] = ('Local quantities/Auxiliary/Excitation/Level densities', 268 + i, float)
        #'NH','NH2', 'NH+','NCO', 'NCI','NC+',
        for el, ind in zip(['h', 'h2', 'hd', 'c', 'co', 'oh','o','hp','cp','electr'], [0, 2, 3, 5, 46, 36, 34,92,100,238]):
            self.pardict[f'cd_prof_{el}'] = ('Local quantities/Densities/Column densities', ind, float)
        # total 'NH','NH2', 'NH+','NCO', 'NCI','NC+'
        for el, ind in zip(['h', 'h2', 'hd', 'c', 'co', 'oh'], [0, 2, 3, 5,46, 36]):
            self.pardict[f'cd_{el}'] = ('Integrated quantities/Column densities', ind, float)
        # 'NH2j0', 'NH2j1', 'NH2j2', 'NH2j3', 'NH2j4', 'NH2j5', 'NH2j6', 'NH2j7', 'NH2j8', 'NH2j9', 'NH2j10',
        for i in range(11):
            self.pardict['cd_lev_prof_h2_v0_j' + str(i)] = (
            'Local quantities/Auxiliary/Excitation/Level column densities', 9 + i, float)
        #'NCj0', 'NCj1', 'NCj2',
        for i in range(3):
            self.pardict['cd_lev_prof_c_el3p_j' + str(i)] = (
                'Local quantities/Auxiliary/Excitation/Level column densities', 427 + i, float)
        #'NCOj0','NCOj1','NCOj2','NCOj3','NCOj4','NCOj5','NCOj6','NCOj7',
        for i in range(8):
            self.pardict['cd_lev_prof_co_v0_j' + str(i)] = (
            'Local quantities/Auxiliary/Excitation/Level column densities', 268 + i, float)

        #'heat_tot', 'heat_phel','heat_secp', 'heat_phot','heat_chem', 'heat_h2','heat_cr'
        for el, ind in zip(['tot', 'pe', 'secp', 'ph', 'chem', 'h2', 'cr'], [0, 1,5,3,6,2,4]):
            self.pardict['heatrate_{el}'] = ('Local quantities/Auxiliary/Thermal balance', ind, float)
        #'cool_tot', 'cool_o', 'cool_cp', 'cool_c','cool_co', 'cool_elrec', 'cool_free', 'cool_h','cool_h2',
        for el, ind in zip(['tot', 'o', 'cp', 'c', 'co', 'elrecomb', 'freefree','h','h2'], [2,12,20,10,6,1,0,3,4]):
            self.pardict['coolrate_{el}'] = ('Local quantities/Auxiliary/Thermal balance', ind, float)
    def read(self, show_meta=False, show_summary=True, fast=None):
        """
        Read model data from hdf5 file

        :param
            -  show_meta           :  if True, show Metadata table
            -  show_summary        :  if True, show summary

        :return: None
        """
        self.file = h5py.File(self.folder + self.filename, 'r')

        if show_meta:
            self.showMetadata()
        if fast is not None:
            self.fastread = fast

        if 1:
            # >>> model input parameters
            self.Z = self.par('metal')
            #self.P = self.par('gas_pressure_input')
            self.n0 = self.par('proton_density_input')
            self.avmax = self.par('avmax')
            self.uv = self.par('radm_ini')
            self.cr = self.par('zeta')
            self.tcmb = self.par('t_cmb')
            print('zeta',self.cr,self.tcmb)



            # >>> profile of physical quantities
            self.x = self.par('distance')
            for el in ['h', 'h2', 'hd', 'co','c']:
                setattr(self, el, self.par(f'cd_prof_{el}'))
            self.NCO = self.par('cd_co')
            self.av = self.par('av')
            self.Av = self.par('avmax')
            self.tgas = self.par('tgas')
            self.pgas = self.par('pgas')
            self.n = self.par('ntot')
            self.nH = self.par('protdens')
            self.uv_flux = self.par('uv_flux')
            self.uv_dens = self.par('uv_dens')
            self.h2t01= self.par('h2t01')
            #self.h2t01_cd = self.par('h2t01_cd')
            self.OPR = self.par('OPR')
            #self.OPR_cd = self.par('OPR_cd')
            self.h2fr = self.par('h2fr')
           # self.t01 = self.par('t01')

            self.readspecies()


            self.file.close()
            self.file = None

            if show_summary:
                self.show_summary()

    def par(self, par=None,vertype=1):
        #print('par',par)
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
                attr, ind, typ = self.pardict['codeversion']
                codeversion = self.file[attr][:, ind]
                if codeversion[0] == 'PDR_1.5.4_rev_2083_210215':
                    vertype = 0
                elif codeversion[0] == 'PDR_1.5.4_rev_2053_200303':
                    vertype = 1
                if vertype:
                    ind = np.where(meta[:, 3] == par.encode())[0]
                    if len(ind) > 0:
                        attr = meta[ind, 0][0].decode() + '/' + meta[ind, 1][0].decode()
                        typ = {'string': str, 'real': float, 'integer': int}[meta[ind, 5][0].decode()]
                        ind = int(meta[ind, 2][0].decode())
                    else:
                        return None
                else:
                    ind = np.where(meta[:, 3] == par)[0]
                    if len(ind) > 0:
                        attr = meta[ind, 0][0] + '/' + meta[ind, 1][0]
                        typ = {'string': str, 'real': float, 'integer': int}[meta[ind, 5][0]]
                        ind = int(meta[ind, 2][0])
                    else:
                        return None
        x = self.file[attr][:, ind]
        if len(x) == 1:
            if isinstance(x[0], float):
                return x[0]
            else:
                return typ(x[0])
            #if vertype:
            #    return typ(x[0].decode())
            #else:
            #    return typ(x[0])
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

    def show_summary(self, pars=['Z', 'n', 'uv'], output=False):
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
                   legend=True, limit=None,band=None, pyfit=False, lev = 0, ylevels = False):
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
                self.plot_phys_cond(pars=p, logx=logx, ax=ax[m], legend=legend, parx=parx, limit=limit,band=band)

        if species is not None:
            for n, sp in enumerate(species):
                p = parx
                if 'CIj0/CI' in sp: p = 'ci'
                self.plot_profiles(species=sp, logx=logx, logy=logy, ax=ax[n+1+m], label=None, legend=legend, parx=p, limit=limit,normed=False,band=band)
                if 'CIj0/CI' in sp:
                    if pyfit:
                        self.plot_popratio(ax=ax[n+1+m], logx=logx, logy=logy, parx=p, sp = 'CI')
                if 'COj1/COj0' in sp:
                    if pyfit:
                        self.plot_popratio(ax=ax[n+1+m], logx=logx, logy=logy, parx=p, sp = 'CO',lev=lev)
                if ylevels:
                    ax[n + 1 + m].axhline(14)
                    ax[n + 1 + m].axhline(15)
                    ax[n + 1 + m].axhline(16)
                    ax[n + 1 + m].axhline(17)
                    ax[n + 1 + m].axhline(18)


#        def plot_popratio(self, species=None, ax=None, parx='av', logx=False, logy=False):

        if excs is not None:
            self.plot_exciation(ax=ax[n+1+m+k],logN=excs)

        fig.suptitle(self.name, fontsize=14)
        return fig

    def plot_phys_cond(self, pars=['tgas', 'n', 'av','h2t01','uv_dens'], logx=True, logy = True, ax=None, legend=False,
                       parx='x', limit=None, band=None, yscale='dec', ls='-', fontsize=12, ylabel='$\\log$\\,Physical cond.',xlabel=True):
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
        if xlabel:
            if parx == 'av':
                xlabel = 'Av'
            elif parx == 'x':
                xlabel = '$\log$(Distance), cm'
            elif parx == 'h2':
                xlabel = 'log(NH2), cm-2'
            elif parx == 'h':
                xlabel = 'log(NHI), cm-2'
            elif parx == 'ci':
                xlabel = 'log(NCI), cm-2'
            elif parx == 'co':
                xlabel = 'log(NCO), cm-2'
            elif parx == 'pgas':
                xlabel = 'pgas'

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
                v = self.sp[list(limit.keys())[0]]
                mask *= v < list(limit.values())[0]
            else:
                warnings.warn(
                    'limit key {:s} is not found in the model. Mask did not applied'.format(list(limit.keys())[0]))

        if logx:
            x = np.log10(getattr(self, parx)[mask])
        else:
            x = getattr(self, parx)[mask]


        lines = []
        thb_names = ['cool_tot', 'cool_o','cool_cp', 'cool_c', 'cool_co', 'cool_elrec', 'cool_free', 'cool_h', 'cool_h2',
          'heat_tot', 'heat_phel','heat_secp', 'heat_phot','heat_chem', 'heat_h2', 'heat_cr']
        label = ''
        for i, p in enumerate(pars):
            axi = ax
            ylabel=''
            if p == 'tgas':
                ylabel = '$\\log T_{\\rm kin}$, K'
                label = '$T_{\\rm gas}$'
            #    yscale = 'dec'
            #    bot,top = 10,1e3
            elif p == 'n' or p =='nH':
                ylabel = 'Density, cm'
                label = '$n_{\\rm gas}$'
            #    yscale = 'dec'
            elif p in ['OPR']:
                ylabel = 'OPR'
            #    yscale = 'dec'
            #    bot, top = 1, 15
            elif p in thb_names:
                ylabel = 'thb_rates'
            #    yscale = 'dec'
            elif p == 'heat_tot':
                ls = '--'
            elif p == 'pgas':
                label = '$P_{\\rm gas}$'
            if p in ['tgas', 'n', 'nH', 'av', 'h2t01','uv_dens','uv_flux','H2_photo_dest_prob','OPR','h2fr','pgas']:
                y = getattr(self, p)[mask]
            else:
                if p == 'tgas_m':
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
                    label = '$T_{\\rm 01}(NH_2)$'
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
                    label = '$T_{\\rm 01}(H_2)$'
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
                #    if p in ['heat_tot', 'heat_phel','heat_secp', 'heat_phot','heat_chem', 'heat_h2', 'heat_cr']:
                    y = self.sp[p][mask]/(self.sp['H'][mask]+self.sp['H2'][mask])**2.0
                #    else:
                #        y = self.sp[p][mask]/(self.sp['H'][mask]+self.sp['H2'][mask])**2.0
                #    if p in ['heat_tot','cool_tot']:
                #        bot, top = -0.5*np.max(y),2*np.max(y)
                elif p in ['H2_diss']:
                    y = np.log10(self.sp[p][mask])
                else:
                    if 'N_' in p:
                        y = np.log10(integrate.cumtrapz(self.sp[p.replace('N_', '')][mask], 10**x, initial=0))
                    else:
                        y = self.sp[p][mask]

            color = plt.cm.tab10(i / 10)
            if i==0 :
                #axi.set_ylabel(ylabel, color=color,fontsize=fontsize)
                axi.set_ylim(1,5)
            if logy:
                line, = axi.plot(x, np.log10(y), color=color, label=label, ls=ls)
            else:
                line, = axi.plot(x, y, color=color, label=label, ls=ls)
            lines.append(line)

            if 1:
                axi.tick_params(which='both', width=1, direction='in', right='True', top='True')
                axi.tick_params(which='major', length=5, direction='in', right='True', top='True')
                axi.tick_params(which='minor', length=4, direction='in')
                axi.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')

            if yscale == 'log':
                ax.set_yscale('log')
                axi.set_yscale('log')
        if ylabel is not None:
            axi.set_ylabel(ylabel, fontsize=fontsize)

        ax.set_xlim([x[0], x[-1]])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)

        if legend:
            ax.legend(handles=lines, loc='best',fontsize=fontsize)

        if band is not None:
            bot, top = ax.get_ylim()
            y1 = np.arange(bot-1, top+1, 0.5)

            for k in band.keys():
                if k == 'NH2': color = 'orange'
                if k == 'NCO': color = 'red'
                if k == 'NCI': color = 'green'
                if k in self.sp.keys():
                    mask1 = list(mask)
                    #v = getattr(self, list(limit.keys())[0])
                    v = band[k]
                    logN = self.sp[k]
                    mask1 *= logN < v
                    v1 = np.log10(getattr(self, parx)[mask1][-1])
                    mask1 *= logN < v/10
                    v2 = np.log10(getattr(self, parx)[mask1][-1])
                    print('band x-axis:', v1,v2)
                    ax.fill_betweenx(y1,v1,v2, color=color, alpha=0.2)

        return ax

    def plot_profiles(self, species=None, logx=False, logy=False, label=None, ylabel=True, ax=None, legend=True, ls='-', lw=1, parx='av',
                      limit=None, band=None, normed=False ,fontsize=12, tuneaxes=True, colors = None):
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
            xlabel = '$\\log({\\rm Distance})$, cm'
        elif parx == 'h2':
            xlabel = '$\\log(N_{\\rm H_2}),{\\rm\\,cm}^{-2}$'
        elif parx == 'h':
            xlabel = 'log(NHI), cm-2'
        elif parx == 'ci':
            xlabel = 'log(NCI), cm-2'
        elif parx == 'co':
            xlabel = '$\\log(N_{\\rm CO}),{\\rm\\,cm}^{-2}$'
        elif parx == 'pgas':
            xlabel = 'pgas'
        elif parx == 'tgas':
            xlabel = 'tgas'


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
                v = self.sp[list(limit.keys())[0]]
                mask *= v < list(limit.values())[0]
            else:
                warnings.warn(
                    'limit key {:s} is not found in the model. Mask did not applied'.format(list(limit.keys())[0]))

        if logx:
            x = np.log10(getattr(self, parx)[mask])
        else:
            x = getattr(self, parx)[mask]



        lab = label
        for k_sp,s in enumerate(species):
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
                logy = True
                ax.set_ylim(1,3)
                z = []
                for e in y:
                    z1 = root(poly, x0=0.2, args=(e))
                    print('root',z1.x, poly(z1.x,e))
                    z.append(z1.x)
                y = -170.5/np.log(z)
            else:
                y = self.sp[s][mask]
            if label is None:
                lab = s
                if s == 'H2':
                    lab = 'H$_2$'
            else:
                if k_sp>0:
                    lab ='_nolegend_'
            if normed:
                y = y/self.n0
            if logy:
                y =  np.log10(y)
            if colors is None:
                ax.plot(x, y, ls=ls, label=lab, lw=lw)
            else:
                ax.plot(x, y, ls=ls, label=lab, lw=lw,color=colors[k_sp])



        if 1:
            ax.set_xlim([x[0], x[-1]])
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)

        if legend:
            ax.legend(fontsize=fontsize)

        if band is not None:
            bot, top = ax.get_ylim()
            y1 = np.arange(bot-1, top+1, 0.5)

            for k in band.keys():
                if k == 'NH2': color = 'orange'
                if k == 'NCO': color = 'red'
                if k == 'NCI': color = 'green'
                if k in self.sp.keys():
                    mask1 = list(mask)
                    #v = getattr(self, list(limit.keys())[0])
                    v = band[k]
                    logN = self.sp[k]
                    mask1 *= logN < v
                    v1 = np.log10(getattr(self, parx)[mask1][-1])
                    mask1 *= logN < v/10
                    v2 = np.log10(getattr(self, parx)[mask1][-1])
                    ax.fill_betweenx(y1,v1,v2, color=color, alpha=0.2)

        if tuneaxes:
            ax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True', top='True')
            ax.tick_params(which='major', length=5)
            ax.tick_params(which='minor', length=3)
        return ax

    def plot_popratio(self,  ax=None,parx='av',logx=False, logy=False, ls = '--', colors=None,sp='CI',lev=-1):

        #if species is None:
        #    species = self.species
        if parx == 'av':
            xlabel = 'Av'
        elif parx == 'x':
            xlabel = 'log(Distance), cm'
        elif parx == 'h2':
            xlabel = 'log(NH2), cm-2'
        elif parx == 'h':
            xlabel = 'log(NHI), cm-2'
        elif parx == 'ci':
            xlabel = 'log(NCI), cm-2'
        elif parx == 'co':
            xlabel = 'log(NCI), cm-2'

        if ax is None:
            fig, ax = plt.subplots(figsize=(4.5, 6))

        if logx:
            mask = getattr(self, parx) > 0
        else:
            mask = getattr(self, parx) > -1
        mask*=np.log10(self.h2) < np.log10(self.h2[-1]/2)
        if logx:
            x = np.log10(getattr(self, parx)[mask])
        else:
            x = getattr(self, parx)[mask]

        tkin = self.tgas[mask]
        ntot = self.nH[mask]
        fr = self.h2fr[mask]
        uv = self.uv*(self.uv_flux/self.uv_flux[0])[mask]
        if 0:
            tau, beta_l, beta_r = [], [], []
            file_path = pathH2exc+'/data/beta_co10.dat'
            with open(file_path, 'r') as f:
                for k, line in enumerate(f):
                    if line[0] is not '#':
                        values = [s for s in line.split()]
                        if not (values[0][0] == '#'):
                            tau.append(float(values[0]))  # wavelenght
                            beta_l.append(float(values[3]))
                            beta_r.append(float(values[4]))

            tau = np.array(tau)
            beta_t = np.array((beta_l)) + np.array((beta_r))
            av = -2.5 * np.log10(np.exp(-tau))
            beta_coj0 = np.interp(self.av, av,beta_t)[mask]

            tau, beta_l, beta_r = [], [], []
            file_path = pathH2exc + '/data/beta_co21.dat'
            with open(file_path, 'r') as f:
                for k, line in enumerate(f):
                    if line[0] is not '#':
                        values = [s for s in line.split()]
                        if not (values[0][0] == '#'):
                            tau.append(float(values[0]))  # wavelenght
                            beta_l.append(float(values[3]))
                            beta_r.append(float(values[4]))

            tau = np.array(tau)
            beta_t = np.array((beta_l)) + np.array((beta_r))
            av = -2.5 * np.log10(np.exp(-tau))
            beta_coj1 = np.interp(self.av, av, beta_t)[mask]

            tau, beta_l, beta_r = [], [], []
            file_path = pathH2exc + '/data/beta_co32.dat'
            with open(file_path, 'r') as f:
                for k, line in enumerate(f):
                    if line[0] is not '#':
                        values = [s for s in line.split()]
                        if not (values[0][0] == '#'):
                            tau.append(float(values[0]))  # wavelenght
                            beta_l.append(float(values[3]))
                            beta_r.append(float(values[4]))

            tau = np.array(tau)
            beta_t = np.array((beta_l)) + np.array((beta_r))
            av = -2.5 * np.log10(np.exp(-tau))
            beta_coj2 = np.interp(self.av, av, beta_t)[mask]

            tau, beta_l, beta_r = [], [], []
            file_path = pathH2exc + '/data/beta_co43.dat'
            with open(file_path, 'r') as f:
                for k, line in enumerate(f):
                    if line[0] is not '#':
                        values = [s for s in line.split()]
                        if not (values[0][0] == '#'):
                            tau.append(float(values[0]))  # wavelenght
                            beta_l.append(float(values[3]))
                            beta_r.append(float(values[4]))

            tau = np.array(tau)
            beta_t = np.array((beta_l)) + np.array((beta_r))
            av = -2.5 * np.log10(np.exp(-tau))
            beta_coj3 = np.interp(self.av, av, beta_t)[mask]

        jsp = [1, 2,3,4]
        zabs = self.tcmb/2.725 - 1


        if sp == 'CI':
            pr = pyratio(z=0)
            numlevels = 3
            pr.add_spec('CI', num=numlevels)
            pr.set_pars(['rad', 'n', 'T', 'f','UV'])
            spec = {}
            for k in range(numlevels):
                spec['CI' + str(k)] = []
            x0 = []
            for i in range(0, len(tkin), 20):
                pr.pars['n'].value = np.log10(ntot[i])
                pr.pars['T'].value = np.log10(tkin[i])
                pr.pars['f'].value = np.log10(fr[i])
                pr.pars['rad'].value = np.log10(uv[i])
                results = pr.predict(level=-1)
                #print(i, [pr.pars[el].value for el in ['rad', 'n', 'T', 'f']])
                #print(i, results)
                x0.append(x[i])
                for k in range(numlevels):
                    spec['CI' + str(k)].append(results[k])

            colors = ['blue', 'orange', 'green']
            for k in range(numlevels):
                ax.plot(np.array(x0), np.array(spec['CI' + str(k)]), ls=ls, color=colors[k])
            ax.set_xlim(10, 18)
            ax.set_ylim(-2.5, 0.1)

            if 1:
                pr = pyratio(z=0)
                numlevels = 3
                pr.add_spec('CI', num=numlevels)
                pr.set_pars(['rad', 'n', 'T', 'f', 'UV','b_trap'])
                spec = {}
                for k in range(numlevels):
                    spec['CI' + str(k)] = []
                x0 = []
                for i in range(0, len(tkin), 20):
                    pr.pars['n'].value = np.log10(ntot[i])
                    pr.pars['T'].value = np.log10(tkin[i])
                    pr.pars['f'].value = np.log10(fr[i])
                    pr.pars['rad'].value = np.log10(uv[i])
                    pr.pars['b_trap'].value = -0.2*logN[i]/logN[-1]
                    results = pr.predict(level=-1)
                    #print(i, [pr.pars[el].value for el in ['rad', 'n', 'T', 'f']])
                    #print(i, results)
                    x0.append(x[i])
                    for k in range(numlevels):
                        spec['CI' + str(k)].append(results[k])


                for k in range(numlevels):
                    ax.plot(np.array(x0), np.array(spec['CI' + str(k)]), ls='-.', color=colors[k])
                ax.set_xlim(10,18)
                ax.set_ylim(-2.5, 0.1)

        if sp == 'CO':
            # calculation with photon trapping
            if 0:
                pr = pyratio(z=zabs)
                numlevels = 10
                pr.add_spec('CO', num=numlevels)
                pr.set_pars(['rad', 'nH', 'T', 'f','trap'])
                legend = 'CMB + Coll + trapping'
                spec = {}
                for k in range(numlevels):
                    spec['COj' + str(k)] = []
                x0 = []
                for i in range(0, len(tkin), 30):
                    pr.pars['nH'].value = np.log10(ntot[i])
                    pr.pars['T'].value = np.log10(tkin[i])
                    pr.pars['f'].value = np.log10(fr[i])
                    pr.pars['rad'].value = np.log10(uv[i])
                    pr.pars['trap'].value = [beta_coj0[i],beta_coj1[i],beta_coj2[i],beta_coj3[i]]
                    results = pr.predict(level=lev)
                    for k in range(numlevels):
                        spec['COj' + str(k)].append(results[k])
                    x0.append(x[i])

                '''
                #    for j in jsp:
                        pr.pars['n'].value = np.log10(ntot[i])
                        pr.pars['T'].value = np.log10(tkin[i])
                        pr.pars['f'].value = np.log10(fr[i])
                        pr.pars['rad'].value = np.log10(uv[i])
                        pr.pars['b_trap'].value = tau0[i] #0.5*1/(1+0.5*tau0[j][i])
                        results[j] = pr.predict(level=0)
                        spec['COj' + j].append(results[j][int(j)])
                        if 0:
                            pr.balance(debug='C')
                '''



                for k in range(1,len(jsp)+1):
                    if colors is None:
                        ax.plot(np.array(x0), np.array(spec['COj' + str(k)]), ls='--', lw=1)
                    else:
                        #print(k)
                        ax.plot(np.array(x0), np.array(spec['COj' + str(k)]), ls='--', lw=1, color=colors[k - 1],
                                label=legend)
                        legend = "_nolegend_"
                #pr.pars['b_trap'].value = 1.0
                #MC = pr.balance(debug='C')
                #MR = pr.balance(debug='rad')
                #MI = pr.balance(debug='IR') + pr.balance(debug='A')
            if 1:
                pr = pyratio(z=zabs)
                numlevels = 10
                pr.add_spec('CO', num=numlevels)
                pr.set_pars(['rad', 'nH', 'T', 'f'])
                legend = 'CMB+Coll'
                spec = {}
                for k in range(numlevels):
                    spec['COj' + str(k)] = []
                x0 = []
                for i in range(0, len(tkin)):
                    pr.pars['nH'].value = np.log10(ntot[i])
                    pr.pars['T'].value = np.log10(tkin[i])
                    pr.pars['f'].value = np.log10(fr[i])
                    pr.pars['rad'].value = np.log10(uv[i])
                    results = pr.predict(level=lev)
                    x0.append(x[i])
                    for k in range(numlevels):
                        spec['COj' + str(k)].append(results[k])
                for k in jsp:
                    if colors is None:
                        ax.plot(np.array(x0), np.array(spec['COj' + str(k)]), ls='dotted', lw=2,label=legend)
                    else:
                        ax.plot(np.array(x0), np.array(spec['COj' + str(k)]), ls='dotted', lw=1, color=colors[k - 1],label=legend)
                        legend = "_nolegend_"

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

        if logN == np.inf:
            for s in species:
                cols[s] = np.log10(integrate.cumtrapz(self.sp[s], x=self.x))
                #cols[s] = np.log10(np.trapz(self.sp[s], x=self.x))
        else:
            if logN is not None:
                if sides != 0:
                    self.set_mask(species=list(logN.keys())[0], logN=logN[list(logN.keys())[0]] - np.log10(sides),
                                  sides=sides)
                else:
                    self.set_mask(species=list(logN.keys())[0], logN=logN[list(logN.keys())[0]], sides =sides)

#            if logN is not None:
#                if logN[list(logN.keys())[0]] is not None:
#                    #logN[list(logN.keys())[0]] -= np.log10(sides)
#                    self.set_mask(species=list(logN.keys())[0], logN=logN[list(logN.keys())[0]] - np.log10(sides), sides=sides)
#                else:
#                    self.set_mask(species=list(logN.keys())[0], logN=logN[list(logN.keys())[0]])

#        if logN is not None:
#            logN[list(logN.keys())[0]] -= np.log10(sides)
#            self.set_mask(logN=logN, sides=sides)

            for s in species:
                cols[s] = np.log10(np.trapz(self.sp[s][self.mask], x=self.x[self.mask])) + np.log10((sides > 1) + 1)


        self.cols = cols

        return self.cols

    def set_mask(self, species='H', logN=None, sides=2):
        """
        Calculate mask for a given threshold

        :param:
            -  logN          :  column density threshold

        :return: None
        """
        cols = np.log10(self.sp['N' + species])

        if logN is not None and sides != 0:
            l = int(len(self.x) / sides) + 1 if sides > 1 else len(self.x)
            if logN > cols[l - 1]:
                logN = cols[l - 1]
            self.mask = cols < logN
        else:
            self.mask = cols > -1

    def lnLike(self, species={}, syst=0, verbose=False, relative=None):
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

            if syst > 0:
                v1 *= a(0, syst, syst, 'l')
            if verbose:
                print(np.log10(self.uv), np.log10(self.n0), self.cols[k], v1.log(), v1.lnL(self.cols[k]))
            if v.type in ['m', 'u', 'l']:
                lnL += v1.lnL(s)

        self.lnL = lnL
        return self.lnL

    def calc_mean_pars(self, pars=['tgas'], logH2=None,logCI=None, sides=2,logscale=True):
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
        print('n0,uv:', self.n0, self.uv)
        if pars is not None:
            if logH2 is not None:
                logH2[list(logH2.keys())[0]] -= np.log10(sides)
                self.set_mask(logN=logH2, sides=sides)
                maskH2 = self.mask
                logH2[list(logH2.keys())[0]] -= 1
                self.set_mask(logN=logH2, sides=sides)
                maskH2 *= np.invert(self.mask)
                #maskH2=self.mask*(self.h2fr>1e-2)
            if logCI is not None:
                logCI[list(logCI.keys())[0]] -= np.log10(sides)
                self.set_mask(logN=logCI, sides=sides)
                maskCI = self.mask
                logCI[list(logCI.keys())[0]] -= 1
                self.set_mask(logN=logCI, sides=sides)
                maskCI *=np.invert(self.mask)
            elif logH2 is not None:
                logH2[list(logH2.keys())[0]] -= np.log10(sides)
                self.set_mask(logN=logH2, sides=sides)
                maskH2 = self.mask
                logH2[list(logH2.keys())[0]] -= 1
                self.set_mask(logN=logH2, sides=sides)
                maskH2 *= np.invert(self.mask)
            else:
                maskH2 = self.mask
                self.set_mask(logN={'H2':20}, sides=sides)
                maskH2 *= np.invert(self.mask)

            for p in pars:
                if p in ['T01']:
                    mpars[p] = 0
                if p == 'pgas':
                    x0 = self.x[maskCI]
                    p0 = getattr(self, p)[maskCI]
                    mpars[p] = np.trapz(p0, x=x0)/(x0[-1]-x0[0])
                    print('grid pgas_ci:', np.log10([mpars[p],x0[-1], x0[0]]))
                elif p == 'pgas_h2':
                    x0 = self.x[maskH2]
                    p0 = getattr(self, 'pgas')[maskH2]
                    mpars[p] = np.trapz(p0, x=x0) / (x0[-1] - x0[0])
                    print('grid pgas_h2:',np.log10([mpars[p],x0[-1], x0[0]]))
                    #print('n0,uv:',self.n0,self.uv)
                    #print('pgas',mpars[p],'xrange',np.log10(x0[0]),np.log10(x0[-1]),np.log10(p0[0]),np.log10(p0[-1]) )
                else:
                    x0 = self.x[maskH2]
                    p0 = getattr(self, p)[maskH2]
                    mpars[p] = np.trapz(p0, x=x0)/(x0[-1]-x0[0])
            #else:
            #    for p in pars:
            #        if p in ['av']:
            #            mpars[p] = 0
            #        else:
            #            mpars[p] = np.array(integrate.cumtrapz(getattr(self, p), x=self.x)) / self.x[-1]

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
        self.species = ['H', 'H+', 'H2',  'HD','CI', 'C+', 'CO','O','el',
                        'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7', 'H2j8', 'H2j9', 'H2j10',
                        'COj0','COj1','COj2','COj3','COj4','COj5','COj6','COj7',
                        'HDj0', 'HDj1',
                        'CIj0', 'CIj1', 'CIj2',
                        'NH','NH2', 'NH+','NCO', 'NCI','NC+',
                        'NH2j0', 'NH2j1', 'NH2j2', 'NH2j3', 'NH2j4', 'NH2j5', 'NH2j6', 'NH2j7', 'NH2j8', 'NH2j9', 'NH2j10',
                        'NCj0', 'NCj1', 'NCj2',
                        'NCOj0','NCOj1','NCOj2','NCOj3','NCOj4','NCOj5','NCOj6','NCOj7',
                        'cool_tot', 'cool_o', 'cool_cp', 'cool_c','cool_co', 'cool_elrec', 'cool_free', 'cool_h','cool_h2',
                        'heat_tot', 'heat_phel','heat_secp', 'heat_phot','heat_chem', 'heat_h2','heat_cr']
        self.readH2database(H2database)

    def readH2database(self, data='all'):
        import sys
        sys.path.append('/home/slava/science/codes/python/H2_excitation')
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
        if data == 'CO':
            self.H2.append(H2_summary.load_co_sample())

    def readmodel(self, filename=None, show_summary=False, folder=None, printname=True,show_meta=False):
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
            m = model(folder=folder, filename=filename, species=self.species, show_summary=show_summary,show_meta=show_meta)
            self.models[m.name] = m
            self.current = m.name

            #if printname:
            #    print(m.name, 'n0=', m.n0, 'uv=', m.uv, 'me=',m.Z, 'zeta=', m.cr, 'av=', m.av[-1],'tcmb=',m.tcmb)

            if show_summary:
                m.showSummary()

    def readfolder(self, verbose=False):
        """
        Read list of models from the folder
        """
        if 1:
            for (dirpath, dirname, filenames) in os.walk(self.folder):
                print(dirpath, dirname, filenames)
                for k,f in enumerate(filenames):
                    if f.endswith('.hdf5'):
                        print(k,'from', len(filenames))
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

        for name, model in self.models.items():
            for k, v in fixed.items():
                if getattr(model, k) != v and v != 'all':
                    break
            else:
                for p in pars:
                    self.grid[p].append(getattr(model, p))
                self.mask.append(name)

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

    #def compare(self, object='', species='H2', spmode = 'abs', mpars = ['tgas','pgas'], models='current', syst=0.0,  syst_factor=1,levels=[], others='ignore', sides=2):
    def compare(self, object='', species='',  mpars = ['tgas','pgas'], models='current', syst=0.0, syst_factor=1, levels=[], others='ignore',  relative=False, sides=2):
        """
        Calculate the column densities of H2 rotational levels for the list of models given the total H2 column density.
        and also log of likelihood

        :param:
            -  object            :  object name
            - species            : which species to use. Can be 'H2' or 'CI'??
            -  models            :  names of the models, can be list or string
            -  syst              :  add systematic uncertainty to the calculation of the likelihood
            -  levels            :  levels that used to constraint, if empty list used all avaliable

        :return: None
            column densities are stored in the dictionary <cols> attribute for each model
            log of likelihood value is stored in <lnL> attribute
        """
        # syst factor (?)

        q = self.comp(object)
        if species in ['H2', 'CI', 'CO']:
            if len(levels) > 0:
                label = species + 'j'
                full_keys = [s for s in q.e.keys() if (label in s) and ('v' not in s)]
                label = species + 'j{:}'
                keys = [label.format(i) for i in levels if label.format(i) in full_keys]
                spec = OrderedDict([(s, a(q.e[s].col.log().val, q.e[s].col.log().plus * syst_factor, q.e[s].col.log().minus * syst_factor, 'l') * a(0.0, syst, syst)) for s in keys])
                if others in ['lower', 'upper']:
                    for k in full_keys:
                        if k not in keys:
                            v = a(q.e[k].col.val, q.e[k].col.plus * syst_factor, q.e[k].col.minus * syst_factor)
                            if syst > 0:
                                v = v * a(0.0, syst, syst)
                            if others == 'lower':
                                spec[k] = a(v.val - v.minus, t=others[0])
                            else:
                                spec[k] = a(v.val + v.plus, t=others[0])
        else:
            keys, full_keys, logN = [], [], None

        #spec = OrderedDict([(s, q.e[s].col * a(0.0, syst, syst)) for s in full_keys])

        #for model in self.listofmodels(models):
        #    model.calc_cols(spec.keys(), logN=logN, sides=sides)
        #    relative = 'CIj0' if species == 'CI' else None
        #    model.lnLike(spec, relative=relative)

        for model in self.listofmodels(models):
            if relative:
                relative = species+'j0'
                logN = {'H2': q.e['H2'].col.val}
            else:
                relative =None
                logN = {species: q.e[species].col.val}

            #if spmode == 'abs':
            #    logN = {species: q.e[species].col.val}

            #if spmode == 'abs':
            #    relative = None
            #    logN = {species: q.e[species].col.val}
            #elif spmode == 'rel':
            #    relative = species+'j0'
            #    logN = {'H2': q.e['H2'].col.val}
            #if spmode == 'test':
            #    relative = None
            #    #logN = np.inf
            #    logN = {species: q.e[species].col.val}

            print(logN, sides, relative)

            model.calc_cols(spec.keys(), logN=logN,sides=sides)
            model.lnLike(spec, relative=relative)
            if mpars is not None:
                if 'H2' in q.e.keys():
                    logH2 = {'H2':q.e['H2'].col.val}
                else:
                    logH2 = None
                if 'CI' in q.e.keys():
                    logCI = {'CI':q.e['CI'].col.val}
                else:
                    logCI = None
                print(model.name,'logH2,logCI', logH2,logCI)
                model.calc_mean_pars(pars=mpars, logH2=logH2,logCI=logCI)

    #def comparegrid(self, object='', species='H2', spmode = 'abs', pars=[], fixed={}, syst=0.0, plot=True, show_best=True, levels='all', others='ignore'):
    #def comparegrid(self, object='', species='H2', spmode = 'abs', pars=[], fixed={}, syst=0.0, syst_factor=1.0, plot=True, show_best=True, levels='all', others='ignore', sides=2):
    def comparegrid(self,  object='', species='H2', pars=[], fixed={}, syst=0.0, syst_factor=1.0, plot=True, show_best=True, levels='all', others='ignore', relative=False, sides=2):
        print('comparegrid', syst, syst_factor, pars, fixed, 'sides', sides)
        self.setgrid(pars=pars, fixed=fixed, show=False)
        for el in ['H2', 'CI', 'CO']:
            if el in self.comp(object).e.keys():
                self.grid['N'+el+'tot'] = self.comp(object).e[el].col.val
            else:
                self.grid['N' + el + 'tot'] = None

        #self.compare(object, species=species, spmode = spmode, models=self.mask, syst=syst, syst_factor=syst_factor, levels=levels, others=others, mpars=None,sides=sides)
        self.compare(object, species=species, models=self.mask, syst=syst, syst_factor=syst_factor, levels=levels, others=others, mpars=None,
                     relative=relative, sides=sides)

        self.grid['lnL'] = np.asarray([self.models[m].lnL for m in self.mask])
        self.grid['cols'] = np.asarray([self.models[m].cols for m in self.mask])
        self.grid['dims'] = len(pars)
        #self.grid['mpars'] = np.asarray([self.models[m].mpars for m in self.mask])
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

    def plot_objects(self, objects=[], species='H2', ax=None, plotstyle='scatter', legend=False,syst=None,label=None,msize=3,color='black', tuneaxes=True,fontsize=12):
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
            if species is None or len(species) == 0 or species == 'H2':
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
                if species == 'CI':
                    y = [(q.e[sort + str(i)].col/q.e[sort + '0'].col)*a(0.0, syst, syst)  / stat[i] for i in j]
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
                        #print('err_H2_',k,column(y, 2)[k], column(y, 1)[k])
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
        if tuneaxes:
            ax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True', top='True')
            ax.tick_params(which='major', length=5)
            ax.tick_params(which='minor', length=2)
            if species == 'H2':
                species = 'H_2'
            ax.set_ylabel('$\log~N_{\\rm{J}}({\\rm{' + species + '}})/g_{\\rm{J}}$', fontsize=fontsize)
            if species == 'CI':
                ax.set_ylabel('$\log~N_{\\rm{J}}({\\rm{' + species + '}})/g_{\\rm{J}}-\log~N_{\\rm{0}}/g_{\\rm{0}}$', fontsize=fontsize)
            ax.set_xlabel('Energy of ${\\rm{'+species+'}}$ levels, cm$^{-1}$', fontsize=fontsize)
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

    def plot_one_parameter_vary(self, parx='x', pary='NH2', ax=None, pars=None, fixed={}):
        if ax is None:
            fig, ax = plt.subplots()

        species = [pary]

        models = self.setgrid(pars=[pars], fixed=fixed)
        for m in self.listofmodels(models):
            print(m.name)
            m.plot_profiles(species=species, logx=True, logy=True, label=m.name, ax=ax, legend=False, parx=parx,
                            limit=None)

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
        data[0, int(meta[ind, 2][0].decode())] = d[0:3] + b'60' + d[5:]
        #data[0, int(meta[ind, 2][0].decode())] = d[0:3] + b'160' + d[6:]
        self.file.close()

    def best(self, object='', models='all', syst=0.0):
        models = self.listofmodels(models)
        self.compare(object=object, models=[m.name for m in models], syst=syst)
        return models[np.argmax([m.lnL for m in models])].name

    def calc_T01_grid(self,species='H2', pars=[], fixed={},plot=True,case=1):
        self.setgrid(pars=pars, fixed=fixed, show=False)
        self.grid['NH2th'] = np.asarray([self.models[m].calc_T01_limit(limith2={'H2': 21.5},case=case) for m in self.mask])
        if plot:
            if len(pars) == 2:
                fig, ax = plt.subplots()
                for v1, v2, l in zip(self.grid[pars[0]],self.grid[pars[1]], self.grid['NH2th']):
                    v1,v2 = np.log10(v1),np.log10(v2)
                    ax.scatter(v1, v2, 0)
                    ax.text(v1, v2, '{:.1f}'.format(l), size=20)

    def calc_logN_grid(self,species='H2', pars=[], fixed={},plot=False):
        self.setgrid(pars=pars, fixed=fixed, show=False)
        self.grid['N'+species] = np.asarray([np.log10(self.models[m].sp['N'+species][-1]) for m in self.mask])
        if plot:
            if len(pars) == 2:
                fig, ax = plt.subplots(2,1)
                print(pars[0])
                for v1, v2,l in zip(self.grid[pars[0]],self.grid[pars[1]], self.grid['N'+species]):
                    v1,v2 = np.log10(v1),v2
                    ax[0].scatter(v1, v2, 0)
                    ax[0].text(v1, v2, '{:.1f}'.format(l), size=12)
                    ax[1].scatter(v1,l,s=10,color='red')
                ax[0].set_xlabel('nH')
                ax[0].set_ylabel('Av max')
                ax[1].set_xlabel('nH')
                ax[1].set_ylabel('log N(CO) max')

                #plt.show()

    def calc_pars_grid(self,pars = [], models='current', logN = {'H2': 20}):
        self.setgrid(pars=pars, show=False)
        for m in self.mask:
            self.models[m].calc_mean_pars(pars=None, logN=logN)
            #self.models[m].calc_mean_pars(pars=['tgas','pgas'], logN=logN)
        self.grid['mpars'] = np.asarray([self.models[m].mpars for m in self.mask])



if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])

    labelsize=12

    if 0:
        fig, ax = plt.subplots(1,2)
        fig2, ax2 = plt.subplots(1,2)
        H2 = H2_exc(folder='./data/sample/1_5_4/tmp3/')
        H2.readfolder()
        colors = ['red','blue','green']
        for k,m in enumerate(H2.listofmodels()):
            if 1:
                m.plot_phys_cond(pars=['tgas', 'pgas'], logx=True, logy = True, ax=ax[0], legend=True, parx='co')
            if 1:
                m.plot_profiles(ax=ax[1],species=['H','H2','CI','CO'], logx=True, logy=True,parx='co')
                m.plot_profiles(ax=ax2[0], species=['NH2', 'NCI','NH','NCO'], logx=True, logy=True, parx='av')
                #m.plot_profiles(ax=ax2[1], species=['NH2j0/NH2', 'NH2j1/NH2', 'NH2j2/NH2'], logx=True, logy=True, parx='tgas',limit={'H2':21})

    if 1:
        if 1:
            H2 = H2_exc(folder='data/sample/1_5_4/ver2083/grid-n-av/')
            H2.readmodel(filename='pdr_grid_av0_1_n1e2_uv03_s_20.hdf5',show_meta=True)
            for m in H2.listofmodels():
                m.plot_phys_cond(legend=True)


        if 0:
            H2 = H2_exc(folder='data/sample/1_5_4/ver2083/test/')
            H2.readfolder()
            pars = {'n0': 'x', 'uv': 'y', 'avmax': 'z'}
            H2.setgrid(pars=list(pars.keys()), show=False)
            grid = H2.grid
            x, y, z = np.log10(grid[list(pars.keys())[list(pars.values()).index('x')]]), np.log10(
                grid[list(pars.keys())[list(pars.values()).index('y')]]), np.log10(
                grid[list(pars.keys())[list(pars.values()).index('z')]])
            names = list(pars.keys())
            if 0:
                t = np.linspace(0, 20, 100)
                x, y, z = np.cos(t), np.sin(t), t
            import plotly.graph_objects as go

            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                               name = 'a',
                                               mode='markers',
                                               marker=dict(
                                                   size=12,
                                                   color=z,  # set color to an array/list of desired values
                                                   colorscale='Viridis',  # choose a colorscale
                                                   opacity=0.8
                                               )
                                               )])
            fig.update_layout(scene=dict(
                xaxis_title=names[0],
                yaxis_title=names[1],
                zaxis_title=names[2]),
                #width=700,
                margin=dict(r=20, b=10, l=10, t=10)
            )

            fig.show()

        if 0:
            H2.calc_logN_grid(species='CO', pars=list(pars.keys()), fixed={}, plot=True)
        if 0:
            H2.calc_T01_grid(pars=pars)
        if 0:
            H2.logN = Rbf(x, y, H2.grid['NCO'], function='multiquadric', smooth=0.1)

            delta = 0.2
            num = 100
            xmin, xmax, ymin, ymax = np.min(x) - delta, np.max(x) + delta, np.min(y) - delta, np.max(y) + delta
            x1, y1 = x, y
            z1 = np.zeros_like(x1)
            for i, xi in enumerate(x):
                z1[i] = H2.logN(xi,y[i])
            x, y = np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num)
            X, Y = np.meshgrid(x, y)
            z = np.zeros_like(X)
            for i, xi in enumerate(x):
                for k, yi in enumerate(y):
                    z[k, i] = H2.logN(xi, yi)
            if 1:
                fig, ax = plt.subplots()
                c = ax.pcolor(X, Y, z, cmap='Purples') #, vmin=1.0, vmax=4.0)
                cax = fig.add_axes([0.93, 0.27, 0.01, 0.47])
                fig.colorbar(c, cax=cax, orientation='vertical') #, ticks=[-3, -2.5, -2, -1.5])
                for v1, v2, l in zip(x1, y1, z1):
                    ax.text(v1, v2, '{:.1f}'.format(l), size=10, color='black')
                ax.set_xlabel('log nH')
                ax.set_ylabel('log Iuv')

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