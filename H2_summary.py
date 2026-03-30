 #!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import sys
sys.path.append('C:/science/python')
from spectro.a_unc import a
from spectro.atomic import e
from spectro.atomic import metallicity as me
from spectro.excitation_temp import ExcitationTemp

class sy:
    def __init__(self, z_abs, err, errm=None):
        self.z = z_abs
        if errm is None:
            self.z_err = err
        else:
            self.z_err = (err, errm)

        self.e = {}
        self.physcond = {}
        self.exc_h2_j02_path = ''
        self.exc_h2_j05_path = ''
        self.exc_h2_j02_path_to_logN = ''
        self.exc_h2_j05_path_to_logN = ''
        self.exc_CI_path = ''
        self.e['size'] = e('size',0,0,0)



    def el(self, *args, **kwargs):
        el = e(*args, **kwargs)
        self.e[el.name] = el
        setattr(self, el.name, el)

    def set_physcond(self, *args, **kwargs):
        el = e(*args, **kwargs)
        self.physcond[el.name] = el
        setattr(self, el.name, el)


    def __eq__(self,other):
        return self.z_abs == other.z_abs

    def __repr__(self):
        return "{0.z!r}$\pm${0.z_err!r}".format(self)
    
    def CI(self):
        for i in range(len(self.el)):
            if self.el[i].name+'_'+self.el[i].ion == 'C_I':
                return self.el[i].col

    def H2(self,ind=-1):
        for i in range(len(self.el)):
            if self.el[i].name == 'H2' and self.el[i].rot == ind:
                return self.el[i].col



class qso:
    def __init__(self, name, z_em, z_dla):
        self.name = name
        self.z_em = z_em
        self.z_dla = z_dla
        self.coord = []
        self.SIMBAD = ''        # SIMBAD Identifier
        self.m = {}
        self.progID = []        # ID of the observational programs
        self.e = {}
        if 0:
            species = ('HI', 'H2', 'f', 'Av', 'HD', 'CO', 'CI', 'CII', 'MgI',
                   'MgII', 'SI', 'SII', 'SiII','SiIII', 'ClI', 'PII', 'ZnII', 'NiII',
                   'CrII', 'TiII', 'FeII', 'NI', 'NII', 'OI', 'NaI', 'CaII',
                   'ArI', 'NV')
        if 1:
            species = ()
        for s in species:
            self.el(s, 0, 0, 0)
        self.el('Me', 0.0, 0.0, 0.0)
        self.Me_ind = ''
        self.comp = []
        self.telescope = ''     # telescope, where high-res spectrum was obtained
        self.year = ''          # year of H2 detection
        self.ref = []           # list of relevant references
        self.comment = ''
        self.SDSS = ''          # SDSS Identifier
        self.full = 'n'         # Specified how much data is loaded: 'y' - fully, 'p' - partially, 'n' - not seen refered paper yet
        self.mangastatus = ''

    def el(self, *args, **kwargs):
        el = e(*args, **kwargs)
        self.e[el.name] = el
        setattr(self, el.name, el)
        
    def H_tot(self):
        if self.H2.type == 'm':
            return self.H2.col*2 + self.HI.col
        if self.H2.type == 'd':
            return [log10(10**self.H2.col[0]+10**self.HI.col[0]), log10(10**self.H2.col[1]+10**self.HI.col[0]), 0.0]
            
    def molec(self):
        if self.H2.type == 'm':
            return self.H2.col*2/self.H_tot()
        if self.H2.type == 'd':
            return [10**(0.3+self.H2.col[0]-self.H_tot()[0]), 10**(0.3+self.H2.col[1]-self.H_tot()[0]), 0.0]

    def sumcompcoldens(self, species, ion):
        n = a()
        for c in self.comp:
            n += sum([e.col for e in c.el if  e.name == species and e.ion == ion and e.type == 'm'])
        return n
    def sumincompcoldens(self, comp, species):
        return sum([e.col for e in self.comp[comp].el if e.name == species and e.ion !=-1 and e.type == 'm'])


class sample(OrderedDict):
    def __init__(self):
        super().__init__()

    def get(self, name):
        values = [v for k, v in self.items() if name in k]
        if len(values) == 1:
            return values[0]
        elif len(values) == 0:
            return None
        elif len(values) > 1:
            raise KeyError('multiple entry for searched key: {:s}'.format(name))

    def all(self, species='H2'):
        lst = []
        for q in self.values():
            for i, c in enumerate(q.comp):
                if species in c.e.keys():
                    lst.append(q.name+'_'+str(i))
        return lst
    def allsys(self, species='H2'):
        lst = []
        for q in self.values():
            lst.append(q.name)
        return lst
    def append(self, qso):
        if isinstance(qso, sample):
            for q in qso.values():
                self[q.name] = q
        else:
            self[qso.name] = qso

    def remove(self,q):
        if isinstance(q, qso):
            del self[q.name]


    def makelist(self, pars=['name'], sys=[], view='list'):
        lst = []
        if len(sys) == 0:
            for k in self.values():
                d = []
                for p in pars:
                    d.append(str(getattr(k, p)))
                lst.append(d)
        else:
            for s in sys:
                q = self.get(s.split('_')[0])
                comp = q.comp[int(s.split('_')[1])]
                d = [s]
                for p in pars:
                    attr = None if any([st not in p for st in ['__val']]) else p.split('__')[1]
                    p = p.split('__')[0]
                    unit = q if any([p.startswith(st) for st in ['z_dla', 'Me', 'CO']]) else comp
                    if attr is None:
                        d.append(str(getattr(unit, p)))
                    else:
                        print('attr',attr,d[0],p)
                        if p in unit.e.keys():
                            d.append(getattr(getattr(unit, p).col, attr))
                        else:
                            d.append(-999)
                lst.append(d)
        if view == 'numpy':
            lst = np.array([tuple(l) for l in lst], dtype=[('name', 'U20')] + [(p, float) for p in pars])

        return lst

    def getcomp(self, name):
        """
        Return component object from sample
        :param:
            -  name         :  object name.
                                    Examples: '0643' - will search for the 0643 im quasar names. Return first component.
                                              '0643_1' - will search for the 0643 im quasar names. Return second component
        :return: q
            -  q              :  qso.comp object
        """
        qso = self.get(name.split('_')[0])
        if len(name.split('_')) > 1:
            q = qso.comp[int(name.split('_')[1])]
        else:
            q = qso.comp[0]

        return q
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# add QSO H2 data     

def load_empty():
    return sample()


def load_QSO():
     global sy
     QSO = sample()

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0000+0048
     q = qso('J0000+0048', 3., 2.525)
     q.telescope = 'KECK/UVES'
     q.year = 2016
     q.coord = ['J0000', '0000+0048']
     q.SIMBAD = 'SDSS J000015.17+004833.2'
     q.m = {'u': 22.357, 'g': 19.238, 'r': 19.238, 'i': 19.130, 'z': 18.898}
     q.progID.append('093.A-0126(A)')
     q.progID.append('096.A-0354(A)')
     q.progID.append('096.A-0924(A)')
     q.ref.append('Noterdaeme2017')
     q.el('HI', 20.80, 0.10, 0.10)
     q.el('H2', 20.44, 0.03, 0.03)
     q.el('H', 21.07, 0.06, 0.05)
     q.el('f', -0.33, 0.06, 0.07)
     q.el('HD', 16.64, 0.31, 0.21)
     q.el('CO', 14.95, 0.05, 0.05)
     q.el('CI', 16.10, 0.08, 0.08)
     q.el('T01', 50, 2, 2, f='d')
     q.el('n', 1.85, 0.18, 0.15)
     q.el('ZnII', 14.09, 0.45, 0.45)
     q.el('FeII', 15.14, 0.03, 0.03)
     q.el('Me', 0.46, 0.45, 0.45)
     q.el('EBV', 0.05,0.01,0.01,f='d')
     q.el('Av', 0.23,0.01,0.01,f='d')
     q.el('Rv', 4.13,0.4,0.4,f='d')
     q.Me_ind = 'Zn'
     q.comment = 'not published yet'
     s = []
     #s.append(co)
     co = sy(2.524432, 0)
     co.el('T01', 50, 2, 2, f='d')
     co.el('CIj0', 12.76, 0.05, 0.05)
     co.el('CIj1', 12.74, 0.11, 0.11)
     co.el('CIj2', 12.69, 0.07, 0.07)
     co.el('n', 3.20, 0.31, 0.18)
     s.append(co)
     co = sy(2.524876, 0)
     co.el('T01', 50, 2, 2, f='d')
     co.el('CIj0', 13.05, 0.04, 0.04)
     co.el('CIj1', 12.73, 0.07, 0.07)
     co.el('CIj2', 12.38, 0.16, 0.16)
     co.el('n', 2.22, 0.12, 0.12)
     s.append(co)
     co = sy(2.524993, 0)
     co.el('T01', 50, 2, 2, f='d')
     co.el('CIj0', 13.61, 0.07, 0.07)
     co.el('CIj1', 13.31, 0.03, 0.03)
     co.el('CIj2', 12.67, 0.07, 0.07)
     co.el('n', 2.16, 0.06, 0.12)
     s.append(co)
     co = sy(2.525348, 0)
     co.el('T01', 50, 2, 2, f='d')
     co.el('CIj0', 13.65, 0.06, 0.06)
     co.el('CIj1', 13.46, 0.06, 0.06)
     co.el('CIj2', 12.46, 0.14, 0.14)
     co.el('n', 2.04, 0.18, 0.12)
     s.append(co)
     co = sy(2.525458, 0)
     co.el('T01', 52, 2, 2, f='d')
     co.el('T02', 99, 2, 2, f='d')
     co.el('CI', 16.21, 0.07,0.07)
     co.el('CIj0', 16.10, 0.08, 0.08)
     co.el('CIj1', 15.54, 0.14, 0.14)
     co.el('CIj2', 14.67, 0.11, 0.11)
     co.el('H2', 20.44, 0.03, 0.03)
     co.el('H2j0', 20.29, 0.02, 0.02, b=(1.0, 0.0, 0.0))
     co.el('H2j1', 19.81, 0.02, 0.02, b=(1.0, 0.0, 0.0))
     co.el('H2j2', 18.77, 0.03, 0.03, b=(1.0, 0.0, 0.0))
     co.el('H2j3', 18.67, 0.02, 0.02, b=(1.0, 0.0, 0.0))
     co.el('H2j4', 17.32, 0.12, 0.12, b=(1.0, 0.0, 0.0))
     co.el('H2j5', 14.50, 0.36, 0.36, b=(1.0, 0.0, 0.0))
     co.el('CO', 14.95, 0.05,0.05)
     co.el('COj0', 14.43, 0.12, 0.12)
     co.el('COj1', 14.52, 0.08, 0.08)
     co.el('COj2', 14.33, 0.06, 0.06)
     co.el('COj3', 13.73, 0.05, 0.05)
     co.el('COj4', 13.14, 0.13, 0.13)
     co.el('n', 1.61, 0.24, 0.24)
     co.el('PDRnH',1.31,0.24,0.42)
     co.el('PDRuv',-0.19, 0.28, 0.24)
     co.el('T_co', 10.5,0.6,0.5, f='d')
     co.el('P_co', 4.38, 0.36, 1.43)
     #co.el('n_co', 2.67, 0.37, 1.48)
     co.el('P_ci', 3.31, 0.18, 0.20)
     co.el('n_ci', 1.49, 0.25, 0.70) # from Klimenko,Ivanchik+ 2020
     co.el('ci_cmb_exc', 0.70)
     co.el('T03_co', 9.85, 0.71,0.56)
     co.el('Tcmbcorr', 9.82, 0.68, 0.59)
     co.el('Tcmb_ci', 11.1, 1.5, 6.6,f='d')
     #
     co.el('n_co_pdr', 2.75, 0.2, 0.6)
     co.el('n_ci_pdr', 1.37, 0.2, 0.25)
     co.el('n_co_3dpdr', 2.36, 0.22, 0.74)
     co.el('uv_co_3dpdr', -0.11, 0.23, 0.28)
     co.el('n_ci_3dpdr', 1.52, 0.20, 0.20)
     co.el('uv_ci_3dpdr', -0.33, 0.27, 0.26)
     #co.el('xco', 22.01, 0.09, 0.09)
     #co.el('wco', -1.57, 0.09, 0.09)
     co.el('xco', 22.94, 0.34, 0.21)
     co.el('wco', -2.50, 0.21, 0.34)
     co.el('n_co_3dpdr', 2.75, 0.2, 0.6)
     # pdr estimate in CO region
     co.el('tgas', 1.82, 0.25,0.06)
     co.el('ngas(co)', 1.99, 0.52, 1.18)
     co.el('pgas', 3.71, 0.59, 0.84)
     co.el('ngas(ci)', 1.29, 0.29,0.25)
     co.el('tgas(ci)', 2.0, 0.16,0.12)
     s.append(co)
     q.comp = s
     q.full = 'u'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add 0013-0029
     q = qso('0013-0029', 2.09, 1.973)
     # q.coord = ['J2000', 130.0020, 0.002112]
     q.telescope = 'UVES'
     q.year = 1997
     q.coord = ['J2000', 001602.406, -001225.08]
     q.SIMBAD = 'LBQS 0013-0029'
     q.m['V'] = 18.36
     q.progID.append('66.A-0624(A)')
     q.progID.append('267.A-5714(A)')
     q.ref += ['Ge1997', 'Petitjean2002', 'Ledoux2003', 'Noterdaeme2008']
     q.el('HI', 20.83, 0.05, 0.05)
     q.el('H2', 18.86, 1.14, 1.14)
     q.el('H', 20.84, 0.11, 0.05)
     q.el('f', -1.73, 1.14, 1.14)
     q.el('Me', -0.59, 0.05, 0.05)
     q.el('Fe/X', -0.83, 0.01, 0.01)
     q.Me_ind = 'Zn'
     q.SDSS = 'J001602.40-001225.0'
     q.full = 'n'
     s =[]
     co = sy(1.966780, 0)
     co.el('H2', 15.86, 0.1, 0.1)
     co.el('H2', 14.67, 0.05, 0.05, J=0, b=(5.00, 0.00, 0.00))
     co.el('H2', 15.70, 0.05, 0.05, J=1)
     co.el('H2', '<18',J=2)
     co.el('H2', 15.07, 0.05, 0.05, J=3)
     co.el('T01', 134, 41, 25, f='d')
     co.el('LFR', 0.30, 0.10, 0.10,f='d')
     co.el('f_cov', 0.1, 0.30, 0.10, f= 'd')
     #co.el('LFR', 0.23, 0.03, 0.05,f='d')
     #co.el('f_cov', 0.31, 0.15, 0.09, f= 'd')
     #s.append(co)
     #co = sy(1.9667967, 0)
     co.el('CIj0', 13.50, 0.11, 0.11)
     co.el('CIj1', 12.78, 0.04, 0.04)
     co.el('CIj2', 11.79, 0.30, 0.30)
     s.append(co)
     co = sy(1.966898, 0)
     co.el('H2', 15.86, 0.15, 0.15)
     co.el('H2', 15.02, 0.04, 0.05, J=0, b=(5.96, 0.29, 0.44))
     co.el('H2', 15.60, 0.07, 0.07, J=1)
     co.el('H2', '<18', J=2)
     co.el('H2', 15.19, 0.04, 0.03, J=3)
     co.el('CIj0', 12.87, 0.11, 0.11)
     co.el('CIj1', 12.65, 0.05, 0.05)
     co.el('CIj2', 12.41, 0.08, 0.08)
     co.el('LFR', 0.24, 0.04, 0.04,f='d')
     co.el('f_cov', 0.28, 0.12, 0.12, f= 'd')
     #co.el('LFR', 0.19, 0.02, 0.03,f='d')
     #co.el('f_cov', 0.31, 0.06, 0.09,f='d')
     #co.el('n', 2.06,0.12,0.12)
     s.append(co)
     co = sy(1.9681759, 0)
     co.el('H2', 15.40, 0.4, 0.50)
     co.el('H2', 14.55, 0.38, 0.64, J=0, b=(3.0, 0.26, 0.00))
     co.el('H2', 14.97, 0.10, 0.10, J=1)
     co.el('H2', '<18', J=2)
     co.el('H2', 14.90, 0.10, 0.10, J=3)
     co.el('CIj0', 12.60, 0.08, 0.08)
     co.el('CIj1', 12.50, 0.08, 0.08)
     co.el('CIj2', 12.4, 0.09, 0.09)
     co.el('LFR', 0.30, 0.10, 0.10,f='d')
     co.el('f_cov', 0.1, 0.30, 0.10, f= 'd')
     #co.el('LFR', 0.19, 0.03, 0.03,f='d')
     #co.el('f_cov', 0.37, 0.10, 0.10,f='d')
     s.append(co)
     co = sy(1.9602503, 0)
     co.el('H2', 16.46, 0.45, 0.4)
     co.el('H2', 16.15, 0.35, 0.30, J=0, b=(3.00, 0.11, 0.22))
     co.el('H2', 15.85, 0.22, 0.16, J=1)
     co.el('H2', '<18', J=2)
     co.el('H2', 15.79, 0.16, 0.11, J=3)
     co.el('CIj0', 13.42, 0.07, 0.07)
     co.el('CIj1', 13.09, 0.04, 0.04)
     co.el('CIj2', 12.68, 0.06, 0.06)
     co.el('LFR', 0.25, 0.04, 0.04,f='d')
     co.el('f_cov', 0.25, 0.12, 0.12, f= 'd')
     #co.el('LFR', 0.22, 0.03, 0.03,f='d')
     #co.el('f_cov', 0.34, 0.30, 0.30,f='d')
     s.append(co)
     co = sy(1.97290, 0)
     co.el('H2', 18.3, 0.30, 0.30)
     co.el('LFR', '<0.05',f='d')
     co.el('f_cov', '>0.85',f='d')
     s.append(co)
     co = sy(1.97380, 0)
     co.el('H2', 18.1, 0.30, 0.30)
     co.el('LFR', '<0.07',f='d')
     co.el('f_cov', '>0.79',f='d')
     s.append(co)

     q.comp = s
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B0027-1836
     q = qso('B0027-1836', 2.56, 2.402)
     # q.coord = ['J1950', 130.0020, 0.002112]
     q.telescope = 'UVES'
     q.year = 2007
     q.coord = ['J2000', 003023.63, -181956.0]
     q.SIMBAD = 'QSO B0027-1836'
     q.m['V'] = 17.9
     q.progID.append('072.A-0442(A)')
     q.progID.append('073.A-0071(A)')
     q.progID.append('074.A-0201(A)')
     q.progID.append('185.A-0745(B)')
     q.progID.append('185.A-0745(G)')
     q.ref.append('Noterdaeme2007')
     q.ref.append('Noterdaeme2008')
     q.ref.append('Rahmani2014')
     q.el('HI', 21.75, 0.10, 0.10)
     q.el('H2', 17.30, 0.07, 0.07)
     q.el('H', 21.75, 0.10, 0.10)
     q.el('f', -4.15, 0.12, 0.13)
     q.el('Me', -1.63, 0.10, 0.10)
     q.el('Fe/X', -0.65, 0.03, 0.03)
     q.Me_ind = 'Zn'
     q.ClI = e('ClI', '<12.71')
     s = []
     co = sy(2.40183, 4)
     co.el('H2', 17.30, 0.07, 0.07)
     co.el('H2', 16.75, 0.06, 0.06, J=0, b=(1.06, 0.13, 0.13))
     co.el('H2', 17.15, 0.07, 0.07, J=1, b=(1.46, 0.14, 0.14))
     co.el('H2', 14.91, 0.02, 0.02, J=2, b=(2.67, 0.08, 0.08))
     co.el('H2', 14.91, 0.01, 0.01, J=3, b=(3.77, 0.07, 0.07))
     co.el('H2', 14.22, 0.01, 0.01, J=4, b=(4.61, 0.25, 0.25))
     co.el('H2', 14.02, 0.03, 0.03, J=5, b=(6.17, 0.69, 0.69))
     co.el('H2', 13.53, 0.00, 0.00, J=6)
     #co.el('T01', 134, 41, 25, f='d')
     co.el('T01', 134, 27, 19, f='d')
     co.el('T02', 87, 2, 2, f='d')
     co.el('CIj0', 12.25, 0.09, 0.15)
     co.el('CIj1', '<12.27')
     #co.el('n', '<2.21') #Slava
     co.el('LFR','<5',f='d')
     co.el('f_cov', '>0.84', f='d')
     #co.el('n', 1.47, 0.3, 0.3) #Noterdaeme
     # note that there is slight shift between H2 and metals
     co.el('NI', 15.04, 0.09, 0.09)
     co.el('MgI', 12.41, 0.05, 0.05)
     co.el('MgII', 15.81, 0.02, 0.02)
     co.el('SiII', 15.45, 0.02, 0.02)
     co.el('PII', 12.87, 0.40, 0.40)
     co.el('SII', 14.98, 0.03, 0.03)
     co.el('ArI', 14.24, 0.02, 0.02)
     co.el('TiII', 12.41, 0.03, 0.03)
     co.el('CrII', 13.08, 0.01, 0.01)
     co.el('MnII', 12.60, 0.02, 0.02)
     co.el('FeII', 14.66, 0.03, 0.03)
     co.el('NiII', 13.42, 0.02, 0.02)
     co.el('ZnII', 12.60, 0.01, 0.01)
     s.append(co)
     co = sy(2.40150, 4)
     co.el('NI', 14.53, 0.24, 0.24)
     co.el('MgI', 11.52, 0.09, 0.09)
     co.el('MgII', 15.32, 0.14, 0.14)
     co.el('SiII', 14.84, 0.11, 0.11)
     co.el('PII', 12.62, 0.30, 0.30)
     co.el('SII', 14.41, 0.10, 0.10)
     co.el('ArI', 13.64, 0.06, 0.06)
     co.el('TiII', 11.87, 0.29, 0.29)
     co.el('CrII', 12.71, 0.04, 0.04)
     co.el('MnII', 11.93, 0.16, 0.16)
     co.el('FeII', 14.36, 0.05, 0.05)
     co.el('NiII', 12.88, 0.10, 0.10)
     co.el('ZnII', 11.99, 0.10, 0.10)
     co.el('CII*', 12.65, 0.30, 0.30)
     s.append(co)
     co = sy(2.40159, 6)
     co.el('NI', 14.51, 0.22, 0.22)
     co.el('MgI', 11.96, 0.09, 0.09)
     co.el('MgII', 15.14, 0.17, 0.17)
     co.el('SiII', 15.06, 0.07, 0.07)
     co.el('PII', 12.50, 0.30, 0.30)
     co.el('SII', 14.68, 0.03, 0.03)
     co.el('ArI', 13.64, 0.06, 0.06)
     co.el('TiII', 11.90, 0.11, 0.11)
     co.el('CrII', 12.81, 0.04, 0.04)
     co.el('MnII', 12.31, 0.08, 0.08)
     co.el('FeII', 14.39, 0.05, 0.05)
     co.el('NiII', 13.21, 0.05, 0.05)
     co.el('ZnII', 12.09, 0.06, 0.06)
     co.el('CII*', 13.80, 0.10, 0.10)
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B0107-0232
     q = qso('B0107-0232', 0.728, 0.56)
     q.telescope = 'HST/COS'
     q.year = 2013
     q.coord = ['J2000', 011014.40, -021658.0]
     q.SIMBAD = 'LBQS 0107-0232'
     q.m = {'B': 17.8, 'V': 18.4, 'R': 18.3, 'u': 19.039, 'g': 18.597, 'r': 18.420, 'i': 18.382, 'z': 18.247}
     q.ref.append('Crighton2013')
     q.ref.append('Muzahid2015')
     q.el('HI', 19.50, 0.20, 0.20)
     q.el('H2', 17.25, 0.21, 0.17)
     q.el('H', 19.50, 0.20, 0.20)
     q.el('f', -1.95, 0.29, 0.27)
     q.el('OI', 15.53, 0.24, 0.25)
     q.el('NII', 14.73, 0.17, 0.19)
     q.el('SiII', 14.79, 0.23, 0.64)
     q.el('CrII', 12.01, 0.09, 0.09)
     q.el('Me', -0.72, 0.32, 0.32)
     q.Me_ind == 'O'
     s = []
     co = sy(0.5571530, 0)
     co.el('H2', 17.17, 0.23, 0.20)
     co.el('H2', 16.17, 0.25, 0.25, J=0)
     co.el('H2', 17.05, 0.28, 0.28, J=1)
     co.el('H2', 16.19, 0.19, 0.19, J=2)
     co.el('H2', 15.77, 0.12, 0.12, J=3)
     co.el('H2', '<14.50', J=4)
     co.el('H2', '<14.50', J=5)
     co.el('T01', '<124', f='d')
     co.el('T02', 326, 260, 106, f='d')
     s.append(co)
     co = sy(0.5572885, 0)
     co.el('H2', 16.57, 0.32, 0.25)
     co.el('H2', 15.63, 0.39, 0.39, J=0)
     co.el('H2', 16.42, 0.40, 0.40, J=1)
     co.el('H2', 15.65, 0.25, 0.25, J=2)
     co.el('H2', 15.47, 0.18, 0.18, J=3)
     co.el('H2', '<14.50', J=4)
     co.el('H2', '<14.30', J=5)
     co.el('T01', 451, np.Inf, 373, f='d')
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B0120-28
     q = qso('B0120-28', 0.436, 0.18562)
     q.coord = ['J2000', 012236.7683, -284321.432]
     q.telescope = 'HST/COS'
     q.year = 2014
     q.SIMBAD = 'QSO B0120-28'
     q.m = {'B': 16.5, 'V': 15.71, 'R': 17.800, 'J': 15.479, 'H': 14.713, 'K': 13.844}
     q.ref.append('Oliveira2014')
     q.ref.append('Muzahid2015')
     q.el('HI', 20.50, 0.10, 0.10)
     q.el('H2', 20.00, 0.10, 0.10)
     q.el('H', 20.71, 0.08, 0.07)
     q.el('f', -0.41, 0.12, 0.13)
     q.el('HD', 14.82, 0.15, 0.15)
     q.el('SII', 14.67, 0.13, 0.20)
     q.el('SiII', 15.07, 0.05, 0.05)
     q.el('FeII', 14.68, 0.08, 0.06)
     q.el('Me', -1.19, 0.15, 0.21)
     q.Me_ind = 'S'
     s = []
     # +13 km/s
     co = sy(0.1856713, 0)
     co.el('H2', 19.95, 0.02, 0.02)
     co.el('H2', 19.72, 0.02, 0.02, J=0)
     co.el('H2', 19.53, 0.03, 0.03, J=1)
     co.el('H2', 18.40, 0.04, 0.04, J=2)
     co.el('H2', 17.60, 0.05, 0.05, J=3)
     co.el('T01', 65, 3, 3, f='d')
     co.el('T02', 110, 3, 3, f='d')
     co.el('HD', 14.53, 0.10, 0.10)
     s.append(co)
     # -20 km/s
     co = sy(0.1855541, 0)
     co.el('H2', 19.03, 0.06, 0.06)
     co.el('H2', 16.81, 0.87, 0.22, J=0)
     co.el('H2', 18.91, 0.07, 0.07, J=1)
     co.el('H2', 18.32, 0.05, 0.05, J=2)
     co.el('H2', 17.73, 0.05, 0.05, J=3)
     co.el('HD', 14.14, 0.10, 0.10)
     s.append(co)
     # -96 km/s
     co = sy(0.1852406, 0)
     co.el('H2', 17.54, 0.07, 0.07)
     co.el('H2', 16.80, 0.13, 0.13, J=0)
     co.el('H2', 17.45, 0.08, 0.08, J=1)
     co.el('H2', 14.55, 0.16, 0.08, J=2)
     co.el('H2', 14.21, 0.07, 0.07, J=3)
     co.el('T01', 243, 211, 86, f='d')
     co.el('T02', 75, 6, 3, f='d')
     s.append(co)
     # -170 km/s
     co = sy(0.1849481, 0)
     co.el('H2', 17.26, 0.08, 0.08)
     co.el('H2', 16.14, 0.14, 0.14, J=0)
     co.el('H2', 17.23, 0.08, 0.08, J=1)
     #co.el('T01', '<869', f='d')
     #co.el('T02', '<543', f='d')
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0203+1134
     q = qso('J0203+1134', 3.6100, 3.38714)
     q.telescope = 'XShooter'
     q.year = 2012
     q.coord = ['J2000', 020346.65706, +113445.4096]
     q.SIMBAD = 'QSO B0201+113'
     q.m = {'B': 19.5, 'V': 19.5, 'R': 19.41}
     q.ref.append('Srianand2012')
     q.ref.append('Ellison2001')
     # col = a.dec(np.array([15.21,0.07, 0.07])
     # col2 = a.dec(np.array([21.26,0.07, 0.07])
     # print(a.logg(a.ratio(col, col2))
     q.el('HI', 21.26, 0.07, 0.08)
     q.el('H2', 15.60, 0.77, 0.77)
     q.el('H', 21.26, 0.07, 0.08)
     q.el('f', -5.36, 0.77, 0.77)
     q.el('Me', -1.25, 0.10, 0.10)
     q.Me_ind = 'S'
     q.SDSS = 'J091826.16+163609.0'
     # q.comment = ''
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B0347-3819
     q = qso('B0347-3819', 3.22, 3.025)
     # q.coord = ['J2000', 130.0020, 0.002112]
     q.telescope = 'UVES'
     q.year = 2002
     q.coord = ['J2000', 034943.68, -381031.1]
     q.SIMBAD = 'QSO B0347-383'
     q.m = {'B': 17.600, 'V': 17.3, 'R': 17.400, 'J': 16.357, 'H': 15.991, 'K': 15.199}
     q.progID.append('60.A-9022(A)')
     q.progID.append('68.B-0115(A)')
     q.progID.append('083.A-0733(A)')
     q.ref.append('Levshakov2002')
     q.ref.append('Ledoux2003')
     q.ref.append('Noterdaeme2008')
     q.el('HI', 20.73, 0.05, 0.05)
     q.el('H2', 14.53, 0.06, 0.06)
     q.el('H', 20.73, 0.05, 0.05)
     q.el('CI', 11.73, 0.26, 0.26)
     q.el('f', -5.90, 0.08, 0.08)
     q.el('Me', -1.17, 0.07, 0.07)
     q.el('Fe/X', -0.71, 0.05, 0.05)
     q.Me_ind = 'Zn'
     s = []
     co = sy(3.025, 0)
     co.el('H2', 14.53, 0.06, 0.06)
     co.el('H2', 13.25, 0.08, 0.08, J=0)
     co.el('H2', 14.26, 0.06, 0.06, J=1)
     co.el('H2', 13.65, 0.04, 0.04, J=2)
     co.el('H2', 13.90, 0.04, 0.04, J=3)
     co.el('H2', 13.12, 0.12, 0.12, J=4)
     co.el('LFR', '<10', f='d')
     co.el('f_cov', '>0.68', f='d')
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B0405-4418
     q = qso('B0405-4418', 3.02, 2.59475)
     # q.coord = ['J2000', 130.0020, 0.002112]
     q.telescope = 'UVES'
     q.year = 2003
     q.coord = ['J2000', 040718.08, -441014.1]
     q.SIMBAD = 'QSO J0407-4410'
     q.m = {'B': 17.6, 'V': 17.6, 'R': 17.100, 'J': 16.126, 'H': 15.514, 'K': 14.832}
     q.progID.append('68.A-0600(A)')
     q.progID.append('70.A-0017(A)')
     q.progID.append('185.A-0745(C)')
     q.progID.append('68.A-0361(A)')
     q.ref.append('Ledoux2003')
     q.ref.append('Noterdaeme2008')
     q.el('HI', 21.05, 0.1, 0.1)
     q.el('H2', 18.14, 0.07, 0.10)
     q.el('H', 21.05, 0.10, 0.10)
     q.el('CI', '<12.23')
     q.el('f', -2.61, 0.12, 0.15)
     q.el('Me', -1.12, 0.1, 0.1)
     q.el('Fe/X', -0.34, 0.05, 0.05)
     q.Me_ind = 'Zn'
     q.ClI = e('ClI', '<12.71')
     s = []
     co = sy(2.59475, 0)
     co.el('H2', 18.14, 0.07, 0.10)
     co.el('H2', 17.73, 0.15, 0.05, J=0)
     co.el('H2', 17.95, 0.20, 0.05, J=1)
     co.el('H2', 15.71, 0.90, 0.49, J=2)
     co.el('H2', 14.70, 0.70, 0.22, J=3)
     co.el('H2', 13.12, 0.12, 0.12, J=4)
     co.el('T01', 101, 39, 15, f='d')
     co.el('T02', 81, 42, 11, f='d')
     #co.el('LFR', 0.07, 0.01, 0.01,f='d')
     #co.el('f_cov', 0.93, 0.03, 0.03,f='d')
     co.el('LFR', '<0.07', f='d')
     co.el('f_cov', '>0.69', f='d')
     co.el('PDRnH',1.81,0.46,1.42)
     co.el('PDRuv',-0.01, 0.34, -0.60)
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B0515-4414
     q = qso('B0515-4414', 1.71, 1.15079)
     q.telescope = 'HST'
     q.year = 2003
     q.coord = ['J2000', 051707.63, -441055.5]
     q.SIMBAD = 'QSO B0515-4414'
     q.m = {'U': 14.74, 'B': 15.49, 'V': 15.16, 'R': 14.000, 'J': 13.578, 'H': 13.031, 'K': 12.620}
     q.ref.append('Reimers2003')
     q.el('HI', 19.88, 0.05, 0.05)
     q.el('H2', 16.94, 0.16, 0.24)
     q.el('H', 19.88, 0.05, 0.05)
     q.el('f', -2.64, 0.31, 0.28)
     q.el('ZnII', 11.99, 0.02, 0.02)
     q.el('CrII', 12.01, 0.09, 0.09)
     q.el('Me', -0.49, 0.10, 0.10)
     q.Me_ind == 'Zn'
     s = []
     co = sy(1.15079, 0)
     co.el('H2', 16.94, 0.16, 0.24)
     co.el('H2', 16.47, 0.23, 0.47, J=0)
     co.el('H2', 16.60, 0.25, 0.60, J=1)
     co.el('H2', 15.85, 0.10, 0.15, J=2)
     co.el('H2', 16.00, 0.18, 0.10, J=3)
     co.el('H2', 15.00, 0.30, 0.15, J=4)
     co.el('H2', 14.48, 0.12, 0.18, J=5)
     co.el('T01', 90, np.Inf, 47, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add Q0528-2505
     q = qso('B0528-2505', 2.77, 2.811)
     q.telescope = 'UVES'
     q.year = 1985
     q.coord = ['J2000', 053007.960, -250329.84]
     q.SIMBAD = 'QSO B0528-2505'
     q.m = {'B': 18.17, 'V': 17.34, 'R': 17.7, 'J': 16.338, 'H': 15.782}
     q.progID.append('60.A-9022(A)')
     q.progID.append('66.A-0594(A)')
     q.progID.append('68.A-0600(A)')
     q.progID.append('68.A-0106(A)')
     q.progID.append('082.A-0087(A)')
     q.ref.append('Levshakov1985')
     q.ref.append('Noterdaeme2008')
     q.ref.append('Klimenko2015')
     q.el('HI', 21.35, 0.07, 0.07)
     q.el('H2', 18.22, 0.11, 0.12)
     q.el('H', 21.35, 0.07, 0.07)
     q.el('HD', 13.33, 0.02, 0.02)
     q.el('CI', 12.36, 0.10, 0.10)
     q.el('f', -2.83, 0.13, 0.14)
     q.el('Me', -0.91, 0.07, 0.07)
     q.el('Fe/X', -0.46, 0.01, 0.01)
     q.Me_ind = 'Zn'
     s = []
     co = sy(2.810995, 2)
     co.el('H2', 18.10, 0.01, 0.01)
     co.el('H2', 17.50, 0.02, 0.02, J=0)
     co.el('H2', 17.93, 0.01, 0.01, J=1)
     co.el('H2', 16.87, 0.03, 0.03, J=2)
     co.el('H2', 15.97, 0.07, 0.07, J=3)
     co.el('H2', 14.18, 0.01, 0.01, J=4)
     co.el('H2', 13.58, 0.02, 0.02, J=5)
     co.el('T01', 141, 6, 6, f='d')
     co.el('T02', 167, 4, 4, f='d')
     co.el('ClI', 11.92, 0.08, 0.08, b=(4.1, 1.5, 1.5))
     co.el('CIj0', 11.67, 0.07, 0.07, b=(3.1, 0.9, 0.9))
     co.el('CIj1', 11.91, 0.06, 0.06, b=(3.1, 0.9, 0.9))
     co.el('CIj2', 11.10, 0.43, 0.43, b=(3.1, 0.9, 0.9))
     co.el('SiII', 14.85, 0.15, 0.15)  # not measured
     co.el('SiII*', '<11.0')
     co.el('n', 2.61, 0.29, 0.20) #logntot 2.6,0.3,0.2 # it is log n_tot = n1 + 2n2
     co.el('LFR', '<0.04',f='d')
     co.el('f_cov', '>0.86',f='d')
     #co.el('f_cov', 0.978, 0.005, 0.005, f='d')
     #co.el('n', 2.73, 0.14, 0.12)
     co.el('n_ci', 2.49, 0.07, 0.11)
     co.el('ci_cmb_exc', 0.26)
     co.el('PDRnH',2.22,0.21,0.22)
     co.el('PDRuv',0.90, 0.14, 0.15)
     co.el('Tcmb_ci', '<20',f='d')
     s.append(co)
     co = sy(2.811124, 2)
     co.el('H2', 17.83, 0.02, 0.02)
     co.el('H2', 17.16, 0.03, 0.03, J=0)
     co.el('H2', 17.67, 0.02, 0.02, J=1)
     co.el('H2', 16.64, 0.03, 0.03, J=2)
     co.el('H2', 16.24, 0.06, 0.06, J=3)
     co.el('H2', 14.20, 0.01, 0.01, J=4)
     co.el('H2', 13.60, 0.02, 0.02, J=5)
     co.el('T01', 167, 14, 13, f='d')
     co.el('T02', 182, 70, 58, f='d')
     co.el('ClI', 11.73, 0.11, 0.11, b=(4.2, 2.0, 2.0))
     co.el('CIj0', 12.07, 0.03, 0.03, b=(2.5, 0.3, 0.3))
     co.el('CIj1', 12.34, 0.02, 0.02, b=(2.5, 0.3, 0.3))
     co.el('CIj2', 11.98, 0.04, 0.04, b=(2.5, 0.3, 0.3))
     co.el('SiII', 14.9, 0.15, 0.15)
     co.el('SiII*', 11.37, 0.03, 0.03)
     # co.el('HD', 13.33, 0.02, 0.02)
     co.el('HD', 13.33, 0.02, 0.02, J=0)
     co.el('HD', '<12.90', J=1)
     co.el('n', 2.56, 0.06, 0.04) #logntot// serj 2.62,0.10
     co.el('LFR', '<0.04',f='d')
    #co.el('f_cov', 0.978, 0.005, 0.005,f='d')
     co.el('f_cov', '>0.86',f='d')
     co.el('PDRnH',2.26,0.08,0.05)
     co.el('PDRuv',1.04, 0.08, 0.08)
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add 0551-3638
     q = qso('0551-3638', 2.32, 1.962)
     q.telescope = 'UVES'
     q.year = 2002
     q.coord = ['J2000', 055246.19, -363727.6]
     q.SIMBAD = 'QSO B0551-36'
     q.m = {'B': 17.72, 'V': 17.57, 'R': 17.200, 'J': 15.728, 'H': 15.093, 'K': 14.245}
     q.progID.append('66.A-0624(A)')
     q.ref.append('Ledoux2002')
     q.ref.append('Noterdaeme2008')
     q.el('HI', 20.70, 0.08, 0.08)
     q.el('H2', 17.42, 0.45, 0.73)
     q.el('CI', 13.03, 0.11, 0.11)
     q.el('H', 20.70, 0.08, 0.08)
     q.el('f', -2.83, 0.13, 0.14)
     q.el('Me', -0.35, 0.08, 0.08)
     q.el('Fe/X', -0.80, 0.04, 0.04)
     q.Me_ind = 'Zn'
     q.ClI = e('ClI', '<12.40')
     s = []
     co = sy(1.96168, 0)
     co.el('H2', 15.64, 0.26, 0.07)
     co.el('H2', 15.19, 0.46, 0.13, J=0)
     co.el('H2', 15.23, 0.33, 0.14, J=1)
     co.el('H2', 14.76, 0.27, 0.09, J=2)
     co.el('H2', 14.72, 0.18, 0.06, J=3)
     co.el('H2', '<14.19', J=4)
     co.el('H2', '<14.34', J=5)
     co.el('T01', 81, 95, 34, f='d')
     s.append(co)
     co = sy(1.96214, 0)
     co.el('H2', 17.40, 0.43, 0.33)
     co.el('H2', 16.83, 0.47, 1.09, J=0)
     co.el('H2', 17.12, 0.46, 0.69, J=1)
     co.el('H2', 16.56, 0.84, 0.80, J=2)
     co.el('H2', 16.24, 1.12, 0.59, J=3)
     co.el('H2', 14.35, 0.05, 0.13, J=4)
     co.el('H2', '<14.34', J=5)
     #co.el('T01', 111, np.Inf, 71, f='d')
     co.el('T01', '<40', f='d')
     co.el('T02', '<98', f='d')
     co.el('CIj0', 12.66, 0.12, 0.12, b=(2.1, 0.8, 0.8))
     co.el('CIj1', 12.69, 0.11, 0.11, b=(2.1, 0.8, 0.8))
     co.el('CIj2', 12.11, 0.34, 0.34, b=(2.1, 0.8, 0.8))
     co.el('n',2.18,0.3,0.3) #logntot
     co.el('LFR', '<0.09',f='d')
     #co.el('PDRnH',1.72,0.30,0.28)
     #co.el('PDRuv',0.85, 0.21, 0.22)
     s.append(co)
     co = sy(1.96221, 0)
     co.el('H2', 15.58, 0.02, 0.05)
     co.el('H2', 14.74, 0.05, 0.13, J=0)
     co.el('H2', 15.18, 0.03, 0.05, J=1)
     co.el('H2', 14.93, 0.02, 0.16, J=2)
     co.el('H2', 14.96, 0.01, 0.11, J=3)
     co.el('T01', 144, 83, 23, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0643-5041
     q = qso('J0643-5041', 3.09, 2.65863)
     q.coord = ['J2000', 064327.0024, -504112.804]
     q.telescope = 'UVES'
     q.SIMBAD = 'QSO B0642-5038'
     q.m['V'] = 18.5
     q.progID.append('60.A-9022(A)')
     q.progID.append('074.A-0201(A)')
     q.year = 2008
     q.ref.append('Noterdaeme2008')
     q.ref.append('Albornoz2014')
     q.el('HI', 20.95, 0.08, 0.08)
     q.el('H2', 18.54, 0.01, 0.01)
     q.el('H', 20.95, 0.08, 0.08)
     q.el('f', -2.11, 0.08, 0.09)
     q.el('HD', 13.65, 0.00, 0.01)
     q.el('Me', -0.91, 0.09, 0.09)
     q.el('Fe/X', -0.30, 0.04, 0.04)
     q.el('CI', 12.57,0.10,0.10)
     q.Me_ind = 'Zn'
     q.ClI = e('ClI', 12.51, 0.05, 0.05)
     s = []
     co = sy(2.6586, 0)
     co.el('H2', 18.54, 0.01, 0.01)
     co.el('H2', 18.22, 0.01, 0.01, J=0, b=(1.59, 1.10, 1.10))
     co.el('H2', 18.25, 0.01, 0.01, J=1, b=(1.39, 0.70, 0.70))
     co.el('H2', 16.62, 0.12, 0.12, J=2, b=(1.41, 0.29, 0.29))
     co.el('H2', 14.84, 0.05, 0.05, J=3, b=(2.14, 0.61, 0.61))
     co.el('H2', 13.94, 0.02, 0.02, J=4, b=(3.45, 1.01, 1.01))
     co.el('H2', 13.86, 0.07, 0.07, J=5, b=(9.49, 2.37, 2.37))
     co.el('H2', '<13.70', J=6)
     co.el('H2', '<13.50', J=7)
     co.el('CIj0', 12.57, 0.09, 0.09, b=(0.71,0.3,0.3))
     co.el('T01', 80, 1, 1, f='d')
     co.el('T02', 96, 7, 6, f='d')
     co.el('HD', 13.65, 0.00, 0.00)
     co.el('ClI', 12.51, 0.05, 0.05, b=(5.8, 1.4, 1.4))
     co.el('LFR', 0.06, 0.01, 0.01,f='d')
     #co.el('f_cov', 0.94, 0.1, 0.1,f='d')
     co.el('f_cov', '>0.74', f='d')
     co.el('PDRnH',2.76,0.65,0.50)
     co.el('PDRuv',0.85, 0.14, 0.26)
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0812+3208
     q = qso('J0812+3208', 2.71, 2.626)
     q.coord = ['J2000', 081240.6837, +320808.577]
     q.telescope = 'KECK'
     q.year = 2009
     q.SIMBAD = 'QSO J0812+32'
     q.m = {'B': 18.16, 'V': 17.75, 'J': 16.186, 'H': 15.885, 'K': 15.313, 'u': 19.638, 'g': 17.836, 'r': 17.450,
            'i': 17.281, 'z': 17.144}
     q.ref.append('Jorgenson2009')
     q.ref.append('Balashev2010')
     q.SDSS = 'J081240.68+320808.6'
     q.el('HI', 21.35, 0.10, 0.10)
     q.el('H2', 19.93, 0.05, 0.05)
     q.el('H', 21.38, 0.09, 0.09)
     q.e['H2'].ref = 'Balashev2010'
     q.el('CI', 13.63, 0.10, 0.10)
     q.el('f', -1.15, 0.11, 0.11)
     q.el('HD', 15.7, 0.07, 0.07)
     q.el('CO', '<12.81')
     q.el('Me', -0.81, 0.10, 0.10)
     q.Me_ind = 'Zn'
     s = []
     co = sy(2.626443, 2)
     co.el('H2', 19.93, 0.05, 0.05)
     co.el('H2', 19.83, 0.05, 0.05, J=0)
     co.el('H2', 19.25, 0.02, 0.02, J=1)
     co.el('H2', 16.47, 0.10, 0.10, J=2)
     co.el('H2', 15.15, 0.25, 0.25, J=3)
     co.el('H2', 13.95, 0.12, 0.12, J=4)
     co.el('T01', 48, 2, 2, f='d')
     co.el('T02',54,2,2, f='d')
     co.el('HD', 15.7, 0.07, 0.07)
     co.el('HD', 15.70, 0.07, 0.07, J=0)
     co.el('HD', 13.77, 0.15, 0.15, J=1)
     co.el('CI',  13.52, 0.15, 0.15)
     co.el('CIj0', 13.30, 0.23, 0.23)
     co.el('CIj1', 13.02, 0.03, 0.03)
     co.el('CIj2', 12.47, 0.05, 0.05)
     co.el('n', 2.60, 0.14, 0.14) #logntot 2.56,0.15,0.15
     co.el('ClI', 13.78, 0.27, 0.27, b=(0.17, 0.05, 0.05))
     co.el('LFR', 0.04, 0.01, 0.01,f='d')
     co.el('f_cov', 0.88, 0.03, 0.03,f='d')
     co.el('n_ci', 2.55, 0.16, 0.18)
     co.el('PDRnH',2.26,0.23,0.21)
     co.el('PDRuv',-0.14, 0.17, 0.17)
     co.el('Tcmb_ci', '<20',f='d')
     s.append(co)
     co = sy(2.626276, 2)
     co.el('H2', 18.82, 0.37, 0.37)
     co.el('H2', 18.71, 0.45, 0.45, J=0)
     co.el('H2', 18.19, 0.19, 0.19, J=1)
     co.el('H2', 16.79, 0.10, 0.10, J=2)
     co.el('H2', 14.62, 0.09, 0.09, J=3)
     co.el('H2', 13.39, 0.14, 0.14, J=4)
     co.el('T01', 50, 25, 12, f='d')
     co.el('T02', 84, 18, 12, f='d')
     co.el('HD', 12.98, 0.22, 0.22)
     co.el('CIj0', 12.70, 0.02, 0.02)
     co.el('CIj1', 12.32, 0.04, 0.04)
     #co.el('CIj2', '<12.39')
     co.el('ClI', 12.79, 0.05, 0.05, b=(2.0, 0.6, 0.6))
     co.el('LFR', 0.04, 0.01, 0.01, f='d')
     co.el('f_cov', 0.88, 0.03, 0.03, f='d')
     co.el('n',1.71, 0.17,0.21)
     co.el('n_ci', 1.79, 0.24, 0.49)
     co.el('ci_cmb_exc', 0.46)
     co.el('Tcmb_ci', 10.8, 1.4,3.3,f='d')
     co.el('PDRnH',0.86,0.24,0.42)
     co.el('PDRuv',-0.95, 0.34, 0.05)
     s.append(co)
     co = sy(2.625808, 2)
     co.el('H2', 15.98, 0.29, 0.23)
     co.el('H2', 15.03, 0.15, 0.15, J=0)
     co.el('H2', 15.26, 0.15, 0.15, J=1)
     co.el('H2', 14.04, 0.06, 0.06, J=2)
     co.el('H2', 13.39, 0.10, 0.10, J=3)
     co.el('H2', '<12.64', J=4)
     co.el('H2', 12.22, 0.61, 0.61, J=5)
     co.el('T01', 147, 13, 11, f='d')
     co.el('CIj0', 12.13, 0.05, 0.05)
     co.el('CIj1', 11.68, 0.16, 0.16)
     co.el('CIj2', 11.37, 0.27, 0.27)
     co.el('LFR', '<0.3', f='d')
     co.el('f_cov', '>0.05', f='d')
     co.el('n', 1.69, 0.26, 0.23) #logntot#1.7,0.2,0.27
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0816+1446
     q = qso('J0816+1446', 3.84563, 3.287)
     q.telescope = 'UVES'
     q.year = 2012
     q.coord = ['J2000', 081634.389, +144612.47]
     q.SIMBAD = 'SDSS J081634.40+144612.9'
     q.m = {'B': 21.91, 'V': 20.39, 'u': 25.27, 'g': 21.20, 'r': 19.028, 'i': 18.845, 'z': 18.79}
     q.progID.append('081.A-0334(A)')
     q.progID.append('282.A-5030(A)')
     q.ref.append('Guimaraes2012')
     q.el('HI', 22.0, 0.10, 0.10)
     q.el('H2', 18.66, 0.20, 0.16)
     q.el('H', 22.00, 0.10, 0.10)
     q.el('f', -3.04, 0.23, 0.19)
     q.el('SII', 15.39, 0.06, 0.06)
     q.el('ZnII', 13.53, 0.01, 0.01)
     q.el('SiII', 16.31, 0.01, 0.01)
     q.el('CrII', 14.07, 0.02, 0.02)
     q.el('FeII', 15.89, 0.02, 0.02)
     q.el('NiII', 14.54, 0.01, 0.01)
     q.el('Me', -1.10, 0.10, 0.10)
     q.el('CI', 13.67, 0.03,0.03)
     q.Me_ind = 'Zn'
     q.SDSS = 'J081634.40+144612.9'
     # q.comment = ''
     s = []
     co = sy(3.28742, 0)
     co.el('H2', 18.62, 0.21, 0.17)
     co.el('H2', 18.19, 0.35, 0.35, J=0, b=(2, 2, 1))
     co.el('H2', 18.41, 0.20, 0.20, J=1, b=(2, 2, 1))
     co.el('H2', 16.21, 0.25, 0.25, J=2, b=(8.3, 0.4, 0.4))
     co.el('H2', 15.75, 0.10, 0.10, J=3, b=(6.5, 0.4, 0.4))
     co.el('T01', 101, 94, 38, f='d')
     co.el('T02', 82, 15, 11, f='d')
     co.el('CIj0', 13.43, 0.01, 0.01)
     co.el('CIj1', 13.24, 0.02, 0.02)
     co.el('CIj2', 12.47, 0.07, 0.07)
     co.el('ClI', '<13.65')
     co.el('SiII', 14.90, 0.04, 0.04)
     co.el('CrII', 12.63, 0.08, 0.08)
     co.el('NiII', 13.33, 0.03, 0.03)
     co.el('ZnII', 12.40, 0.07, 0.07)
     co.el('FeII', 14.57, 0.06, 0.06)
     co.el('n',1.88,0.05,0.05)
     co.el('n_ci', 1.77, 0.45, 0.80)
     co.el('ci_cmb_exc', 0.57)
     co.el('Tcmb_ci', 15.2, 1.0, 4.2, f='d')
     co.el('PDRnH',1.63,0.08,0.12)
     co.el('PDRuv',-0.32, 0.20, -0.16)
     # wrong estimate
     #co.el('PDRnH',0.95,0.26,0.37)
     #co.el('PDRuv',-1.00, 0.31, 0.00)
     s.append(co)
     co = sy(3.28667, 0)
     co.el('H2', 17.60, 0.29, 0.27)
     co.el('H2', 16.59, 0.50, 0.50, J=0, b=(2, 2, 1))
     co.el('H2', 17.55, 0.30, 0.30, J=1, b=(2, 2, 1))
     co.el('H2', 14.71, 0.20, 0.20, J=2, b=(8.3, 0.4, 0.4))
     co.el('H2', 15.13, 0.10, 0.10, J=4, b=(6.5, 0.4, 0.4))
     #co.el('T01', 69, 10, 8, f='d')
     co.el('T01', '<90', f='d')
     co.el('T02', 86, 23,14, f='d')
     co.el('ClI', '<12.76')
     co.el('SiII', 15.99, 0.04, 0.04)
     co.el('CrII', 13.71, 0.02, 0.02)
     co.el('NiII', 14.15, 0.01, 0.01)
     co.el('ZnII', 13.12, 0.02, 0.02)
     co.el('FeII', 15.48, 0.03, 0.03)
     s.append(co)
     co = sy(3.286481, 0)
     co.el('SiII', 15.73, 0.03, 0.03)
     co.el('CrII', 13.43, 0.02, 0.02)
     co.el('NiII', 13.87, 0.01, 0.01)
     co.el('ZnII', 13.04, 0.02, 0.02)
     co.el('FeII', 15.09, 0.04, 0.04)
     s.append(co)
     co = sy(3.286998, 0)
     co.el('SiII', 14.90, 0.10, 0.10)
     co.el('CrII', 12.68, 0.11, 0.11)
     co.el('NiII', 13.18, 0.10, 0.10)
     co.el('ZnII', 12.19, 0.10, 0.10)
     co.el('FeII', 14.79, 0.04, 0.04)
     s.append(co)
     co = sy(3.287260, 0)
     co.el('SiII', 14.99, 0.05, 0.05)
     co.el('CrII', 12.83, 0.06, 0.06)
     co.el('NiII', 13.44, 0.04, 0.04)
     co.el('ZnII', 12.40, 0.05, 0.05)
     co.el('FeII', 14.85, 0.04, 0.04)
     s.append(co)
     co = sy(3.287260, 0)
     co.el('SiII', 15.29, 0.04, 0.04)
     co.el('CrII', 13.23, 0.05, 0.05)
     co.el('NiII', 13.73, 0.03, 0.03)
     co.el('ZnII', 12.45, 0.06, 0.06)
     co.el('FeII', 14.91, 0.07, 0.07)
     s.append(co)
     co = sy(3.288169, 0)
     co.el('SiII', 14.89, 0.07, 0.07)
     co.el('CrII', 12.69, 0.09, 0.09)
     co.el('NiII', 13.23, 0.05, 0.05)
     co.el('ZnII', 11.73, 0.16, 0.16)
     co.el('FeII', 14.41, 0.13, 0.13)
     s.append(co)
     co = sy(3.288169, 0)
     co.el('SiII', 14.30, 0.25, 0.25)
     co.el('NiII', 12.30, 0.48, 0.48)
     co.el('ZnII', 11.34, 0.38, 0.38)
     co.el('FeII', 14.82, 0.05, 0.05)
     s.append(co)
     q.comp = s
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0843+0221
     q = qso('J0843+0221', 2.92, 2.7865)
     q.telescope = 'UVES'
     q.year = 2015
     q.coord = ['J2000', 084312.72, +022117.28]
     q.SIMBAD = 'SDSS J084312.72+022117.2'
     q.m = {'u': 22.917, 'g': 20.469, 'r': 19.902, 'i': 19.745, 'z': 19.44}
     q.progID.append('092.A-0345(A)')
     q.ref.append('Balashev2015')
     q.el('HI', 21.82, 0.11, 0.11)
     q.el('H2', 21.21, 0.02, 0.02)
     q.el('H', 21.99, 0.08, 0.07)
     q.el('f', -0.48, 0.08, 0.08)
     q.el('HD', 17.35, 0.15, 0.34)
     q.el('HD', 17.34, 0.13, 0.37, J=0)
     q.el('HD', 15.87, 0.72, 0.49, J=1)
     q.el('CO', '<13.40')
     q.el('SII', 15.57, 0.05, 0.02)
     q.el('ZnII', 13.03, 0.03, 0.06)
     q.el('SiII', 15.77, 0.04, 0.06)
     q.el('CrII', 13.23, 0.05, 0.06)
     q.el('FeII', 14.96, 0.04, 0.04)
     q.el('NiII', 13.69, 0.05, 0.05)
     q.el('MnII', 13.67, 0.07, 0.06)
     q.el('MgII', 15.87, 0.06, 0.05)
     q.el('ClI', 13.63, 0.20, 0.05)
     q.el('SiII*', 11.80, 0.07, 0.11)
     q.el('CI', 13.53, 0.04, 0.02)
     q.el('CI*', 13.61, 0.02, 0.02)
     q.el('CI**', 13.34, 0.04, 0.03)
     q.el('CII', 16.93, 0.10, 0.10)
     q.el('CII*', 15.55, 0.16, 0.18)
     q.el('T01', 123, 9, 8, f='d')
     q.el('n', 2.58, 0.03, 0.06) ##Balashev
     q.el('Me', -1.52, 0.08, 0.10)
     q.el('Fe/X', -1.01, 0.07, -0.05)
     q.el('EBV', 0.04,0.03,0.03,f='d')
     q.el('AV', 0.09, 0.1, 0.1,f='d')
     q.Me_ind = 'Zn'
     q.SDSS = 'J084312.72+022117.28'
     s = []
     co = sy(2.78644, 0)
     co.el('H2', 21.21, 0.02, 0.02)
     # co.el('H2', '<20.71', J=0)
     # co.el('H2', '<21.05', J=1)
     co.el('H2', 20.71, 0.05, 0.05,J=0)
     co.el('H2', 21.05, 0.05, 0.05,J=1)
     co.el('H2', 19.42, 0.10, 0.05, J=2)
     co.el('H2', 17.39, 0.30, 0.06, J=3)
     co.el('H2', 16.41, 0.11, 0.08, J=4)
     co.el('H2', 15.81, 0.04, 0.08, J=5)
     co.el('H2', 14.68, 0.04, 0.03, J=6)
     co.el('H2', 14.50, 0.04, 0.05, J=7)
     co.el('H2', 13.90, 0.09, 0.13, J=8)
     #co.el('H2', 13.55, 0.09, 0.07, J=0, nu=1)
     #co.el('H2', '<13.67', J=1, nu=1)
     #co.el('H2', '<13.72', J=2, nu=1)
     co.el('H2', 13.80, 0.07, 0.29, J=9)
     co.el('H2', '<13.25', J=10)
     co.el('T01', 123, 15, 12, f='d')
     co.el('T02', 111, 8, 3, f='d')
     co.el('HD', 15.15, 0.24, 0.12, J=0)
     co.el('HD', 14.92, 0.06, 0.05, J=1)
     # print(a(15.26, 0.37, 0.23)+ a(14.92, 0.06, 0.06)+a(17.39, 0.18, 0.34)+a(16.68, 0.25, 0.25)
     co.el('CIj0', 13.16, 0.11, 0.09)
     co.el('CIj1', 13.25, 0.13, 0.09)
     co.el('CIj2', 13.01, 0.12, 0.11)
     co.el('n', 2.57, 0.22, 0.20)
     co.el('LFR', '<0.05', f='d')
     co.el('f_cov', '>0.84', f='d')
     co.el('MgII', 14.45, 1.08, 1.90)
     co.el('SiII', 14.92, 0.14, 0.12)
     co.el('SiII*', 11.32, 0.14, 0.21)
     co.el('SII', 14.16, 0.54, 2.12)
     co.el('ClI', 13.18, 0.18, 0.24)
     co.el('TiII', '<12.50')
     co.el('CrII', 13.11, 0.22, 0.30)
     co.el('MnII', 13.17, 0.30, 0.38)
     co.el('FeII', 14.75, 0.06, 0.10)
     co.el('NiII', 13.49, 0.11, 0.11)
     co.el('ZnII', 12.49, 0.15, 0.21)
     co.el('n_ci', 1.94, 0.12, 0.10)
     co.el('Tcmb_ci', '<16', f='d')
     co.el('ci_cmb_exc', 0.28)
     co.el('PDRnH',1.85,0.07,0.09)
     co.el('PDRuv',1.80, 0.15, 0.14)
     s.append(co)
     co = sy(2.786574, 0)
     co.el('CO', '<13.5')
     co.el('H2', 21.21, 0.02, 0.02)
     # co.el('H2', '<20.71', J=0)
     # co.el('H2', '<21.05', J=1)
     co.el('H2', 20.71, 0.05, 0.05,J=0)
     co.el('H2', 21.05, 0.05, 0.05,J=1)
     co.el('H2', 19.15, 0.14, 0.18, J=2)
     co.el('H2', 18.68, 0.02, 0.04, J=3)
     co.el('H2', 17.09, 0.15, 0.13, J=4)
     co.el('H2', 15.95, 0.09, 0.05, J=5)
     co.el('H2', 14.19, 0.09, 0.09, J=6)
     co.el('H2', 14.20, 0.04, 0.15, J=7)
     co.el('H2', 13.71, 0.22, 0.35, J=8)
     #co.el('H2', '<13.42', J=0, nu=1)
     #co.el('H2', '<13.64', J=1, nu=1)
     #co.el('H2', '<13.50', J=2, nu=1)
     co.el('H2', 13.70, 0.15, 0.17, J=9)
     co.el('H2', '<13.40', J=10)
     co.el('T01', 123, 15, 12, f='d')
     co.el('T02', 98, 6, 8, f='d')
     co.el('HD', 17.16, 0.27, 0.54, J=0)
     co.el('HD', 15.37, 0.55, 0.25, J=1)
     co.el('CIj0', 13.36, 0.06, 0.08)
     co.el('CIj1', 13.42, 0.05, 0.10)
     co.el('CIj2', 13.02, 0.07, 0.16)
     co.el('CII*', 15.46, 0.20, 0.20)
     co.el('n', 2.39, 0.14, 0.18)
     co.el('MgII', 15.86, 0.06, 0.20)
     co.el('SiII', 15.58, 0.06, 0.08)
     co.el('SiII*', 11.62, 0.08, 0.14)
     co.el('SII', 15.50, 0.03, 0.03)
     co.el('ClI', 13.27, 0.13, 0.15)
     co.el('TiII', 12.71, 0.72, 0.14)
     co.el('CrII', 12.80, 0.37, 0.63)
     co.el('MnII', 13.29, 0.56, 0.42)
     co.el('FeII', 14.24, 0.16, 0.15)
     co.el('NiII', 13.00, 0.20, 0.18)
     co.el('ZnII', 12.82, 0.13, 0.13)
     #co.el('n_ci', 1.94, 0.12, 0.10)
     co.el('PDRnH',1.94,0.08,0.07)
     co.el('PDRuv',1.80, 0.12, 0.09)
     s.append(co)
     co = sy(2.786733, 0)
     co.el('H2', 21.21, 0.02, 0.02)
     # co.el('H2', '<20.71', J=0)
     # co.el('H2', '<21.05', J=1)
     co.el('H2', 20.71, 0.05, 0.05,J=0)
     co.el('H2', 21.05, 0.05, 0.05,J=1)
     co.el('H2', 15.85, 0.50, 2.30, J=2, f='l')
     co.el('H2', 14.63, 0.11, 1.11, J=3, f='l')
     co.el('H2', 14.71, 0.08, 0.08, J=4)
     co.el('H2', 15.17, 0.05, 0.06, J=5)
     co.el('H2', 14.30, 0.09, 0.03, J=6)
     co.el('H2', 14.50, 0.06, 0.06, J=7)
     co.el('H2', '<13.82', J=8)
     co.el('H2', '<13.36', J=0, nu=1)
     co.el('H2', '<13.52', J=1, nu=1)
     co.el('H2', '<13.70', J=2, nu=1)
     co.el('H2', '<13.45', J=9)
     co.el('H2', '<12.92', J=10)
     #co.el('T01', 123, 9, 8, f='d')
     #co.el('T02', 98, 6, 8, f='d')
     co.el('MgII', 13.57, 1.11, 1.35)
     co.el('SiII', 14.98, 0.16, 0.12)
     co.el('SiII*', 10.9, 0.5, 0.5)
     co.el('SII', 14.89, 0.12, 0.09)
     co.el('ClI', 12.85, 0.35, 0.43)
     co.el('TiII', '<12.60')
     co.el('CrII', 11.75, 0.75, 0.70)
     co.el('MnII', 12.86, 0.25, 1.31)
     co.el('FeII', 14.06, 0.21, 0.23)
     co.el('NiII', 13.10, 0.21, 0.41)
     co.el('ZnII', 12.42, 0.25, 0.42)
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0857+1855
     q = qso('J0857+1855', 1.89, 1.7293)
     q.telescope = 'UVES'
     q.year = 2011
     q.coord = ['J0857', '0857+1855']
     q.SIMBAD = 'SDSS J085726+185524'
     q.ref.append('Noterdaeme2011')
     q.el('CO', 13.51, 0.05, 0.05)
     q.el('CI', 14.16, 0.06, 0.06)
     #q.el('H2', '<21')
     q.el('EBV', 0.02,0.05,0.05, f = 'd')
     q.el('Rv', 2.74, 0., 0,f='d')
     q.el('Av', 0.06, 0.10, 0.06,f='d')
     #q.el('ZnII', 14.09, 0.45, 0.45)
     #q.el('FeII', 15.14, 0.03, 0.03)
     #q.el('Me', 0.46, 0.45, 0.45)
     #q.Me_ind = 'Zn'
     q.comment = ''
     s = []
     co = sy(1.7293299, 0)
     co.el('H2','<21')
     co.el('CI', 14.16,0.06,0.06)
     co.el('CIj0', 13.90, 0.10, 0.10, b=(5.6,0.2,0.2))
     co.el('CIj1', 13.67, 0.08, 0.07)
     co.el('CIj2', 13.23, 0.07, 0.07)
     co.el('CO', 13.51,0.05,0.05)
     co.el('COj0', 13.08, 0.08, 0.08)
     co.el('COj1', 13.01, 0.09, 0.08)
     co.el('COj2', 12.91, 0.08, 0.09)
     co.el('COj3', 12.30, 0.22, 0.54)
     co.el('n_co', 2.42, 0.40,0.30)
     co.el('n_ci', 2.10, 0.35, 0.25)
     #co.el('P_co', 4.36, 0.36, 1.25) # derived by pyratio
     #co.el('n_co', 2.34, 0.38, 1.30)
     #co.el('P_ci', 4.31, 0.70, 0.20)
     co.el('T_co', 8.7,1.3,1.1, f='d')
     co.el('T03_co', 8.9, 1.5,1.2, f='d')
     co.el('Tcmbcorr', 7.9, 1.7, 1.4, f='d')
     ##################################### results with new version PDR:
     co.el('n_co_pdr', 2.42,0.42,0.39)
     co.el('n_ci_pdr', 2.3, 0.46, 0.20)
     co.el('n_co_3dpdr', 2.15,0.38,0.28)
     co.el('n_ci_3dpdr', 2.15, 0.38, 0.28)
     co.el('wco', -3.8, 0.30, 0.30,'l')
     # pdr estimate in CO region
     co.el('tgas', 2.13, 0.07,0.33)
     co.el('ngas(co)', 1.97,0.57,0.61)
     co.el('ngas(ci)', 2.13, 0.07,0.33)
     co.el('tgas(ci)', 1.97, 0.57,0.61)
     s.append(co)
     co = sy(1.729167, 0)
     co.el('CIj0', 13.11, 0.07, 0.07, b=(5.17,0.45,0.43))
     co.el('CIj1', 13.24, 0.07, 0.07)
     co.el('CIj2', 13.18, 0.07, 0.07)
     s.append(co)
     q.comp = s
     q.full = 'u'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0917+0154
     q = qso('J0917+0154', 2.1763, 2.107)
     q.telescope = 'XShooter'
     q.year = 2018
     q.coord = ['J2000', 091721.36, +015448.12]
     q.progID.append('084.A-0699(A)')
     q.SIMBAD = '[VV2006] J091721.4+015448'
     q.m = {'B': 20.91, 'V': 20.43, 'G': 20.126, 'u': 21.61, 'g': 20.587, 'r': 19.953, 'i': 19.665, 'z': 19.217}
     q.ref.append('Noterdaeme2018')
     q.el('HI', 21.10, 0.01, 0.01)
     q.el('H2', 20.11, 0.06, 0.06)
     q.el('H', 21.18, 0.01, 0.01)
     q.el('f', -0.77, 0.06, 0.06)
     q.el('Me', -0.5, 0.5, 0.5)
     q.el('EBV', 0.132, 0.03,0.03, f='d')
     # q.Me_ind = 'Zn'
     q.SDSS = 'J091721.36+015448.1'
     # q.comment = ''
     s = []
     co = sy(2.107, 0)
     co.el('H2',20.11, 0.06, 0.06)
     co.el('CI', 14.32, 0.06, 0.06)
     co.el('CO', '<14.07')
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0918+1636
     q = qso('J0918+1636', 3.0727, 2.58)
     q.telescope = 'XShooter'
     q.year = 2011
     q.coord = ['J2000', 091826.1, +163609]
     q.SIMBAD = 'QSO J0918+1636'
     q.m = {'B': 21.19, 'V': 20.58, 'u': 24.141, 'g': 20.646, 'r': 19.916, 'i': 19.709, 'z': 19.541}
     q.ref.append('Fynbo2011')
     q.el('HI', 20.96, 0.05, 0.05)
     q.el('H2', 17.6, 1.45, 1.45)
     q.el('H', 20.96, 0.05, 0.05)
     q.el('f', -3.07, 1.45, 1.45)
     q.el('Me', -0.12, 0.05, 0.05)
     q.Me_ind = 'Zn'
     q.SDSS = 'J091826.16+163609.0'
     # q.comment = ''
     q.full = 'n'
     QSO.append(q)


     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1047+2057
     q = qso('J1047+2057', 2.01, 1.7738)
     q.telescope = 'UVES'
     q.year = 2011
     q.coord = ['J1047', '1047+2057']
     q.SIMBAD = 'SDSS J1047+2057'
     q.ref.append('Noterdaeme2011')
     q.el('CO', 14.95, 0.05, 0.05)
     q.el('EBV', 0.18,0.06,0.06, f = 'd')
     q.el('Av', 0.49, 0.18, 0.18,f='d')
     #q.el('Av', 0.18, 0.05, 0.05, f='d')
     q.el('Rv', 2.74, 0., 0,f='d')
     #q.el('H2', 20., 0.5, 0.5)
     #q.el('HI', 17.6, 1.45, 1.45)
     # q.el('ZnII', 14.09, 0.45, 0.45)
     # q.el('FeII', 15.14, 0.03, 0.03)
     #q.el('Me', 0.46, 0.45, 0.45)
     # q.Me_ind = 'Zn'
     q.comment = ''
     s = []
     co = sy(1.7738, 0)
     #co.el('CI', 13, 0.07, 0.07) #?????
     co.el('H2', '<21')
     co.el('CI', '>14.9')
     co.el('CO', 14.97, 0.12,0.12)
     co.el('COj0', 14.53, 0.21, 0.24, b=(0.8,0.11,0.11))
     co.el('COj1', 14.63, 0.18, 0.15)
     co.el('COj2', 14.19, 0.09, 0.09)
     co.el('COj3', 13.23, 0.11, 0.14)
     co.el('n_co', 2.26,0.43,0.30)
     #co.el('P_co', '<3.6') derived by popratio
     #co.el('n_co', '<1.6')
     co.el('T_co', 6.7,0.7,0.6, f='d')
     co.el('T03_co', 6.87, 0.70,0.70, f='d')
     co.el('Tcmbcorr', 6.6, 1.2, 1.1)
     #co.el('P_ci', 3.68, 0.15, 0.15)
     ##################################### results with new version PDR:
     co.el('n_co_3dpdr', '<2.2')
     co.el('wco', -3.41, 0.18, 0.15)
     #co.el('n_co_3dpdr', 2.26, 0.43, 0.29)
     # pdr estimate in CO region
     co.el('tgas', 1.97, 0.16, 0.22)
     co.el('ngas(co)', '<1.9')
     co.el('pgas', 3.0, 0.7, 0.2)
     s.append(co)
     q.comp = s
     q.full = 'u'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1117+1437
     q = qso('J1117+1437', 3.09184, 2.001)
     q.telescope = 'UVES'
     q.year = 2018
     q.coord = ['J2000', 111109.69, +144237.8]
     q.progID.append('081.A-0334(A)')
     q.SIMBAD = 'SDSS J111109.65+144238.2'
     q.m = {'B': 19.60, 'V': 19.23, 'G': 19.0126, 'u': 20.818, 'g': 19.159, 'r': 18.762, 'i': 18.681, 'z': 18.587}
     q.ref.append('Noterdaeme2018')
     q.el('HI', 19.879, 0.025, 0.025)
     q.el('H2', 18.0, 0.5, 0.5)
     q.el('H', 19.89, 0.03, 0.03)
     q.el('f', -1.59, 0.40, 0.40)
     q.el('Me', -0.5, 0.5, 0.5)
     q.el('EBV', 0.05,0.03,0.03, f='d')
     q.el('CO','<13.13')
     # q.Me_ind = 'Zn'
     q.SDSS = 'J111109.65+144238.2'
     # q.comment = ''
     s = []
     co = sy(2.001, 0)
     co.el('CI', 14.40,0.03,0.03)
     co.el('CO', '<13.13')
     q.el('H2', 18,0.5,0.5)
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add Q1232+0815
     q = qso('Q1232+0815', 2.57, 2.3377)
     q.telescope = 'UVES'
     q.year = 2001
     q.SIMBAD = 'LBQS 1232+0815'
     q.m = {'V': 18.4, 'J': 16.865, 'H': 16.371, 'K': 15.594, 'u': 19.491, 'g': 18.507, 'r': 18.448, 'i': 18.373,
            'z': 18.054}
     q.progID.append('60.A-9022(A)')
     q.progID.append('68.A-0106(A)')
     q.progID.append('65.P-0038(A)')
     q.progID.append('70.A-0017(A)')
     q.progID.append('69.A-0061(A)')
     q.progID.append('71.B-0136(A)')
     q.progID.append('69.A-0061(A)')
     q.ref.append('Ge2001')
     q.ref.append('Noterdaeme2008')
     q.ref.append('Ivanchik2010')
     q.ref.append('Balashev2011')
     q.el('HI', 20.90, 0.08, 0.08)
     q.el('H2', 19.57, 0.10, 0.10)
     q.el('H', 20.94, 0.07, 0.07)
     q.el('HD', 15.52, 0.17, 0.17)
     q.el('CO', '<12.55')
     q.el('f', -1.07, 0.12, 0.13)
     q.el('Me', -1.35, 0.12, 0.12)
     q.el('Fe/X', -0.45, 0.01, 0.01)
     q.Me_ind == 'S'
     q.ClI = e('ClI', 13.49, 0.08, 0.08)
     q.el('CI',14.07,0.03,0.03)
     s = []
     co = sy(2.3377, 0)
     co.el('H2', 19.57, 0.10, 0.10)
     co.el('H2', 19.45, 0.10, 0.10, J=0)
     co.el('H2', 19.29, 0.15, 0.15, J=1)
     co.el('H2', 16.78, 0.24, 0.24, J=2)
     co.el('H2', 16.36, 0.10, 0.10, J=3)
     co.el('H2', 14.70, 0.06, 0.06, J=4)
     co.el('H2', 14.36, 0.07, 0.07, J=5)
     co.el('T01', 66, 13, 9, f='d')
     co.el('T02', 66, 6, 5, f='d')
     co.el('CIj0', 13.87, 0.05, 0.05)
     co.el('CIj1', 13.56, 0.04, 0.04)
     co.el('CIj2', 12.82, 0.07, 0.07)
     #co.el('n', 1.9261745, 0.1409396, 0.16107383) #Balashev
     co.el('n', 1.93,0.1,0.1) #logntot Klimenko
     co.el('ClI', 13.49, 0.08, 0.08, b=(0.8, 0.2, 0.2))
     co.el('LFR', 0.06, 0.03, 0.03, f='d')
     co.el('f_cov', 0.82, 0.10, 0.10, f='d')
     co.el('n_ci', 2.03, 0.17, 0.18)
     co.el('Tcmb_ci', '<9.4', f='d')
     co.el('ci_cmb_exc', 0.38)
     co.el('PDRnH',1.58,0.14,0.11)
     co.el('PDRuv',-0.55,0.21,0.20)
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1237+0647
     # Metals component have not been added
     q = qso('J1237+0647', 2.78, 2.69)
     q.telescope = 'UVES'
     q.year = 2010
     q.coord = ['J2000', 123714.606, +064759.56]
     q.SIMBAD = '[VV2010] J123714.6+064759'
     q.m = {'B': 19.66, 'V': 19.15, 'J': 17.757, 'H': 17.513, 'K': 17.444, 'u': 21.308, 'g': 19.186, 'r': 18.594,
            'i': 18.370, 'z': 18.218}
     q.progID.append('082.A-0544(A)')
     q.progID.append('091.A-0124(A)')
     q.progID.append('093.A-0373(A)')
     q.ref.append('Noterdaeme2010')
     q.el('HI', 20.0, 0.15, 0.15)
     q.el('H2', 19.21, 0.13, 0.12)
     q.el('H', 20.12, 0.12, 0.11)
     q.el('f', -0.61, 0.17, 0.18)
     q.el('HD', 14.48, 0.05, 0.05)
     q.el('CO', 14.17, 0.09, 0.09)
     q.el('SII', 15.39, 0.06, 0.06)
     q.el('ZnII', 13.02, 0.02, 0.02)
     q.el('SiII', 15.15, 0.02, 0.02)
     q.el('FeII', 14.57, 0.01, 0.01)
     q.el('NiII', 13.48, 0.03, 0.03)
     q.el('T01', 108, 52, 28, f='d')
     q.el('Me', 0.34, 0.12, 0.12)
     q.el('Av',0.46,0.20,0.20,f='d') #Ledoux
     q.el('Rv', 2.74, 0., 0,f='d')
     q.el('EBV', 0.17, 0.07, 0.07,f='d')
     q.Me_ind == 'Zn'
     #q.el('EBV', 0.143,0.01,0.01, f = 'd') #Ledoux2015
     q.ClI = e('ClI', 13.01, 0.02, 0.02)
     q.SDSS = 'J123714.60+064759.5'
     s = []
     case = 'Klimenko'
     if case == 'Noterdaeme':
             co = sy(2.68801, 0)
             co.el('H2', 16.28, 0.10, 0.10)
             co.el('H2', 15.51, 0.05, 0.05, J=0, b=(3.3, 0.2, 0.2))
             co.el('H2', 16.08, 0.10, 0.10, J=1, b=(3.3, 0.2, 0.2))
             co.el('H2', 15.39, 0.20, 0.20, J=2, b=(3.3, 0.2, 0.2))
             co.el('H2', 15.13, 0.05, 0.05, J=3, b=(3.3, 0.2, 0.2))
             co.el('T01', 193, 134, 56, f='d')
             co.el('CIj0', 13.24, 0.04, 0.04)
             co.el('CIj1', 12.80, 0.03, 0.03)
             co.el('n',1.0,0.3,0.9)
             #co.el('n', '<1.3')
             co.el('LFR', '<0.08', f='d')
             co.el('f_cov', '>0.74', f='d')
             s.append(co)
             co = sy(2.68868, 0)
             co.el('H2', 17.62, 0.08, 0.11)
             co.el('H2', 16.94, 0.20, 0.44, J=0, b=(0.4, 0.8, 0.1))
             co.el('H2', 17.51, 0.05, 0.05, J=1, b=(1.2, 0.1, 0.1))
             co.el('H2', 15.25, 0.03, 0.03, J=2, b=(3.3, 0.2, 0.2))
             co.el('H2', 14.89, 0.02, 0.02, J=3, b=(4.9, 0.2, 0.2))
             #co.el('T01', 193, np.Inf, 82, f='d')
             co.el('T01', '>110', f='d')
             co.el('T02', 93, 26,5, f='d')
             co.el('CIj0', 12.84, 0.03, 0.03)
             co.el('CIj1', 12.70, 0.05, 0.05)
             co.el('n', 1.87,0.11,0.11)
             co.el('LFR', '<0.08', f='d')
             co.el('f_cov', '>0.74', f='d')
             co.el('PDRnH',1.85,0.16,0.20)
             co.el('PDRuv',-0.10, 0.14, 0.11)
             s.append(co)
             co = sy(2.68955, 0)
             co.el('H2', 19.20, 0.13, 0.12)
             co.el('H2', 18.65, 0.20, 0.20, J=0, bF=(6.0, 0.1, 0.1))
             co.el('H2', 18.92, 0.10, 0.10, J=1, b=(6.0, 0.1, 0.1))
             co.el('H2', 18.18, 0.10, 0.10, J=2, b=(6.0, 0.1, 0.1))
             co.el('H2', 18.21, 0.10, 0.10, J=3, b=(6.0, 0.1, 0.1))
             co.el('H2', 15.43, 0.10, 0.10, J=4, b=(7.9, 0.1, 0.1))
             co.el('H2', 14.95, 0.05, 0.05, J=5, b=(7.9, 0.1, 0.1))
             co.el('T01', 108, 52, 27, f='d')
             co.el('T02', 189, 45, 30, f='d')
             co.el('CO', 14.16,0.01,0.02)
             co.el('CO', 13.53, 0.04, 0.04, J=0)
             co.el('CO', 13.77, 0.02, 0.02, J=1)
             co.el('CO', 13.54, 0.03, 0.03, J=2)
             co.el('CO', 13.21, 0.04, 0.04, J=3)
             co.el('CO', 12.64, 0.16, 0.16, J=4)
             co.el('T01', 108, 92, 34, f='d')
             co.el('HD', 14.48, 0.05, 0.05)
             co.el('HD', 14.48, 0.05, 0.05, J=0)
             co.el('HD', '<13.60', J=1)
             #co.el('CIj0', 14.71, 0.08, 0.08)
             #co.el('CIj1', 14.26, 0.07, 0.06)
             #co.el('CIj2', 13.66, 0.06, 0.04)
             co.el('CI', 14.90,0.03,0.03)
             co.el('CIj0', 14.67, 0.04, 0.04)
             co.el('CIj1', 14.46, 0.03, 0.03)
             co.el('CIj2', 13.64, 0.02, 0.02)
             co.el('ClI', 13.01, 0.02, 0.02, b=(4.5, 0.4, 0.4))
             #co.el('n', 1.89, 0.16, 0.18) #Balashev
             co.el('n', 1.65,0.1,0.1)  # Klimenko
             co.el('LFR', '<0.03', f='d')
             co.el('f_cov', '>0.90', f='d')
             co.el('PDRnH',1.27,0.14,0.10)
             co.el('PDRuv',0.99,0.11,0.15)
             co.el('T_co', 12.5, 0.5, 0.5, f='d')
             co.el('P_co', 4.79, 0.10, 0.12)
             co.el('P_ci', 3.75, 0.10, 0.10)
             co.el('T03_co', 10.5, 0.81, 0.62, f='d')
             co.el('Tcmbcorr', 10.4, 0.75, 0.66, f='d')
             co.el('n_ci', 1.19, 0.18, 0.17)
             co.el('Tcmb_ci', '<13.8', f='d')
             co.el('ci_cmb_exc', 0.67)
             s.append(co)
     elif case == 'Klimenko':
         co = sy(2.689517, 0)
         co.el('H2', 19.34, 0.10, 0.10)
         co.el('H2', 19.01, 0.03, 0.03, J=0, b=(7.2,0,0))
         co.el('H2', 18.98, 0.02, 0.02, J=1, b=(7.2,0,0))
         co.el('H2', 18.12, 0.04, 0.04, J=2, b=(7.2,0,0))
         co.el('H2', 18.10, 0.04, 0.04, J=3, b=(7.2,0,0))
         co.el('T01', 193, 134, 56, f='d')
         co.el('CI', 13.88,0.07,0.08)
         co.el('CIj0', 13.50, 0.13, 0.21, b=(4.66, 0.70,0.70))
         co.el('CIj1', 13.53, 0.09, 0.13, b=(4.66, 0.70,0.70))
         co.el('CIj2', 12.99, 0.10, 0.10, b=(4.66, 0,0))
         co.el('ClI', 12.10,0.0, 0, b=(7.2,0,0))
         #co.el('n', 1.0, 0.3, 0.9)
         #co.el('n', '<1.3')
         co.el('LFR', '<0.08', f='d')
         co.el('f_cov', '>0.74', f='d')
         s.append(co)
         co = sy(2.6895698, 0)
         co.el('H2', 19.34, 0.10, 0.10)
         co.el('H2', 19.01, 0.03, 0.03, J=0, b=(7.2, 0, 0))
         co.el('H2', 18.98, 0.02, 0.02, J=1, b=(7.2, 0, 0))
         co.el('H2', 18.12, 0.04, 0.04, J=2, b=(7.2, 0, 0))
         co.el('H2', 18.10, 0.04, 0.04, J=3, b=(7.2, 0, 0))
         #co.el('T01', 193, 134, 56, f='d')
         co.el('T01', 108, 92, 34, f='d') #Noterdaeme fit
         co.el('CI', 15.33,0.30,0.20)
         co.el('CIj0', 15.24, 0.34, 0.27, b=(1.12, 0.14,0.14))
         co.el('CIj1', 14.52, 0.21, 0.10, b=(1.12, 0.14,0.14))
         co.el('CIj2', 13.71, 0.08, 0.11, b=(1.12, 0.14,0.14))
         co.el('ClI', 13.14, 0.0, 0, b=(1.0, 0,0))
         co.el('CO', 14.21, 0.02,0.02)
         co.el('CO', 13.60, 0.03, 0.04, J=0)
         co.el('CO', 13.83, 0.03, 0.03, J=1)
         co.el('CO', 13.63, 0.03, 0.04, J=2)
         co.el('CO', 13.12, 0.04, 0.04, J=3)
         #co.el('CO', '<12.4', J=4)
         co.el('CO', 12.4,0.1,0.1, J=4)
         #co.el('n', 1.87, 0.11, 0.11)
         co.el('LFR', '<0.08', f='d')
         co.el('f_cov', '>0.74', f='d')
         co.el('P_co', 4.38,0.26,0.48)
         #co.el('n_co', 2.01, 0.34, 0.51)
         co.el('T_co', 11.0, 0.2, 0.2, f='d')
         co.el('P_ci', 3.62,0.27,0.26)
         co.el('PDRnH', 1.87, 0.17, 0.13)
         co.el('PDRuv', 0.09, 0.13, 0.14)
         #
         co.el('n_co_pdr', 2.46, 0.24,-0.43)
         co.el('n_ci_pdr', '<1.5')
         #co.el('xco', 21.60,0.10,0.10)
         #co.el('wco', -2.26, 0.03, 0.03)
         co.el('xco', 22.52, 0.10, 0.10)
         co.el('wco', -3.18, 0.03, 0.03)
         co.el('n_co_3dpdr', 2.24, 0.28, 0.61)
         co.el('uv_co_3dpdr', 0.31, 0.37, 0.26)
         co.el('n_ci_3dpdr', '<1.35')
         co.el('uv_ci_3dpdr', 0.09, 0.47, 0.21)
         # pdr estimate in CO region
         co.el('tgas', 2.33, 0.15,0.27)
         co.el('ngas(co)', 1.92, 0.54,0.62)
         co.el('pgas', 4.31, 0.31,0.69)
         co.el('tgas(ci)', 2.42,0.24)
         co.el('ngas(ci)', '<1.34')
         s.append(co)
         co = sy(2.68961400, 0)
         co.el('H2', 19.34, 0.10, 0.10)
         co.el('H2', 19.01, 0.03, 0.03, J=0, b=(7.2, 0, 0))
         co.el('H2', 18.98, 0.02, 0.02, J=1, b=(7.2, 0, 0))
         co.el('H2', 18.12, 0.04, 0.04, J=2, b=(7.2, 0, 0))
         co.el('H2', 18.10, 0.04, 0.04, J=3, b=(7.2, 0, 0))
         co.el('T01', 193, 134, 56, f='d')
         co.el('CIj0', 13.94, 0.06, 0.10, b=(8.33, 0.7,1.0))
         co.el('CIj1', 13.68, 0.06, 0.09, b=(8.33, 0.7,1.0))
         co.el('CIj2', 12.70, 0.13, 0.17, b=(8.33, 0.7,1.0))
         co.el('ClI', 12.69, 0.0, 0,b=(1.88, 0,0))
         #co.el('n', 1.87, 0.11, 0.11)
         co.el('LFR', '<0.08', f='d')
         co.el('f_cov', '>0.74', f='d')
         #co.el('PDRnH', 1.87, 0.17, 0.13)
         #co.el('PDRuv', 0.09, 0.13, 0.14)

         s.append(co)
     elif case == 'model':
         co = sy(2.689517, 0)
         co.el('H2', 19.34, 0.10, 0.10)
         co.el('H2', 19.01, 0.03, 0.03, J=0, b=(7.2, 0, 0))
         co.el('H2', 18.98, 0.02, 0.02, J=1, b=(7.2, 0, 0))
         co.el('H2', 18.12, 0.04, 0.04, J=2, b=(7.2, 0, 0))
         co.el('H2', 18.10, 0.04, 0.04, J=3, b=(7.2, 0, 0))
         co.el('T01', 193, 134, 56, f='d')
         co.el('CI', 13.88, 0.07, 0.08)
         co.el('CIj0', 13.50, 0.13, 0.21, b=(4.66, 0.70, 0.70))
         co.el('CIj1', 13.53, 0.09, 0.13, b=(4.66, 0.70, 0.70))
         co.el('CIj2', 12.99, 0.10, 0.10, b=(4.66, 0, 0))
         co.el('ClI', 12.10, 0.0, 0, b=(7.2, 0, 0))
         # co.el('n', 1.0, 0.3, 0.9)
         # co.el('n', '<1.3')
         co.el('LFR', '<0.08', f='d')
         co.el('f_cov', '>0.74', f='d')
         s.append(co)
         co = sy(2.6895698, 0)
         co.el('H2', 19.34, 0.10, 0.10)
         co.el('H2', 19.01, 0.03, 0.03, J=0, b=(7.2, 0, 0))
         co.el('H2', 18.98, 0.02, 0.02, J=1, b=(7.2, 0, 0))
         co.el('H2', 18.12, 0.04, 0.04, J=2, b=(7.2, 0, 0))
         co.el('H2', 18.10, 0.04, 0.04, J=3, b=(7.2, 0, 0))
         # co.el('T01', 193, 134, 56, f='d')
         co.el('T01', 108, 92, 34, f='d')  # Noterdaeme fit
         co.el('CI', 15.33, 0.30, 0.20)
         co.el('CIj0', 15.24, 0.34, 0.27, b=(1.12, 0.14, 0.14))
         co.el('CIj1', 14.52, 0.21, 0.10, b=(1.12, 0.14, 0.14))
         co.el('CIj2', 13.71, 0.08, 0.11, b=(1.12, 0.14, 0.14))
         co.el('ClI', 13.14, 0.0, 0, b=(1.0, 0, 0))
         co.el('CO', 14.21, 0.01, 0.01)
         co.el('CO', 13.60, 0.01, 0.01, J=0)
         co.el('CO', 13.846, 0.01, 0.01, J=1)
         co.el('CO', 13.596, 0.01, 0.01, J=2)
         co.el('CO', 13.032, 0.01, 0.01, J=3)
         co.el('CO', 12.197, 0.01, 0.01, J=4)
         # co.el('n', 1.87, 0.11, 0.11)
         co.el('LFR', '<0.08', f='d')
         co.el('f_cov', '>0.74', f='d')
         co.el('P_co', 4.38, 0.26, 0.48)
         # co.el('n_co', 2.01, 0.34, 0.51)
         co.el('T_co', 11.0, 0.4, 0.4, f='d')
         co.el('P_ci', 3.62, 0.27, 0.26)
         co.el('PDRnH', 1.87, 0.17, 0.13)
         co.el('PDRuv', 0.09, 0.13, 0.14)
         #
         co.el('n_co_pdr', 2.46, 0.24, -0.43)
         co.el('n_ci_pdr', '<1.5')
         co.el('xco', 22.15, 0.08, 0.08)
         co.el('wco', -2.26, 0.03, 0.03)
         co.el('n_co_3dpdr', 2.46, 0.24, -0.43)
         s.append(co)
         co = sy(2.68961400, 0)
         co.el('H2', 19.34, 0.10, 0.10)
         co.el('H2', 19.01, 0.03, 0.03, J=0, b=(7.2, 0, 0))
         co.el('H2', 18.98, 0.02, 0.02, J=1, b=(7.2, 0, 0))
         co.el('H2', 18.12, 0.04, 0.04, J=2, b=(7.2, 0, 0))
         co.el('H2', 18.10, 0.04, 0.04, J=3, b=(7.2, 0, 0))
         co.el('T01', 193, 134, 56, f='d')
         co.el('CIj0', 13.94, 0.06, 0.10, b=(8.33, 0.7, 1.0))
         co.el('CIj1', 13.68, 0.06, 0.09, b=(8.33, 0.7, 1.0))
         co.el('CIj2', 12.70, 0.13, 0.17, b=(8.33, 0.7, 1.0))
         co.el('ClI', 12.69, 0.0, 0, b=(1.88, 0, 0))
         # co.el('n', 1.87, 0.11, 0.11)
         co.el('LFR', '<0.08', f='d')
         co.el('f_cov', '>0.74', f='d')
         # co.el('PDRnH', 1.87, 0.17, 0.13)
         # co.el('PDRuv', 0.09, 0.13, 0.14)
         s.append(co)
     q.comp = s
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1311+2225
     q = qso('J1311+2225', 3.13661, 3.092)
     q.telescope = 'UVES'
     q.year = 2018
     q.coord = ['J2000', 131129.07, +222552.0]
     q.progID.append('083.A-0454(A)')
     q.SIMBAD = 'SDSS J131129.11+222552.5'
     q.m = {'B': 20.62, 'V': 20.03, 'G': 19.4319, 'u': 23.794, 'g': 20.047, 'r': 19.374, 'i': 19.082, 'z': 19.108}
     q.ref.append('Noterdaeme2018')
     q.el('HI', 20.68, 0.14, 0.14)
     q.el('H2', 19.69, 0.01, 0.01)
     q.el('H', 20.76, 0.12, 0.11)
     q.el('f', -0.77, 0.11, 0.12)
     q.el('Me', -0.5, 0.5, 0.5)
     q.el('EBV',0.047,0.036,0.036, f='d')
     # q.Me_ind = 'Zn'
     q.SDSS = 'J131129.11+222552.5'
     # q.comment = ''
     s = []
     co = sy(3.092, 0)
     co.el('H2', 19.69, 0.02, 0.02)
     co.el('CI', 14.30, 0.02, 0.02)
     co.el('CO', '<13.4')
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B1331+0170
     q = qso('B1331+0170', 2.08, 1.7765)
     q.telescope = 'HST'
     q.year = 2005
     q.coord = ['J2000', 133335.78380, +164904.0330]
     q.SIMBAD = 'QSO B1331+170'
     q.m = {'B': 16.84, 'V': 16.71, 'R': 16.41, 'J': 15.183, 'H': 14.342, 'K': 13.649, 'u': 16.474, 'g': 16.254,
            'r': 16.112, 'i': 15.988, 'z': 15.840}
     q.ref.append('Cui2005')
     q.ref.append('Balashev2010')
     q.ref.append('Carswell2011')
     q.el('HI', 21.17, 0.07, 0.07)
     q.el('H2', 19.71, 0.07, 0.07)
     q.el('HD', 15.03, 0.13, 0.11)
     q.el('H', 21.20, 0.07, 0.07)
     q.el('CI', 13.56, 0.10, 0.10)
     q.el('f', -1.19, 0.10, 0.10)
     q.el('ZnII', 12.61, 0.01, 0.01)
     q.el('CrII', 12.92, 0.02, 0.02)
     q.el('SiII', 15.29, 0.01, 0.01)
     q.el('FeII', 14.60, 0.01, 0.01)
     q.el('NiII', 13.24, 0.01, 0.01)
     q.el('ClI', 12.87, 0.02, 0.02)
     q.el('Me', -1.22, 0.04, 0.04)
     q.Me_ind = 'Zn'
     s = []
     source = 'Carswell'
     if source == 'Balashev':
         co = sy(1.77637, 2)
         co.el('H2', 19.42, 0.10, 0.10)
         co.el('HD', 14.82, 0.15, 0.15)
         co.el('LFR', 0.05, 0.05, 0.05, f='d')
         co.el('f_cov', 0.83, 0.17, 0.17, f='d')
         s.append(co)
         co = sy(1.77670, 2)
         co.el('H2', 19.39, 0.11, 0.11)
         co.el('HD', 14.61, 0.20, 0.20)
         s.append(co)
     elif source =='Carswell':
         co = sy(1.77637, 2)
         co.el('H2', 19.06, 0.30, 0.30)
         co.el('H2', 18.56, 0.28, 0.28, J=0, b=(8.7, 0.5, 0.5))
         co.el('H2', 18.90, 0.22, 0.22, J=1, b=(8.7, 0.5, 0.5))
         co.el('H2', 16.68, 0.24, 0.24, J=2, b=(8.7, 0.5, 0.5))
         co.el('H2', 15.94, 0.14, 0.14, J=3, b=(8.7, 0.5, 0.5))
         co.el('H2', 14.95, 0.06, 0.06, J=4, b=(8.7, 0.5, 0.5))
         co.el('H2', 14.46, 0.07, 0.07, J=5, b=(8.7, 0.5, 0.5))
         co.el('CIj0', 13.08,0.02,0.02, b=(5.75,0.23,0.23))
         co.el('CIj1', 12.59, 0.02, 0.02, b=(5.75, 0.23, 0.23))
         co.el('CIj2', 11.90, 0.14, 0.14, b=(5.75, 0.23, 0.23))
         co.el('n',1.54,0.05,0.05)
         co.el('LFR', '<0.10', f='d')
         co.el('f_cov', '>0.66', f='d')
         co.el('T01', 121, 51,54, f='d')
         co.el('T02', 86, 14,10, f='d')
         #co.el('LFR', 0.00, 0.05, 0.05, f='d')
         #co.el('f_cov', 0.83, 0.17, 0.17, f='d')
         s.append(co)
         co = sy(1.77652, 2)
         #co.el('H2', 19.63, 0.15, 0.15)
         co.el('H2', 18.99, 0.16, 0.16, J=0, b=(1.0, 0.0, 0.0))
         co.el('H2', 19.46, 0.08, 0.08, J=1, b=(1.0, 0.0, 0.0))
         co.el('H2', 18.44, 0.11, 0.11, J=2, b=(1.0, 0.0, 0.0))
         co.el('H2', 18.20, 0.14, 0.14, J=3, b=(1.0, 0.0, 0.0))
         co.el('H2', 14.42, 0.35, 0.35, J=4, b=(1.0, 0.0, 0.0))
         co.el('H2', 13.61, 0.33, 0.33, J=5, b=(1.0, 0.0, 0.0))
         co.el('CIj0', 12.51,0.11,0.11)
         co.el('CIj1', '<12.2')
         co.el('CIj2', '<12.25')
         #co.el('n','<1.82')
         co.el('LFR', '<0.10', f='d')
         co.el('f_cov', '>0.66', f='d')
         s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1337+3152
     q = qso('J1337+3152', 3.173, 3.174441)
     q.telescope = 'VLT'
     q.year = 2010
     q.coord = ['J2000', 133724.693, +315254.734]
     q.SIMBAD = 'QSO J1337+3152'
     q.m = {'B': 19.40, 'V': 19.01, 'u': 22.956, 'g': 18.905, 'r': 18.524, 'i': 18.475, 'z': 18.454}
     q.progID.append('082.A-0544(A)')
     q.progID.append('083.A-0454(A)')
     q.ref.append('Srianand2010')
     q.el('HI', 21.36, 0.10, 0.10)
     q.el('H2', 14.09, 0.03, 0.03)
     q.el('H', 21.36, 0.10, 0.10)
     q.el('f', -6.97, 0.11, 0.11)
     q.el('ZnII', 12.26, 0.26, 0.26)
     q.el('Me', -1.45, 0.22, 0.22)
     q.Me_ind = 'S'
     q.SDSS = 'J133724.69+315254.55'
     q.comment = ''
     s = []
     co = sy(3.174441, 5)
     co.el('H2', 14.09, 0.03, 0.03)
     co.el('H2', 12.88, 0.16, 0.16, J=0)
     co.el('H2', 13.85, 0.04, 0.04, J=1)
     co.el('H2', 13.65, 0.04, 0.04, J=2)
     co.el('H2', '<13.47', J=3)
     s.append(co)
     q.comp = s
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1439+1118
     q = qso('J1439+1118', 2.58, 2.41837)
     q.telescope = 'UVES'
     q.year = 2008
     q.coord = ['J2000', 143912.0420, +111740.485]
     q.SIMBAD = 'QSO J1439+1117'
     q.m = {'B': 19.27, 'V': 18.92, 'J': 17.588, 'H': 16.004, 'K': 15.056, 'u': 20.225, 'g': 18.779, 'r': 18.473,
            'i': 18.330, 'z': 17.827}
     q.progID.append('278.A-5062(A)')
     q.ref.append('Srianand2008')
     q.ref.append('Noterdaeme2008b')
     q.el('HI', 20.10, 0.10, 0.08)
     q.el('H2', 19.38, 0.10, 0.10)
     q.el('H', 20.24, 0.08, 0.06)
     q.el('f', -0.56, 0.12, 0.13)
     q.el('T01', 110, 46, 25, f='d')
     q.el('HD', 14.87, 0.03, 0.03)
     q.el('CO', 13.89, 0.02, 0.02)
     q.el('SII', 15.27, 0.06, 0.06)
     q.el('ZnII', 12.39, 0.04, 0.04)
     q.el('SiII', 14.80, 0.04, 0.04)
     q.el('FeII', 14.28, 0.05, 0.05)
     q.el('CO', 13.27, 0.03, 0.03, J=0)
     q.el('CO', 13.48, 0.02, 0.02, J=1)
     q.el('CO', 13.18, 0.06, 0.06, J=2)
     q.el('CO', 12.96, 0.06, 0.06, J=3)
     q.el('CO', '<12.5', J=4)
     q.el('NI', '>15.71')
     q.el('ClI', '<13.25')
     q.el('Me', 0.16, 0.11, 0.11)
     q.el('EBV', 0.20,0.14,0.14,f='d')
     q.el('Av', 0.54, 0.38, 0.38,f='d') #Ledoux
     q.el('Rv', 2.85, 0., 0,f='d')
     q.Me_ind == 'Zn'
     q.SDSS = 'J143912.04+111740.5'
     s = []
     # this system includes total H2 content, the exact values in each component are not reported in literature.
     case = 'Klimenko'
     if case == 'Noterdaeme':
         co = sy(2.41837, 0)
         co.el('H2', 19.52, 0.07, 0.07)
         co.el('H2', 19.04, 0.06, 0.06, J=0)
         co.el('H2', 19.30, 0.10, 0.10, J=1)
         co.el('H2', 18.02, 0.24, 0.24, J=2)
         co.el('H2', 17.54, 0.30, 0.30, J=3)
         co.el('H2', 16.29, 0.40, 0.40, J=4)
         co.el('CI', 14.90,0.03,0.03)
         co.el('CIj0', 14.26, 0.02, 0.02)
         co.el('CIj1', 14.02, 0.01, 0.01)
         co.el('CIj2', 13.10, 0.02, 0.02)
         #co.el('CIj0', 14.477, 0.03, 0.03)
         #co.el('CIj1', 14.089, 0.028, 0.028)
         #co.el('CIj2', 13.280, 0.23, 0.55)
         co.el('CO', 13.81,0.01,0.01)
         co.el('CO', 13.27, 0.03, 0.03, J=0)
         co.el('CO', 13.48, 0.02, 0.02, J=1)
         co.el('CO', 13.18, 0.06, 0.06, J=2)
         co.el('T01', 107, 22, 15, f='d')
         co.el('T02', 129, 22, 16, f='d')
         co.el('LFR', 0.07, 0.02, 0.02, f='d')
         co.el('f_cov', 0.70, 0.15, 0.15, f='d')
         co.el('n',1.58,0.1,0.1)
         co.el('PDRnH',0.90,0.15,0.18)
         co.el('PDRuv',0.63, 0.19, 0.17)
         co.el('P_co', '<3.5')
         co.el('P_ci', 3.68,0.15,0.15)
         co.el('T_co', 9.1,0.7,0.7, f='d')
         co.el('T03_co', 9.09, 0.85, 0.69)
         co.el('Tcmbcorr', 9.00, 0.87, 0.67)
         co.el('n_ci', 0.98, 0.20, 0.25)
         co.el('Tcmb_ci', '<13.7', f='d')
         co.el('ci_cmb_exc', 0.73)
         s.append(co)
         co = sy(2.41835, 0)
         co.el('HD', 13.89, 0.08, 0.08, b=(2.9, 0.8, 0.8))
         s.append(co)
         co = sy(2.41851, 0)
         co.el('HD', 14.57, 0.04, 0.04, b=(5.0, 1.0, 1.0))
         s.append(co)
         co = sy(2.41866, 0)
         co.el('HD', 14.46, 0.03, 0.03, b=(3.5, 0.3, 0.3))
         s.append(co)
         co = sy(2.41793,0)
         co.el('CIj0', '<12.5')
         s.append(co)
         co = sy(2.4180991,0)
         co.el('CIj0', 13.03,0.03,0.03, b = (3.49,0.3,0.3))
         co.el('CIj1', 12.85,0.04,0.04, b = (3.49,0.3,0.3))
         co.el('CIj2', 11.45,0.22,0.22, b = (3.49,0.3,0.3))
         co.el('n', 1.86, 0.10,0.10)
         #s.append(co)
         co = sy(2.4182606,0)
         co.el('CIj0', 13.78,0.10,0.10, b = (2.45,0.2,0.2))
         co.el('CIj1', 13.18,0.03,0.03, b = (2.45,0.2,0.2))
         co.el('CIj2', 12.77,0.06,0.06, b = (2.45,0.2,0.2))
         #co.el('n', 2.01, 0.11, 0.11)
         s.append(co)
         co = sy(2.4183678,0)
         co.el('H2', 19.15, 0.12, 0.12)
         co.el('CIj0', 14.35,0.15,0.15, b = (2.12,0.12,0.12))
         co.el('CIj1', 13.96,0.05,0.05, b = (2.12,0.12,0.12))
         co.el('CIj2', 13.20,0.04,0.04, b = (2.12,0.12,0.12))
         co.el('n', 1.74, 0.12, 0.12)
         co.el('LFR', 0.07, 0.02, 0.02, f='d')
         co.el('f_cov', 0.70, 0.15, 0.15, f='d')
         #s.append(co)
         co = sy(2.4185158,0)
         co.el('H2', 18.34, 0.35, 0.35)
         co.el('CIj0', 14.05,0.07,0.07, b = (3.42,0.15,0.15))
         co.el('CIj1', 13.65,0.02,0.02, b = (3.42,0.15,0.15))
         co.el('CIj2', 13.05,0.04,0.04, b = (3.42,0.15,0.15))
         co.el('n', 1.84, 0.09, 0.09)
         co.el('LFR', 0.07, 0.02, 0.02, f='d')
         co.el('f_cov', 0.70, 0.15, 0.15, f='d')
         #s.append(co)
         co = sy(2.4186494,0)
         co.el('H2', 19.09, 0.03, 0.03)
         co.el('CIj0', 13.63,0.09,0.09, b = (2.38,0.24,0.24))
         co.el('CIj1', 13.21,0.02,0.02, b = (2.5,0.24,0.24))
         co.el('CIj2', 12.48,0.07,0.06, b = (2.5,0.24,0.24))
         co.el('n', 1.71, 0.12, 0.12)
         co.el('LFR', '<0.10', f='d')
         co.el('f_cov', '>0.85', f='d')
         #s.append(co)
     elif case == 'Klimenko':
         co = sy(2.41837, 0)
         co.el('H2', 19.52, 0.07, 0.07)
         co.el('H2', 19.04, 0.06, 0.06, J=0)
         co.el('H2', 19.30, 0.10, 0.10, J=1)
         co.el('H2', 18.02, 0.24, 0.24, J=2)
         co.el('H2', 17.54, 0.30, 0.30, J=3)
         co.el('H2', 16.29, 0.40, 0.40, J=4)
         co.el('CI', 14.57,0.02,0.02)
         co.el('CIj0', 14.38, 0.03, 0.03)
         co.el('CIj1', 14.04, 0.03, 0.03)
         co.el('CIj2', 13.26, 0.03, 0.03)
         co.el('CO', 13.84, 0.02,0.02)
         co.el('CO', 13.22, 0.03, 0.03, J=0)
         co.el('CO', 13.45, 0.03, 0.03, J=1)
         co.el('CO', 13.17, 0.04, 0.04, J=2)
         co.el('CO', 12.96, 0.06, 0.06, J=3)
         co.el('CO', '<12.6', J=4)
         #co.el('CO', 11.5, 0.9, 1.1, J=4)
         co.el('T01', 107, 33, 20, f='d')
         co.el('LFR', 0.07, 0.02, 0.02, f='d')
         co.el('f_cov', 0.70, 0.15, 0.15, f='d')
         co.el('n', 1.58, 0.1, 0.1)
         co.el('PDRnH', 0.90, 0.15, 0.18)
         co.el('PDRuv', 0.63, 0.19, 0.17)
         co.el('P_co', 4.72,0.12,0.14)
         #co.el('n_co', 2.64, 0.16, 0.16)
         co.el('P_ci', 3.71,0.05,0.05)
         co.el('T_co', 12.05,0.6,0.6, f='d')
         #
         co.el('n_co_pdr', 2.87,0.3,0.59)
         co.el('n_ci_pdr', 1.58, 0.2, 0.2)
         co.el('n_co_3dpdr', 2.55, 0.16, 0.24)
         co.el('uv_co_3dpdr', 0.43, 0.22, 0.25)
         co.el('n_ci_3dpdr', 1.70, 0.14, 0.15)
         co.el('uv_ci_3dpdr', -0.11, 0.26, 0.20)
         #co.el('xco', 22.15,0.08,0.08)
         #co.el('wco', -2.63, 0.03, 0.03)
         co.el('xco', 22.66, 0.08, 0.08)
         co.el('wco', -3.14, 0.03, 0.03)
         co.el('n_co_3dpdr', 2.87, 0.3, 0.59)
         # pdr estimate in CO region
         co.el('tgas', 2.19, 0.10,0.12)
         co.el('ngas(co)', 2.33, 0.25,0.30)
         co.el('pgas', 4.52, 0.29,0.37)
         co.el('ngas(ci)', 1.52, 0.22,0.15)
         co.el('tgas(ci)', 2.19, 0.15,0.13)
         s.append(co)

     q.comp = s
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1443+2724 or B1441+2737
     q = qso('J1443+2724', 4.42, 4.224)
     q.telescope = 'UVES'
     q.year = 2006
     q.coord = ['J2000', 144331.170, +272436.79]
     q.SIMBAD = 'QSO J1443+2724'
     q.m = {'V': 19.3, 'u': 25.584, 'g': 22.099, 'r': 19.645, 'i': 19.012, 'z': 18.82}
     q.progID.append('60.A-9022(A)')
     q.progID.append('090.A-0304(A)')
     q.progID.append('077.A-0148(A)')
     q.progID.append('072.A-0346(B)')
     q.ref.append('Ledoux2006')
     q.ref.append('Noterdaeme2008')
     q.el('HI', 20.95, 0.10, 0.10)
     q.el('H2', 18.29, 0.07, 0.07)
     q.el('H', 20.95, 0.10, 0.10)
     q.el('f', -2.36, 0.12, 0.13)
     q.el('Me', -0.63, 0.10, 0.10)
     q.Me_ind = 'Zn'
     s = []
     co = sy(4.22371, 0)
     co.el('H2', 17.91, 0.02, 0.03)
     co.el('H2', 17.41, 0.02, 0.05, J=0)
     co.el('H2', 17.59, 0.03, 0.03, J=1)
     co.el('H2', 17.15, 0.05, 0.13, J=2)
     co.el('H2', 16.53, 0.17, 0.91, J=3)
     co.el('H2', '<13.6', J=4)
     co.el('T01', 96, 10, 3, f='d')
     co.el('T02', 230, 15, 29, f='d')
     co.el('ClI', '<12.66')
     co.el('LFR', '<0.03', f='d')
     co.el('f_cov', '>0.90', f='d')
     co.el('PDRnH',0.72,0.34,0.72)
     co.el('PDRuv',-0.37, 0.40, 0.16)
     s.append(co)
     co = sy(4.22401, 0)
     co.el('H2', 18.05, 0.05, 0.05)
     co.el('H2', 17.34, 0.09, 0.07, J=0)
     co.el('H2', 17.75, 0.06, 0.08, J=1)
     co.el('H2', 17.25, 0.13, 0.14, J=2)
     co.el('H2', 17.21, 0.13, 0.13, J=3)
     co.el('H2', 14.02, 0.15, 0.07, J=4)
     co.el('T01', 136, 20, 27, f='d')
     co.el('T02', 281, 56, 51, f='d')
     co.el('ClI', '<12.86')
     co.el('LFR', '<0.03', f='d')
     co.el('f_cov', '>0.90', f='d')
     #co.el('PDRnH',2.94,0.37,1.53)
     #co.el('PDRuv',1.71, 0.21, 0.74)
     s.append(co)
     co = sy(4.22416, 0)
     co.el('H2', 15.18, 0.03, 0.07)
     co.el('H2', 14.34, 0.01, 0.12, J=0)
     co.el('H2', 14.86, 0.04, 0.13, J=1)
     co.el('H2', 14.61, 0.06, 0.15, J=2)
     co.el('H2', 14.18, 0.01, 0.16, J=3)
     co.el('H2', '<13.6', J=4)
     co.el('T01', 171, 118, 51, f='d')
     #s.append(co)
     q.comp = s
     q.SDSS = 'J144331.17+272436.7'
     q.comment = 'old name is B1441+2737'
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B1444+0126
     q = qso('B1444+0126', 2.21, 2.087)
     q.telescope = 'UVES'
     q.year = 2003
     q.coord = ['J2000', 144653.0, +011355.]
     q.SIMBAD = 'LBQS 1444+0126'
     q.m = {'B': 18.71, 'V': 18.38, 'u': 19.276, 'g': 18.575, 'r': 18.453, 'i': 18.335, 'z': 18.132}
     q.progID.append('67.A-0078(A)')
     q.progID.append('69.B-0108(A)')
     q.progID.append('65.O-0158(A)')
     q.progID.append('71.B-0136(A)')
     q.ref.append('Ledoux2003')
     q.ref.append('Noterdaeme2008')
     q.el('HI', 20.25, 0.07, 0.07)
     q.el('H2', 18.16, 0.14, 0.12)
     q.el('H', 20.26, 0.07, 0.07)
     q.el('f', -1.79, 0.16, 0.14)
     q.el('ClI', '<12.42')
     q.el('Me', -0.8, 0.09, 0.09)
     q.el('Fe/X', -0.95, 0.07, 0.07)
     q.el('CI', 13.18,0.04,0.04)
     q.Me_ind = 'Zn'
     s = []
     co = sy(2.08680, 0)
     co.el('H2', 16.49, 0.23, 0.09)
     co.el('H2', 15.68, 0.22, 0.08, J=0)
     co.el('H2', 16.37, 0.27, 0.11, J=1)
     co.el('H2', 15.31, 0.06, 0.03, J=2)
     co.el('H2', 14.80, 0.06, 0.04, J=3)
     co.el('H2', '<14.15', J=4)
     co.el('H2', '<13.75', J=5)
     co.el('T01', 280, np.Inf, 159, f='d')
     co.el('LFR', '<0.08', f='d')
     co.el('f_cov', '>0.74', f='d')
     s.append(co)
     co = sy(2.08696, 0)
     co.el('H2', 18.15, 0.10, 0.09)
     co.el('H2', 17.46, 0.10, 0.11, J=0)
     co.el('H2', 18.03, 0.12, 0.12, J=1)
     co.el('H2', 16.63, 0.57, 0.46, J=2)
     co.el('H2', 15.26, 0.07, 0.08, J=3)
     co.el('H2', '<14.20', J=4)
     co.el('H2', '<13.75', J=5)
     co.el('T01', 193, 282, 70, f='d')
     co.el('T02', 145, 96, 33, f='d')
     co.el('CIj0', 12.82, 0.11, 0.11)
     co.el('CIj1', 12.42, 0.12, 0.12)
     co.el('CIj2', 12.78, 0.20, 0.20)
     co.el('n_ci', 2.16, 0.27, 0.26)
     co.el('Tcmb_ci', '<10.5', f='d')
     co.el('LFR', '<0.08', f='d')
     co.el('f_cov', '>0.74', f='d')
     co.el('PDRnH',1.94,0.26,0.21)
     co.el('PDRuv',0.49, 0.16, 0.19)

     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1456+1609
     q = qso('J1456+1609', 3.68, 3.35)
     q.telescope = 'VLT'
     q.year = 2015
     q.coord = ['J2000', 145646.48, +160939.3]
     q.SIMBAD = '[VV2010] J145646.5+160939'
     q.m = {'B': 20.89, 'V': 20.04, 'u': 23.342, 'g': 20.333, 'r': 19.191, 'i': 19.017, 'z': 18.95}
     q.progID.append('091.A-0370(A)')
     q.ref.append('Noterdaeme2015')
     # col = a.dec(np.array([15.21,0.07, 0.07])
     # col2 = a.dec(np.array([21.26,0.07, 0.07])
     # print(a.logg(a.ratio(col, col2))
     q.el('HI', 21.70, 0.10, 0.10)
     q.el('H2', 17.10, 0.09, 0.09)
     q.el('H', 21.70, 0.10, 0.10)
     q.el('f', -4.30, 0.14, 0.14)
     q.el('SiII', 15.81, 0.05, 0.05)
     q.el('CrII', 13.75, 0.04, 0.04)
     q.el('FeII', 15.39, 0.06, 0.06)
     q.el('NiII', 14.09, 0.03, 0.03)
     q.el('ZnII', 12.94, 0.04, 0.04)
     q.el('Me', -1.32, 0.11, 0.11)
     q.Me_ind = 'Zn'
     q.SDSS = 'J145646.48+160939.3'
     s = []
     co = sy(3.351829, 0)
     co.el('H2', 17.10, 0.09, 0.09)
     co.el('H2', 16.09, 0.26, 0.26, J=0)
     co.el('H2', 16.75, 0.05, 0.05, J=1)
     co.el('H2', 16.29, 0.10, 0.10, J=2)
     co.el('H2', 16.49, 0.06, 0.06, J=3)
     co.el('H2', 15.78, 0.16, 0.16, J=4)
     co.el('T01', 252, np.Inf, 140, f='d')
     co.el('T02', 440, 160, 195, f='d')
     s.append(co)
     q.comp = s
     # q.comment = ''
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1513+0352
     q = qso('J1513+0352', 2.684, 2.4636)
     q.telescope = 'XShooter'
     q.year = 2017
     q.ref.append('Ranjan in prep.')
     q.coord = ['J2000', 151349.52, +035211.64]
     q.SIMBAD = '[VV2006] J151349.5+035211'
     q.m = {'B': 21.51, 'V': 20.98, 'J': 18.74, 'H': 19.481, 'K': 17.54, 'u': 23.46, 'g': 21.107, 'r': 20.55,
            'i': 20.40, 'z': 19.961}
     # q.progID.append('xxx')
     # q.ref.append('xxx')
     q.el('HI', 21.82, 0.02, 0.02)
     q.el('H2', 21.31, 0.02, 0.02)
     q.el('H', 22.03, 0.01, 0.01)
     q.el('f', -0.42, 0.03, 0.02)
     q.el('HD', 17.28, 0.18, 0.18)
     q.el('CI', 15.02,0.02,0.03)
     # q.el('HD', 17.34, 0.13, 0.37, J=0)
     # q.el('HD', 15.87, 0.72, 0.49, J=1)
     # q.el('CO', '<13.40')
     # q.el('n', 2.58, 0.03, 0.06)
     q.el('Me', -1.22, 0.08, 0.10)
     q.el('EBV', 0.43/3,0.11/3,0.11/3,f='d')
     q.Me_ind = 'Zn'
     q.SDSS = 'J151349.52+035211.64'
     s = []
     co = sy(2.463622, 0)
     co.el('H2', 21.31, 0.01, 0.01)
     co.el('H2', 20.97, 0.02, 0.02, J=0)
     co.el('H2', 21.03, 0.02, 0.02, J=1)
     co.el('H2', 19.25, 0.04, 0.04, J=2)
     co.el('H2', 18.94, 0.04, 0.04, J=3)
     co.el('H2', 18.20, 0.09, 0.09, J=4)
     co.el('H2', 16.36, 0.24, 0.24, J=5)
     co.el('T01', 82, 3, 3, f='d')
     co.el('T02', 91, 2, 2, f='d')
     co.el('CO', '<13.6')
     # co.el('CI*', 14.60, 0.06, 0.06)
     # co.el('CI**', 14.03, 0.06, 0.06)
     co.el('CIj0', 14.78, 0.08, 0.08)
     co.el('CIj1', 14.57, 0.06, 0.06)
     co.el('CIj2', 13.88, 0.04, 0.04)
     co.el('n', 2.07, 0.11, 0.10)
     co.el('HD', 16.40, 0.23, 0.23)
     co.el('HD', 16.36, 0.24, 0.24, J=0)
     co.el('HD', 15.30, 0.23, 0.23, J=1)
     co.el('PDRnH',1.90,0.13,0.12)
     #co.el('PDRnH', 2.35, 0.13, 0.12)
     co.el('PDRuv',0.45, 0.24, 0.20)
     co.el('n_ci', 1.95, 0.16, 0.36)
     co.el('Tcmb_ci', '<12', f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1604+2057
     q = qso('J1604+2203', 1.98, 1.64)
     q.telescope = 'UVES'
     q.year = 2011
     q.coord = ['J1604', '1604+2203']
     q.SIMBAD = 'SDSS J1604+2203'
     q.ref.append('Noterdaeme2009')
     #q.el('H2', '<21')
     q.el('CI', '>15.14')
     q.el('CO', 13.89, 0.02, 0.02)
     q.el('EBV', 0.27, 0.02, 0.02, f='d')
     q.el('Av', 0.72, 0.05, 0.05, f='d')
     q.el('Rv', 2.7, 0., 0, f='d')
     q.comment = ''
     s = []
     co = sy(1.63967, 0)
     co.el('SI', 12.32,0.20,0.30)
     s.append(co)
     co = sy(1.6399247, 0)
     co.el('SI', 13.15,0.03,0.03)
     s.append(co)
     co = sy(1.640296, 0)
     co.el('SI', 12.60,0.10,0.10)
     s.append(co)
     co = sy(1.640498, 0)
     co.el('SI', 13.03, 0.04, 0.04)
     co.el('CO', 13.97, 0.05, 0.05)
     co.el('COj0', 13.18, 0.06, 0.11, b=(6.9, 0.7, 0.8))
     co.el('COj1', 13.59, 0.07, 0.11)
     co.el('COj2', 13.60, 0.08, 0.10)
     s.append(co)
     co = sy(1.6407145, 0)
     co.el('SI', 15.06, 0.10, 0.15)
     co.el('CO', 14.65, 0.13, 0.10)
     co.el('COj0', 13.97, 0.20, 0.20, b=(0.5, 0.1, 0.1))
     co.el('COj1', 14.40, 0.20, 0.17)
     co.el('COj2', 14.01, 0.10, 0.10)
     s.append(co)
     co = sy(1.6408467, 0)
     co.el('SI', 13.18, 0.05, 0.05)
     co.el('CO', 14.47, 0.05, 0.05)
     co.el('COj0', 14.00, 0.05, 0.07, b=(2.7, 0.2, 0.2))
     co.el('COj1', 14.16, 0.03, 0.03)
     co.el('COj2', 13.34, 0.06, 0.08)
     co.el('COj3', 13.46, 0.06, 0.04)
     #co.el('H2', 21.00, 1.0, 1.0)
     co.el('CI', '>15.14')
     co = sy(1.6408467, 0)
     co.el('SI', 13.18, 0.05, 0.05)
     co.el('CO', 14.47, 0.05, 0.05)
     co.el('COj0', 14.23, 0.05, 0.07, b=(2.7, 0.2, 0.2))
     co.el('COj1', 14.18, 0.03, 0.03)
     co.el('COj2', 13.72, 0.06, 0.08)
     co.el('COj3', 13.65, 0.06, 0.04)
     s.append(co)
     q.comp = s
     q.full = 'u'
     QSO.append(q)


     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1646+2329
     q = qso('J1646+2329', 2.05715, 1.998)
     q.telescope = 'UVES'
     q.year = 2018
     q.coord = ['J2000', 164610.20, +232923.5]
     q.progID.append('082.A-0544(A)')
     q.SIMBAD = '[VV2006] J164610.2+232922'
     q.m = {'B': 19.03, 'V': 18.76, 'G': 18.3740, 'u': 18.858, 'g': 18.518, 'r': 18.341, 'i': 18.171, 'z': 17.947}
     q.ref.append('Noterdaeme2018')
     q.el('HI', 19.786, 0.020, 0.020)
     q.el('H2', 18.02, 0.11, 0.11)
     q.el('H', 19.80, 0.02, 0.02)
     q.el('f', -1.48, 0.11, 0.11)
     q.el('Me', -0.5, 0.5, 0.5)
     q.el('EBV', 0.02, 0.04, 0.04, f='d')
     q.el('CI', 14.22, 0.06, 0.06)
     q.el('CO', '<13.4')
     # q.Me_ind = 'Zn'
     q.SDSS = 'J164610.20+232922.9'
     s = []
     co = sy(1.998, 0)
     # co.el('H2', '<21')
     co.el('CI', 14.22, 0.06, 0.06)
     co.el('H2', 18.02, 0.11, 0.11)
     co.el('CO', '<13.4')
     s.append(co)
     q.comp = s
     # q.comment = ''
     q.full = 'n'
     QSO.append(q)

     # add J1705+3543
     q = qso('J1705+3543', 2.01, 2.038)
     q.telescope = 'UVES'
     q.year = 2017
     q.ref.append('Noterdaeme 2011')
     q.el('f', -0.42, 0.03, 0.02)
     q.el('CI', 15.02, 0.02, 0.03)
     q.el('CO', 14.12, 0.06,0.05)
     q.el('Rv', 2.75, 0, 0, f='d')
     q.el('EBV', 0.154,0.075,0.075, f = 'd')
     q.el('Av', 0.42, 0.20, 0.20, f='d')
     #q.el('H2','<21')
     #q.SDSS = 'J151349.52+035211.64'
     s = []
     co = sy(2.038, 0)
     #co.el('H2', '<21')
     co.el('CI', 14.89, 0.04, 0.04)
     co.el('CIj0', 14.57, 0.07, 0.07)
     co.el('CIj1', 14.49, 0.05, 0.05)
     co.el('CIj2', 13.98, 0.05, 0.05)
     co.el('CO', 14.12, 0.06, 0.05)
     co.el('COj0', 13.465, 0.07, 0.07)
     co.el('COj1', 13.747, 0.09, 0.09)
     co.el('COj2', 13.372, 0.09, 0.09)
     co.el('COj3', 13.365, 0.16, 0.16)
     co.el('T_co', 12.4, 1.5, 1.5, f='d')
     co.el('P_co', 4.93, 0.20, 0.20)
     #co.el('n_co', 2.91, 0.26, 0.23)
     ##################################### results with new version PDR:
     co.el('n_co_3dpdr', '>3')
     co.el('uv_co_3dpdr', -0.19,0.22,0.16)
     co.el('n_ci_3dpdr', '>3')
     co.el('tgas',1.81,0.12,0.12)
     co.el('ngas(co)', 3.58, 0.20, 0.89)
     co.el('tgas(ci)', 1.81,0.12)
     co.el('ngas(ci)', 3.58, 0.20, 0.89)
     co.el('wco', -2.7, 0.09, 0.08)
     #co.el('n_co_3dpdr', 2.95, 0.20, 0.25)
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)


     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2100-0641
     q = qso('J2100-0641', 3.14, 3.09145)
     q.coord = ['J2000', 210025.0293, -064145.987]
     q.telescope = 'KECK'
     q.year = 2015
     q.SIMBAD = 'QSO J2100-0641'
     q.m = {'B': 19.02, 'V': 18.54, 'J': 16.761, 'H': 16.385, 'K': 15.753, 'u': 21.61, 'g': 18.740, 'r': 18.172,
            'i': 17.996, 'z': 17.969}
     q.ref.append('Balashev2015')
     q.ref.append('Ivanchik2015')
     q.ref.append('Jorgenson2010')
     q.el('HI', 21.05, 0.15, 0.15)
     q.el('H2', 18.76, 0.03, 0.03)
     q.el('H', 21.05, 0.15, 0.15)
     q.el('f', -1.99, 0.15, 0.16)
     q.el('HD', 13.83, 0.06, 0.06)
     q.el('CI', 13.17, 0.10, 0.10)
     q.el('ClI', '<12.86')
     q.el('ZnII', '<13.14')
     q.el('Me', -0.73, 0.15, 0.15)
     q.Me_ind = 'Si'
     q.SDSS = 'J210025.03-064145.9'
     s = []
     co = sy(3.09145, 0)
     co.el('H2', 18.76, 0.03, 0.03)
     co.el('H2', 18.15, 0.06, 0.06, J=0)
     co.el('H2', 18.64, 0.04, 0.04, J=1)
     co.el('H2', 16.19, 0.09, 0.09, J=2)
     co.el('H2', 16.33, 0.06, 0.06, J=3)
     co.el('H2', 15.16, 0.02, 0.02, J=4)
     co.el('H2', 14.85, 0.01, 0.01, J=5)
     co.el('T01', 159, 29, 21, f='d')
     co.el('T02', 83, 35, 33, f='d')
     co.el('CIj0', 12.57, 0.03, 0.03)
     co.el('CIj1', 12.31, 0.08, 0.08)
     co.el('n', 1.49,0.38,0.81)# Klimenko
     co.el('n_ci', 2.02, 0.15, 0.93)
     co.el('Tcmb_ci', 12.9, 3.3, 4.5, f='d')
     #co.el('n', 2.11, 0.34, 0.66) #Balashev
     co.el('LFR', 0.09, 0.02, 0.02, f='d')
     co.el('f_cov', 0.73, 0.10, 0.10, f='d')
     co.el('PDRnH',1.40,0.28,0.35)
     co.el('PDRuv',-0.50, 0.22, 0.34)
     s.append(co)
     q.comp = s
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2123-0050
     q = qso('J2123-0050', 2.26, 2.059)
     q.telescope = 'KECK/UVES'
     q.year = 2009
     q.coord = ['J2000', 212329.4674, -005052.897]
     q.SIMBAD = 'QSO J2123-0050'
     q.m = {'B': 16.91, 'V': 16.62, 'R': 16.200, 'J': 15.180, 'H': 14.616, 'K': 13.904, 'u': 17.160, 'g': 16.648,
            'r': 16.434, 'i': 16.338, 'z': 16.121}
     q.progID.append('081.A-0242(A)')
     q.ref.append('Malec2009')
     q.ref.append('Tumlinson2009')
     q.ref.append('Milutinovic2010')
     q.el('HI', 19.18, 0.15, 0.15)
     q.el('H2', 17.94, 0.01, 0.01)
     q.el('H', 19.23, 0.14, 0.13)
     q.el('f', -1.26, 0.21, 0.21)
     q.el('HD', 13.84, 0.20, 0.20)
     q.el('SII', 14.70, 0.02, 0.02)
     q.el('SiII', 14.69, 0.02, 0.02)
     q.el('FeII', 14.12, 0.02, 0.02)
     q.el('NI', 14.53, 0.02, 0.02)
     q.el('ClI', 12.27, 0.06, 0.06)
     q.el('Me', -0.19, 0.15, 0.15)
     q.el('CI', 13.96,0.02,0.02)
     q.el('CO', '<13.07')
     q.el('EBV', 0,0.04,0.04, f='d')
     q.Me_ind = 'S'
     q.comment = 'metallicity corrected for ionization, see Milutinovic2011'
     q.comp = []
     co = sy(2.05930, 0)
     co.el('H2', 17.94, 0.01, 0.01)
     co.el('H2', 17.37, 0.02, 0.02, J=0, b=(1.71, 0.04, 0.04))
     co.el('H2', 17.79, 0.01, 0.01, J=1, b=(2.07, 0.05, 0.05))
     co.el('H2', 16.00, 0.03, 0.03, J=2, b=(2.97, 0.05, 0.05))
     co.el('H2', 15.54, 0.02, 0.02, J=3, b=(3.56, 0.06, 0.06))
     co.el('H2', 14.16, 0.01, 0.01, J=4, b=(4.64, 0.23, 0.23))
     co.el('H2', 13.75, 0.03, 0.03, J=5, b=(5.10, 0.51, 0.51))
     co.el('T01', 139, 6, 6, f='d')
     co.el('T02', 107, 2, 2, f='d')
     co.el('HD', 13.87, 0.06, 0.06)
     co.el('CIj0', 13.84, 0.03, 0.03)
     co.el('CIj1', 13.27, 0.02, 0.02)
     co.el('CIj2', 12.63, 0.04, 0.04)
     co.el('n', 1.53, 0.04, 0.05)
     co.el('SI', 12.08, 0.05, 0.05)
     co.el('ClI', 12.27, 0.06, 0.06, b=(2.6, 0.5, 0.5))
     co.el('LFR', 0.027, 0.005, 0.005, f='d')
     co.el('f_cov', 0.90, 0.04, 0.04, f='d')
     co.el('PDRnH',0.95,0.13,0.10)
     co.el('PDRuv',-0.41, 0.17, 0.16)
     co.el('CO', '<13.07')
     q.comp.append(co)
     co = sy(2.05955, 0)
     co.el('H2', 15.16, 0.02, 0.02)
     co.el('H2', 14.00, 0.02, 0.02, J=0, b=(5.18, 0.51, 0.51))
     co.el('H2', 14.84, 0.01, 0.01, J=1, b=(5.24, 0.11, 0.11))
     co.el('H2', 14.41, 0.01, 0.01, J=2, b=(5.02, 0.18, 0.18))
     co.el('H2', 14.50, 0.01, 0.01, J=3, b=(4.71, 0.30, 0.30))
     co.el('H2', 13.84, 0.03, 0.03, J=4, b=(7.86, 0.57, 0.57))
     co.el('H2', 13.50, 0.06, 0.06, J=5, b=(7.36, 0.80, 0.80))
     co.el('T01', 648, 247, 140, f='d')
     co.el('CIj0', 12.71, 0.03, 0.03)
     co.el('CIj1', 12.43, 0.04, 0.04)
     co.el('CIj2', 11.90, 0.18, 0.18)
     co.el('n', 1.83, 0.14, 0.12)
     co.el('LFR', '<0.04', f='d')
     co.el('f_cov', '>0.86', f='d')
     q.comp.append(co)
     # Slava data
     q.full = 'p'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2140-0321
     q = qso('J2140-0321', 2.48, 2.34)
     q.telescope = 'VLT'
     q.year = 2015
     q.coord = ['J2000', 214043.02, -032139.2]
     q.SIMBAD = 'SDSS J214043.02-032139.2'
     q.m = {'u': 21.443, 'g': 20.306, 'r': 19.724, 'i': 19.475, 'z': 19.09}
     q.progID.append('091.A-0370(A)')
     q.ref.append('Noterdaeme2015')
     # col = a.dec(np.array([15.21,0.07, 0.07])
     # col2 = a.dec(np.array([21.26,0.07, 0.07])
     # print(a.logg(a.ratio(col, col2))
     q.el('HI', 22.40, 0.10, 0.10)
     q.el('H2', 20.13, 0.07, 0.07)
     q.el('H', 22.40, 0.10, 0.10)
     q.el('f', -1.97, 0.12, 0.13)
     q.el('T01', 75, 12, 9, f='d')
     q.el('n', 3.01, 0.20, 0.14)
     q.el('CO', '<13.73')
     q.CI = e('CI', 13.57, 0.03, 0.03)
     q.el('Me', -1.05, 0.13, 0.13)
     q.Me_ind = 'P'
     q.el('ClI', 13.67, 0.15, 0.15)
     q.el('CI', 13.57, 0.03, 0.03)
     q.el('CIj0', 13.03, 0.04, 0.04)
     q.el('CIj1', 13.20, 0.04, 0.04)
     q.el('CIj2', 13.02, 0.05, 0.05)
     q.el('PII', 14.76, 0.08, 0.08)
     q.el('TiII', 13.26, 0.05, 0.05)
     q.el('FeII', 15.64, 0.03, 0.03)
     q.el('NiII', 14.40, 0.05, 0.05)
     q.el('ZnII', 13.72, 0.92, 0.92)
     q.SDSS = 'J214043.02-032139.2'
     s = []
     co = sy(2.33990, 0)  # redshift of J=0 rot level, see Noterdaeme2015
     co.el('H2', 20.13, 0.07, 0.07)
     co.el('H2', 19.84, 0.09, 0.09, J=0)
     co.el('H2', 19.81, 0.04, 0.04, J=1)
     co.el('H2', 17.96, 0.14, 0.14, J=2)
     co.el('H2', 17.76, 0.40, 0.40, J=3)
     co.el('H2', 15.88, 0.26, 0.26, J=4)
     co.el('H2', 15.17, 0.16, 0.16, J=5)
     co.el('H2', '<14.72', J=6)
     co.el('T01', 75, 9, 6, f='d')
     co.el('T02', 85, 6, 5, f='d')
     co.el('CIj0', 13.03, 0.04, 0.04)
     co.el('CIj1', 13.20, 0.04, 0.04)
     co.el('CIj2', 13.02, 0.05, 0.05)
     co.el('n', 3.01, 0.20, 0.14)
     co.el('OI', '<17.9')
     co.el('OI*', 13.89, 0.11, 0.11)
     co.el('OI**', 13.85, 0.08, 0.08)
     co.el('SiII', '>16.16')
     co.el('SiII*', 12.81, 0.03, 0.03)
     co.el('PDRnH',2.42,0.11,0.09)
     co.el('PDRuv',1.64, 0.19, 0.21)
     co.el('n_ci', 2.94, 0.23, 0.18)
     co.el('Tcmb_ci', '<20', f='d')
     s.append(co)
     q.comp = s
     q.comment = 'Note that Noterdaeme2015 used different z for rot levels of H2'
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2225+0527
     q = qso('J2225+0527', 2.32, 2.13123)
     q.telescope = 'XShooter'
     q.year = 2016
     q.coord = ['J2000', 222514.97, 052709.1]
     q.SIMBAD = '4C 05.84'
     q.m = {'B': 18.400, 'V': 18.52, 'R': 17.600, 'J': 15.813, 'H': 15.087, 'K': 14.249, 'u': 20.66, 'g': 18.843,
            'r': 18.106, 'i': 17.599, 'z': 17.096}
     q.ref.append('Krogager2016')
     q.el('HI', 20.69, 0.05, 0.05)
     q.el('H2', 19.4, 0.10, 0.10)
     q.el('H', 20.73, 0.05, 0.05)
     q.el('f', -4, 0.06, 0.06)
     q.el('SiII', 15.49, 0.04, 0.04)
     q.el('SII', 15.41, 0.02, 0.02)
     q.el('CrII', 13.04, 0.08, 0.08)
     q.el('MnII', 12.91, 0.02, 0.02)
     q.el('FeII', 14.87, 0.01, 0.01)
     q.el('NiII', 13.90, 0.07, 0.07)
     q.el('ZnII', 13.16, 0.02, 0.02)
     q.el('ClI', 13.34, 0.03, 0.03)
     q.el('MgI', 12.93, 0.02, 0.02)
     q.el('Av', 0.39, 0.13, 0.10, f='d')
     q.el('Me', -0.09, 0.05, 0.05)
     q.Me_ind = 'Zn'
     s = []
     co = sy(2.13123, 0)  # redshift of J=0 rot level, see Noterdaeme2015
     co.el('CI', 13.86, 0.04, 0.04)
     co.el('CI*', 13.80, 0.04, 0.04)
     co.el('CI**', 12.98, 0.11, 0.11)
     q.comp = s
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2257-1001
     q = qso('J2257-1001', 2.07967, 1.836)
     q.telescope = 'UVES'
     q.year = 2018
     q.coord = ['J2000', 225719.03, -100104.0]
     q.progID.append('081.A-0334(B)')
     q.SIMBAD = '[VV2006] J225719.1-100105'
     q.m = {'B': 18.85, 'V': 18.57, 'G': 17.7541, 'u': 18.846, 'g': 18.541, 'r': 18.340, 'i': 18.061, 'z': 17.796}
     q.ref.append('Noterdaeme2018')
     q.el('HI', 20.415, 0.009, 0.009)
     q.el('H2', 19.50, 0.10, 0.10)
     q.el('H', 20.51, 0.02, 0.02)
     q.el('f', -0.71, 0.10, 0.10)
     q.el('Me', -0.5, 0.5, 0.5)
     q.el('EBV',0.052,0.039,0.039, f='d')
     q.el('CI', 14.65, 0.02,0.02)
     q.el('CO', '<13.09')
     # q.Me_ind = 'Zn'
     q.SDSS = 'J225719.04-100104.7'
     # q.comment = ''
     s = []
     co = sy(1.836, 0)
     co.el('CI', 14.65, 0.02, 0.02)
     co.el('CO', '<13.09')
     co.el('H2', 19.50, 0.10, 0.10)
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B2318-1107
     q = qso('B2318-1107', 2.96, 1.989)
     q.telescope = 'UVES'
     q.year = 2007
     q.coord = ['J2000', 232128.80, -105121.2]
     q.SIMBAD = 'QSO B2318-1107'
     q.ref.append('Noterdaeme2007')
     q.ref.append('Noterdaeme2008')
     q.el('HI', 20.68, 0.05, 0.05)
     q.el('H2', 15.49, 0.03, 0.03)
     q.el('H', 20.68, 0.05, 0.05)
     q.el('f', -4.89, 0.06, 0.06)
     q.el('Me', -0.85, 0.06, 0.06)
     q.el('Fe/X', -0.42, 0.03, 0.03)
     q.Me_ind = 'Zn'
     s = []
     co = sy(1.98888, 2)
     co.el('H2', 15.49, 0.03, 0.03)
     co.el('H2', 14.72, 0.03, 0.03, J=0, b=(3.6, 0.2, 0.2))
     co.el('H2', 15.28, 0.03, 0.03, J=2, b=(3.6, 0.2, 0.2))
     co.el('H2', 14.84, 0.03, 0.03, J=3, b=(3.6, 0.2, 0.2))
     co.el('H2', '<14.70', J=4)
     co.el('T01', 188, 34, 25, f='d')
     co.el('CIj0', 12.63, 0.02, 0.02)
     co.el('CIj1', 12.30, 0.04, 0.04)
     co.el('NI', 14.55, 0.02, 0.02)
     co.el('MgI', 12.32, 0.05, 0.05)
     co.el('MgII', 14.94, 0.04, 0.04)
     co.el('SiII', 14.72, 0.01, 0.01)
     co.el('SII', 14.50, 0.04, 0.04)
     co.el('CrII', 12.60, 0.01, 0.01)
     co.el('FeII', 14.35, 0.01, 0.01)
     co.el('NiII', 13.18, 0.01, 0.01)
     co.el('ZnII', 11.97, 0.01, 0.01)
     co.el('n',1.74,0.08,0.08)
     s.append(co)
     co = sy(1.98807, 8)
     co.el('NI', '<14.23')
     co.el('MgI', '<11.50')
     co.el('SiII', 13.73, 0.06, 0.06)
     co.el('SII', 13.33, 0.06, 0.06)
     co.el('CrII', '<11.70')
     co.el('FeII', 13.52, 0.05, 0.05)
     co.el('NiII', '<12.30')
     co.el('ZnII', '<10.80')
     s.append(co)
     co = sy(1.98827, 8)
     co.el('NI', '<13.79')
     co.el('MgI', '<11.50')
     co.el('SiII', 14.05, 0.02, 0.02)
     co.el('SII', 13.75, 0.02, 0.02)
     co.el('CrII', 11.96, 0.08, 0.08)
     co.el('FeII', 13.71, 0.04, 0.04)
     co.el('NiII', 12.58, 0.05, 0.05)
     co.el('ZnII', 11.18, 0.12, 0.12)
     s.append(co)
     co = sy(1.98845, 3)
     co.el('NI', '<13.53')
     co.el('MgI', '<11.50')
     co.el('SiII', 14.23, 0.05, 0.05)
     co.el('SII', 14.00, 0.04, 0.04)
     co.el('CrII', '<11.70')
     co.el('FeII', 13.43, 0.03, 0.03)
     co.el('NiII', 12.63, 0.04, 0.04)
     co.el('ZnII', 11.53, 0.05, 0.05)
     s.append(co)
     co = sy(1.98865, 0)
     co.el('NI', '<13.62')
     co.el('MgI', '<11.50')
     co.el('SiII', 14.62, 0.05, 0.05)
     co.el('SII', 14.43, 0.05, 0.05)
     co.el('CrII', 12.37, 0.03, 0.03)
     co.el('FeII', 14.13, 0.02, 0.02)
     co.el('NiII', 13.11, 0.02, 0.02)
     co.el('ZnII', 11.72, 0.04, 0.04)
     s.append(co)
     co = sy(1.98899, 3)
     co.el('NI', '<13.57')
     co.el('MgI', '<11.50')
     co.el('MgII', '<14.00')
     co.el('SiII', 14.03, 0.02, 0.02)
     co.el('SII', 13.87, 0.02, 0.02)
     co.el('CrII', 11.94, 0.06, 0.06)
     co.el('FeII', 13.55, 0.05, 0.05)
     co.el('NiII', 12.51, 0.05, 0.05)
     co.el('ZnII', 10.99, 0.13, 0.13)
     s.append(co)
     co = sy(1.98918, 3)
     co.el('NI', '<14.09')
     co.el('MgI', '<11.50')
     co.el('MgII', '<14.00')
     co.el('SiII', 14.31, 0.02, 0.02)
     co.el('SII', 14.18, 0.02, 0.02)
     co.el('CrII', 11.83, 0.11, 0.11)
     co.el('FeII', 13.92, 0.08, 0.08)
     co.el('NiII', 12.90, 0.02, 0.02)
     co.el('ZnII', 11.63, 0.04, 0.04)
     s.append(co)
     co = sy(1.98942, 1)
     co.el('NI', '<14.01')
     co.el('MgI', 11.80, 0.13, 0.13)
     co.el('MgII', '<14.00')
     co.el('SiII', 14.40, 0.03, 0.03)
     co.el('SII', 14.07, 0.06, 0.06)
     co.el('CrII', 12.17, 0.05, 0.05)
     co.el('FeII', 14.03, 0.03, 0.03)
     co.el('NiII', 12.96, 0.02, 0.02)
     co.el('ZnII', 11.44, 0.13, 0.13)
     s.append(co)
     co = sy(1.98956, 4)
     co.el('NI', '<13.39')
     co.el('MgI', '<11.50')
     co.el('MgII', '<14.48')
     co.el('SiII', 13.85, 0.08, 0.08)
     co.el('SII', 13.61, 0.08, 0.08)
     co.el('CrII', '<11.70')
     co.el('FeII', 13.42, 0.04, 0.04)
     co.el('NiII', '<12.30')
     co.el('ZnII', '<10.80')
     s.append(co)
     co = sy(1.98970, 7)
     co.el('NI', '<14.01')
     co.el('MgI', 12.04, 0.08, 0.08)
     co.el('SiII', 14.42, 0.03, 0.03)
     co.el('SII', 13.95, 0.04, 0.04)
     co.el('CrII', 12.26, 0.04, 0.04)
     co.el('FeII', 13.99, 0.08, 0.08)
     co.el('NiII', 12.84, 0.03, 0.03)
     co.el('ZnII', 11.53, 0.05, 0.05)
     q.comp = s
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2331-0908
     q = qso('J2331-0908', 2.66061, 2.143)
     q.telescope = 'UVES'
     q.year = 2018
     q.coord = ['J2000', 233156.47, -090801.2]
     q.progID.append('080.A-0795(A)')
     q.SIMBAD = '[VV2006] J233156.5-090802'
     q.m = {'B': 20.07, 'V': 19.56, 'G': 19.1894, 'u': 21.58, 'g': 19.723, 'r': 19.063, 'i': 18.674, 'z': 18.27}
     q.ref.append('Noterdaeme2018')
     q.el('HI', 21.204, 0.018, 0.018)
     q.el('H2', 20.57, 0.05, 0.05)
     q.el('H', 21.37, 0.02, 0.02)
     q.el('f', -0.50, 0.05, 0.05)
     q.el('Me', -0.5, 0.5, 0.5)
     q.el('EBV', 0.14,0.08,0.08, f='d')
     q.el('CI', 14.6,0.1)
     q.el('CO', 13.65)
     # q.Me_ind = 'Zn'
     q.SDSS = 'J233156.49-090802.0'
     # q.comment = ''
     s = []
     co = sy( 2.1422719614657466, 4)
     co.el('CI', 14.25,0.05) #b =3.98\pm0.4
     co.el('CIj0', 13.92, 0.03)
     co.el('CIj1', 13.85, 0.03)
     co.el('CIj2', 13.43, 0.03)
     co.el('H2j0', 20.14, 0.04)
     co.el('H2j1', 20.42, 0.05)
     co.el('H2', 20.57, 0.05)
     co.el('T01', 264, 20, f='d')
     co.el('CO',14.05,0.05)
     #co.el('COj0', 13.19, 0.06)
     #co.el('COj1', 13.37, 0.05)
     #co.el('COj2', 12.91, 0.19)
     co.el('COj0', 13.21, 0.06)
     co.el('COj1', 13.44, 0.05)
     co.el('COj2', 12.67, 0.29,0.58)
     #co.el('COj2', 12.67, 0.8, 0.6)
     co.el('COj3', 13.19, 0.10)
     co.el('COj4', 13.22, 0.10)
     co.el('T_co', 24.4, 2.8,2.3, f='d')
     co.el('n_co_3dpdr', 3.09, 0.08, 0.08)
     co.el('uv_co_3dpdr', '>1')
     co.el('n_ci_3dpdr', 2.39, 0.09, 0.07)
     #co.el('uv_ci_3dpdr', 0.80, 0.18, 0.14)
     co.el('uv_ci_3dpdr', '>0.64')
     co.el('tgas', 2.20, 0.10, 0.10)
     co.el('ngas(co)',2.86,0.13,0.11)
     co.el('tgas(ci)', 2.24, 0.07, 0.10)
     co.el('ngas(ci)',2.19,0.12,0.09)
     s.append(co)
     co = sy(2.14218844962, 4)
     co.el('CI', 14.27,0.05) #b =20\pm1
     co.el('CIj0', 13.94, 0.03)
     co.el('CIj1', 13.84, 0.03)
     co.el('CIj2', 13.52, 0.03)
     s.append(co)
     co = sy(2.142561597665765, 4)
     co.el('CI', 13.70,0.05) #b =4.45\pm0.4
     co.el('CIj0', 13.07, 0.03)
     co.el('CIj1', 13.37, 0.03)
     co.el('CIj2', 13.20, 0.03)
     co.el('H2j0',20.14,0.04)
     co.el('H2j1', 20.42, 0.05)
     co.el('H2',20.57,0.05)
     co.el('T01', 264,20,f='d')
     co.el('n_ci_3dpdr', 2.51, 0.19, 0.15)
     co.el('uv_ci_3dpdr', 0.78, 0.15, 0.15)
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2336-1058
     q = qso('J2336-1058', 2.04108, 1.829)
     q.telescope = 'XShooter'
     q.year = 2018
     q.coord = ['J2000', 233633.87, -105842.3]
     q.progID.append('084.A-0699(A)')
     q.SIMBAD = '[VV2006] J233633.8-105841'
     q.m = {'B': 19.30, 'V': 19.01, 'G': 18.1974, 'u': 19.136, 'g': 18.910, 'r': 18.754, 'i': 18.573, 'z': 18.31}
     q.ref.append('Noterdaeme2018')
     q.el('HI', 20.381, 0.005, 0.005)
     q.el('H2', 19.00, 0.12, 0.12)
     q.el('H', 20.42, 0.01, 0.01)
     q.el('f', -1.11, 0.12, 0.12)
     q.el('Me', -0.5, 0.5, 0.5)
     q.el('EBV', 0.01,0.04,0.04, f='d')
     q.el('CI', 14.07,0.02,0.02)
     q.el('CO', '<12.93')
     # q.Me_ind = 'Zn'
     q.SDSS = 'J233633.81-105841.5 '
     # q.comment = ''
     s = []
     co = sy(2.143, 0)
     co.el('CI', 14.07, 0.02, 0.02)
     co.el('CO', '<12.93')
     co.el('H2', 19.00, 0.12, 0.12)
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2350-0052
     q = qso('J2350-0052', 3.03, 2.426)
     q.telescope = 'XShooter'
     #q.year = 2018
     #q.coord = ['J2000', 233633.87, -105842.3]
     #q.progID.append('084.A-0699(A)')
     #q.SIMBAD = '[VV2006] J233633.8-105841'
     #q.m = {'B': 19.30, 'V': 19.01, 'G': 18.1974, 'u': 19.136, 'g': 18.910, 'r': 18.754, 'i': 18.573, 'z': 18.31}
     #q.ref.append('Noterdaeme2018')
     #q.el('HI', 20.381, 0.005, 0.005)
     q.el('H2', 18.52, 0.2, 0.3)
     #q.el('H', 20.42, 0.01, 0.01)
     #q.el('f', -1.11, 0.12, 0.12)
     #q.el('Me', -0.5, 0.5, 0.5)
     q.el('EBV', 0.016, 0.06, 0.06, f='d')
     q.el('CI', 14.36, 0.01, 0.01)
     q.el('CO', '<12.94')
     # q.Me_ind = 'Zn'
     q.SDSS = 'J2350-0052'
     # q.comment = ''
     s = []
     co = sy(2.426, 0)
     co.el('CI', 14.36, 0.01, 0.01)
     co.el('CO', '<12.94')
     co.el('H2', 18.52, 0.2, 0.3)
     s.append(co)
     q.comp = s
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2340-0053
     q = qso('J2340-0053', 2.08503, 2.05452)
     q.telescope = 'KECK'
     q.year = 2010
     q.coord = ['J2000', 234023.6701, -005326.998]
     q.SIMBAD = 'QSO J2340-0053'
     q.m = {'B': 18.26, 'V': 17.93, 'R': 17.800, 'J': 16.860, 'H': 16.663, 'K': 16.492, 'u': 18.168, 'g': 17.764,
            'r': 17.465, 'i': 17.142, 'z': 16.897}
     q.progID.append('085.A-0569(B)')
     q.progID.append('082.A-0569(A)')
     q.ref.append('Jorgenson2010')
     q.el('HI', 20.35, 0.15, 0.15)
     q.el('H2', 18.20, 0.20, 0.20)
     q.el('H2', 17.41, 0.06, 0.06, J=0)
     q.el('H2', 18.07, 0.04, 0.04, J=1)
     q.el('H', 20.36, 0.15, 0.15)
     q.el('f', -1.86, 0.25, 0.25)
     #q.el('EBV', 0.058,0.04,0.04)
     q.el('CI', 14.09, 0.20, 0.20)
     q.el('CO', '<12.58')
     q.el('CIj0', 13.40, 0.19, 0.08)
     q.el('CIj1', 13.11, 0.16, 0.10)
     q.el('CIj2', 12.60, 0.10, 0.08)
     q.el('T01', 252, 132, 64, f='d')
     q.el('n', 2.00, 0.15, 0.24)
     q.el('Me', -0.74, 0.16, 0.16)
     q.Me_ind = 'Si'
     s = []
     source = 'Klimenko'
     if source == 'Jorgenson':
         co = sy(2.054165, 0)
         co.el('H2', 16.061, 0.039, 0.038)
         co.el('H2', 15.262, 0.040, 0.040, J=0)
         co.el('H2', 15.940, 0.049, 0.049, J=1)
         co.el('H2', 14.904, 0.065, 0.065, J=2)
         co.el('H2', 14.251, 0.049, 0.049, J=3)
         co.el('H2', '<13.39', J=4)
         co.el('H2', '<12.90', J=5)
         co.el('T01', 268, 127, 65, f='d')
         co.el('CIj0', 12.32, 0.04, 0.04)
         co.el('CIj1', 11.96, 0.12, 0.12)
         co.el('n', 1.46,0.30,0.32)
         s.append(co)
         co = sy(2.054291, 0)
         co.el('H2', 15.464, 0.058, 0.055)
         co.el('H2', 14.657, 0.073, 0.073, J=0)
         co.el('H2', 15.292, 0.079, 0.079, J=1)
         co.el('H2', 14.478, 0.079, 0.079, J=2)
         co.el('H2', 14.299, 0.059, 0.059, J=3)
         co.el('H2', '<13.08', J=4)
         co.el('H2', '<13.23', J=5)
         co.el('T01', 232, 209, 75, f='d')
         co.el('CIj0', 12.54, 0.17, 0.17)
         co.el('CIj1', 12.28, 0.09, 0.09)
         co.el('CIj2', 11.84, 0.14, 0.14)
         co.el('n', 1.89, 0.23, 0.16)
         s.append(co)
         co = sy(2.054573, 0)
         co.el('H2', 18.079, 0.047, 0.045)
         co.el('H2', 17.269, 0.082, 0.082, J=0)
         co.el('H2', 17.955, 0.054, 0.054, J=1)
         co.el('H2', 17.045, 0.129, 0.129, J=2)
         co.el('H2', 15.326, 0.028, 0.028, J=3)
         co.el('H2', 13.812, 0.041, 0.041, J=4)
         co.el('H2', 13.317, 0.139, 0.139, J=5)
         co.el('T01', 276, 290, 93, f='d')
         co.el('CIj0', 12.97, 0.12, 0.12)
         co.el('CIj1', 12.53, 0.12, 0.12)
         co.el('CIj2', 12.12, 0.13, 0.13)
         co.el('n', 1.72, 0.15, 0.18)
         s.append(co)
         co = sy(2.054714, 0)
         co.el('H2', 16.979, 0.049, 0.047)
         co.el('H2', 15.997, 0.047, 0.047, J=0)
         co.el('H2', 16.825, 0.065, 0.065, J=1)
         co.el('H2', 16.059, 0.039, 0.039, J=2)
         co.el('H2', 15.836, 0.019, 0.019, J=3)
         co.el('H2', 14.355, 0.015, 0.015, J=4)
         co.el('H2', 13.932, 0.040, 0.040, J=5)
         co.el('T01', 587, 5058, 277, f='d')
         co.el('CIj0', 12.54, 0.67, 0.67)
         co.el('CIj1', 12.64, 0.36, 0.36)
         co.el('CIj2', 11.96, 0.26, 0.26)
         co.el('n', 1.67, 0.83, 0.63)
         s.append(co)
         co = sy(2.054986, 0)
         co.el('H2', 16.671, 0.054, 0.052)
         co.el('H2', 15.76, 0.042, 0.042, J=0)
         co.el('H2', 16.55, 0.067, 0.067, J=1)
         co.el('H2', 15.63, 0.047, 0.047, J=2)
         co.el('H2', 15.13, 0.027, 0.027, J=3)
         co.el('H2', 13.75, 0.046, 0.046, J=4)
         co.el('T01', 451, 954, 182, f='d')
         co.el('CI', 12.56, 0.07, 0.07)
         co.el('CI*', 11.96, 0.16, 0.16)
         co.el('CI**', 11.70, 0.29, 0.29)
         co.el('n', 1.36, 0.22, 0.25)
         s.append(co)
         co = sy(2.055135, 0)
         co.el('H2', 17.351, 0.046, 0.044)
         co.el('H2', 16.735, 0.074, 0.074, J=0)
         co.el('H2', 17.200, 0.057, 0.057, J=1)
         co.el('H2', 16.035, 0.088, 0.088, J=2)
         co.el('H2', 14.753, 0.049, 0.049, J=3)
         co.el('H2', '<13.10', J=4)
         co.el('H2', '<13.02', J=5)
         co.el('T01', 151, 55, 32, f='d')
         co.el('CI', 12.47, 0.07, 0.07)
         co.el('CI*', 12.15, 0.08, 0.08)
         co.el('CI**', 11.72, 0.16, 0.16)
         co.el('n', 1.85, 0.14, 0.12)
         s.append(co)
     elif source == 'Balashev':
         co = sy(2.0541537, 35)
         co.el('H2', 16.061, 0.039, 0.038)
         co.el('H2', 15.262, 0.040, 0.040, J=0)
         co.el('H2', 15.940, 0.049, 0.049, J=1)
         co.el('H2', 14.904, 0.065, 0.065, J=2)
         co.el('H2', 14.251, 0.049, 0.049, J=3)
         co.el('H2', '<13.39', J=4)
         co.el('H2', '<12.90', J=5)
         co.el('T01', 268, 127, 65, f='d')
         co.el('CIj0', 12.27, 0.05, 0.05)
         co.el('CIj1', 11.99, 0.09, 0.18)
         co.el('CIj2', 10.47, 0.44, 0.38)
         co.el('n', 2.15, 0.32, 0.42)
         s.append(co)
         co = sy(2.0542886, 20)
         co.el('H2', 15.464, 0.058, 0.055)
         co.el('H2', 14.657, 0.073, 0.073, J=0)
         co.el('H2', 15.292, 0.079, 0.079, J=1)
         co.el('H2', 14.478, 0.079, 0.079, J=2)
         co.el('H2', 14.299, 0.059, 0.059, J=3)
         co.el('H2', '<13.08', J=4)
         co.el('H2', '<13.23', J=5)
         co.el('T01', 232, 209, 75, f='d')
         co.el('CIj0', 13.01, 0.35, 0.23)
         co.el('CIj1', 12.50, 0.15, 0.10)
         co.el('CIj2', 12.24, 0.19, 0.17)
         co.el('n', 1.99, 0.26, 0.30)
         s.append(co)
         co = sy(2.0545306, 8)
         co.el('H2', 18.079, 0.047, 0.045)
         co.el('H2', 17.269, 0.082, 0.082, J=0)
         co.el('H2', 17.955, 0.054, 0.054, J=1)
         co.el('H2', 17.045, 0.129, 0.129, J=2)
         co.el('H2', 15.326, 0.028, 0.028, J=3)
         co.el('H2', 13.812, 0.041, 0.041, J=4)
         co.el('H2', 13.317, 0.139, 0.139, J=5)
         co.el('T01', 276, 290, 93, f='d')
         co.el('CIj0', 13.52, 0.05, 0.04)
         co.el('CIj1', 13.05, 0.02, 0.01)
         co.el('CIj2', 11.04, 0.58, 0.43)
         co.el('n', 1.79, 0.24, 0.28)
         co.el('LFR', 0.12, 0.02, 0.02, f='d')
         co.el('f_cov', 0.69, 0.06, 0.06, f='d')
         s.append(co)
         co = sy(2.0546076, 20)
         co.el('H2', 16.979, 0.049, 0.047)
         co.el('H2', 15.997, 0.047, 0.047, J=0)
         co.el('H2', 16.825, 0.065, 0.065, J=1)
         co.el('H2', 16.059, 0.039, 0.039, J=2)
         co.el('H2', 15.836, 0.019, 0.019, J=3)
         co.el('H2', 14.355, 0.015, 0.015, J=4)
         co.el('H2', 13.932, 0.040, 0.040, J=5)
         co.el('T01', 587, 5058, 277, f='d')
         co.el('CIj01', 13.20, 0.05, 0.05)
         co.el('CIj1', 12.67, 0.03, 0.03)
         co.el('CIj2', 11.16, 0.11, 0.89)
         co.el('n', 2.05, 1.07, 1.05)
         s.append(co)
         co = sy(2.0546541, 60)
         co.el('H2', 16.671, 0.054, 0.052)
         co.el('H2', 15.76, 0.042, 0.042, J=0)
         co.el('H2', 16.55, 0.067, 0.067, J=1)
         co.el('H2', 15.63, 0.047, 0.047, J=2)
         co.el('H2', 15.13, 0.027, 0.027, J=3)
         co.el('H2', 13.75, 0.046, 0.046, J=4)
         co.el('T01', 451, 954, 182, f='d')
         co.el('CIj0', 12.51, 1.38, 0.27)
         co.el('CIj1', 11.94, 0.27, 0.35)
         co.el('CIj2', 11.26, 0.42, 0.63)
         co.el('n', 1.48, 0.34, 0.36)
         s.append(co)
         co = sy(2.055135, 0)
         co.el('H2', 17.351, 0.046, 0.044)
         co.el('H2', 16.735, 0.074, 0.074, J=0)
         co.el('H2', 17.200, 0.057, 0.057, J=1)
         co.el('H2', 16.035, 0.088, 0.088, J=2)
         co.el('H2', 14.753, 0.049, 0.049, J=3)
         co.el('H2', '<13.10', J=4)
         co.el('H2', '<13.02', J=5)
         co.el('T01', 151, 55, 32, f='d')
         co.el('CIj0', 12.47, 0.07, 0.07)
         co.el('CIj1', 12.15, 0.08, 0.08)
         co.el('CIj2', 11.72, 0.16, 0.16)
         co.el('n', 1.97, 0.18, 0.18)
         s.append(co)
     elif source == 'Rawlins':
         co = sy(2.054168, 1)
         co.el('H2', 15.96, 0.03, 0.03)
         co.el('H2', 15.35, 0.05, 0.05, J=0, b=(2.5, 0.1, 0.1))
         co.el('H2', 15.74, 0.05, 0.05, J=1, b=(2.5, 0.1, 0.1))
         co.el('H2', 15.03, 0.06, 0.06, J=2, b=(2.5, 0.1, 0.1))
         co.el('H2', 14.30, 0.07, 0.07, J=3, b=(2.5, 0.1, 0.1))
         co.el('H2', 13.17, 0.36, 0.36, J=4, b=(2.5, 0.1, 0.1))
         co.el('H2', '<13.67', J=5, b=(2.5, 0.1, 0.1))
         co.el('T01', 131, 28, 20, f='d')
         s.append(co)
         co = sy(2.054293, 2)
         co.el('H2', 15.49, 0.05)
         co.el('H2', 14.77, 0.10, J=0, b=(1.5, 0.2, 0.2))
         co.el('H2', 15.15, 0.08, J=1, b=(1.5, 0.2, 0.2))
         co.el('H2', 14.79, 0.12, J=2, b=(1.5, 0.2, 0.2))
         co.el('H2', 14.62, 0.08, J=3, b=(1.5, 0.2, 0.2))
         co.el('H2', 13.55, 0.18, J=4, b=(1.5, 0.2, 0.2))
         co.el('H2', '<13.90', J=5, b=(1.5, 0.2, 0.2))
         co.el('T01', 129, 59, 31, f='d')
         s.append(co)
         co = sy(2.054509, 4)
         co.el('H2', 17.79, 0.08)
         co.el('H2', 17.38, 0.11, J=0, b=(0.9, 0.2, 0.2))
         co.el('H2', 17.57, 0.11, J=1, b=(0.9, 0.2, 0.2))
         co.el('H2', '<15.90', J=2, b=(0.9, 0.2, 0.2))
         co.el('H2', 15.24, 0.21, J=3, b=(0.9, 0.2, 0.2))
         co.el('H2', 13.77, 0.15, J=4, b=(0.9, 0.2, 0.2))
         co.el('H2', 13.92, 0.15, J=5, b=(0.9, 0.2, 0.2))
         co.el('T01', 97, 39, 22, f='d')
         s.append(co)
         co = sy(2.054599, 7)
         co.el('H2', 16.83, 0.06)
         co.el('H2', 16.06, 0.07, J=0, b=(7.5, 0.5, 0.5))
         co.el('H2', 16.54, 0.12, J=1, b=(7.5, 0.5, 0.5))
         co.el('H2', 16.29, 0.04, J=2, b=(7.5, 0.5, 0.5))
         co.el('H2', 15.36, 0.04, J=3, b=(7.5, 0.5, 0.5))
         co.el('H2', 13.78, 0.15, J=4, b=(7.5, 0.5, 0.5))
         co.el('H2', '<13.91', J=5, b=(7.5, 0.5, 0.5))
         co.el('T01', 156, 109, 45, f='d')
         s.append(co)
         co = sy(2.054727, 3)
         co.el('H2', 16.80, 0.06)
         co.el('H2', 15.76, 0.06, J=0, b=(5.0, 0.2, 0.2))
         co.el('H2', 16.60, 0.09, J=1, b=(5.0, 0.2, 0.2))
         co.el('H2', 16.07, 0.06, J=2, b=(5.0, 0.2, 0.2))
         co.el('H2', 15.74, 0.03, J=3, b=(5.0, 0.2, 0.2))
         co.el('H2', 14.37, 0.04, J=4, b=(5.0, 0.2, 0.2))
         co.el('H2', 14.14, 0.09, J=5, b=(5.0, 0.2, 0.2))
         co.el('T01', '<271', f='d')
         s.append(co)
         co = sy(2.054991, 1)
         co.el('H2', 16.39, 0.03)
         co.el('H2', 15.57, 0.03, J=0, b=(4.2, 0.1, 0.1))
         co.el('H2', 16.18, 0.04, J=1, b=(4.2, 0.1, 0.1))
         co.el('H2', 15.66, 0.06, J=2, b=(4.2, 0.1, 0.1))
         co.el('H2', 15.13, 0.02, J=3, b=(4.2, 0.1, 0.1))
         co.el('H2', 13.77, 0.12, J=4, b=(4.2, 0.1, 0.1))
         co.el('H2', '<13.88', J=5, b=(4.2, 0.1, 0.1))
         co.el('T01', 215, 56, 37, f='d')
         s.append(co)
         co = sy(2.055137, 1)
         co.el('H2', 17.29, 0.09)
         co.el('H2', 16.43, 0.17, J=0, b=(1.9, 0.2, 0.2))
         co.el('H2', 17.20, 0.11, J=1, b=(1.9, 0.2, 0.2))
         co.el('H2', 16.03, 0.11, J=2, b=(1.9, 0.2, 0.2))
         co.el('H2', 14.69, 0.06, J=3, b=(1.9, 0.2, 0.2))
         co.el('H2', 13.29, 0.30, J=4, b=(1.9, 0.2, 0.2))
         co.el('H2', 13.55, 0.30, J=5, b=(1.9, 0.2, 0.2))
         co.el('T01', '<161', f='d')
         s.append(co)
     elif source == 'Klimenko':
         co = sy(2.0541639, 0)
         co.el('H2', 17.43, 0.02, 0.02)
         co.el('H2', 16.85, 0.04, 0.03, J=0)
         co.el('H2', 17.30, 0.02, 0.02, J=1)
         co.el('T01', 147, 9, 12, f='d')
         co.el('CIj0', 12.31, 0.08, 0.08)
         co.el('CIj1', 12.14, 0.12, 0.12)
         co.el('CIj2', 11.32, 0.16, 0.21)
         co.el('n', 1.86, 0.16, 0.18)
         co.el('LFR', '<0.10', f='d')
         co.el('f_cov', '>0.68', f='d')
         #co.el('LFR', '<0.05', f='d')
         #co.el('f_cov', '>0.84', f='d')
         s.append(co)
         co = sy(2.0542886, 20)
         co.el('H2', 16.15, 0.20, 0.10)
         co.el('H2', 15.01, 0.15, 0.12, J=0)
         co.el('H2', 16.11, 0.20, 0.10, J=1)
         co.el('T01', 500, 1000, 400, f='d')
         co.el('CIj0', 12.16, 0.10, 0.11)
         co.el('CIj1', 12.27, 0.10, 0.11)
         co.el('CIj2', 11.93, 0.19, 0.16)
         co.el('n', 2.17, 0.26, 0.30)
         co.el('LFR', '<0.30', f='d')
         co.el('f_cov', '>0.05', f='d')
         #co.el('LFR', '<0.10', f='d')
         #co.el('f_cov', '>0.68', f='d')
         s.append(co)
         co = sy(2.0545306, 8) #final
         #co.el('H2', 18.18, 0.02, 0.02)
         co.el('H2', 17.66, 0.044, 0.076, J=0)
         co.el('H2', 18.04, 0.025, 0.025, J=1)
         co.el('T01', 128, 15, 14, f='d')
         co.el('CIj0', 13.45, 0.02, 0.02)
         co.el('CIj1', 13.03, 0.02, 0.03)
         co.el('CIj2', 11.51, 0.25, 0.35)
         co.el('n', 1.55, 0.07, 0.06)
         co.el('LFR', 0.17, 0.03, 0.02, f='d')
         co.el('f_cov', 0.63, 0.07, 0.07, f='d')
         #co.el('CO', '<12.58')
         s.append(co)
         co = sy(2.0545970, 20) #final
         co.el('H2', 17.02, 0.42, 0.42)
         co.el('H2', 16.80, 0.80, 0.80, J=0)
         co.el('H2', 16.50, 0.90, 0.90, J=1)
         co.el('T01', 59, 5000, 30, f='d')
         co.el('CIj01', 13.05, 0.03, 0.03)
         co.el('CIj1', 12.56, 0.05, 0.05)
         co.el('CIj2', 12.02, 0.16, 0.20)
         co.el('LFR', 0.15, 0.03, 0.07, f='d')
         co.el('n', 1.70, 0.7, 0.5)
         co.el('f_cov', 0.67, 0.15, 0.07, f='d')
         s.append(co)
         co = sy(2.0547098, 60) #final
         co.el('H2', 17.61, 0.04, 0.04)
         co.el('H2', 17.00, 0.10, 0.10, J=0)
         co.el('H2', 17.50, 0.04, 0.04, J=1)
         co.el('T01', 163, 954, 182, f='d')
         co.el('CIj0', 13.30, 0.02, 0.02)
         co.el('CIj1', 13.03, 0.02, 0.02)
         co.el('CIj2', 12.34, 0.06, 0.06)
         co.el('n', 1.77, 0.05, 0.05)
         co.el('LFR', '<7', f='d')
         co.el('f_cov', '>0.84', f='d')
         s.append(co)
         co = sy(2.0549960, 0)
         co.el('H2', 17.94, 0.03, 0.02)
         co.el('H2', 17.32, 0.02, 0.10, J=0)
         co.el('H2', 17.83, 0.04, 0.02, J=1)
         co.el('T01', 176, 32, 20, f='d')
         co.el('CIj0', 12.31, 0.07, 0.07)
         co.el('CIj1', 11.96, 0.14, 0.20)
         co.el('CIj2', 11.68, 0.23, 0.18)
         co.el('n', 1.93, 0.18, 0.20)
         co.el('LFR', '<0.05', f='d')
         co.el('f_cov', '>0.84', f='d')
         #co.el('CO', '<12.58')
         s.append(co)
         co = sy(2.0551350, 0)
         co.el('H2', 17.09, 0.05, 0.05)
         co.el('H2', 16.34, 0.10, 0.08, J=0)
         co.el('H2', 17.00, 0.05, 0.05, J=1)
         co.el('T01', 250, 150, 100, f='d')
         co.el('CIj0', 12.50, 0.04, 0.04)
         co.el('CIj1', 12.12, 0.12, 0.15)
         co.el('CIj2', 11.80, 0.22, 0.22)
         co.el('n', 1.73, 0.18, 0.18)
         co.el('LFR', '<0.05', f='d')
         co.el('f_cov', '>0.84', f='d')
         s.append(co)
     q.comp = s
     q.comment = 'stange H2 excitation diagrams'
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B2343+125
     q = qso('B2343+125', 2.51, 2.431)
     q.telescope = 'UVES'
     q.year = 2006
     q.coord = ['J2000', 234628.2, +124900]
     q.SIMBAD = 'QSO B2343+125'
     q.m = {'V': 17.5}
     q.progID.append('60.A-9022(A)')
     q.progID.append('075.A-0018(A)')
     q.progID.append('67.A-0022(A)')
     q.progID.append('079.B-0469(A)')
     q.progID.append('072.A-0346(A)')
     q.progID.append('65.O-0299(A)')
     q.progID.append('68.A-0216(A)')
     q.progID.append('69.A-0204(A)')
     q.ref.append('Petitjean2006')
     q.ref.append('Noterdaeme2007')
     q.ref.append('Noterdaeme2008')
     q.el('HI', 20.40, 0.07, 0.07)
     q.el('H2', 13.69, 0.09, 0.09)
     q.el('H', 20.40, 0.07, 0.07)
     q.el('f', -6.41, 0.12, 0.12)
     q.el('Me', -0.87, 0.10, 0.10)
     q.Me_ind = 'Zn'
     s = []
     co = sy(2.43128, 3)
     co.el('H2', 13.69, 0.09, 0.09)
     co.el('H2', 12.97, 0.04, 0.04, J=0)
     co.el('H2', 13.60, 0.10, 0.10, J=1)
     co.el('H2', 13.10, 0.00, 0.00, J=2)
     co.el('T01', 228, 198, 72, f='d')
     co.el('CI', '<12.10')
     co.el('CI*', '<12.40')
     s.append(co)
     co = sy(2.43105, 7)
     co.el('NI', 13.49, 0.02, 0.02)
     co.el('OI', 15.46, 0.02, 0.02)
     co.el('MgI', '<11.55')
     co.el('SiII', 14.31, 0.08, 0.08)
     co.el('PII', 12.16, 0.09, 0.09)
     co.el('SII', 13.41, 0.13, 0.13)
     co.el('ArI', 11.64, 0.26, 0.26)
     co.el('CrII', 12.00, 0.09, 0.09)
     co.el('MnII', 11.51, 0.10, 0.10)
     co.el('FeII', 13.57, 0.08, 0.08)
     co.el('NiII', 12.39, 0.14, 0.14)
     co.el('ZnII', '<11.00')
     s.append(co)
     co = sy(2.43116, 9)
     co.el('NI', 14.20, 0.01, 0.01)
     co.el('OI', '>15.72')
     co.el('MgI', 11.94, 0.06, 0.06)
     co.el('SiII', 14.55, 0.07, 0.07)
     co.el('PII', 12.53, 0.06, 0.06)
     co.el('SII', 14.20, 0.04, 0.04)
     co.el('ArI', 12.82, 0.02, 0.02)
     co.el('CrII', 12.31, 0.06, 0.06)
     co.el('MnII', 11.67, 0.08, 0.08)
     co.el('FeII', 13.96, 0.05, 0.05)
     co.el('NiII', 12.84, 0.06, 0.06)
     co.el('ZnII', 11.71, 0.04, 0.04)
     s.append(co)
     co = sy(2.43129, 3)
     co.el('NI', 14.19, 0.01, 0.01)
     co.el('OI', '>15.92')
     co.el('MgI', 11.87, 0.08, 0.08)
     co.el('SiII', 14.70, 0.04, 0.04)
     co.el('PII', 12.64, 0.04, 0.04)
     co.el('SII', 14.24, 0.04, 0.04)
     co.el('ArI', 12.78, 0.02, 0.02)
     co.el('CrII', 12.34, 0.05, 0.05)
     co.el('MnII', 11.89, 0.04, 0.04)
     co.el('FeII', 14.00, 0.04, 0.04)
     co.el('NiII', 12.97, 0.04, 0.04)
     co.el('ZnII', 11.74, 0.04, 0.04)
     s.append(co)
     co = sy(2.43143, 8)
     co.el('NI', 13.78, 0.01, 0.01)
     co.el('OI', '>15.81')
     co.el('MgI', 11.57, 0.13, 0.13)
     co.el('SiII', 14.45, 0.04, 0.04)
     co.el('PII', 12.22, 0.06, 0.06)
     co.el('SII', 13.88, 0.04, 0.04)
     co.el('ArI', 12.32, 0.04, 0.04)
     co.el('CrII', 12.14, 0.05, 0.05)
     co.el('MnII', 11.66, 0.05, 0.05)
     co.el('FeII', 13.90, 0.03, 0.03)
     co.el('NiII', 12.71, 0.05, 0.05)
     co.el('ZnII', 11.57, 0.04, 0.04)
     s.append(co)
     co = sy(2.43158, 1)
     co.el('NI', 13.18, 0.03, 0.03)
     co.el('OI', 14.82, 0.06, 0.06)
     co.el('MgI', '<11.55')
     co.el('SiII', 13.80, 0.09, 0.09)
     co.el('PII', '<11.85')
     co.el('SII', 13.33, 0.08, 0.08)
     co.el('ArI', 11.69, 0.17, 0.17)
     co.el('CrII', 11.92, 0.05, 0.05)
     co.el('MnII', 11.28, 0.09, 0.09)
     co.el('FeII', 13.40, 0.04, 0.04)
     co.el('NiII', 12.45, 0.06, 0.06)
     co.el('ZnII', '<11.00')
     s.append(co)
     q.comp = s
     q.full = 'y'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add B2348-0108
     q = qso('B2348-0108', 3.01, 2.426)
     q.telescope = 'UVES'
     q.year = 2006
     q.SIMBAD = 'QSO B2348-0108'
     q.m = {'B': 19.23, 'V': 18.77, 'R': 18.7, 'J': 18.149, 'H': 17.986, 'K': 18.124, 'u': 21.266, 'g': 19.120,
            'r': 18.679, 'i': 18.561, 'z': 18.49}
     q.progID.append('079.A-0404(A)')
     q.progID.append('072.A-0346(A)')
     q.ref.append('Petitjean2006')
     q.ref.append('Noterdaeme2008')
     q.el('HI', 20.50, 0.10, 0.10)
     q.el('H2', 18.52, 0.29, 0.49)
     q.el('H', 20.51, 0.10, 0.10)
     q.el('f', -1.69, 0.31, 0.51)
     q.el('ClI', '<13.86')
     q.el('Me', -0.62, 0.10, 0.10)
     q.el('Fe/X', -0.55, 0.02, 0.02)
     q.Me_ind == 'S'
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # >>>>> Muzahid et al 2015 low z sample (B0120-28 in Oliveira et al. 2015, and Q0107-0232 in Chighton et al. 2013)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1241+2852
     q = qso('J1241+2852', 0.589, 0.06650)
     q.telescope = 'HST'
     q.year = 2015
     q.SIMBAD = 'QSO J1241+2852'
     q.ref.append('Muzahid2015')
     q.el('HI', 19.18, 0.10, 0.10)
     q.el('H2', 16.45, 0.12, 0.12)
     q.el('H', 19.18, 0.10, 0.10)
     q.el('f', -2.43, 0.16, 0.16)
     q.el('Me', -0.62, 0.13, 0.13)
     q.Me_ind == 'S'
     s = []
     co = sy(0.06650, 0)
     co.el('H2', 16.45, 0.12, 0.12, b=(17.1, 1.1, 1.1))
     co.el('H2', 15.72, 0.08, 0.08, J=0)
     co.el('H2', 16.30, 0.06, 0.06, J=1)
     co.el('H2', 15.45, 0.08, 0.08, J=2)
     co.el('H2', '<14.8', J=3)
     co.el('H2', '<14.7', J=4)
     co.el('T01', 198, 120, 54, f='d')
     s.append(co)
     q.comp = s
     q.comment = 'PI: T.Heckman, PID:12603'
     q.full = 'n'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1619+3342
     q = qso('J1619+3342', 0.472, 0.09630)
     q.telescope = 'HST'
     q.year = 2015
     q.SIMBAD = 'QSO J1619+3342'
     q.ref.append('Muzahid2015')
     q.el('HI', 20.55, 0.10, 0.10)
     q.el('H2', 18.57, 0.06, 0.06)
     q.el('H', 20.56, 0.10, 0.10)
     q.el('f', -1.69, 0.11, 0.11)
     q.el('Me', -0.62, 0.13, 0.13)
     # q.Me_ind == 'S' see Battisti et al 2012
     s = []
     co = sy(0.09630, 0)
     co.el('H2', 18.57, 0.03, 0.03, b=(4.1, 0.4, 0.4))
     co.el('H2', 18.17, 0.04, 0.04, J=0)
     co.el('H2', 18.36, 0.04, 0.04, J=1)
     co.el('H2', 15.97, 0.25, 0.25, J=2)
     co.el('H2', 15.27, 0.29, 0.29, J=3)
     co.el('H2', '<14.1', J=4)
     co.el('T01', 97, 7, 6, f='d')
     co.el('T02', 76, 7, 6, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     q.comment = 'PI: J. Tumlinson, PID:11598'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add Q0439-433
     q = qso('Q0439-433', 0.594, 0.10115)
     q.telescope = 'HST'
     q.year = 2015
     q.SIMBAD = 'QSO J0441-4313'
     q.ref.append('Muzahid2015')
     q.ref.append('Dutta2015')
     q.el('HI', 19.63, 0.08, 0.08)
     q.el('H2', 16.64, 0.05, 0.05)
     q.el('H', 19.63, 0.08, 0.08)
     q.el('FeII', 14.92, 0.03, 0.03)
     q.el('PII', 13.47, 0.06, 0.06)
     q.el('SII', 15.03, 0.02, 0.02)
     q.el('ArI', 13.27, 0.14, 0.14)
     q.el('CaII', 12.66, 0.02, 0.02)
     q.el('NI', 14.98, 0.02, 0.02)
     q.el('NaI', 12.28, 0.02, 0.02)
     q.el('f', -2.69, 0.09, 0.09)
     q.el('Me', +0.32, 0.14, 0.14)
     q.Me_ind == 'S'
     s = []
     co = sy(0.10091, 0)
     co.el('H2', 15.51, 0.02, 0.02, b=(32.7, 2.6, 2.6))
     co.el('H2', 14.67, 0.05, 0.05, J=0)
     co.el('H2', 15.21, 0.02, 0.02, J=1)
     co.el('H2', 14.91, 0.04, 0.04, J=2)
     co.el('H2', 14.52, 0.07, 0.07, J=3)
     co.el('H2', '<14.1', J=4)
     co.el('T01', 179, 40, 28, f='d')
     s.append(co)
     co = sy(0.10115, 0)
     co.el('H2', 16.63, 0.03, 0.03, b=(12.0, 0.5, 0.5))
     co.el('H2', 15.98, 0.06, 0.06, J=0)
     co.el('H2', 16.38, 0.05, 0.05, J=1)
     co.el('H2', 15.70, 0.04, 0.04, J=2)
     co.el('H2', 15.56, 0.04, 0.04, J=3)
     co.el('H2', '<14.1', J=4)
     co.el('T01', 134, 33, 22, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     q.comment = 'PI: V. Kulkarni, PID:12536'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add Q0850+440
     q = qso('Q0850+440', 0.515, 0.16375)
     q.telescope = 'HST'
     q.year = 2015
     q.SIMBAD = '2MASS J08533423+4349023'
     q.ref.append('Muzahid2015')
     q.el('HI', 19.67, 0.10, 0.10)
     q.el('H2', 15.05, 0.07, 0.07)
     q.el('H', 19.67, 0.10, 0.10)
     q.el('f', -4.32, 0.12, 0.12)
     q.el('Me', -1.36, 0.10, 0.10)
     # q.Me_ind == 'S' from LAnzetta et al 1997
     s = []
     co = sy(0.16375, 0)
     co.el('H2', 15.05, 0.07, 0.07, b=(11.9, 0.8, 0.8))
     co.el('H2', 14.40, 0.03, 0.03, J=0)
     co.el('H2', 14.83, 0.02, 0.02, J=1)
     co.el('H2', 14.33, 0.06, 0.06, J=2)
     co.el('H2', '<13.7', J=3)
     co.el('H2', '<13.6', J=4)
     co.el('T01', 141, 15, 12, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     q.comment = 'PI: C. Churchill, PID:13398'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1342-0053
     q = qso('J1342-0053', 0.326, 0.22711)
     q.telescope = 'HST'
     q.year = 2015
     q.SIMBAD = 'LBQS 1340-0038'
     q.ref.append('Muzahid2015')
     q.el('HI', 19.0, 0.5, 0.8)
     q.el('H2', 14.63, 0.06, 0.06)
     q.el('H', 19.00, 0.50, 0.80)
     q.el('f', -4.07, 0.50, 0.80)
     q.el('Me', -0.40, 0.10, 0.10)
     # q.Me_ind == 'S' see Werk et al 2013
     s = []
     co = sy(0.22711, 0)
     co.el('H2', '<14.53', b=(10.1, 2.3, 2.3))
     co.el('H2', '<13.4', J=0)
     co.el('H2', 14.63, 0.06, 0.06, J=1)
     co.el('H2', '<13.6', J=2)
     co.el('H2', '<14.0', J=3)
     co.el('H2', '<14.1', J=4)
     s.append(co)
     q.comp = s
     q.full = 'n'
     q.comment = 'PI: J. Tumlinson, PID:11598'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J0925+4004
     q = qso('J0925+4004', 0.472, 0.24788)
     q.telescope = 'HST'
     q.year = 2015
     q.SIMBAD = 'QSO J0925+4004'
     q.ref.append('Muzahid2015')
     q.el('HI', 19.55, 0.15, 0.15)
     q.el('H2', 18.82, 0.13, 0.13)
     q.el('H', 19.69, 0.12, 0.11)
     q.el('f', -0.57, 0.17, 0.17)
     q.el('Me', -0.29, 0.17, 0.17)
     # q.Me_ind == 'S' see Battisti et al 2012
     s = []
     co = sy(0.24788, 0)
     co.el('H2', 18.82, 0.13, 0.13, b=(8.4, 0.5, 0.5))
     co.el('H2', 18.15, 0.08, 0.08, J=0)
     co.el('H2', 18.63, 0.04, 0.04, J=1)
     co.el('H2', 17.90, 0.10, 0.10, J=2)
     co.el('H2', 16.83, 0.15, 0.15, J=3)
     co.el('H2', '<14.3', J=4)
     co.el('T01', 156, 36, 24, f='d')
     co.el('T02', 233, 38, 29, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     q.comment = 'PI: J. Tumlinson, PID:11598'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J1616+4154
     q = qso('J1616+4154', 0.441, 0.32110)
     q.telescope = 'HST'
     q.year = 2015
     q.SIMBAD = 'QSO J1616+4154'
     q.ref.append('Muzahid2015')
     q.el('HI', 20.60, 0.20, 0.20)
     q.el('H2', 19.26, 0.02, 0.02)
     q.el('H', 20.64, 0.19, 0.18)
     q.el('f', -1.08, 0.20, 0.20)
     q.el('Me', -0.38, 0.23, 0.23)
     # q.Me_ind == 'S' see Battisti et al 2012
     s = []
     co = sy(0.32110, 0)
     co.el('H2', 19.26, 0.02, 0.02, b=(6.9, 0.5, 0.5))
     co.el('H2', 18.95, 0.02, 0.02, J=0)
     co.el('H2', 18.93, 0.02, 0.02, J=1)
     co.el('H2', 17.83, 0.09, 0.09, J=2)
     co.el('H2', 16.99, 0.12, 0.12, J=3)
     co.el('H2', '<14.1', J=4)
     co.el('T01', 76, 3, 3, f='d')
     co.el('T02', 120, 6, 6, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     q.comment = 'PI: J. Tumlinson, PID:11598'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add Q1241+176
     q = qso('Q1241+176', 1.273, 0.55048)
     q.telescope = 'HST'
     q.year = 2015
     q.SIMBAD = 'QSO J1244+1721'
     q.ref.append('Muzahid2015')
     q.el('HI', '>19.60')
     q.el('H2', 15.81, 0.17, 0.17)
     q.el('H', '>19.60')
     q.el('f', '<-2.89')
     q.el('Me', '<+0.18')
     q.Me_ind == 'S'
     s = []
     co = sy(0.55048, 0)
     co.el('H2', 15.81, 0.17, 0.17, b=(7.8, 0.4, 0.4))
     co.el('H2', 15.35, 0.13, 0.13, J=0)
     co.el('H2', 15.42, 0.06, 0.06, J=1)
     co.el('H2', 14.95, 0.05, 0.05, J=2)
     co.el('H2', 14.80, 0.08, 0.08, J=3)
     co.el('H2', '<14.0', J=4)
     co.el('T01', 84, 25, 16, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     q.comment = 'PI: J. Charlton, PID:12466'
     QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add Q2128-123
     q = qso('Q2128-123', 0.5010, 0.4298)
     q.telescope = 'HST'
     q.year = 2016
     q.SIMBAD = 'PKS 2128-123'
     q.ref.append('Muzahid2016')
     q.el('HI', 19.50, 0.15, 0.15)
     q.el('H2', 16.36, 0.08, 0.08)
     q.el('H', 19.50, 0.15, 0.15)
     q.el('f', -2.84, 0.17, 0.17)
     q.el('Me', -0.26, 0.19, 0.19)
     q.Me_ind == 'O'
     s = []
     co = sy(0.429807, 0)
     co.el('H2', 16.36, 0.08, 0.08, b=(7.1, 0.3, 0.3))
     co.el('H2', 15.74, 0.08, 0.08, J=0)
     co.el('H2', 16.22, 0.09, 0.09, J=1)
     co.el('H2', 15.23, 0.03, 0.03, J=2)
     co.el('H2', 14.83, 0.03, 0.03, J=3)
     co.el('H2', '<13.8', J=4)
     co.el('T01', 156, 47, 47, f='d')
     co.el('CI', 13.70, 0.10, 0.10)
     co.el('CI*', 13.60, 0.10, 0.10)
     co.el('CI**', 13.07, 0.10, 0.10)
     s.append(co)
     co = sy(0.429807, 0)
     co.el('H2', 18.27, 0.03, 0.03, b=(2.8, 0.3, 0.3))
     co.el('H2', 17.65, 0.04, 0.04, J=0)
     co.el('H2', 18.09, 0.02, 0.02, J=1)
     co.el('H2', 17.23, 0.07, 0.07, J=2)
     co.el('H2', 15.23, 0.03, 0.03, J=3)
     co.el('H2', '<13.5', J=4)
     co.el('T01', 143, 17, 17, f='d')
     co.el('T02', 198, 15, 13, f='d')
     s.append(co)
     q.comp = s
     q.full = 'n'
     q.comment = 'PID: GO-13398, GO-12536'
     QSO.append(q)

     # print(len(QSO))

     return QSO

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # unbublished (secret! systems):


def load_Secret():
     global sy
     QSO = sample()

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # add J2340-0053
     q = qso('J2340-0053', 2.08503, 2.05452)
     q.telescope = 'KECK'
     q.year = 2010
     q.coord = ['J2000', 234023.6701, -005326.998]
     q.SIMBAD = 'QSO J2340-0053'
     q.m = {'B': 18.26, 'V': 17.93, 'R': 17.800, 'J': 16.860, 'H': 16.663, 'K': 16.492, 'u': 18.168, 'g': 17.764,
            'r': 17.465, 'i': 17.142, 'z': 16.897}
     q.progID.append('085.A-0569(B)')
     q.progID.append('082.A-0569(A)')
     q.ref.append('Jorgenson2010')
     q.el('HI', 20.35, 0.15, 0.15)
     q.el('H2', 18.20, 0.20, 0.20)
     q.el('H2', 17.41, 0.06, 0.06, J=0)
     q.el('H2', 18.07, 0.04, 0.04, J=1)
     q.el('H', 20.36, 0.15, 0.15)
     q.el('f', -1.86, 0.25, 0.25)
     # q.el('CI', 14.09, 0.20, 0.20)
     q.el('CI', 13.40, 0.19, 0.08)
     q.el('CI*', 13.11, 0.16, 0.10)
     q.el('CI**', 12.60, 0.10, 0.08)
     q.el('T01', 252, 132, 64, f='d')
     q.el('n', 2.00, 0.15, 0.24)
     q.el('Me', -0.74, 0.16, 0.16)
     q.Me_ind = 'Si'
     s = []
     q.comp = []
     co = sy(2.0541536, 48)
     co.el('T01', 100, 50, 50, f='d')
     co.el('CI', 12.44, 0.05, 0.06)
     co.el('CI', 12.27, 0.04, 0.06, b=(1.8, 1.2, 0.8))
     co.el('CI*', 11.95, 0.12, 0.16, b=(1.8, 1.2, 0.8))
     co.el('CI**', 10.5, 0.4, 0.4, b=(1.8, 1.2, 0.8))
     q.comp.append(co)
     co = sy(2.0542890, 31)
     co.el('T01', 100, 50, 50, f='d')
     co.el('CI', 13.19, 0.21, 0.11)
     co.el('CI', 13.02, 0.28, 0.17, b=(0.170, 0.026, 0.023))
     co.el('CI*', 12.52, 0.13, 0.10, b=(0.170, 0.026, 0.023))
     co.el('CI**', 12.27, 0.20, 0.13, b=(0.170, 0.026, 0.023))
     q.comp.append(co)
     co = sy(2.0545300, 10)
     co.el('T01', 100, 50, 50, f='d')
     co.el('CI', 13.727, 0.026, 0.034)
     co.el('CI', 13.62, 0.03, 0.04, b=(1.67, 0.09, 0.07))
     co.el('CI*', 13.065, 0.013, 0.016, b=(1.67, 0.09, 0.07))
     co.el('CI**', 10.47, 0.79, 0.28, b=(1.67, 0.09, 0.07))
     q.comp.append(co)
     co = sy(2.0546055, 30)
     co.el('T01', 100, 50, 50, f='d')
     co.el('CI', 13.281, 0.037, 0.026)
     co.el('CI', 13.16, 0.05, 0.03, b=(1.49, 0.20, 0.18))
     co.el('CI*', 12.665, 0.023, 0.035, b=(1.49, 0.20, 0.18))
     co.el('CI**', 10.8, 0.5, 0.6, b=(1.49, 0.20, 0.18))
     q.comp.append(co)
     co = sy(2.054663, 14)
     co.el('T01', 100, 50, 50, f='d')
     co.el('CI', 12.71, 0.15, 0.14)
     co.el('CI', 12.52, 0.20, 0.19, b=(0.6, 0.4, 0.5))
     co.el('CI*', 12.14, 0.21, 0.26, b=(0.6, 0.4, 0.5))
     co.el('CI**', 11.65, 0.22, 0.44, b=(0.6, 0.4, 0.5))
     q.comp.append(co)
     co = sy(2.0547245, 36)
     co.el('T01', 100, 50, 50, f='d')
     co.el('CI', 13.485, 0.025, 0.024)
     co.el('CI', 13.31, 0.03, 0.03, b=(2.2, 0.4, 0.4))
     co.el('CI*', 12.941, 0.027, 0.025, b=(2.2, 0.4, 0.4))
     co.el('CI**', 12.16, 0.05, 0.10, b=(2.2, 0.4, 0.4))
     q.comp.append(co)
     co = sy(2.055001, 8)
     co.el('T01', 100, 50, 50, f='d')
     co.el('CI', 12.45, 0.06, 0.07)
     co.el('CI', 12.34, 0.06, 0.06, b=(4.0, 1.4, 1.1))
     co.el('CI*', 11.75, 0.15, 0.29, b=(4.0, 1.4, 1.1))
     co.el('CI**', 10.8, 0.3, 0.6, b=(4.0, 1.4, 1.1))
     q.comp.append(co)
     co = sy(2.055135, 5)
     co.el('T01', 100, 50, 50, f='d')
     co.el('CI', 12.63, 0.03, 0.04)
     co.el('CI', 12.42, 0.03, 0.05, b=(3.3, 0.6, 1.5))
     co.el('CI*', 12.20, 0.07, 0.08, b=(3.3, 0.6, 1.5))
     co.el('CI**', 10.25, 0.41, 0.19, b=(3.3, 0.6, 1.5))
     q.comp.append(co)
     q.comment = 'this is my H2 and CI fit'
     q.full = 'n'
     QSO.append(q)
     return QSO



 # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# add XShooter P94 H2 data

def load_P94():
    global sy
    QSO = sample()
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J013644.02+044039.00
    q = qso('J0136+0440', 2.78, 2.7792369)
    q.telescope = 'Xshooter'
    q.year = 2019
    q.SIMBAD = 'SDSS J013644.02+044039.0'
    q.progID.append('094.A-0362')
    #q.ref.append('Balashev2018')
    q.el('HI', 20.73, 0.01, 0.01)
    q.el('H2', 18.65, 0.06, 0.07)
    q.el('H', 20.74, 0.01, 0.01)
    q.el('f', -1.79, 0.06, 0.07)
    q.el('CI', 14.61, 0.08, 0.09)
    q.el('SiII', 15.45, 0.04, 0.03)
    q.el('SII', 15.28, 0.03, 0.02)
    q.el('FeII', 14.96, 0.01, 0.01)
    q.el('Me', -0.58, 0.03, 0.03)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(2.77901, 7, 4)
    co.el('H2', 16.97, 0.21, 0.14)
    co.el('H2j0', 16.49, 0.32, 0.27, b=(13.2, 1.7, 3.2))
    co.el('H2j1', 16.74, 0.15, 0.16, b=(13.2, 1.7, 3.2))
    co.el('H2j2', 15.79, 0.07, 0.17, b=(13.2, 1.7, 3.2))
    co.el('H2j3', 15.57, 0.12, 0.10, b=(13.2, 1.7, 3.2))
    co.el('H2j4', 14.24, 0.27, 0.50, b=(13.2, 1.7, 3.2))
    co.el('T01', 105, 161, 43, f='d')
    q.comp.append(co)
    co = sy(2.77943, 4, 7)
    co.el('H2', 18.64, 0.06, 0.08)
    co.el('H2j0', 18.42, 0.08, 0.08, b=(7.7, 2.4, 1.9))
    co.el('H2j1', 18.18, 0.13, 0.09, b=(7.7, 2.4, 1.9))
    co.el('H2j2', 15.81, 0.17, 0.17, b=(7.7, 2.4, 1.9))
    co.el('H2j3', 15.73, 0.19, 0.14, b=(7.7, 2.4, 1.9))
    co.el('H2j4', 15.10, 0.10, 0.06, b=(7.7, 2.4, 1.9))
    co.el('T01', 62, 13, 8, f='d')
    co.el('n_H', '$2.16^{+0.13}_{-0.14}$')
    co.el('CI', 14.44, 0.10, 0.12, b=(4.4, 0.2, 0.5))
    co.el('CI*', 14.08, 0.05, 0.05, b=(4.4, 0.2, 0.5))
    co.el('CI**', 13.31, 0.06, 0.06, b=(4.4, 0.2, 0.5))
    co.el('UV', '$-0.19^{+0.17}_{-0.13}$')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J085859.67+174925.32
    q = qso('J0858+1749', 2.65, 2.6252678)
    q.telescope = 'Xshooter'
    q.year = 2019
    q.SIMBAD = 'SDSS J085859.66+174925.1'
    q.progID.append('094.A-0362')
    #q.ref.append('Balashev2018')
    q.el('HI', 20.40, 0.01, 0.01)
    q.el('H2', 19.72, 0.01, 0.02)
    q.el('H', 20.55, 0.01, 0.01)
    q.el('f', -0.53, 0.01, 0.02)
    q.el('CI', 14.39, 0.02, 0.03)
    q.el('SiII', 15.10, 0.02, 0.03)
    q.el('ZnII', 12.64, 0.04, 0.02)
    q.el('SII', 15.12, 0.03, 0.04)
    q.el('FeII', 13.75, 0.02, 0.01)
    q.el('Me', -0.55, 0.03, 0.04)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(2.625251, 18)
    co.el('H2', 19.72, 0.01, 0.02)
    co.el('H2j0', 19.28, 0.02, 0.03, b=(7.8, 0.4, 0.4))
    co.el('H2j1', 19.51, 0.01, 0.03, b=(7.8, 0.4, 0.4))
    co.el('H2j2', 17.78, 0.22, 0.18, b=(7.8, 0.4, 0.4))
    co.el('H2j3', 16.99, 0.22, 0.107, b=(7.8, 0.4, 0.4))
    co.el('H2j4', 15.14, 0.07, 0.09, b=(7.8, 0.4, 0.4))
    co.el('H2j5', 14.53, 0.10, 0.07, b=(7.8, 0.4, 0.4))
    co.el('T01', 102, 6, 7, f='d')
    co.el('CI', 14.26, 0.02, 0.04, b=(3.8, 0.1, 0.1))
    co.el('CI*', 13.74, 0.02, 0.02, b=(3.8, 0.1, 0.1))
    co.el('CI**', 12.96, 0.02, 0.03, b=(3.8, 0.1, 0.1))
    co.el('n_H', '$1.82^{+0.04}_{-0.07}$')
    co.el('UV', '$0.13^{+0.16}_{-0.19}$')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J090609.46+054818.72
    q = qso('J0906+0548', 2.79, 2.5678688)
    q.telescope = 'Xshooter'
    q.year = 2019
    q.SIMBAD = 'SDSS J090609.45+054818.7'
    q.progID.append('094.A-0362')
    #q.ref.append('Balashev2017') 
    q.el('HI', 20.13, 0.01, 0.01)
    q.el('H2', 18.88, 0.02, 0.02)
    q.el('H', 20.18, 0.01, 0.01)
    q.el('f', -1.00, 0.02, 0.02)
    q.el('CI', 13.90, 0.04, 0.03)
    q.el('SiII', 14.99, 0.04, 0.05)
    q.el('ZnII', '<12.70')
    q.el('SII', 15.14, 0.05, 0.08)
    q.el('FeII', 14.24, 0.03, 0.02)
    q.el('Me', -0.16, 0.05, 0.08)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(2.566980, 9, 4)
    co.el('H2', 16.70, 0.11, 0.06)
    co.el('H2j0', 15.91, 0.13, 0.15, b=(7.5, 0.3, 0.3))
    co.el('H2j1', 16.43, 0.09, 0.07, b=(7.5, 0.3, 0.3))
    co.el('H2j2', 15.69, 0.13, 0.10, b=(7.5, 0.3, 0.3))
    co.el('H2j3', 15.95, 0.12, 0.09, b=(7.5, 0.3, 0.3))
    co.el('H2j4', 14.74, 0.05, 0.04, b=(7.5, 0.3, 0.3))
    co.el('H2j5', 14.34, 0.10, 0.04, b=(7.5, 0.3, 0.3))
    co.el('T01', 172, 221, 55, f='d')
    q.comp.append(co)
    co = sy(2.5691809, 11, 73)
    co.el('H2', 18.901, 0.014, 0.014)
    co.el('H2j0', 18.38, 0.01, 0.07, b=(6.8, 0.1, 0.1))
    co.el('H2j1', 18.70, 0.05, 0.01, b=(6.8, 0.1, 0.1))
    co.el('H2j2', 16.87, 0.01, 0.01, b=(6.8, 0.1, 0.1))
    co.el('H2j3', 16.15, 0.07, 0.07, b=(6.8, 0.1, 0.1))
    co.el('H2j4', 15.20, 0.05, 0.04, b=(6.8, 0.1, 0.1))
    co.el('H2j5', 15.00, 0.05, 0.03, b=(6.8, 0.1, 0.1))
    co.el('T01', 116, 26, 4, f='d')
    co.el('CI', 13.50, 0.08, 0.06, b=(3.8, 0.6, 0.5))
    co.el('CI*', 13.61, 0.03, 0.03, b=(3.8, 0.6, 0.5))
    co.el('CI**', 12.89, 0.07, 0.08, b=(3.8, 0.6, 0.5))
    co.el('n_H', '$2.63^{+0.07}_{-0.09}$')
    co.el('UV', '$0.65^{+0.14}_{-0.16}$')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J094649.47+121628.56
    q = qso('J0946+1216', 2.67, 2.6054133)
    q.telescope = 'Xshooter'
    q.year = 2019
    q.SIMBAD = 'SDSS J094649.47+121628.4'
    q.progID.append('094.A-0362')
    #q.ref.append('Balashev2017') 
    q.el('HI', 21.15, 0.02, 0.02)
    q.el('H2', 19.97, 0.01, 0.02)
    q.el('H', 21.20, 0.02, 0.02)
    q.el('f', -0.93, 0.02, 0.03)
    q.el('CI', 14.42, 0.04, 0.02)
    q.el('SiII', 16.06, 0.01, 0.01)
    q.el('ZnII', 13.52, 0.02, 0.01)
    q.el('SII', 15.86, 0.01, 0.01)
    q.el('NiII', 14.36, 0.02, 0.02)
    q.el('CrII', 13.81, 0.02, 0.02)
    q.el('FeII', 15.48, 0.01, 0.01)
    q.el('Me', -0.46, 0.02, 0.02)
    q.Me_ind == 'S'
    #q.Me_ind == 'S'
    q.comp = []
    co = sy(2.606406, 30, 8)
    co.el('H2', 19.96, 0.01, 0.02)
    co.el('H2j0', 19.42, 0.04, 0.02, b=(9.8, 0.8, 0.3))
    co.el('H2j1', 19.81, 0.02, 0.02, b=(9.8, 0.8, 0.3))
    co.el('H2j2', 18.10, 0.12, 0.17, b=(9.8, 0.8, 0.3))
    co.el('H2j3', 16.53, 0.13, 0.15, b=(9.8, 0.8, 0.3))
    co.el('H2j4', 15.19, 0.09, 0.04, b=(9.8, 0.8, 0.3))
    co.el('H2j5', 14.70, 0.09, 0.10, b=(9.8, 0.8, 0.3))
    co.el('T01', 131, 10, 14, f='d')
    co.el('CI', 14.10, 0.05, 0.05, b=(4.6, 0.3, 0.3))
    co.el('CI*', 13.89, 0.03, 0.04, b=(4.6, 0.3, 0.3))
    co.el('CI**', 13.28, 0.05, 0.04, b=(4.6, 0.3, 0.3))
    co.el('n_H', '$2.28^{+0.06}_{-0.06}$')
    co.el('UV', '$0.50^{+0.16}_{-0.17}$')
    q.comp.append(co)
    co = sy(2.607083, 16, 5)
    co.el('H2', 17.26, 0.31, 0.08)
    co.el('H2j0', 16.34, 0.44, 0.21, b=(17.6, 1.6, 0.7))
    co.el('H2j1', 17.10, 0.37, 0.08, b=(17.6, 1.6, 0.7))
    co.el('H2j2', 16.33, 0.08, 0.13, b=(17.6, 1.6, 0.7))
    co.el('H2j3', 15.74, 0.07, 0.05, b=(17.6, 1.6, 0.7))
    co.el('H2j4', 14.50, 0.11, 0.13, b=(17.6, 1.6, 0.7))
    co.el('H2j5', 14.46, 0.12, 0.08, b=(17.6, 1.6, 0.7))
    co.el('T01', '>90', f='d')
    co.el('CI', 13.22, 0.05, 0.06, b=(12.6, 3.1, 1.8))
    co.el('CI*', 13.35, 0.03, 0.03, b=(12.6, 3.1, 1.8))
    co.el('CI**', 12.83, 0.10, 0.18, b=(12.6, 3.1, 1.8))
    co.el('n_H', '$2.79^{+0.10}_{-0.16}$')
    co.el('UV', '$1.55^{+0.20}_{-0.22}$')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J114638.95+074311.28
    q = qso('J1146+0743', 3.02, 2.8403571)
    q.telescope = 'Xshooter'
    q.year = 2019
    q.SIMBAD = 'SDSS J114638.95+074311.2'
    q.progID.append('094.A-0362')
    #q.ref.append('Balashev2017') 
    q.el('HI', 21.54, 0.01, 0.01)
    q.el('H2', 18.82, 0.03, 0.02)
    q.el('H', 21.54, 0.01, 0.01)
    q.el('f', -2.42, 0.03, 0.02)
    q.el('CI', 13.85, 0.03, 0.03)
    q.el('SiII', 16.36, 0.02, 0.02)
    q.el('ZnII', 13.53, 0.02, 0.02)
    q.el('FeII', 15.80, 0.01, 0.01)
    q.el('CrII', 14.13, 0.02, 0.02)
    q.el('NiII', 14.63, 0.01, 0.01)
    q.el('PII', 14.24, 0.04, 0.05)
    #q.el('SII', 16.62, 0.22, 0.20)
    q.el('Me', -0.56, 0.02, 0.02)
    q.Me_ind == 'Zn'
    q.comp = []
    co = sy(2.839459, 19, 8)
    co.el('H2', 18.76, 0.01, 0.01)
    co.el('H2j0', 18.38, 0.01, 0.01, b=(7.6, 0.1, 0.4))
    co.el('H2j1', 18.52, 0.01, 0.01, b=(7.6, 0.1, 0.4))
    co.el('H2j2', 16.57, 0.09, 0.17, b=(7.6, 0.1, 0.4))
    co.el('H2j3', 15.97, 0.09, 0.05, b=(7.6, 0.1, 0.4))
    co.el('H2j4', 14.70, 0.05, 0.10, b=(7.6, 0.1, 0.4))
    co.el('H2j5', '<14.0', b=(7.6, 0.1, 0.4))
    co.el('T01', 91, 3, 2, f='d')
    co.el('CI', 13.28, 0.05, 0.04, b=(16.8, 2.6, 4.0))
    co.el('CI*', 13.10, 0.09, 0.10, b=(16.8, 2.6, 4.0))
    co.el('CI**', 12.93, 0.12, 0.18, b=(16.8, 2.6, 4.0))
    co.el('n_H', '$2.60^{+0.13}_{-0.14}$')
    co.el('UV', '$0.55^{+0.18}_{-0.17}$')
    q.comp.append(co)
    co = sy(2.841629, 10, 14)
    co.el('H2', 17.75, 0.02, 0.05)
    co.el('H2j0', 17.75, 0.02, 0.05, b=(11.4, 0.5, 0.7))
    co.el('H2j1', 17.40, 0.28, 0.37, b=(11.4, 0.5, 0.7))
    co.el('H2j2', 16.06, 0.11, 0.10, b=(11.4, 0.5, 0.7))
    co.el('H2j3', 16.05, 0.09, 0.05, b=(11.4, 0.5, 0.7))
    co.el('H2j4', 15.14, 0.07, 0.05, b=(11.4, 0.5, 0.7))
    co.el('H2j5', 14.62, 0.07, 0.08, b=(11.4, 0.5, 0.7))
    co.el('T01', 57, 25, 16, f='d')
    co.el('CI', 13.17, 0.10, 0.11, b=(2.7, 14.9, 1.9))
    co.el('CI*', 13.07, 0.05, 0.08, b=(2.7, 14.9, 1.9))
    co.el('CI**', 12.85, 0.12, 0.11, b=(2.7, 14.9, 1.9))
    co.el('n_H', '$2.91^{+0.17}_{-0.15}$')
    co.el('UV', '$0.92^{+0.19}_{-0.20}$')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J123602.11+001024.60
    q = qso('J1236+0010', 3.03, 3.0316079)
    q.telescope = 'Xshooter'
    q.year = 2019
    q.SIMBAD = 'SDSS J123602.11+001024.6'
    q.progID.append('094.A-0362')
    #q.ref.append('Balashev2017') 
    q.el('HI', 20.78, 0.01, 0.01)
    q.el('H2', 19.76, 0.01, 0.01)
    q.el('H', 20.86, 0.01, 0.01)
    q.el('f', -0.80, 0.01, 0.01)
    q.el('CI', 14.35, 0.33, 0.25)
    q.el('SiII', 15.69, 0.06, 0.05)
    q.el('OI', 16.77, 0.44, 0.213)
    q.el('SII', 15.39, 0.03, 0.04)
    q.el('FeII', 14.66, 0.03, 0.03)
    q.el('Me', -0.51, 0.03, 0.04)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(3.03292, 27, 18)
    co.el('H2', 19.76, 0.01, 0.01)
    co.el('H2j0', 19.21, 0.03, 0.02, b=(2.3, 0.2, 0.2))
    co.el('H2j1', 19.56, 0.02, 0.01, b=(2.3, 0.2, 0.2))
    co.el('H2j2', 18.46, 0.03, 0.02, b=(2.3, 0.2, 0.2))
    co.el('H2j3', 18.10, 0.04, 0.02, b=(2.3, 0.2, 0.2))
    co.el('H2j4', 16.10, 0.49, 0.16, b=(2.3, 0.2, 0.2))
    co.el('H2j5', 15.11, 0.41, 0.28, b=(2.3, 0.2, 0.2))
    co.el('T01', 122, 9, 7, f='d')
    co.el('CI', 14.28, 0.36, 0.37, b=(2.0, 0.4, 0.2))
    co.el('CI*', 13.70, 0.11, 0.08, b=(2.0, 0.4, 0.2))
    co.el('CI**', 13.18, 0.11, 0.16, b=(2.0, 0.4, 0.2))
    co.el('n_H', '$1.50^{+0.55}_{-0.55}$')
    co.el('UV', '$0.44^{+0.34}_{-0.41}$')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J234730.76-005131.68
    q = qso('J2347-0051', 2.62, 2.5872537)
    q.telescope = 'Xshooter'
    q.year = 2019
    q.SIMBAD = 'SDSS J234730.76-005131.5'
    q.progID.append('094.A-0362')
    #q.ref.append('Balashev2017') 
    q.el('HI', 20.47, 0.01, 0.01)
    q.el('H2', 19.44, 0.01, 0.01)
    q.el('H', 20.54, 0.01, 0.01)
    q.el('f', -0.80, 0.01, 0.01)
    q.el('CI', 14.35, 0.35, 0.25)
    q.el('SiII', 15.14, 0.05, 0.03)
    q.el('ZnII', 12.50, 0.06, 0.09)
    q.el('SII', 14.85, 0.05, 0.03)
    q.el('FeII', 14.51, 0.01, 0.02)
    q.el('NiII', 13.54, 0.07, 0.06)
    q.el('CrII', 12.95, 0.15, 0.09)
    q.el('Me', -0.81, 0.05, 0.04)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(2.587969, 10, 9)
    co.el('H2', 19.44, 0.01, 0.01)
    co.el('H2j0', 19.14, 0.01, 0.01, b=(6.1, 0.2, 0.1))
    co.el('H2j1', 19.12, 0.01, 0.01, b=(6.1, 0.2, 0.1))
    co.el('H2j2', 16.23, 0.07, 0.10, b=(6.1, 0.2, 0.1))
    co.el('H2j3', 15.92, 0.07, 0.06, b=(6.1, 0.2, 0.1))
    co.el('H2j4', 14.77, 0.04, 0.05, b=(6.1, 0.2, 0.1))
    co.el('H2j5', 14.10, 0.09, 0.10, b=(6.1, 0.2, 0.1))
    co.el('T01', 76, 2, 2, f='d')
    co.el('CI', 14.28, 0.36, 0.37, b=(2.0, 0.4, 0.2))
    co.el('CI*', 13.70, 0.11, 0.08, b=(2.0, 0.4, 0.2))
    co.el('CI**', 13.18, 0.11, 0.16, b=(2.0, 0.4, 0.2))
    co.el('n_H', '$2.00^{+0.42}_{-0.52}$')
    co.el('UV', '$-0.40^{+0.32}_{-0.41}$')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)
    
    return QSO
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# add GRB H2 data
    
def load_GRB():    
    global sy
    GRB = sample()
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add GRB 080607
    q = qso('GRB 080607', 3.036, 3.036)
    #q.coord = ['J2000', 130.0020, 0.002112]
    q.telescope = 'KECK/LRIS'
    q.year = 2009
    #q.coord = ['J2000', 00.0 -00.0]
    #q.SIMBAD = ''
    q.ref.append('Prochaska2009') 
    q.el('HI', 22.7, 0.15, 0.15)
    q.el('H2', 21.2, 0.2, 0.2)
    q.el('CO', 16.5, 0.3, 0.3)
    q.el('f', -1.23, 0.25, 0.26)
    q.el('Me', '>-0.2')
    q.Me_ind = 'O'
    q.full = 'n'
    GRB.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add GRB 120815
    q = qso('GRB 120815', 2.36, 2.36)
    #q.coord = ['J2000', 00.0, 0.0]
    # q.SIMBAD = ''
    q.telescope = 'VLT/XShooter'
    q.year = 2013
    #q.coord = ['J2000', 001602.406 -001225.08]
    #q.SIMBAD = 'LBQS 0013-0029'
    q.ref.append('Kruhler2013') 
    q.el('HI', 21.95, 0.1, 0.1)
    q.el('H2', 20.52, 0.04, 0.04)
    q.el('CO', '<15.0')
    q.el('f', -1.16, 0.10, 0.11)
    q.el('Me', -1.15, 0.12, 0.12)
    q.Me_ind = 'Zn'
    q.full = 'n'
    GRB.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add GRB 121024A
    q = qso('GRB 121024A', 2.3, 2.3)
    #q.coord = ['J2000', 130.0020, 0.002112]
    q.telescope = 'VLT/XShooter'
    q.year = 2014
    #q.coord = ['J2000', 00.0 -00.0]
    #q.SIMBAD = ''
    q.ref.append('Friis2015') 
    q.el('HI', 21.88, 0.10, 0.10)
    q.el('H2', 19.90, 0.2, 0.2)
    q.el('CO', '<14.4')
    q.el('f', -1.61, 0.25, 0.26)
    q.el('Me', -0.6, 0.2, 0.2)
    q.Me_ind = 'Zn'
    q.full = 'n'
    GRB.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add GRB 120327A
    q = qso('GRB 120327A', 2.8145, 2.8145)
    #q.coord = ['J2000', 130.0020, 0.002112]
    q.telescope = 'VLT/XShooter'
    q.year = 2014
    #q.coord = ['J2000', 00.0-00.0]
    #q.SIMBAD = ''
    q.ref.append('D\'Elia2014') 
    q.el('HI', 22.01, 0.09, 0.09)
    q.el('H2', 16.5, 1.2, 1.2)
    q.el('f', -5.0, 1.2, 1.2)
    q.el('Me', -1.17, 0.11, 0.11)
    q.Me_ind = 'Zn'
    q.full = 'n'
    GRB.append(q)
    
    return GRB

def load_LMC():
    global sy
    QSO = sample()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    q = qso('SK-6705', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Tumlinson2002')
    q.el('HI', 20.88, 0.12,0.15)
    q.el('H2', 19.46, 0.05, 0.05)
    q.el('CI',13.89,0.02,0.02)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'Zn'
    q.comp = []
    co = sy(288.3/3e5, 0, 0) #comp_1, from table 4 in Welty 2016
    co.el('T01', 57, 5, 4, f='d')
    co.el('H2', 19.46, 0.05, 0.05, b=(6.2, 1.4, 1.1))
    co.el('H2', 19.30, 0.05, 0.05, J=0)
    co.el('H2', 18.96, 0.06, 0.06, J=1)
    co.el('H2', 15.66, 0.32, 0.20, J=2)
    co.el('H2', 15.29, 0.22, 0.16, J=3)
    co.el('H2', 14.62, 0.08, 0.07, J=4)
    co.el('CI', 13.62,0.02)
    co.el('CIj0', 13.41, 0.04, 0.04)
    co.el('CIj1', 13.10, 0.06, 0.07)
    co.el('CIj2', 12.57, 0.09, 0.11)
    co.el('PDRnH', 2.22, 0.13, 0.10)
    co.el('PDRuv', -0.68, 0.16, 0.20)
    q.comp.append(co)
    co = sy(288.3 / 3e5, 0, 0)  # comp_1, from table 4 in Welty 2016
    co.el('T01', 57, 5, 4, f='d')
    co.el('H2', 19.46, 0.05, 0.05, b=(6.2, 1.4, 1.1))
    co.el('H2', 19.30, 0.05, 0.05, J=0)
    co.el('H2', 18.96, 0.06, 0.06, J=1)
    co.el('H2', 15.66, 0.32, 0.20, J=2)
    co.el('H2', 15.29, 0.22, 0.16, J=3)
    co.el('H2', 14.62, 0.08, 0.07, J=4)
    co.el('CI', 13.15, 0.04)
    co.el('CIj0', 12.98, 0.07, 0.08)
    co.el('CIj1', 12.51, 0.15, 0.22)
    co.el('CIj2', 12.10, 0.17, 0.26)
    co.el('PDRnH', 2.13, 0.19, 0.19)
    co.el('PDRuv', -0.86, 0.20, 0.14)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-70115', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Tumlinson2001')
    q.el('HI', 21.37, 0.12,0.15)
    q.el('H2', 19.94, 0.07, 0.07)
    q.el('CI',13.89,0.03,0.03)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'Zn'
    q.comp = []
    co = sy(24.2/3e5, 0, 0) # comp_1
    co.el('H2', 19.94, 0.07, 0.07)
    co.el('T01', 53, 6, 5, f='d')
    co.el('H2', 19.80, 0.07, J=0)
    co.el('H2', 19.37, 0.07, J=1)
    co.el('H2', 17.55, 0.34, J=2)
    co.el('H2', 17.12, 0.14, 0.81, J=3)
    co.el('H2', 15.40, 0.75, 0.43, J=4)
    co.el('H2', 14.80, 0.63, 0.24, J=5)
    co.el('CI', 13.25, 0.04, 0.04)
    co.el('CIj0', 13.14, 0.05, 0.05)
    co.el('CIj1', 12.45, 0.13, 0.17)
    co.el('CIj2', 12.10, 0.16, 0.25)
    co.el('PDRnH', 1.63, 0.18, 0.17)
    co.el('PDRuv', -0.37, 0.35, 0.31)
    q.comp.append(co)
    co = sy(30.5/3e5, 0, 0) # comp_2
    co.el('H2', 19.94, 0.07, 0.07)
    co.el('T01', 53, 6, 5, f='d')
    co.el('H2', 19.80, 0.07, J=0)
    co.el('H2', 19.37, 0.07, J=1)
    co.el('H2', 17.55, 0.34, J=2)
    co.el('H2', 17.12, 0.14, 0.81, J=3)
    co.el('H2', 15.40, 0.75, 0.43, J=4)
    co.el('H2', 14.80, 0.63, 0.24, J=5)
    co.el('CI', 13.35, 0.03, 0.03)
    co.el('CIj0', 13.19, 0.04, 0.04)
    co.el('CIj1', 12.73, 0.07, 0.08)
    co.el('CIj2', 12.20, 0.16, 0.24)
    co.el('PDRnH', 1.81, 0.12, 0.13)
    co.el('PDRuv', -0.28, 0.29, 0.27)
    q.comp.append(co)
    co = sy(220.2 / 3e5, 0, 0)  # comp_2
    co.el('H2', 19.94, 0.07, 0.07)
    co.el('T01', 53, 6, 5, f='d')
    co.el('H2', 19.80, 0.07, J=0)
    co.el('H2', 19.37, 0.07, J=1)
    co.el('H2', 17.55, 0.34, J=2)
    co.el('H2', 17.12, 0.14, 0.81, J=3)
    co.el('H2', 15.40, 0.75, 0.43, J=4)
    co.el('H2', 14.80, 0.63, 0.24, J=5)
    co.el('CI', 13.70, 0.03, 0.03)
    co.el('CIj0', 13.38, 0.06, 0.07)
    co.el('CIj1', 13.26, 0.07, 0.08)
    co.el('CIj2', 12.90, 0.14, 0.21)
    co.el('PDRnH', 2.31, 0.16, 0.15)
    co.el('PDRuv', 0.04, 0.26, 0.20)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)


    q = qso('HD 32109', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Tumlinson2002')
    q.el('H2', 18.67, 0.22, 0.26)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 18.67, 0.22, 0.26)
    co.el('H2', 18.40, 0.13, 0.18, J=0)
    co.el('H2', 18.33, 0.26, 0.37, J=1)
    co.el('H2', 15.51, 1.79, 0.22, J=2)
    co.el('H2', 15.55, 1.86, 0.26, J=3)
    co.el('H2', 14.93, 1.57, 0.13, J=4)
    co.el('H2', 14.58, 0.68, 0.07, J=5)
    co.el('T01', 72, 5, 2, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-6521', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Tumlinson2002')
    q.el('H2', 18.21, 0.14, 0.35)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 18.21, 0.14, 0.35)
    co.el('H2', 17.81, 0.13, 0.41, J=0)
    co.el('H2', 17.94, 0.09, 0.27, J=1)
    co.el('H2', 16.81, 0.40, 1.48, J=2)
    co.el('H2', 16.70, 0.43, 1.47, J=3)
    co.el('H2', 14.23, 1.62, 0.15, J=4)
    co.el('T01', 89, 3, 3, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-6852', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Tumlinson2002')
    q.el('H2', 19.47, 0.06, 0.05)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 19.47, 0.06, 0.05)
    co.el('T01', 60, 5, 4, f='d')
    co.el('H2', 19.28, 0.05, J=0)
    co.el('H2', 19.01, 0.05, J=1)
    co.el('H2', 15.13, 1.72,0.37, J=2)
    co.el('H2', 15.39, 1.72,0.60, J=3)
    co.el('H2', 14.79, 1.90,0.21, J=4)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-7145', 0, 0)
    q.telescope = 'FUSE'
    # q.ref.append('Tumlinson2001')
    q.el('H2', 18.63, 0.09, 0.19)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 18.63, 0.09, 0.19)
    co.el('T01', 98, 13, 11, f='d')
    co.el('H2', 18.06, 0.09, J=0)
    co.el('H2', 18.26, 0.03, J=1)
    co.el('H2', 17.89, 0.17, 1.14, J=2)
    co.el('H2', 17.63, 0.15, 1.11, J=3)
    co.el('H2', 16.61, 0.25, 1.05, J=4)
    co.el('H2', 15.85, 0.11, 0.09, J=5)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('HD37680', 0, 0)
    q.telescope = 'FUSE'
    # q.ref.append('Tumlinson2001')
    q.el('H2', 18.94, 0.06, 0.19)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 18.94, 0.06, 0.19)
    co.el('T01', 65, 3, 17, f='d')
    co.el('H2', 18.71, 0.03,0.09, J=0)
    co.el('H2', 18.52, 0.07,0.36, J=1)
    co.el('H2', 16.87, 0.52, 1.16, J=2)
    co.el('H2', 17.11, 0.37, 1.53, J=3)
    co.el('H2', 14.88, 0.41, 0.21, J=4)
    co.el('H2', 14.33, 0.15, 0.10, J=5)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-66172', 0, 0)
    q.telescope = 'FUSE'
    # q.ref.append('Tumlinson2001')
    q.el('H2', 18.21, 0.39, 0.32)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 18.21, 0.39, 0.32)
    co.el('T01', 41, 35, 15, f='d')
    co.el('H2', 18.15, 0.15, 0.27, J=0)
    co.el('H2', 17.32, 0.84, 1.05, J=1)
    co.el('H2', 15.45, 1.94, 0.26, J=2)
    co.el('H2', 15.76, 1.76, 0.51, J=3)
    co.el('H2', 14.50, 1.04, 0.08, J=4)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-68135', 0, 0)
    q.telescope = 'FUSE'
    # q.ref.append('Tumlinson2001')
    q.el('H2', 19.87, 0.07, 0.07)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 19.87, 0.07, 0.07)
    co.el('T01', 91, 16, 12, f='d')
    co.el('H2', 19.47, 0.12, 0.12, J=0)
    co.el('H2', 19.61, 0.03, 0.03, J=1)
    co.el('H2', 18.42, 0.09, 0.23, J=2)
    co.el('H2', 18.05, 0.11, 0.35, J=3)
    co.el('H2', 17.17, 0.49, 1.19, J=4)
    co.el('H2', 16.91, 0.75, 1.11, J=5)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-69246', 0, 0)
    q.telescope = 'FUSE'
    # q.ref.append('Tumlinson2001')
    q.el('H2', 19.71, 0.03, 0.03)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 19.71, 0.03, 0.03)
    co.el('T01', 74, 3, 3, f='d')
    co.el('H2', 19.42, 0.02, 0.02, J=0)
    co.el('H2', 19.38, 0.03, 0.03, J=1)
    co.el('H2', 17.75, 0.05, 1.19, J=2)
    co.el('H2', 17.83, 0.08, 1.44, J=3)
    co.el('H2', 15.88, 0.31, 0.65, J=4)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)




    q = qso('SK-68 73', 0, 0)
    q.telescope = 'HST/STIS'
    q.ref.append('Welty2016')
    q.el('H2', 20.09, 0.20, 0.20)
    q.el('Me', -0.4, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 20.09, 0.20, 0.20)
    co.el('T01', 57, 0, 0, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)


    return QSO

def load_SMC():
    global sy
    QSO = sample()


    #SMC
    q = qso('SK13', 0, 0) #AV18 Cartledge_ApJ_630_3555_2005
    q.telescope = 'HST/STIS'
    q.ref.append('Welty2016')
    #q.el('HI', 20.88, 0.12,0.15)
    q.el('H2', 20.36, 0.07, 0.07)
    q.el('CI',14.00,0.04,0.03) #13.29
    q.el('Me', -0.7,0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(121.3/3e5, 0, 0) #comp_1 in Welty2016 from table 6
    co.el('H2', 20.36, 0.07, 0.07)
    co.el('T01', 66, 5, 5, f='d')
    co.el('H2', 20.13, 0.1, 0.1, J=0)
    co.el('H2', 19.97, 0.1, 0.1, J=1)
    co.el('H2', 17.58, 0.3, 0.3, J=2)
    co.el('H2', 17.79, 0.3, 0.3, J=3)
    co.el('H2', 16.06, 0.3, 0.3, J=4)
    co.el('H2', 15.36, 0.3, 0.3, J=5)
    co.el('CI', 13.12, 0.04, 0.04)
    co.el('CIj0', 12.92, 0.07, 0.07)
    co.el('CIj1', 12.48, 0.10, 0.11)
    co.el('CIj2', 12.27, 0.16, 0.25)
    co.el('PDRnH', 2.08, 0.14, 0.17)
    co.el('PDRuv', -0.23, 0.26, 0.25)
    q.comp.append(co)

    co = sy(145.7 / 3e5, 0, 0)  # comp_2 in Welty2016 from table 6
    co.el('H2', 20.36, 0.07, 0.07)
    co.el('T01', 66, 5, 5, f='d')
    co.el('H2', 20.13, 0.1, 0.1, J=0)
    co.el('H2', 19.97, 0.1, 0.1, J=1)
    co.el('H2', 17.58, 0.3, 0.3, J=2)
    co.el('H2', 17.79, 0.3, 0.3, J=3)
    co.el('H2', 16.06, 0.3, 0.3, J=4)
    co.el('H2', 15.36, 0.3, 0.3, J=5)
    co.el('CI', 13.51, 0.09, 0.09)
    co.el('CIj0', 13.17, 0.16, 0.19)
    co.el('CIj1', 12.99, 0.19, 0.26)
    co.el('CIj2', 12.89, 0.16, 0.19)
    co.el('PDRnH', 2.67, 0.30, 0.24)
    co.el('PDRuv', 0.18, 0.24, 0.23)
    q.comp.append(co)

    co = sy(147.6 / 3e5, 0, 0)  # comp_3 in Welty2016 from table 6
    co.el('H2', 20.36, 0.07, 0.07)
    co.el('T01', 66, 5, 5, f='d')
    co.el('H2', 20.13, 0.1, 0.1, J=0)
    co.el('H2', 19.97, 0.1, 0.1, J=1)
    co.el('H2', 17.58, 0.3, 0.3, J=2)
    co.el('H2', 17.79, 0.3, 0.3, J=3)
    co.el('H2', 16.06, 0.3, 0.3, J=4)
    co.el('H2', 15.36, 0.3, 0.3, J=5)
    co.el('CI', 13.54, 0.05)
    co.el('CIj0', 12.97, 0.13, 0.17)
    co.el('CIj1', 13.23, 0.07)
    co.el('CIj2', 12.92, 0.11, 0.13)
    co.el('PDRnH', 3.12, 0.25, 0.49)
    co.el('PDRuv', 0.18, 0.10, 1.18)
    q.comp.append(co)

    co = sy(149.9 / 3e5, 0, 0)  # comp_4 in Welty2016 from table 6
    co.el('H2', 20.36, 0.07, 0.07)
    co.el('T01', 66, 5, 5, f='d')
    co.el('H2', 20.13, 0.1, 0.1, J=0)
    co.el('H2', 19.97, 0.1, 0.1, J=1)
    co.el('H2', 17.58, 0.3, 0.3, J=2)
    co.el('H2', 17.79, 0.3, 0.3, J=3)
    co.el('H2', 16.06, 0.3, 0.3, J=4)
    co.el('H2', 15.36, 0.3, 0.3, J=5)
    co.el('CI', 13.31, 0.06)
    co.el('CIj0', 12.85, 0.15, 0.20)
    co.el('CIj1', 12.88, 0.11, 0.13)
    co.el('CIj2', 12.76, 0.14, 0.18)
    co.el('PDRnH', 2.94, 0.31, 0.26)
    co.el('PDRuv', 0.31, 0.24, 0.21)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK18', 0, 0) #AV26
    q.telescope = 'FHST/STIS'
    q.ref.append('Welty2016')
    q.el('HI',21.9,0.15)
    q.el('H2', 20.63, 0.05, 0.05)
    q.el('CI',14.04,0.03,0.03) #13.58
    q.el('Me', -0.7,0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(13.6/3e5, 0, 0) # comp_1 in Welty2016 from table 6
    co.el('H2', 20.63, 0.05, 0.05)
    co.el('T01', 53, 4, 4, f='d')
    co.el('H2', 20.49, 0.04, 0.04, J=0)
    co.el('H2', 20.06, 0.07, 0.07, J=1)
    co.el('H2', 17.99, 0.04, 1.58,'l', J=2)
    co.el('H2', 17.91, 0.04, 1.56,'l', J=3)
    co.el('H2', 16.74, 0.45, 1.75,'l', J=4)
    co.el('CI', 13.13, 0.06, 0.06)
    co.el('CIj0', 13.01, 0.08, 0.08)
    co.el('CIj1', 12.45, 0.17, 0.25)
    co.el('CIj2', 11.73, 0.25, 0.06)
    co.el('PDRnH', 1.94, 0.16, 0.16)
    co.el('PDRuv',-0.23, 0.25, 0.36)
    q.comp.append(co)

    co = sy(123.9/3e5, 0, 0) # comp_2 in Welty2016 from table 6
    co.el('H2', 20.63, 0.05, 0.05)
    co.el('T01', 53, 4, 4, f='d')
    co.el('H2', 20.49, 0.04, 0.04, J=0)
    co.el('H2', 20.06, 0.07, 0.07, J=1)
    co.el('H2', 17.99, 0.04, 1.58, J=2)
    co.el('H2', 17.91, 0.04, 1.56, J=3)
    co.el('H2', 16.74, 0.45, 1.75, J=4)
    co.el('CI', 14.04, 0.06, 0.06)
    co.el('CIj0', 13.64, 0.06, 0.06)
    co.el('CIj1', 13.65, 0.05, 0.05)
    co.el('CIj2', 13.32, 0.05, 0.06)
    co.el('PDRnH', 2.76, 0.13, 0.10)
    co.el('PDRuv', 0.27, 0.16, 0.27)
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-143', 0, 0)
    q.telescope = 'HST/STIS'
    q.ref.append('Welty2016')
    q.el('H2', 20.93, 0.09, 0.09)
    q.el('Me', -0.7, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 20.93, 0.09, 0.09)
    co.el('T01', 45, 0, 0, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK-155', 0, 0)
    q.telescope = 'HST/STIS'
    q.ref.append('Welty2016')
    q.el('H2', 19.15, 0.10, 0.10)
    q.el('Me', -0.7, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 19.15, 0.10, 0.10)
    co.el('T01', 82, 0, 0, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('AV69', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Rachford2002')
    q.el('H2', 18.73, 0.16)
    q.el('Me', -0.7, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 18.73, 0.16)
    co.el('H2', 18.36, 0.13, J=0)
    co.el('H2', 18.49, 0.16, J=1)
    co.el('T01', 90, 30, 18, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('AV75', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Rachford2002')
    q.el('H2', 18.51, 0.21, 0.62)
    q.el('Me', -0.7, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 18.51, 0.21, 0.62)
    co.el('H2', 18.42, 0.20,0.58, J=0)
    co.el('H2', 17.80, 0.26,0.88, J=1)
    co.el('T01', 47, 23, 18, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('AV95', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Rachford2002')
    q.el('H2', 19.40, 0.08)
    q.el('Me', -0.7, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 19.40, 0.08)
    co.el('H2', 19.09, 0.09, 0.09, J=0)
    co.el('H2', 19.10, 0.06, 0.06, J=1)
    co.el('T01', 78, 10, 7, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('AV207', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Rachford2002')
    q.el('H2', 19.40, 0.08)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 19.40, 0.08)
    co.el('H2', 19.09, 0.10, J=0)
    co.el('H2', 19.09, 0.06, J=1)
    co.el('T01', 78, 10, 9, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK388', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Rachford2002')
    q.el('H2', 19.40, 0.23)
    q.el('Me', -0.7, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 19.40, 0.23)
    co.el('H2', 19.19, 0.11, J=0)
    co.el('H2', 18.96, 0.38, J=1)
    co.el('T01', 63, 30, 15, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    q = qso('SK159', 0, 0)
    q.telescope = 'FUSE'
    q.ref.append('Rachford2002')
    q.el('H2', 18.94, 0.14, 0.29)
    q.el('Me', -0.7, 0.1)
    q.Me_ind == 'S'
    q.comp = []
    co = sy(0, 0, 0)
    co.el('H2', 18.94, 0.14, 0.29)
    co.el('H2', 18.60, 0.13,0.23, J=0)
    co.el('H2', 18.67, 0.13,0.29, J=1)
    co.el('T01', 84, 30, 20, f='d')
    q.comp.append(co)
    q.full = 'n'
    QSO.append(q)

    return QSO

def load_Magellan():
    global sy
    QSO = load_LMC()
    QSO.append(load_SMC())
    ex = load_LMC()
    ex.append(load_SMC())
    for q in QSO.values():
        if q.CI.col.val==0.0:
            ex.remove(q)

    return ex

def load_MV():
     global sy
     QSO = sample()
     c = 3e+05  # sight velocity

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




     ################################################33
     # Sonnentrucker_2007+ sample
     if 1:
         q = qso('HD24534', 0., 0.) #X Per
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '163.08-17.14']  # from table 1
         q.ref.append('Jensen2010')
         q.ref.append('Sonnetrucker2007')
         q.ref.append('Jenkins2011')
         #q.ref.append('Sheffer2008')
         q.el('HI', 20.73, 0.06, 0.06)
         q.el('H2', 20.92, 0.03, 0.03)
         q.el('T01', 57, 3, 3, f='d')
         q.el('EBV', 0.59, 0.0, 0.0, f='d') #ref De Cia 2021
         q.el('Av', 2.05, 0.0, 0.0, f='d')
         q.el('Rv', 3.1, 0.0, 0.0, f='d')
         q.el('CI', 14.98, 0.10, 0.10) #ref 'Burgh2010'
         q.el('Me', 0.155, 0.07, 0.07) #ref Ritchey
         q.el('Fstar', 1.285,0.093,0.093,f='d') #Ritchey2023
         q.el('Me_ISM', 0.155, 0.069, 0.068,f='d')  # Ritchey2023
         q.el('Me_[O/H]',-0.19,0.08,0.08)# Ritchey2023
         q.el('D', 792, 35, 35, f='d') #ref De Cia 2021
         q.el('CO', 16.01,0.08,0.08)
         q.el('PII', 14.42,0.05,0.05) #Lebouteiller, Kuassivi & Ferlet 2018
         s = []
         # v = 10      # if you dont know it write 0
         co = sy(0, 0)
         co.el('T01', 57, 3, 3, f='d')
         co.el('T02', 72, 1, 1, f='d')  # calc boot
         co.el('TC2', 45, 10, 10,  f='d')
         co.el('H2', 20.92, 0.03, 0.03)  # ref 'Jensen2010'
         co.el('H2j0', 20.76, 0.03, 0.03, b=(2.6, 0.1, 0.1))
         co.el('H2j1', 20.42, 0.06, 0.06, b=(2.6, 0.1, 0.1))
         co.el('H2j2', 18.40, 0.01, 0.01, b=(2.6, 0.1, 0.1))
         co.el('H2j3', 17.07, 0.03, 0.03, b=(2.6, 0.1, 0.1))
         co.el('H2j4', 15.20, 0.05, 0.04, b=(2.6, 0.1, 0.1))
         co.el('H2j5', 14.21, 0.07, 0.07, b=(2.6, 0.1, 0.1))
         co.el('CI', 14.81, 0.03, 0.03) # ref 'Burgh2010'
         #co.el('CI', 13.97, 0.03, 0.03)  # ref 'Jenkins2011 table'
         co.el('CIj0', 13.65, 0.05, 0.05) # ref 'Jenkins201' The total NCI < Burgh value
         co.el('CIj1', 13.56, 0.05, 0.05)
         co.el('CIj2', 13.07, 0.05, 0.05)
         co.el('CO', 16.01, 0.08, 0.09)# ref "Sonnentrucker2007
         co.el('COj0', 15.69, 0.14,0.20)
         co.el('COj1', 15.64, 0.08,0.10)
         co.el('COj2', 14.96, 0.06, 0.07)
         co.el('COj3', 13.83, 0.15, 0.26)
         co.el('PDRnH',2.62,0.10,0.06) # PDR fit
         co.el('PDRuv',-0.05, 0.11, 0.11)
         co.el('P_ci', 4.56, 0.06, 0.06) #Ci+H2 PDR fit
         co.el('n_ci', 2.35, 0.11, 0.11)
         co.el('uv_ci', 0.40, 0.07, 0.05)
         co.el('P_co', 4.49, 0.13, 0.12) #CO+H2 PDR fit
         co.el('n_co', 2.44, 0.07, 0.07)
         co.el('uv_co', 0.45, 0.06, 0.08)
         co.el('T_co', 5.34, 0.46, 0.50, f='d')
         #co.el('CLnH',2.58, 0.09,0.06) #CI+H2 Cloudy fit
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.55, 0.08,0.07)
         co.el('uv_co_pdr', -0.01, 0.18,0.14)
         co.el('n_ci_pdr', 2.55, 0.12,0.09)
         co.el('uv_ci_pdr', 0.03, 0.14,0.13)
         co.el('n_co_goldsmith', 2.2, 0, 0)
         co.el('xco', 21.46, 0.10, 0.09)
         co.el('wco', -0.54, 0.09, 0.11)
         co.el('wco_emission', 5.31, 0, 0,f='d') #form Liszt+ 2008
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.58, 0.11, 0.08)
         co.el('uv_ci_3dpdr', 0.05, 0.15, 0.12)
         co.el('n_co_3dpdr', 2.64, 0.06, 0.08)
         co.el('uv_co_3dpdr', 0.09, 0.06, 0.10)
         co.el('aG',-0.71,0.14,0.16)  #based on Me from Ritchey2023
         # pdr estimate in CO region
         co.el('tgas', 1.77,0.06,0.05)
         co.el('ngas(co)', 2.40,0.09,0.09)
         co.el('pgas', 4.17,0.12,0.12)
         co.el('ngas(ci)', 2.35, 0.13, 0.14)
         co.el('tgas(ci)', 1.88,0.07,0.09)
         #
         co.el('pci-tripp', 4.17, 0.2, 0.2)
         co.el('uvci-tripp', 1.16, 0.2, 0.2)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD 27778
         q = qso('HD27778', 0., 0.) #62 Tau
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '172.76-17.39']
         q.ref.append('Jensen2010')
         q.ref.append('Sonnetrucker2007')
         q.ref.append('Jenkins2011')
         #q.ref.append('Sheffer2008')
         q.ref.append('Burgh2010')
         q.el('HI', 20.95, 0.10, 0.10)
         q.el('H2', 20.79, 0.03, 0.03)
         q.el('T01', 56, 5, 5, f='d')
         q.el('EBV', 0.37, 0.03, 0.03, f='d') #ref De Cia 2021
         q.el('Av', 0.95, 0.08, 0.08, f='d')
         q.el('Rv', 2.59, 0.24, 0.24, f='d') #ref De Cia 2021
         q.el('CI', 15.06, 0.05, 0.05) # ref Burgh2010
         #q.el('Me', -0.44, 0.12, 0.12) #ref De Cia 2021
         q.el('Me', 0.0, 0.1, 0.1)  # ref mean Ritchey
         q.el('MeZou', -0.25, 0.08, 0.10)  # ref de Cia 2021
         q.el('D', 222, 2, 2, f='d') #ref De Cia 2021
         q.el('CO', 16.07,0.04,0.04)
         q.el('O/H', 269,54,54, f='d') #O/H gas Zou 2021
         q.el('Me_[O/H]', -0.15, 0.1, 0.1)  # Ritchey2023
         q.el('Me', -0.27, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 56, 5, 5, f='d')
         co.el('T02', 77, 1.4, 1.4, f='d')
         co.el('TC2', 50, 10, 10,  f='d')
         co.el('H2', 20.79, 0.03, 0.03)  # ref Jensen2010
         co.el('H2j0', 20.64, 0.05, 0.05, b=(3.0, 0.2, 0.1))
         co.el('H2j1', 20.27, 0.10, 0.10, b=(3.0, 0.2, 0.1))
         co.el('H2j2', 18.47, 0.02, 0.02, b=(3.0, 0.2, 0.1))
         co.el('H2j3', 17.55, 0.04, 0.03, b=(3.0, 0.2, 0.1))
         co.el('H2j4', 15.64, 0.06, 0.07, b=(3.0, 0.2, 0.1))
         co.el('H2j5', 14.33, 0.17, 0.19, b=(3.0, 0.2, 0.1))
         #co.el('CI', 15.06, 0.08, 0.07) # ref Burgh2010
         co.el('CI', 15.08, 0.08, 0.07)
         co.el('CIj0', 14.95, 0.10, 0.10)# ref Jenkins2010
         co.el('CIj1', 14.35, 0.10, 0.10)
         co.el('CIj2', 13.94, 0.10, 0.10)
         co.el('CO', 16.07, 0.04,0.04) # ref Sonnentrucker2007
         co.el('COj0', 15.72, 0.13,0.18)
         co.el('COj1', 15.73, 0.04,0.04)
         co.el('COj2', 15.10, 0.05, 0.05)
         co.el('COj3', 14.07, 0.09, 0.11)
         co.el('PDRnH',2.04,0.13,0.09)
         co.el('PDRuv',-0.23, 0.20, 0.17)
         #co.el('P_ci', 3.84, 0.12, 0.13)
         co.el('P_ci', 4.11, 0.06, 0.06)
         co.el('n_ci', 1.85,  0.13, 0.11)
         co.el('uv_ci', 0.22,0.12, 0.10)
         co.el('P_co', 4.61, 0.07, 0.08)
         co.el('n_co', 2.58, 0.11, 0.09)
         co.el('uv_co', 0.22, 0.12, 0.08)
         co.el('T_co', 5.82, 0.39, 0.30, f='d')
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.67, 0.10, 0.07)
         co.el('uv_co_pdr', 0.07, 0.16, 0.14)
         co.el('n_ci_pdr', 2.06, 0.14, 0.13)
         co.el('uv_ci_pdr', -0.23, 0.20, 0.17)
         co.el('n_co_goldsmith', 2.2, 0.1, 0.1)
         co.el('xco', 21.22, 0.06, 0.05)
         co.el('wco', -0.43, 0.06, 0.06)
         co.el('wco_emission', 7.17, 0, 0, f='d')  # form Liszt+ 2008
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.12, 0.11, 0.12)
         co.el('uv_ci_3dpdr', -0.23, 0.20, 0.17)
         co.el('n_co_3dpdr', 2.67, 0.14, 0.12)
         co.el('uv_co_3dpdr', 0.09, 0.14, 0.13)
         #co.el('aG',-0.90,0.20,0.21)#based on Me from Ritchey2023
         co.el('aG', -0.40, 0.20, 0.21)  # based on Me from Ritchey2023
         #pdr estimate in CO region
         co.el('tgas', 1.54,0.05,0.03)
         co.el('ngas(co)', 2.47,0.17)
         co.el('pgas', 3.98,0.23,0.19)
         co.el('tgas(ci)', 1.91, 0.1, 0.07)
         co.el('ngas(ci)', 1.88, 0.21, 0.15)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD147888
         q = qso('HD147888', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '353.65+17.71']
         q.ref.append('Jensen2010')
         q.ref.append('Sonnetrucker2007')
         q.ref.append('Jenkins2011')
         #q.ref.append('Sheffer2008')
         q.ref.append('Burgh2010')
         q.el('HI', 21.34, 0.10, 0.10)
         q.el('H2', 20.48, 0.03, 0.03)
         q.el('T01', 45, 3, 3, f='d')
         q.el('EBV', 0.47, 0.0, 0.0, f='d')
         q.el('Av', 1.91, 0.0, 0.0, f='d')
         q.el('Rv', 4.06, 0.0, 0.0, f='d')
         q.el('D', 124, 6, 6, f='d')
         q.el('CI', 14.23, 0.05, 0.05)
         q.el('MeZou', -0.21, 0.08, 0.07)  # ref de Cia 2021
         q.el('CO',15.30, 0.08,0.09)
         q.el('O/H', 301, 50, 50, f='d')  # O/H gas Zou 2021
         q.el('Fstar', 0.984, 0.095, 0.095, f='d')  # Ritchey2023
         q.el('Me_ISM', -0.081, 0.09, 0.092,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.34, 0.094, 0.094)  # Ritchey2023
         q.el('Megas', -0.56, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.081, 0.092, 0.092)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 45, 3, 3, f='d')
         co.el('T02', 86, 1.7, 1.2, f='d')
         co.el('TC2', 38, 20, 20,  f='d')
         co.el('H2', 20.48, 0.03, 0.03) #ref Jensen2010
         co.el('H2j0', 20.39, 0.04, 0.04, b=(3.1, 0.1, 0.1))
         co.el('H2j1', 19.71, 0.10, 0.10, b=(3.1, 0.1, 0.1))
         co.el('H2j2', 18.51, 0.02, 0.01, b=(3.1, 0.1, 0.1))
         co.el('H2j3', 17.11, 0.05, 0.05, b=(3.1, 0.1, 0.1))
         co.el('H2j4', 15.65, 0.06, 0.07, b=(3.1, 0.1, 0.1))
         co.el('H2j5', 15.13, 0.09, 0.07, b=(3.1, 0.1, 0.1))
         co.el('H2j6', 14.24, 0.16, 0.18, b=(3.1, 0.1, 0.1))
         co.el('CI', 14.70, 0.05, 0.05) #ref Burgh2010
         #co.el('CI', 14.23, 0.04, 0.03)  # ref jenkins2011 table 4
         co.el('CIj0', 13.86,  0.05) # ref jenkins2011
         co.el('CIj1', 13.79,  0.07)
         co.el('CIj2', 13.57, 0.05)
         co.el('CO', 15.30, 0.08,0.09)
         co.el('COj0', 14.73, 0.14,0.21) #ref Sonntr+2007
         co.el('COj1', 14.99, 0.12,0.17)
         co.el('COj2', 14.60, 0.09, 0.11)
         co.el('COj3', 13.96, 0.09, 0.11)
         co.el('PDRnH',2.71,0.07,0.09)
         co.el('PDRuv',0.09, 0.17, 0.13)
         co.el('P_co', 5.05, 0.12, 0.11)
         co.el('n_co', 3.08, 0.11, 0.12)
         co.el('uv_co', 0.31, 0.13, 0.11)
         #co.el('P_ci', 4.51, 0.08, 0.08)
         co.el('P_ci', 4.94, 0.14, 0.11)
         co.el('n_ci', 2.58,0.15,0.11)
         co.el('uv_ci',0.22,0.11,0.10)
         co.el('T_co', 8.85, 1.2, 0.7, f='d')
         co.el('pci-tripp', 4, 0.2, 0.2)
         co.el('uvci-tripp', 1.2, 0.2, 0.2)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 3.07, 0.13, 0.09)
         co.el('uv_co_pdr', 0.31, 0.19, 0.15)
         co.el('n_ci_pdr', 2.71, 0.10, 0.12)
         co.el('uv_ci_pdr', 0.09, 0.16, 0.15)
         co.el('n_co_goldsmith', 2.6, 0.1, 0)
         co.el('xco', 21.59, 0.17, 0.12)
         co.el('wco', -1.11, 0.13, 0.18)
         co.el('wco_emission', 3.49, 0, 0, f='d')  # form Liszt+ 2008
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.76, 0.09, 0.11)
         co.el('uv_ci_3dpdr', 0.09, 0.17, 0.17)
         co.el('n_co_3dpdr', 3.18, 0.11, 0.18)
         co.el('uv_co_3dpdr', 0.43, 0.18, 0.17)
         co.el('aG',-0.90,0.14,0.15)
         #
         co.el('pci-tripp', 3.98, 0.2, 0.2)
         co.el('uvci-tripp', 1.2, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 1.81, 0.05)
         co.el('ngas(co)', 2.96, 0.15)
         co.el('pgas', 4.76, 0.17)
         co.el('tgas(ci)', 1.92, 0.12, 0.07)
         co.el('ngas(ci)', 2.52, 0.17, 0.13)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         # add HD HD 185418
         q = qso('HD185418', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '53.60-2.17']
         q.ref.append('Jensen2010')
         q.ref.append('Sonnetrucker2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.11, 0.15, 0.15)
         q.el('H2', 20.77, 0.03, 0.03)
         q.el('T01', 101, 10, 8, f='d')
         q.el('EBV', 0.51, 0.0, 0.0, f='d')
         q.el('Av', 2.03, 0.0, 0.0, f='d')
         q.el('Rv', 3.98, 0.0, 0.0, f='d')
         q.el('CI', 14.74, 0.05, 0.05) #ref Burgh2010
         q.el('Me', -0.25, 0.3, 0.3)
         q.el('MeZou', -0.10, 0.05, 0.05)  # ref de Cia 2021
         q.el('CO', 14.74, 0.03, 0.03)
         q.el('O/H', 380, 43, 43, f='d')  # O/H gas Zou 2021
         q.el('D', 709, 15, 15, f='d')
         q.el('Fstar', 0.733, 0.138, 0.138, f='d')  # Ritchey2023
         q.el('Me_ISM', 0.053, 0.093, 0.093)  # Ritchey2023
         q.el('Me_[O/H]', -0.03, 0.11, 0.12)  # Ritchey2023
         q.el('Megas', -0.33, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.053, 0.093, 0.093)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 101, 10, 8, f='d')
         co.el('T02', 81, 1, 1.5, f='d')
         co.el('H2', 20.77, 0.03, 0.03) #refJensen2010
         co.el('H2j0', 20.34, 0.04, 0.04, b=(4.2, 0.1, 0.1))
         co.el('H2j1', 20.56, 0.05, 0.05, b=(4.2, 0.1, 0.1))
         co.el('H2j2', 18.32, 0.01, 0.02, b=(4.2, 0.1, 0.1))
         co.el('H2j3', 17.26, 0.02, 0.03, b=(4.2, 0.1, 0.1))
         co.el('H2j4', 15.34, 0.03, 0.03, b=(4.2, 0.1, 0.1))
         co.el('H2j5', 14.55, 0.05, 0.04, b=(4.2, 0.1, 0.1))
         #co.el('CI', 14.74, 0.05, 0.05) #ref Burgh2010
         co.el('CI', 14.83, 0.04, 0.04)  # ref Jenkins2010
         co.el('CIj0', 14.74, 0.05, 0.05) #ref Jenkins2010
         co.el('CIj1', 13.98, 0.07, 0.07)
         co.el('CIj2', 13.40, 0.05, 0.05)
         co.el('CO', 14.74, 0.03, 0.03) #ref Sonn+2007
         co.el('COj0', 14.54, 0.04, 0.04)
         co.el('COj1', 14.30, 0.04, 0.05)
         co.el('COj2', 13.34, 0.22, 0.48)
         co.el('PDRnH', 1.81, 0.06, 0.08) # z=0
         co.el('PDRuv', -0.32, 0.25, 0.17)
         #co.el('PDRnH', 1.72, 0.06, 0.08) z=-0.5
         #co.el('PDRuv', -0.10, 0.25, 0.17)
         co.el('P_co', 3.87, 0.16, 0.19)
         co.el('n_co', 1.85, 0.24, 0.26)
         co.el('uv_co', 0.22,0.12,0.13)
         #co.el('P_ci', 3.58, 0.06, 0.07)
         co.el('P_ci', 3.93, 0.06, 0.07)
         co.el('n_ci', 1.58, 0.10, 0.10)
         co.el('uv_ci', 0.13, 0.12, 0.11)
         co.el('T_co', 3.4, 0.7, 0.47, f='d')
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 1.66, 0.37, 0.61)
         co.el('uv_co_pdr', -0.43, 0.26, 0.33)
         co.el('n_ci_pdr', 1.78, 0.10, 0.09)
         co.el('uv_ci_pdr', -0.35, 0.21, 0.20)
         co.el('n_co_goldsmith', 1.3, 0.3, 0.2)
         co.el('xco', 22.94, 0.14, 0.11)
         co.el('wco', -2.17, 0.11, 0.14)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 1.85, 0.22, 0.23)
         co.el('uv_ci_3dpdr', -0.41, 0.21, 0.22)
         co.el('n_co_3dpdr', 1.91, 0.20, 0.32)
         co.el('uv_co_3dpdr', -0.45, 0.24, 0.27)
         co.el('aG',-0.47,0.31,0.30)
         #
         co.el('pci-tripp', 3.41, 0.2, 0.2)
         co.el('uvci-tripp', 0.23, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 1.87, 0.08)
         co.el('ngas(co)', 1.68, 0.3)
         co.el('pgas', 3.53, 0.27)
         co.el('tgas(ci)', 1.92, 0.14, 0.07)
         co.el('ngas(ci)', 1.64, 0.15, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)




         # add HD HD 192639
         q = qso('HD192639', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '74.09+1.48']
         q.ref.append('Jensen2010')
         q.ref.append('Sonnetrucker2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.32, 0.12, 0.12)
         q.el('H2', 20.69, 0.03, 0.03)
         q.el('T01', 98, 9, 9, f='d')
         q.el('EBV', 0.66, 0.0, 0.0, f='d')
         q.el('Av', 1.87, 0.0, 0.0, f='d')
         q.el('Rv', 2.84, 0.0, 0.0, f='d')
         q.el('CI', 14.73, 0.05, 0.05) #ref Burgh2010
         #q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 446, 110, 110, f='d')  # O/H gas Zou 2021
         q.el('MeZou', -0.03, 0.10, 0.12)  # ref de Cia 2021
         q.el('D', 1960, 120, 120, f='d')
         q.el('Fstar', 0.549, 0.143, 0.143, f='d')  # Ritchey2023
         q.el('Me_ISM', 0.008, 0.112, 0.112,f='d')  # Ritchey2023
         q.el('Megas', -0.25, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.008, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 98, 9, 9, f='d')
         co.el('T02', 87, 1.7, 2.2, f='d')
         co.el('H2', 20.69, 0.03, 0.03) #ref Jensen2010
         co.el('H2j0', 20.28, 0.05, 0.05, b=(6.5, 0.1, 0.2))
         co.el('H2j1', 20.48, 0.05, 0.05, b=(6.5, 0.1, 0.2))
         co.el('H2j2', 18.44, 0.02, 0.03, b=(6.5, 0.1, 0.2))
         co.el('H2j3', 17.52, 0.04, 0.05, b=(6.5, 0.1, 0.2))
         co.el('H2j4', 15.79, 0.03, 0.04, b=(6.5, 0.1, 0.2))
         co.el('H2j5', 15.28, 0.05, 0.04, b=(6.5, 0.1, 0.2))
         co.el('H2j6', 14.06, 0.11, 0.13, b=(6.5, 0.1, 0.2))
         co.el('CI', 14.86,0.11,0.11) #ref Burgh2010
         co.el('CIj0', 14.99, 0.05, 0.05) #ref Jenkins2011 table
         co.el('CIj1', 14.86, 0.05, 0.05)
         co.el('CIj2', 13.74, 0.05, 0.05)
         co.el('CO', 14.13,0.13,0.13) #ref Sonn+2007
         co.el('COj0', 14.00, 0.10,0.14)
         co.el('COj1', 13.54, 0.24,0.59)
         co.el('T_co', 2.57, 0.8, 1.1, f='d')
         co.el('PDRnH',1.94,0.08,0.04)
         co.el('PDRuv',-0.14, 0.22, 0.21)
         co.el('pci-tripp', 3.68,0.2,0.2)
         co.el('uvci-tripp',0.52,0.2,0.2)
         co.el('n_ci_3dpdr', 2.15, 0.08, 0.08)
         co.el('uv_ci_3dpdr', -0.23, 0.20, 0.18)
         co.el('aG', -0.46, 0.19, 0.19)
         co.el('tgas(ci)', 1.96, 0.07, 0.09)
         co.el('ngas(ci)', 1.93, 0.13, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)




         # add HD HD 206267
         q = qso('HD206267', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '99.29+3.74']
         q.ref.append('Jensen2010')
         q.ref.append('Sonnetrucker2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.30, 0.15)
         q.el('H2', 20.86, 0.03, 0.03)
         q.el('T01', 64, 3, 3, f='d')
         q.el('EBV', 0.53, 0.0, 0.0, f='d') # ref de Cia 2021
         q.el('Av', 1.49, 0.09, 0.09, f='d')# ref de Cia 2021
         q.el('Rv', 2.82, 0.16, 0.16, f='d')# ref de Cia 2021
         q.el('CI', 15.32, 0.05, 0.05) #ref Burgh2010
         #q.el('Me', -0.47, 0.13, 0.13) # ref de Cia 2021
         q.el('D', 1190,  543, 543, f='d') # ref de Cia 2021
         q.el('CO', 16.04,0.03,0.03)
         q.el('O/H', 407, 54, 54, f='d')  # O/H gas Zou 2021
         q.el('MeZou', -0.07, 0.05, 0.05)
         q.el('Fstar', 0.789, 0.099, 0.099, f='d')  # Ritchey2023
         q.el('Me_ISM', 0.005, 0.079, 0.079,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.20, 0.08, 0.08)  # Ritchey2023
         q.el('Megas', -0.42, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.005, 0.08, 0.08)  # Rithcey2023

         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 65, 3, 3, f='d')
         co.el('T02', 68, 1, 1, f='d')
         co.el('TC2', 36, 9, 9,  f='d')
         co.el('H2', 20.86, 0.03, 0.03)
         co.el('H2j0', 20.64, 0.03, 0.03, b=(6.6, 0.1, 0.1))
         co.el('H2j1', 20.45, 0.05, 0.05, b=(6.6, 0.1, 0.1))
         co.el('H2j2', 18.11, 0.02, 0.03, b=(6.6, 0.1, 0.1))
         co.el('H2j3', 16.76, 0.04, 0.04, b=(6.6, 0.1, 0.1))
         co.el('H2j4', 15.39, 0.03, 0.03, b=(6.6, 0.1, 0.1))
         co.el('H2j5', 14.91, 0.03, 0.03, b=(6.6, 0.1, 0.1))
         co.el('H2j6', 13.95, 0.12, 0.21, b=(6.6, 0.1, 0.1))
         case = 'Jenkins2001'
         if case == 'Jenkins2011':
             co.el('CI', 15.54, 0.05, 0.05)
             co.el('CIj0', 15.40, 0.05)
             co.el('CIj1', 14.87, 0.05)
             co.el('CIj2', 14.37, 0.05)
             co.el('PDRnH', 2.13, 0.06, 0.08)
             co.el('PDRuv', -0.32, 0.17, 0.13)
         elif case == 'Jenkins2001':
             co.el('CI', 15.30, 0.05, 0.05)
             co.el('CIj0', 15.16, 0.05, 0.05)
             co.el('CIj1', 14.61, 0.05, 0.05)
             co.el('CIj2', 14.13, 0.05, 0.05)
             #co.el('P_ci', 3.84, 0.06, 0.06)
             co.el('P_ci', 4.12, 0.06, 0.06)
             co.el('n_ci', 1.94, 0.09, 0.12)
             co.el('uv_ci',0.13,0.11,0.07)
             co.el('PDRnH', 2.04, 0.07, 0.07)
             co.el('PDRuv', -0.14, 0.19, 0.17)
         co.el('CO', 16.04, 0.03, 0.03) #ref Sonn+2007
         co.el('COj0', 15.64, 0.06, 0.07)
         co.el('COj1', 15.73, 0.05, 0.05)
         co.el('COj2', 15.09, 0.05, 0.05)
         co.el('COj3', 14.15, 0.09, 0.12)
         co.el('P_co', 4.62, 0.06, 0.06)
         co.el('n_co', 2.62, 0.12, 0.08)
         # co.el('P_co', 4.38, 0.06, 0.06)
         co.el('uv_co', 0.18, 0.10, 0.08)
         co.el('T_co', 6.0, 0.4, 0.2, f='d')
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.71, 0.08, 0.09)
         co.el('uv_co_pdr', -0.03, 0.15, 0.12)
         co.el('n_ci_pdr', 2.10, 0.11, 0.09)
         co.el('uv_ci_pdr', -0.35, 0.16, 0.15)
         co.el('n_co_goldsmith', 2.3, 0.1, 0)
         co.el('xco', 21.28, 0.06, 0.06)
         co.el('wco', -0.42, 0.07, 0.07)
         co.el('wco_emission', 2.91, 0, 0, f='d')  # form Liszt+ 2008
         ##################################### results with 3D-PDR:
         #co.el('n_ci_3dpdr', 2.15, 0.19, 0.13)
         #co.el('uv_ci_3dpdr', -0.39, 0.15, 0.13)
         co.el('n_ci_3dpdr', 2.15, 0.09, 0.06)
         co.el('uv_ci_3dpdr', -0.35, 0.15, 0.13)
         co.el('n_co_3dpdr', 2.82, 0.08, 0.08)
         co.el('uv_co_3dpdr', 0.03, 0.14, 0.12)
         co.el('aG', -0.64,0.17,0.14)
         #
         co.el('pci-tripp', 3.64, 0.2, 0.2)
         co.el('uvci-tripp', 0.30, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 1.57, 0.04)
         co.el('ngas(co)', 2.58, 0.14)
         co.el('pgas', 4.15, 0.15)
         co.el('tgas(ci)', 1.88, 0.09, 0.07)
         co.el('ngas(ci)', 1.93, 0.13, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 207198
         q = qso('HD207198', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '103.14+6.99']
         q.ref.append('Jensen2010')
         q.ref.append('Sonnetrucker2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.34, 0.17)
         q.el('H2', 20.83, 0.03, 0.03)
         q.el('T01', 66, 3, 3, f='d')
         q.el('EBV', 0.62, 0.03, 0.03, f='d') #ref de Cia 2021
         q.el('Av', 1.726, 0.22, 0.22, f='d') #ref de Cia 2021
         q.el('Rv', 2.77, 0.35, 0.35, f='d') #ref de Cia 2021
         q.el('CI', 15.26, 0.05, 0.05) #ref Burgh2010
         q.el('Me', -0.54, 0.16, 0.16) #ref de Cia 2021
         q.el('D', 999, 54, 54, f='d') #ref de Cia 2021
         q.el('CO', 15.51,0.04,0.04)
         q.el('O/H', 445, 59, 59, f='d')  # O/H gas Zou 2021
         q.el('MeZou', -0.04, 0.05, 0.05)
         q.el('Fstar', 0.840, 0.094, 0.094, f='d')  # Ritchey2023
         q.el('Me_ISM', 0.066, 0.076, 0.076,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.15, 0.08, 0.08)  # Ritchey2023
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.066, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 66, 3, 3, f='d')
         co.el('T02', 72, 1, 1, f='d')
         co.el('TC2', 43, 20, 20,  f='d')
         co.el('H2', 20.83, 0.03, 0.03) #ref jensen2010
         co.el('H2j0', 20.61, 0.03, 0.03, b=(5.6, 0.1, 0.1))
         co.el('H2j1', 20.44, 0.04, 0.04, b=(5.6, 0.1, 0.1))
         co.el('H2j2', 18.26, 0.03, 0.02, b=(5.6, 0.1, 0.1))
         co.el('H2j3', 17.00, 0.04, 0.05, b=(5.6, 0.1, 0.1))
         co.el('H2j4', 15.73, 0.03, 0.03, b=(5.6, 0.1, 0.1))
         co.el('H2j5', 14.89, 0.05, 0.04, b=(5.6, 0.1, 0.1))
         co.el('CI', 15.26, 0.05, 0.05) #ref Burgh2010
         co.el('CIj0', 15.40, 0.05) #ref Jenkins2011
         co.el('CIj1', 14.82, 0.05)
         co.el('CIj2', 14.29, 0.05)
         co.el('CO', 15.51, 0.04, 0.05) #ref Sonn+2007
         co.el('COj0', 15.26, 0.06, 0.07)
         co.el('COj1', 15.12, 0.05, 0.06)
         co.el('COj2', 14.12, 0.12, 0.16)
         co.el('COj3', '<13.5')
         co.el('PDRnH', 1.94, 0.07, 0.07)
         co.el('PDRuv', -0.10, 0.22, 0.18)
         co.el('P_co', 4.06, 0.11, 0.14)
         co.el('n_co', 2.13, 0.16, 0.17)
         co.el('uv_co',0.22,0.09,0.11)
         #co.el('P_ci', 3.77, 0.06, 0.06)
         co.el('P_ci', 4.06, 0.06, 0.06)
         co.el('n_ci', 1.85, 0.10, 0.09)
         co.el('uv_ci', 0.18, 0.09, 0.11)
         co.el('T_co', 3.9, 0.3, 0.3, f='d')
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.06, 0.15, 0.14)
         co.el('uv_co_pdr', -0.35, 0.19, 0.18)
         co.el('n_ci_pdr', 2.02, 0.06, 0.07)
         co.el('uv_ci_pdr', -0.27, 0.19, 0.17)
         co.el('n_co_goldsmith', 1.7, 0.2, 0.2)
         co.el('xco', 22.03, 0.16, 0.16)
         co.el('wco', -1.2, 0.07, 0.07)
         co.el('wco_emission', 2.63, 0, 0, f='d')  # form Liszt+ 2008
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.09, 0.08, 0.05)
         co.el('uv_ci_3dpdr', -0.33, 0.16, 0.16)
         co.el('n_co_3dpdr', 2.18, 0.15, 0.17)
         co.el('uv_co_3dpdr', -0.31, 0.19, 0.20)
         co.el('aG', -0.58,0.16,0.16)
         #
         co.el('pci-tripp', 3.63, 0.2, 0.2)
         co.el('uvci-tripp', 0.20, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 1.73, 0.05)
         co.el('ngas(co)', 1.96, 0.3)
         co.el('pgas', 3.64, 0.24)
         co.el('tgas(ci)', 1.88, 0.12, 0.07)
         co.el('ngas(ci)', 1.87, 0.13, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)




         # add HD HD 210121
         q = qso('HD210121', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.coord = ['J0000', '56.9-44.4']
         q.ref.append('Sonnetrucker2007')
         q.el('HI', 20.63, 0.10)
         q.el('H2', 20.75, 0.12, 0.12)
         q.el('T01', 51, 11, 11, f='d')
         q.el('EBV', 0.38, 0.0, 0.0, f='d')
         q.el('Av', 0.83, 0.0, 0.0, f='d')
         q.el('Rv', 2.18, 0.0, 0.0, f='d')
         #q.el('CI', 15.26, 0.05, 0.05)
         #q.el('Me', -0.25, 0.28, 0.28)
         q.el('CO', 15.83,0.05,0.05)
         q.el('O/H', 911, 526, 526, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 60, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('MeZou', 0.38, 0.16, 0.27)
         q.el('D', 334, 4, 4, f='d')
         q.el('Me_[O/H]', -0.15, 0.1, 0.1)  # Ritchey2023
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.00, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 51, 11, 11, f='d')
         co.el('T02', 51, 11, 11, f='d') #=T01
         co.el('TC2', 46, 20, 20,  f='d')
         co.el('H2', 20.75, 0.12, 0.12) #ref sonn+2007
         #co.el('H2j0', 20.61, 0.03, 0.03, b=(5.6, 0.1, 0.1))
         #co.el('H2j1', 20.44, 0.04, 0.04, b=(5.6, 0.1, 0.1))
         #co.el('H2j2', 18.26, 0.03, 0.02, b=(5.6, 0.1, 0.1))
         #co.el('H2j3', 17.00, 0.04, 0.05, b=(5.6, 0.1, 0.1))
         #co.el('H2j4', 15.73, 0.03, 0.03, b=(5.6, 0.1, 0.1))
         #co.el('H2j5', 14.89, 0.05, 0.04, b=(5.6, 0.1, 0.1))
         co.el('CI', 15.26, 0.05, 0.05) #ref Burgh2010
         #co.el('CIj0', 15.40, 0.05) #ref Jenkins2011
         #co.el('CIj1', 14.82, 0.05)
         #co.el('CIj2', 14.29, 0.05)
         co.el('CO', 15.83, 0.08, 0.08) #ref Sonn+2007
         co.el('COj0', 15.40, 0.10, 0.10)
         co.el('COj1', 15.50, 0.10, 0.10)
         co.el('COj2', 15.07, 0.12, 0.13)
         co.el('COj3', 13.50, 0.10,0.10)
         #co.el('PDRnH', 2.04, 0.05, 0.76)
         #co.el('PDRuv', -0.28, 0.18, 0.23)
         co.el('P_co', 4.38, 0.10, 0.10)
         co.el('n_co', 2.26, 0.47, 0.29)
         co.el('uv_co', -1,+5,0)
         #co.el('P_ci', 3.77, 0.06, 0.06)
         co.el('T_co', 5.26, 0.29, 0.22, f='d')
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.26, 0.47, 0.29)
         co.el('uv_co_pdr', -1, 0.56, 0.0)
         co.el('xco', 21.43, 0.16, 0.16)
         co.el('wco', -0.68, 0.20, 0.20)
         co.el('wco_emission', 2.91, 0, 0, f='d')  # form Liszt+ 2008
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 2.73, 0.11, 0.14)
         #co.el('uv_co_3dpdr', -1, 0.90, 0.00)
         co.el('uv_co_3dpdr', '<-0.1')
         #co.el('aG',-1.62,0.9,0.1)
         # pdr estimate in CO region
         co.el('tgas', 1.51, 0.28,0.08)
         co.el('ngas(co)', 2.46, 0.35,0.14)
         co.el('pgas', 4.11, 0.15)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)




         # add HD HD 210839
         q = qso('HD210839', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '103.83+2.61']
         q.ref.append('Jensen2010')
         q.ref.append('Sonnetrucker2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.15, 0.10, 0.10)
         q.el('H2', 20.84, 0.03, 0.03)
         q.el('T01', 72, 4, 4, f='d')
         q.el('EBV', 0.57, 0.0, 0.0, f='d')
         q.el('Av', 1.58, 0.0, 0.0, f='d')
         q.el('Rv', 2.78, 0.0, 0.0, f='d')
         q.el('CI', 15.00, 0.05, 0.05) #ref Burgh2010
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('CO', 15.46,0.05,0.05)
         q.el('O/H', 490, 56, 56, f='d')  # O/H gas Zou 2021
         q.el('MeZou', 0.01, 0.05, 0.05)
         q.el('D', 854, 60, 54, f='d')
         q.el('Fstar', 0.815, 0.098, 0.098, f='d')  # Ritchey2023
         q.el('Me_ISM', 0.029, 0.074, 0.074,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.19, 0.08, 0.08)  # Ritchey2023
         q.el('Megas', -0.41, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.029, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 72, 4, 4, f='d')
         co.el('T02', 71, 1.7, 1, f='d')
         co.el('TC2', 16, 6, 6, f= 'd')
         co.el('H2', 20.84, 0.03, 0.03) #ref Jensen2010
         co.el('H2j0', 20.57, 0.04, 0.04, b=(9.7, 0.1, 0.1))
         co.el('H2j1', 20.50, 0.04, 0.04, b=(9.7, 0.1, 0.1))
         co.el('H2j2', 18.15, 0.05, 0.03, b=(9.7, 0.1, 0.1))
         co.el('H2j3', 16.93, 0.05, 0.04, b=(9.7, 0.1, 0.1))
         co.el('H2j4', 16.09, 0.04, 0.05, b=(9.7, 0.1, 0.1))
         co.el('H2j5', 15.58, 0.04, 0.03, b=(9.7, 0.1, 0.1))
         co.el('H2j6', 14.52, 0.02, 0.03, b=(9.7, 0.1, 0.1))
         co.el('H2j7', 14.29, 0.03, 0.03, b=(9.7, 0.1, 0.1))
         case = 'Jenkins2011'
         if case == 'Jenkins2011':
             co.el('CI', 15.09, 0.05, 0.05)
             co.el('CIj0', 14.89, 0.05)
             co.el('CIj1', 14.47, 0.05)
             co.el('CIj2', 14.19, 0.05)
             co.el('P_ci', 5.47, 0.28, 0.18)
             co.el('PDRnH', 3.30, 0.15, 0.10)
             co.el('PDRuv', 0.54, 0.12, 0.11)
             co.el('n_co', 2.13, 0.10, 0.12)
             co.el('uv_co', 0.18, 0.11, 0.07)
         elif case == 'Jenkins2001':
             co.el('CI', 14.98, 0.04, 0.03)
             co.el('CIj0', 14.80, 0.05, 0.05)
             co.el('CIj1', 14.34, 0.05, 0.05)
             co.el('CIj2', 14.01, 0.05, 0.05)
             #co.el('P_ci', 4.07, 0.06, 0.06)  # %Jenkins2001
             co.el('P_ci', 4.35, 0.06, 0.06)  # %Jenkins2001
             co.el('n_ci', 2.48, 0.06,0.06)
             co.el('PDRnH', 2.22, 0.15, 0.10)
             co.el('PDRuv', 0.04, 0.18, 0.17)
         co.el('CO', 15.46, 0.05, 0.05) #ref Sonn+2007
         co.el('COj0', 15.21, 0.09, 0.11)
         co.el('COj1', 15.05, 0.05, 0.06)
         co.el('COj2', 14.12, 0.04, 0.04)
         co.el('COj3', '<13')
         co.el('P_co', 4.17, 0.08, 0.10)
         co.el('n_co', 2.08, 0.09, 0.06)
         co.el('uv_co', 0.13,0.13,0.05)
         co.el('T_co', 4.1, 0.2, 0.24, f='d')
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.18, 0.06, 0.05)
         co.el('uv_co_pdr', -0.35, 0.17, 0.15)
         co.el('n_ci_pdr', 2.34, 0.07, 0.07)
         co.el('uv_ci_pdr', -0.13, 0.19, 0.17)
         co.el('n_co_goldsmith', 1.7, 0, 0)
         co.el('xco', 22.08, 0.07, 0.06)
         co.el('wco', -1.24, 0.07, 0.08)
         co.el('wco_emission', 4.16, 0, 0, f='d')  # form Liszt+ 2008
         co.el('n_co_3dpdr', 2.21, 0.19, 0.21)
         co.el('uv_co_3dpdr', -0.33, 0.16, 0.17)
         co.el('n_ci_3dpdr', 2.39, 0.09, 0.11)
         co.el('uv_ci_3dpdr', -0.21, 0.15, 0.12)
         co.el('tgas', 1.72, 0.04)
         co.el('ngas(co)', 1.98, 0.23)
         co.el('pgas', 3.68, 0.16)
         co.el('tgas(ci)', 1.86, 0.09, 0.07)
         co.el('ngas(ci)', 2.16, 0.14, 0.11)

         ##################################### results with 3D-PDR:


         co.el('aG', -0.74,0.16,0.13)
         #
         co.el('pci-tripp', 4.16, 0.2, 0.2)
         co.el('uvci-tripp', 0.47, 0.2, 0.2)
         #
         #co = sy(0, 0)
         #co.el('T01', 72, 4, 4, f='d')
         #co.el('H2', 20.84, 0.03, 0.03)
         #co.el('H2j0', 20.57, 0.04, 0.04, b=(9.7, 0.1, 0.1))
         #co.el('H2j1', 20.50, 0.04, 0.04, b=(9.7, 0.1, 0.1))
         #co.el('H2j2', 18.15, 0.05, 0.03, b=(9.7, 0.1, 0.1))
         #co.el('H2j3', 16.93, 0.05, 0.04, b=(9.7, 0.1, 0.1))
         #co.el('H2j4', 16.09, 0.04, 0.05, b=(9.7, 0.1, 0.1))
         #co.el('H2j5', 15.58, 0.04, 0.03, b=(9.7, 0.1, 0.1))
         #co.el('H2j6', 14.52, 0.02, 0.03, b=(9.7, 0.1, 0.1))
         #co.el('H2j7', 14.29, 0.03, 0.03, b=(9.7, 0.1, 0.1))
         #co.el('CI', 14.99, 0.05)
         #co.el('CIj0', 14.86, 0.05)
         #co.el('CIj1', 14.28, 0.05)
         #co.el('CIj2', 13.79, 0.05)
         #co.el('PDRnH', 2.04, 0.09, 0.05)
         #co.el('PDRuv', -0.32, 0.20, 0.13)
         # pdr estimate in CO region
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     #SAMPLE Burgh2010
     if 1:
         q = qso('HD37903', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Burgh2010')
         q.el('CO', '<13.7')
         q.el('CI', 14.22, 0.05)
         q.el('H2', 20.92,0.1)
         q.el('HI', 21.17, 0.1)
         q.el('EBV', 0.35, 0.0, 0.0, f='d')
         q.el('Av', 1.31, 0.0, 0.0, f='d')
         q.el('MeZou', -0.25, 0.28, 0.28)
         q.el('O/H', 239, 37, 37, f='d')  # O/H gas Zou 2021
         q.el('MeZou', -0.31, 0.06, 0.07)
         q.el('Megas', -0.69, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.092, 0.08, 0.08)  # Rithcey2023
         s = []
         co = sy(0, 0)
         co.el('CO', '<13.7')  # ref Burgh2010
         co.el('T01', 68, 7, 7, f='d')
         co.el('CI', 14.22, 0.05, 0.05)  # ref Burgh2007
         co.el('H2', 20.92, 0.10, 0.10)  # ref Burgh2010
         co.el('H2j0', 20.68, 0.07, 0.07)  # ref Rachford
         co.el('H2j1', 20.54, 0.05, 0.05)  # ref Rachford
         co.el('CIj0', 13.92, 0.05, 0.05)  # ref Jenkins2011 table
         co.el('CIj1', 13.98, 0.05, 0.05)
         co.el('CIj2', 13.84, 0.05, 0.05)
         co.el('pci-tripp', 4.6,0.2,0.2)
         co.el('uvci-tripp', 1.37, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.79, 0.07, 0.07)
         #co.el('uv_ci_3dpdr', 1, 0.0, 0.15)
         co.el('uv_ci_3dpdr', '>0.85')
         co.el('aG', 0.1, 0.51, 0.41)
         co.el('tgas(ci)', 2.17, 0.04, 0.05)
         co.el('ngas(ci)', 2.55, 0.14, 0.07)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         q = qso('HD69106', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Burgh2010')
         q.el('CO', '<13.5')
         q.el('CI', 14.27, 0.05)
         q.el('H2', 19.73,0.1)
         q.el('HI', 21.08, 0.1)
         q.el('EBV', 0.2, 0.0, 0.0, f='d')
         q.el('Av', 0.61, 0.0, 0.0, f='d')
         q.el('Fstar', 0.473, 0.100, 0.100, f='d')  # Ritchey2023
         q.el('Megas', -0.35, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.134, 0.08, 0.08)  # Rithcey2023
         s = []
         co = sy(0, 0)
         co.el('CO', '<13.5')  # ref Burgh2010
         co.el('T01', 80, 16, 16, f='d')
         co.el('H2', 19.73, 0.10, 0.10)  # ref Burgh2010
         co.el('H2j0', 19.41, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.44, 0.1, 0.1)
         co.el('H2j2', 17.77, 0.5, 0.5)
         co.el('H2j3', 17.61, 0.3, 0.3)  # ref Shull2021 table
         co.el('H2j4', 15.38, 0.1, 0.1)
         co.el('H2j5', 14.46, 0.05, 0.05)
         co.el('CI', 14.27, 0.05, 0.05)  # ref Burgh2007
         co.el('CIj0', 14.42, 0.05, 0.05)  # ref Jenkins2011 table
         co.el('CIj1', 13.81, 0.05, 0.05)
         co.el('CIj2', 13.24, 0.05, 0.05)
         co.el('pci-tripp', 3.55, 0.2, 0.2)
         co.el('uvci-tripp', 0.45, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.91, 0.09, 0.11)
         co.el('uv_ci_3dpdr', -0.43, 0.24, 0.25)
         co.el('aG', -0.31, 0.44, 0.41)
         co.el('tgas(ci)', 1.97, 0.18, 0.08)
         co.el('ngas(ci)', 1.71, 0.13, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD91824', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Burgh2010')
         q.el('CO', '<13.6')
         q.el('CI', 14.47, 0.05)
         q.el('H2', 19.83,0.1)
         q.el('HI', 21.15, 0.1)
         q.el('EBV', 0.24, 0.0, 0.0, f='d')
         q.el('Av', 0.80, 0.0, 0.0, f='d')
         q.el('Me', -0.25,0.28,0.28)
         q.el('O/H', 691, 105, 105, f='d')  # O/H gas Zou 2021
         q.el('MeZou', 0.16, 0.06, 0.07)
         q.el('Fstar', 0.553, 0.109, 0.109, f='d')  # Ritchey2023
         q.el('Megas', -0.14, 0.1, 0.1) #Rithcey2023
         q.el('Me', 0.133, 0.08, 0.08)  # Rithcey2023
         s = []
         co = sy(0, 0)
         co.el('CO', '<13.6')  # ref Burgh2010
         co.el('CI', 14.47, 0.05, 0.05)  # ref Burgh2007
         co.el('H2', 19.83, 0.10, 0.10)  # ref Burgh2010
         co.el('T01', 61, 7, 7, f='d')
         co.el('H2j0', 19.61, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.43, 0.1, 0.1)
         co.el('H2j2', 17.72, 0.5, 0.5)
         co.el('H2j3', 17.17, 0.3, 0.3)  # ref Shull2021 table
         co.el('H2j4', 15.24, 0.1, 0.1)
         co.el('H2j5', 14.64, 0.05, 0.05)
         co.el('CIj0', 14.45, 0.05, 0.05)  # ref Jenkins2011 table
         co.el('CIj1', 13.99, 0.05, 0.05)
         co.el('CIj2', 13.48, 0.05, 0.05)
         co.el('pci-tripp', 3.6, 0.2, 0.2)
         co.el('uvci-tripp', 0.78, 0.2, 0.2)
         #co.el('n_ci_3dpdr', 2.21, 0.15, 0.12) #logNH2=20.5
         #co.el('uv_ci_3dpdr', -0.52, 0.24, 0.21)
         co.el('n_ci_3dpdr', 2.09, 0.10, 0.12)
         co.el('uv_ci_3dpdr', -0.35, 0.38, 0.33)
         co.el('aG', -0.68, 0.51, 0.41)
         co.el('tgas(ci)', 1.93, 0.3, 0.08)
         co.el('ngas(ci)', 1.87, 0.17, 0.12)
         s.append(co)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD93843', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Burgh2010')
         q.el('CO', '<12.7')
         q.el('CI', 14.13, 0.05)
         q.el('H2', 19.61,0.1)
         q.el('HI', 21.32, 0.1)
         q.el('EBV', 0.27, 0.0, 0.0, f='d')
         q.el('Av', 1.05, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 407, 96, 96, f='d')  # O/H gas Zou 2021
         q.el('MeZou', -0.07, 0.09, 0.12)
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.00, 0.08, 0.08)  # Rithcey2023
         s = []
         co = sy(0, 0)
         co.el('CO', '<12.7')  # ref Burgh2010
         co.el('CI', 14.13, 0.05, 0.05)  # ref Burgh2007
         co.el('H2', 19.61, 0.10, 0.10)  # ref Burgh2010
         co.el('H2j0', 19.14, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.43, 0.1, 0.1)
         co.el('H2j2', 16.04, 0.1, 0.1)
         co.el('H2j3', 15.86, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j4', 14.67, 0.1, 0.1)
         co.el('H2j5', 14.40, 0.05, 0.05)
         co.el('T01', 107, 21, 21, f='d')
         co.el('CIj0', 13.77, 0.05, 0.05)  # ref Jenkins2011 table
         co.el('CIj1', 13.53, 0.05, 0.05)
         co.el('CIj2', 13.19, 0.05, 0.05)
         co.el('pci-tripp', 4.1, 0.2, 0.2)
         co.el('uvci-tripp', 0.66, 0.2, 0.2)
         #co.el('n_ci_3dpdr', 2.45, 0.09, 0.08)
         #co.el('uv_ci_3dpdr', -0.76, 0.14, 0.18)
         co.el('n_ci_3dpdr', 2.48, 0.11, 0.08)
         #co.el('uv_ci_3dpdr', -0.90, 0.16, 0.10)
         co.el('uv_ci_3dpdr', '<-0.74')
         co.el('aG', -0.1, 0.29, 0.25)
         co.el('tgas(ci)', 1.79, 0.05, 0.05)
         co.el('ngas(ci)', 2.28, 0.14, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD103779', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Burgh2010')
         q.el('CO', '<12.35')
         q.el('CI', 14.21, 0.05)
         q.el('H2', 19.83,0.1)
         q.el('HI', 21.16, 0.1)
         q.el('EBV', 0.212, 0.0, 0.0, f='d')
         q.el('Av', 0.69, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 389, 108, 108, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 34, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('MeZou', 0.04, 0.08, 0.10)
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.00, 0.08, 0.08)  # Rithcey2023
         s = []
         co = sy(0, 0)
         co.el('CO', '<12.35')  # ref Burgh2010
         co.el('CI', 14.21, 0.05, 0.05)  # ref Burgh2007
         co.el('H2', 19.83, 0.10, 0.10)  # ref Burgh2010
         co.el('T01', 117, 14, 14, f='d')
         co.el('H2j0', 19.33, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.65, 0.1, 0.1)
         co.el('H2j2', 16.25, 0.15, 0.1)
         co.el('H2j3', 15.83, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j4', 14.81, 0.1, 0.1)
         co.el('H2j5', 14.75, 0.05, 0.05)
         co.el('CIj0', 14.31, 0.05, 0.05)  # ref Jenkins2011 table
         co.el('CIj1', 13.53, 0.05, 0.05)
         co.el('CIj2', 12.80, 0.05, 0.05)
         co.el('pci-tripp', 3.3, 0.2, 0.2)
         co.el('uvci-tripp', 0.33, 0.2, 0.2)
         #co.el('n_ci_3dpdr', 1.89, 0.09, 0.13)
         #co.el('uv_ci_3dpdr', -1.0, 0.2, 0.1)
         co.el('n_ci_3dpdr', 1.76, 0.09, 0.12)
         #co.el('uv_ci_3dpdr', -1.0, 0.11, 0.0)
         co.el('uv_ci_3dpdr', '<-0.88')
         co.el('aG', -0.06, 0.64, 0.41)
         co.el('tgas(ci)', 1.92, 0.02, 0.06)
         co.el('ngas(ci)', 1.54, 0.17, 0.16)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD121968', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Burgh2010')
         q.el('CO', '<12.3')
         q.el('CI', 13.36, 0.05)
         q.el('H2', 18.73,0.1)
         q.el('HI', 20.96, 0.1)
         q.el('EBV', 0.07, 0.0, 0.0, f='d')
         q.el('Fstar', 0.613, 0.115, 0.115, f='d')  # Ritchey2023
         q.el('Megas', -0.20, 0.1, 0.1) #Rithcey2023
         q.el('Me', 0.15, 0.14, 0.14)  # Rithcey2023
         s = []
         co = sy(0, 0)
         co.el('CO', '<12.3')  # ref Burgh2010
         co.el('CI', 13.36, 0.05, 0.05)  # ref Burgh2007
         co.el('H2', 18.73, 0.10, 0.10)  # ref Burgh2010
         co.el('H2j0', 18.68, 0.10, 0.10)  # ref Burgh2010
         co.el('H2j1', 17.69, 0.10, 0.10)  # ref Burgh2010
         co.el('T01', 38, 3, 3, f='d')
         co.el('CIj0', 13.26, 0.05, 0.05)  # ref Jenkins2011 table
         co.el('CIj1', 13.03, 0.05, 0.05)
         co.el('CIj2', 12.46, 0.05, 0.05)
         co.el('pci-tripp', 3.8, 0.2, 0.2)
         co.el('uvci-tripp', 1.27, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.58, 0.07, 0.10)
         #co.el('uv_ci_3dpdr', -1.0, 0.12, 0.26)
         co.el('uv_ci_3dpdr', '<-0.88')
         co.el('tgas(ci)', 1.78, 0.10, 0.08)
         co.el('ngas(ci)', 2.37, 0.12, 0.15)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('HD201345', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Burgh2010')
         q.el('CO', '<12.4')
         q.el('CI', 13.93, 0.05)
         q.el('H2', 19.24,0.1)
         q.el('HI', 20.88, 0.1)
         q.el('EBV', 0.18, 0.0, 0.0, f='d')
         q.el('Fstar', 0.400, 0.104, 0.105, f='d')  # Ritchey2023
         q.el('Me_ISM', -0.016, 0.084, 0.084,f='d')  # Ritchey2023
         q.el('Megas', -0.18, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.012, 0.084, 0.084)  # Rithcey2023
         s = []
         co = sy(0, 0)
         co.el('CO', '<12.4')  # ref Burgh2010
         co.el('CI', 13.93, 0.05, 0.05)  # ref Burgh2007
         co.el('H2', 19.24, 0.10, 0.10)  # ref Burgh2010
         co.el('T01', 97, 14, 14, f='d')
         co.el('H2j0', 18.83, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.02, 0.1, 0.1)
         co.el('H2j2', 16.89, 0.5, 0.4)
         co.el('H2j3', 16.49, 0.5, 0.3)  # ref Shull2021 table
         co.el('H2j4', 15.0, 0.1, 0.1)
         co.el('H2j5', 14.57, 0.05, 0.05)
         co.el('CIj0', 14.03, 0.05, 0.05)  # ref Jenkins2011 table
         co.el('CIj1', 13.27, 0.05, 0.05)
         co.el('CIj2', 12.71, 0.05, 0.05)
         co.el('pci-tripp', 3.5, 0.2, 0.2)
         co.el('uvci-tripp', 0.32, 0.2, 0.2)
         #co.el('n_ci_3dpdr', 1.69, 0.16, 0.11)
         #co.el('uv_ci_3dpdr', -0.58, 0.28, 0.34)
         co.el('n_ci_3dpdr', 1.67, 0.14, 0.18)
         co.el('uv_ci_3dpdr', -0.47, 0.41, 0.37)
         co.el('aG', -0.42, 0.31, 0.28)
         co.el('tgas(ci)', 1.99, 0.35, 0.12)
         co.el('ngas(ci)', 1.50, 0.18, 0.20)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         # add HD HD 93205
         q = qso('HD93205', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.33, 0.10, 0.10)
         q.el('H2', 19.86, 0.10, 0.10)
         q.el('T01', 105, 21, 21, f='d')
         q.el('EBV', 0.38, 0.0, 0.0, f='d')
         q.el('Av', 1.23, 0.0, 0.0, f='d')
         q.el('CI', 14.54, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 375, 50, 50, f='d')  # O/H gas Zou 2021
         q.el('MeZou', -0.11, 0.05, 0.05)
         q.el('Fstar', 0.202, 0.108, 0.108, f='d')  # Ritchey2023
         q.el('Megas', -0.12, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.098, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 105, 21, 21, f='d') #ref Burgh2007
         co.el('H2', 19.86, 0.10, 0.10)
         co.el('H2j0', 19.40, 0.10, 0.10)
         co.el('H2j1', 19.65, 0.10, 0.10)
         co.el('CI', 14.54, 0.02, 0.02) #ref Burgh2010
         co.el('CIj0', 14.59, 0.05)#ref Jenkins2011
         co.el('CIj1', 14.16, 0.05)
         co.el('CIj2', 13.76, 0.05)
         co.el('CO', 13.23, 0.06, 0.06) #ref Burgh2007
         co.el('COj0', 13.03, 0.07, 0.07)
         co.el('COj1', 12.77, 0.11, 0.11)
         co.el('P_co', 3.92, 0.43, 1.63)
         co.el('P_ci', 4.04, 0.07, 0.07)
         co.el('T_co', 3.26, 0.7, 0.5, f='d')
         co.el('pci-tripp', 3.6, 1, 1)
         co.el('uvci-tripp', 0.58, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.09, 0.10, 0.12)
         co.el('uv_ci_3dpdr', 0.11, 0.43, 0.29)
         co.el('aG', -0.1, 0.5, 0.33)
         co.el('tgas(ci)', 2.20, 0.17, 0.21)
         co.el('ngas(ci)', 1.88, 0.17, 0.12)
         #co.el('pci-tripp')
         #co.el('pci_jenkins', 4.0, 0.2, 0.2)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 93222
         q = qso('HD93222', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.54, 0.10, 0.10)
         q.el('H2', 19.81, 0.10, 0.10)
         q.el('CO', 13.36, 0.20, 0.20)
         q.el('T01', 77, 11, 11, f='d')
         q.el('EBV', 0.36, 0.0, 0.0, f='d')
         q.el('Av', 1.71, 0.0, 0.0, f='d')
         q.el('CI', 14.36, 0.10) #ref Burgh2010
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 436, 35, 35, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 79, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('MeZou', 0.19, 0.02, 0.02)
         q.el('Fstar', 0.224, 0.103, 0.103, f='d')  # Ritchey2023
         q.el('Megas', -0.16, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.124, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 77, 11, 11, f='d')
         co.el('H2', 19.81, 0.10, 0.10)   #ref Burgh2010
         co.el('H2j0', 19.49, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.44, 0.1, 0.1)
         co.el('H2j2', 15.80, 0.1, 0.1)
         co.el('H2j3', 15.70, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j4', 14.70, 0.1, 0.1)
         co.el('H2j5', 14.46, 0.05, 0.05)
         co.el('CI', 14.36, 0.05, 0.05)#ref Burgh2010
         co.el('CIj0', 14.41, 0.05)#ref Jenkins2011
         co.el('CIj1', 13.97, 0.05)
         co.el('CIj2', 13.55, 0.05)
         co.el('CO', 13.36, 0.20, 0.20)
         co.el('COj0', 13.20, 0.20, 0.20)
         co.el('COj1', 12.79, 0.17, 0.17)
         #co.el('COj2', 11.71, 0.37, 0.37)
         co.el('P_co', 3.41, 0.46, 0.89)
         co.el('P_ci', 3.99, 0.06, 0.06)
         co.el('T_co', 2.7, 1.1, 0.6, f='d')
         co.el('pci-tripp', 4.4, 0.2, 0.2)
         co.el('uvci-tripp', 0.82, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.33, 0.11, 0.10)
         #co.el('uv_ci_3dpdr', -1.0, 0.1, 0.0)
         co.el('uv_ci_3dpdr', '<-0.9')
         co.el('aG', -0.58, 0.17, 0.22)
         co.el('tgas(ci)', 1.80, 0.05, 0.05)
         co.el('ngas(ci)', 2.12, 0.18, 0.15)
         #co.el('T_co02', 3.3, 1.0, 1.0, f='d')
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 99857
         q = qso('HD99857', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.31, 0.10, 0.10)
         q.el('H2', 20.25, 0.10, 0.10)
         q.el('EBV', 0.33, 0.0, 0.0, f='d')
         q.el('CI', 14.59,0.05,0.05) #ref Burgh
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('CO', 13.73, 0.10, 0.10)
         q.el('Fstar', 0.435, 0.091, 0.091, f='d')  # Ritchey2023
         q.el('Megas', -0.23, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.041, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 83, 17, 17, f='d')
         co.el('H2', 20.25, 0.10, 0.10)
         co.el('H2j0', 19.90, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 20.01, 0.1, 0.1)
         co.el('H2j2', 18.19, 0.1, 0.1)
         co.el('H2j3', 17.78, 0.2, 0.2)  # ref Shull2021 table
         co.el('H2j4', 15.05, 0.1, 0.1)
         co.el('H2j5', 14.46, 0.05, 0.05)
         co.el('CI', 14.59, 0.05, 0.05)#ref Burgh
         co.el('CIj0', 14.62, 0.05)#ref Jenkins
         co.el('CIj1', 14.12, 0.05)
         co.el('CIj2', 13.61, 0.05)
         co.el('CO', 13.73, 0.10, 0.10)
         # co.el('COj0', 13.47, 0.07, 0.07)
         # co.el('COj1', 13.15, 0.11, 0.11)
         # co.el('P_co', 5.02, 0.11, 0.12)
         co.el('P_ci', 3.87, 0.07, 0.07)
         co.el('pci-tripp', 3.6, 0.2, 0.2)
         co.el('uvci-tripp', 0.55, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.09, 0.08, 0.12)
         co.el('uv_ci_3dpdr', -0.21, 0.24, 0.24)
         co.el('aG', -0.68, 0.25, 0.15)
         co.el('tgas(ci)', 1.96, 0.2, 0.05)
         co.el('ngas(ci)', 1.85, 0.15, 0.11)
         # co.el('T_co', 3.1, 0.5, 0.5)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 102065
         q = qso('HD102065', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 20.49, 0.10, 0.10)
         q.el('H2', 20.53, 0.10, 0.10)
         q.el('T01', 59, 7, 7, f='d')
         q.el('EBV', 0.17, 0.0, 0.0, f='d')
         q.el('Av', 0.67, 0.0, 0.0, f='d')
         q.el('CI', 14.24, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('CO', 13.62, 0.12, 0.12)
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.00, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 59, 7, 7, f='d')
         co.el('H2', 20.53, 0.10, 0.10)
         co.el('H2j0', 20.45, 0.11, 0.11)
         co.el('H2j1', 20.15, 0.10, 0.10)
         # co.el('CI', 14.23, 0.05, 0.05)
         co.el('CI', 14.36, 0.05, 0.05) #ref Burgh2010
         co.el('CIj0', 14.22, 0.05)
         co.el('CIj1', 13.69, 0.05)
         co.el('CIj2', 13.10, 0.05)
         co.el('CO', 13.62, 0.12, 0.12)
         co.el('COj0', 13.38, 0.14, 0.14)
         co.el('COj1', 13.21, 0.29, 0.29)
         co.el('P_co', 3.92, 0.43, 1.63)
         co.el('P_ci', 3.78, 0.05, 0.06)
         co.el('T_co', 3.7, 3.2, 1.4, f='d')
         co.el('pci-tripp', 3.6, 0.2, 0.2)
         co.el('uvci-tripp', 0.61, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.09, 0.11, 0.15)
         co.el('uv_ci_3dpdr', -0.62, 0.25, 0.20)
         co.el('aG', -0.73, 0.29, 0.22)
         co.el('tgas(ci)', 1.83, 0.2, 0.06)
         co.el('ngas(ci)', 1.88, 0.12, 0.20)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 104705
         q = qso('HD104705', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.11, 0.10, 0.10)
         q.el('H2', 19.98, 0.10, 0.10)
         q.el('T01', 92, 16, 16, f='d')
         q.el('EBV', 0.23, 0.0, 0.0, f='d')
         q.el('Av', 0.65, 0.0, 0.0, f='d')
         q.el('CI', 14.24, 0.10, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 426, 84, 84, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 39, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('MeZou', 0.08, 0.06, 0.07)
         q.el('Fstar', 0.389, 0.105, 0.105, f='d')  # Ritchey2023
         q.el('Megas', -0.21, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.038, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 92, 16, 16, f='d')
         co.el('H2', 19.98, 0.10, 0.10) #ref Burgh
         co.el('H2j0', 19.57, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.77, 0.1, 0.1)
         co.el('H2j2', 17.87, 0.1, 0.1)
         co.el('H2j3', 17.28, 0.2, 0.2)  # ref Shull2021 table
         co.el('H2j4', 15.21, 0.1, 0.1)
         co.el('H2j5', 14.46, 0.05, 0.05)
         co.el('CI', 14.24, 0.10, 0.10) # ref Burgh2010
         co.el('CIj0', 14.36, 0.05) # ref Jenkins
         co.el('CIj1', 13.69, 0.05)
         co.el('CIj2', 13.03, 0.05)
         co.el('CO', 12.98, 0.16, 0.16)
         co.el('COj0', 12.78, 0.17, 0.17)
         co.el('COj1', 12.50, 0.32, 0.32)
         co.el('P_co', '<4.0')
         co.el('P_ci', 3.6, 0.06, 0.06)
         co.el('T_co', 3.2, 2.4, 1.1, f='d')
         co.el('pci-tripp', 3.5, 0.2, 0.2)
         co.el('uvci-tripp', 1.37, 0.2, 0.53)
         co.el('n_ci_3dpdr', 1.79, 0.10, 0.12)
         co.el('uv_ci_3dpdr', -0.41, 0.27, 0.22)
         co.el('aG', -0.46, 0.32, 0.23)
         co.el('tgas(ci)', 2.04, 0.16, 0.13)
         co.el('ngas(ci)', 1.57, 0.17, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 115071
         q = qso('HD115071', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.ref.append('Sheffer2008')
         q.el('HI', 21.38, 0.10, 0.10)
         q.el('H2', 20.69, 0.10, 0.10)
         q.el('T01', 84, 14, 14, f='d')
         q.el('EBV', 0.49, 0.0, 0.0, f='d')
         q.el('CI', 14.69, 0.10)
         q.el('Megas', -0.27, 0.1, 0.1)  #Rithcey2023
         q.el('Me', -0.00, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 84, 14, 14, f='d')
         co.el('H2', 20.69, 0.10, 0.10)
         co.el('H2j0', 20.30, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 20.37, 0.1, 0.1)
         co.el('H2j2', 18.30, 0.1, 0.1)
         co.el('H2j3', 16.86, 0.2, 0.2)  # ref Shull2021 table
         co.el('H2j4', 15.55, 0.1, 0.1)
         co.el('H2j5', 14.93, 0.05, 0.05)
         co.el('CI', 14.69, 0.05, 0.05)
         co.el('CIj0', 14.75, 0.05)
         co.el('CIj1', 14.27, 0.05)
         co.el('CIj2', 13.76, 0.05)
         co.el('CO', 14.53, 0.09, 0.09)
         co.el('COj0', 14.29, 0.12, 0.12)
         co.el('COj1', 14.08, 0.30, 0.30)
         co.el('P_co', 3.79, 0.42, 1.55)
         co.el('P_ci', 3.88, 0.06, 0.06)
         co.el('T_co', 3.5, 2.7, 1.3, f='d')
         co.el('pci-tripp', 3.64, 0.2, 0.2)
         co.el('uvci-tripp', 0.74, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.15, 0.08, 0.09)
         co.el('uv_ci_3dpdr', -0.27, 0.19, 0.16)
         co.el('aG', -0.58, 0.22, 0.17)
         co.el('tgas(ci)', 1.94, 0.08, 0.09)
         co.el('ngas(ci)', 1.93, 0.13, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 116852
         q = qso('HD116852', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 20.96, 0.10, 0.10)
         q.el('H2', 19.85, 0.10, 0.10)
         q.el('T01', 70, 9, 9, f='d')
         q.el('EBV', 0.21, 0.0, 0.0, f='d')
         q.el('Av', 0.51, 0.0, 0.0, f='d')
         q.el('CI', 14.15, 0.10, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 527, 133, 133, f='d')  # O/H gas Zou 2021
         q.el('MeZou', 0.04, 0.10, 0.13)
         q.el('Fstar', 0.312, 0.109, 0.109, f='d')  # Ritchey2023
         q.el('Megas', -0.26, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.158, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 70, 9, 9, f='d')
         co.el('H2', 19.85, 0.10, 0.10)
         co.el('H2j0', 19.50, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.46, 0.1, 0.1)
         co.el('H2j2', 15.99, 0.1, 0.1)
         co.el('H2j3', 15.81, 0.2, 0.2)  # ref Shull2021 table
         co.el('H2j4', 14.64, 0.1, 0.1)
         co.el('H2j5', 14, 0.05, 0.05)
         co.el('CI', 14.15, 0.10, 0.10)
         co.el('CIj0', 14.22, 0.05)
         co.el('CIj1', 13.82, 0.05)
         co.el('CIj2', 13.26, 0.05)
         co.el('CO', 13.28, 0.04, 0.04)
         co.el('COj0', 13.11, 0.05, 0.05)
         co.el('COj1', 12.78, 0.07, 0.07)
         co.el('P_co', '<3.8')
         co.el('P_ci', 3.93, 0.06, 0.06)
         co.el('T_co', 3.0, 0.4, 0.3, f='d')
         co.el('pci-tripp', 3.7, 0.2, 0.2)
         co.el('uvci-tripp', 0.80, 0.2, 0.2)
         #co.el('n_ci_3dpdr', 2.29, 0.09, 0.09)
         #co.el('uv_ci_3dpdr', -1, 0.1, 0.1)
         co.el('n_ci_3dpdr', 2.27, 0.09, 0.10)
         #co.el('uv_ci_3dpdr', -1, 0.1, 0.)
         co.el('uv_ci_3dpdr', '<-0.9')
         co.el('aG', -0.1, 0.33, 0.36)
         co.el('tgas(ci)', 1.82, 0.05, 0.05)
         co.el('ngas(ci)', 2.05, 0.2, 0.14)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD124314', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Burgh2010')
         q.el('HI', 21.34, 0.10, 0.10)
         q.el('H2', 20.52, 0.10, 0.10)
         q.el('T01', 74, 15, 15, f='d')
         q.el('EBV', 0.53, 0.0, 0.0, f='d')
         q.el('CI', 14.67, 0.10, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Fstar', 0.551, 0.097, 0.097, f='d')  # Ritchey2023
         q.el('Megas', -0.20, 0.1, 0.1) #Rithcey2023
         q.el('Me', 0.061, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 74, 15, 15, f='d')
         co.el('H2', 20.52, 0.10, 0.10)
         co.el('H2j0', 20.17, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 20.16, 0.1, 0.1)
         co.el('H2j2', 18.03, 0.2, 0.2)
         co.el('H2j3', 16.99, 0.4, 0.4)  # ref Shull2021 table
         co.el('H2j4', 15.53, 0.1, 0.1)
         co.el('H2j5', 14.75, 0.05, 0.05)
         co.el('CI', 14.67, 0.10, 0.10)
         co.el('CIj0', 14.69, 0.05)
         co.el('CIj1', 14.10, 0.05)
         co.el('CIj2', 13.51, 0.05)
         co.el('CO', 14.20, 0.09, 0.09)
         co.el('pci-tripp', 3.5, 0.2, 0.2)
         co.el('uvci-tripp', 0.58, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.0, 0.11, 0.10)
         co.el('uv_ci_3dpdr', -0.45, 0.25, 0.21)
         co.el('aG', -0.7, 0.18, 0.23)
         co.el('tgas(ci)', 1.92, 0.15, 0.07)
         co.el('ngas(ci)', 1.78, 0.13, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD152590', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Burgh2010')
         q.el('HI', 21.37, 0.10, 0.10)
         q.el('H2', 20.51, 0.10, 0.10)
         q.el('T01', 64, 13, 13, f='d')
         q.el('EBV', 0.38, 0.0, 0.0, f='d')
         q.el('CI', 14.60, 0.10, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Fstar', 0.740, 0.102, 0.102, f='d')  # Ritchey2023
         q.el('Me_ISM', 0.016, 0.089, 0.089,f='d')  # Ritchey2023
         q.el('Megas', -0.34, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.016, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 64, 13, 13, f='d')
         co.el('H2', 20.51, 0.10, 0.10)
         co.el('H2j0', 20.29, 0.10, 0.10)  # ref Burgh2010
         co.el('H2j1', 20.09, 0.10, 0.10)  # ref Burgh2010
         co.el('CI', 14.60, 0.10, 0.10)
         co.el('CIj0', 14.68, 0.05)
         co.el('CIj1', 14.31, 0.05)
         co.el('CIj2', 13.92, 0.05)
         co.el('CO', 13.77, 0.09, 0.09)
         co.el('T_co', 4.1, 2, 2, f='d')
         co.el('pci-tripp', 3.7, 0.2, 0.2)
         co.el('uvci-tripp', 0.77, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.30, 0.09, 0.08)
         co.el('uv_ci_3dpdr', -0.05, 0.25, 0.29)
         co.el('tgas(ci)', 1.92, 0.2, 0.07)
         co.el('ngas(ci)', 2.08, 0.13, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD157857', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Burgh2010')
         q.el('HI', 21.30, 0.10, 0.10)
         q.el('H2', 20.69, 0.10, 0.10)
         q.el('T01', 78, 17, 17, f='d')
         q.el('EBV', 0.43, 0.0, 0.0, f='d')
         q.el('Av', 1.48, 0.0, 0.0, f='d')
         q.el('CI', 14.62, 0.10, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 467, 92, 92, f='d')  # O/H gas Zou 2021
         q.el('Fstar', 0.520, 0.120, 0.120, f='d')  # Ritchey2023
         q.el('Me_ISM', -0.106, 0.11, 0.1,f='d')  # Ritchey2023
         q.el('MegA', -0.35, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.106, 0.08, 0.08)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 78, 17, 17, f='d')
         co.el('H2', 20.69, 0.10, 0.10)
         co.el('H2j0', 20.31, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 20.32, 0.1, 0.1)
         co.el('H2j2', 18.50, 0.1, 0.1)
         co.el('H2j3', 17.73, 0.2, 0.4)  # ref Shull2021 table
         co.el('H2j4', 15.36, 0.1, 0.1)
         co.el('H2j5', 14.80, 0.05, 0.05)
         co.el('CI', 14.62, 0.10, 0.10)
         co.el('CIj0', 14.76, 0.05)
         co.el('CIj1', 14.13, 0.05)
         co.el('CIj2', 13.61, 0.05)
         co.el('CO', 14.08, 0.09, 0.09)
         co.el('pci-tripp', 3.5, 0.2, 0.2)
         co.el('uvci-tripp', 0.46, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.97, 0.10, 0.08)
         co.el('uv_ci_3dpdr', -0.21, 0.25, 0.2)
         co.el('aG', -0.34, 0.32, 0.32)
         co.el('tgas(ci)', 2.0, 0.10, 0.10)
         co.el('ngas(ci)', 1.76, 0.15, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         # add HD HD 177989
         q = qso('HD177989', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.ref.append('Sheffer2008')
         q.el('HI', 20.95, 0.10, 0.10)
         q.el('H2', 20.12, 0.10, 0.10)
         q.el('T01', 61, 5, 5, f='d')
         q.el('EBV', 0.23, 0.0, 0.0, f='d')
         q.el('Av', 0.65, 0.0, 0.0, f='d')
         q.el('CI', 14.68, 0.10, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 436, 71, 71, f='d')  # ppm [O/H]gas ref Zuo 2021
         q.el('Si/H', 49, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('MeZou', 0.12, 0.05, 0.05)
         q.el('Fstar', 0.631, 0.097, 0.097, f='d')  # Ritchey2023
         q.el('Me', 0.009, 0.08, 0.08,f='d')  # Ritchey2023
         q.el('Megas', -0.31, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 61, 5, 5, f='d')
         co.el('H2', 20.12, 0.10, 0.10)
         co.el('H2j0', 19.93, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.67, 0.1, 0.1)
         co.el('H2j2', 17.64, 0.1, 0.2)
         co.el('H2j3', 17.26, 0.2, 0.4)  # ref Shull2021 table
         co.el('H2j4', 15.39, 0.1, 0.1)
         co.el('H2j5', 14.58, 0.05, 0.05)
         co.el('CI', 14.68, 0.10, 0.10)
         co.el('CIj0', 14.70, 0.05)
         co.el('CIj1', 14.14, 0.05)
         co.el('CIj2', 13.56, 0.05)
         co.el('CO', 14.62, 0.17, 0.17)
         co.el('COj0', 14.39, 0.05, 0.05)
         co.el('COj1', 14.21, 0.06, 0.06)
         co.el('P_co', 3.90, 0.22, 0.30)
         co.el('P_ci', 3.77, 0.06, 0.06)
         co.el('T_co', 3.6, 0.5, 0.4, f='d')
         co.el('pci-tripp', 3.6, 0.2, 0.2)
         co.el('uvci-tripp', 0.35, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.0, 0.09, 0.10)
         co.el('uv_ci_3dpdr', -0.58, 0.2, 0.2)
         co.el('aG', -0.91, 0.15, 0.19)
         co.el('tgas(ci)', 1.91, 0.10, 0.06)
         co.el('ngas(ci)', 1.81, 0.14, 0.13)

         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 203374
         q = qso('HD203374', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.11, 0.10, 0.10)
         q.el('H2', 20.68, 0.10, 0.10)
         q.el('EBV', 0.22, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('CI', 14.98, 0.05, 0.05)  # ref Burgh
         q.el('Fstar', 0.776, 0.097, 0.097, f='d')  # Ritchey2023
         q.el('Me', 0.097, 0.076, 0.076,f='d')  # Ritchey2023
         q.el('Megas', -0.30, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 87, 2, 2, f='d')
         co.el('H2', 20.68, 0.10, 0.10)
         co.el('H2j0', 20.32, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 20.42, 0.1, 0.1)
         co.el('H2j2', 18.10, 0.1, 0.2)
         co.el('H2j3', 17.24, 0.4, 0.4)  # ref Shull2021 table
         co.el('H2j4', 15.22, 0.1, 0.1)
         co.el('H2j5', 14.71, 0.05, 0.05)
         co.el('CI', 14.98, 0.05, 0.05)
         co.el('CIj0', 15.08, 0.05)  # ref Jenkins
         co.el('CIj1', 14.48, 0.05)
         co.el('CIj2', 13.98, 0.05)
         co.el('CO', 15.35, 0.10, 0.10)
         co.el('P_ci', 3.79, 0.06, 0.07)
         co.el('pci-tripp', 3.6, 0.2, 0.2)
         co.el('uvci-tripp', 0.31, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.06, 0.11, 0.09)
         co.el('uv_ci_3dpdr', -0.39, 0.19, 0.20)
         co.el('aG', -0.68, 0.19, 0.21)
         co.el('tgas(ci)', 1.90, 0.10, 0.08)
         co.el('ngas(ci)', 1.84, 0.13, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 218915
         q = qso('HD218915', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.11, 0.10, 0.10)
         q.el('H2', 20.16, 0.10, 0.10)
         q.el('T01', 86, 14, 14, f='d')
         q.el('EBV', 0.29, 0.0, 0.0, f='d')
         q.el('CI', 14.54, 0.05, 0.05)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Fstar', 0.705, 0.1, 0.1, f='d')  # Ritchey2023
         q.el('Me', 0.157, 0.095, 0.095,f='d')  # Ritchey2023
         q.el('Megas', -0.26, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 86, 14, 14, f='d')
         co.el('H2', 20.15, 0.10, 0.10)
         co.el('H2j0', 19.78, 0.1, 0.1)  # ref Shull2021 table
         co.el('H2j1', 19.94, 0.1, 0.1)
         co.el('H2j2', 17.95, 0.2, 0.3)
         co.el('H2j3', 17.50, 0.2, 0.4)  # ref Shull2021 table
         co.el('H2j4', 15.54, 0.1, 0.1)
         co.el('H2j5', 14.87, 0.05, 0.05)
         co.el('CI', 14.54, 0.05, 0.05)
         co.el('CIj0', 14.68, 0.05)
         co.el('CIj1', 14.08, 0.05)
         co.el('CIj2', 13.49, 0.05)
         co.el('CO', 13.64, 0.13, 0.13)
         co.el('COj0', 13.40, 0.13, 0.13)
         co.el('COj1', 13.26, 0.13, 0.13)
         co.el('P_co', 4.06, 0.45, 1.00)
         co.el('P_ci', 3.72, 0.06, 0.06)
         co.el('T_co', 3.9, 1.6, 0.8, f='d')
         co.el('pci-tripp', 3.6, 0.2, 0.2)
         co.el('uvci-tripp', 0.33, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.91, 0.10, 0.10)
         co.el('uv_ci_3dpdr', -0.33, 0.28, 0.29)
         co.el('aG', -0.53, 0.20, 0.19)
         co.el('tgas(ci)', 1.97, 0.21, 0.08)
         co.el('ngas(ci)', 1.70, 0.19, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 303308
         q = qso('HDE303308', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2007')
         q.ref.append('Jenkins2011')
         q.ref.append('Burgh2010')
         q.el('HI', 21.45, 0.10, 0.10)
         q.el('H2', 20.35, 0.10, 0.10)
         q.el('T01', 86, 14, 14, f='d')
         q.el('EBV', 0.45, 0.0, 0.0, f='d')
         q.el('Av', 1.36, 0.0, 0.0, f='d')
         q.el('CI', 14.73, 0.10, 0.10)
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('O/H', 423, 35, 35, f='d')  # O/H gas Zou 2021
         q.el('MeZou', -0.06, 0.03, 0.04)
         q.el('Fstar', 0.284, 0.110, 0.110, f='d')  # Ritchey2023
         q.el('Me', -0.093, 0.080, 0.080,f='d')  # Ritchey2023
         q.el('Megas', -0.19, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 86, 14, 14, f='d')
         co.el('H2', 20.35, 0.10, 0.10)
         co.el('H2j0', 20.0, 0.13, 0.13)
         co.el('H2j1', 20.10, 0.13, 0.13)
         co.el('CI', 14.73, 0.05, 0.05)
         co.el('CIj0', 14.76, 0.05)
         co.el('CIj1', 14.36, 0.05)
         co.el('CIj2', 14.14, 0.05)
         co.el('CO', 13.65, 0.06, 0.06)
         co.el('COj0', 13.47, 0.07, 0.07)
         co.el('COj1', 13.15, 0.11, 0.11)
         # co.el('P_co', 5.02, 0.11, 0.12)
         # co.el('P_ci', 4.51, 0.08, 0.08)
         co.el('T_co', 3.0, 0.6, 0.4, f='d')
         co.el('n_ci_3dpdr', 2.30, 0.08, 0.10)
         co.el('uv_ci_3dpdr', 0.37, 0.36, 0.23)
         #
         co.el('pci-tripp', 3.5, 0.5, 0.5)
         co.el('uvci-tripp', 0.59, 0.2, 0.2)
         co.el('aG', 0.07, 0.24, 0.25)
         co.el('tgas(ci)', 2.20, 0.21, 0.20)
         co.el('ngas(ci)', 2.10, 0.1, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)




         # add HD HD 224151
         q = qso('HD224151', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.ref.append('Burgh2010')
         q.el('HI', 21.32, 0.10, 0.10)
         q.el('H2', 20.57, 0.10, 0.10)
         q.el('EBV', 0.44, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('CI', 14.63,0.05,0.05)
         q.el('Fstar', 0.526, 0.079, 0.079, f='d')  # Ritchey2023
         q.el('Me', -0.049, 0.079, 0.079,f='d')  # Ritchey2023
         q.el('Megas', -0.31, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 74, 15, 15, f='d')
         co.el('H2', 20.57, 0.10, 0.10)
         co.el('H2j0', 20.0, 0.13, 0.13)
         co.el('H2j1', 20.07, 0.13, 0.13)
         co.el('CI', 14.63, 0.05, 0.05)
         co.el('CIj0', 14.68, 0.05)
         co.el('CIj1', 14.05, 0.05)
         co.el('CIj2', 13.51, 0.05)
         co.el('CO', 13.85, 0.16, 0.16)
         co.el('P_ci', 3.72, 0.06, 0.07)
         co.el('pci-tripp', 3.8, 0.2, 0.2)
         co.el('uvci-tripp', 0.16, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.91, 0.10, 0.12)
         co.el('uv_ci_3dpdr', 0.13, 0.6, 0.44)
         co.el('aG', -0.89, 0.62, 0.18)
         co.el('tgas(ci)', 2.27, 0.12, 0.37)
         co.el('ngas(ci)', 1.70, 0.12, 0.15)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # SAMPLE Sheffer2008
     if 1:
         # add HD HD 12323
         q = qso('HD12323', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 20.96, 0.10, 0.10)
         q.el('T01', 80, 17, 12, f='d')
         q.el('H2', 20.32, 0.10, 0.10)
         q.el('EBV', 0.23, 0.04, 0.04, f='d') # ref Zuo 2021
         q.el('Rv', 2.74,0.4,0.4,f='d') # ref Zuo 2021
         q.el('Av', 0.63,0.14,0.14, f='d') # ref Zuo 2021
         q.el('O/H', 629.8,104,104, f='d') # ppm [O/H]gas ref Zuo 2021
         q.el('Si/H', 37, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021  O/Hdust is 4 times of Si/Hdust
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('D', 2520, 200, 170, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 14.53, 0.05, 0.05)
         q.el('Fstar', 0.589, 0.12, 0.12, f='d')  # Ritchey2023
         q.el('Me', 0.022, 0.09, 0.09,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.13, 0.09, 0.09)  # Ritchey2023
         q.el('Me_gas', -0.27, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 80, 17, 12, f='d')
         co.el('T02', 99, 14, 10, f='d')
         co.el('H2', 20.32, 0.10, 0.10)
         co.el('T_h210', 82, 10, 10, f='d')
         co.el('T_h220', 101, 10, 10, f='d')
         co.el('T_h230', 142, 10, 10, f='d')
         co.el('T_h240', 217, 10, 10, f='d')
         co.el('H2j0', 20.0, 0.11, 0.11)
         co.el('H2j1', 20.03, 0.12, 0.12)
         co.el('H2j2', 18.47, 0.25, 0.25)
         co.el('H2j3', 18.19, 0.25, 0.25)
         co.el('H2j4', 17.57, 0.19, 0.19)
         # co.el('CI', 14.88, 0.05, 0.05)
         # co.el('CIj0', 14.77, 0.05)
         # co.el('CIj1', 14.14, 0.05)
         # co.el('CIj2', 13.51, 0.05)
         co.el('CO', 14.53, 0.05, 0.05)
         co.el('T_co', 3.0, 0.8, 0.3, f='d')
         co.el('T_co10', 3.0, 0.8, 0.8, f='d')
         co.el('T_co20', 3.3, 0.8, 0.8, f='d')
         co.el('COj0', 14.36, 0.05, 0.05)
         co.el('COj1', 14.00, 0.11, 0.11)
         co.el('COj2', 12.81, 0.36, 0.36)
         co.el('P_co', 3.40, 0.38, 0.95)
         co.el('n_co', 1.72, 0.31, 0.32)
         co.el('uv_co', 0.27, 0.15, 0.14)
         # co.el('n_ci', 2.22, 0.18, 2.20)
         # co.el('uv_ci', 0.31, 0.20, 0.19)
         # co.el('P_ci', 3.67, 0.07, 0.07)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 1.62, 0.27, 0.48)
         co.el('uv_co_pdr', -0.3, 0.26, 0.23)
         co.el('n_co_goldsmith', 1.8, 0, 0)
         co.el('xco', 23.09, 0.32, 0.23)
         co.el('wco', -3.33, 0.18, 0.18)
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 1.52, 0.30, 0.50)
         co.el('uv_co_3dpdr', -0.23, 0.46, 0.40)
         co.el('aG',-0.07,0.36,0.32)
         # pdr estimate in CO region
         co.el('tgas', 1.72, 0.24,0.08)
         co.el('ngas(co)', 1.0, 0.99,0.18)
         co.el('pgas', 2.87, 0.82,0.14)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 15137
         q = qso('HD15137', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.1, 0.30, 0.30)
         q.el('H2', 20.32, 0.10, 0.10)
         q.el('EBV', 0.31, 0.0, 0.0, f='d')
         q.el('Me', -0.26, 0.1, 0.1) #Rithcey2023
         q.el('Rv', 3.1,0.5,0.5,f='d')
         #q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 13.52, 0.05, 0.05)
         q.el('D', 2220, 160, 140, f='d')
         q.el('Fstar', 0.524, 0.11, 0.11, f='d')  # Ritchey2023
         q.el('Me', -0.002, 0.091, 0.091,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.13, 0.08, 0.09)  # Ritchey2023
         q.el('Megas', -0.26, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 104, 28, 18, f='d')
         co.el('T02', 110, 14, 10, f='d')
         co.el('H2', 20.32, 0.10, 0.10)
         co.el('T_h210', 104, 10, 10, f='d')
         co.el('T_h220', 111, 10, 10, f='d')
         co.el('T_h230', 153, 10, 10, f='d')
         co.el('T_h240', 245, 10, 10, f='d')
         co.el('H2j0', 19.87, 0.10, 0.10)
         co.el('H2j1', 20.11, 0.11, 0.11)
         co.el('H2j2', 18.56, 0.21, 0.21)
         co.el('H2j3', 18.30, 0.21, 0.21)
         co.el('H2j4', 17.84, 0.15, 0.15)
         co.el('CI', 14.65, 0.05, 0.05)
         co.el('CIj0', 14.77, 0.05)
         co.el('CIj1', 14.14, 0.05)
         co.el('CIj2', 13.51, 0.05)
         co.el('CO', 13.52, 0.05, 0.05)
         co.el('T_co', 3.8, 1.2, 0.4, f='d')
         co.el('T_co10', 3.1, 0.8, 0.8, f='d')
         co.el('T_co20', 4.2, 0.8, 0.8, f='d')
         co.el('COj0', 13.33, 0.07, 0.07)
         co.el('COj1', 13.01, 0.10, 0.10)
         co.el('COj2', 12.29, 0.22, 0.22)
         co.el('P_co', 4.04, 0.21, 0.26)
         co.el('n_co', 1.90, 0.20, 0.22)
         co.el('uv_co', 0.40, 0.12, 0.11)
         co.el('P_ci', 4.03, 0.07, 0.07)
         co.el('n_ci', 1.54, 0.12, 0.12)
         co.el('uv_ci', 0.31, 0.12, 0.13)
         co.el('PDRnH', 1.58, 0.09, 0.27)  # PDR fit
         co.el('PDRuv', 0.40, 0.64, 0.38)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 1.90, 0.21, 0.26)
         co.el('uv_co_pdr', -0.0, 0.2, 0.20)
         co.el('n_ci_pdr', 1.78, 0.12, 0.10)
         co.el('uv_ci_pdr', -0.01, 0.23, 0.20)
         co.el('n_co_goldsmith', 1.8, 0, 0)
         co.el('xco', 23.65, 0.15, 0.15)
         co.el('wco', -3.33, 0.18, 0.18)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 1.82, 0.11, 0.13)
         co.el('uv_ci_3dpdr', -0.07, 0.50, 0.20)
         co.el('n_co_3dpdr', 2.12, 0.24, 0.31)
         co.el('uv_co_3dpdr', 0.01, 0.45, 0.32)
         co.el('aG', -0.42,0.20,0.21)
         #
         co.el('pci-tripp', 3.61, 0.2, 0.2)
         co.el('uvci-tripp', 0.32, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 2.09, 0.04,0.14)
         co.el('ngas(co)', 1.86, 0.38)
         co.el('pgas', 3.93, 0.33,0.4)
         co.el('tgas(ci)', 2.27, 0.09, 0.33)
         co.el('ngas(ci)', 1.59, 0.21, 0.13)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 23478
         q = qso('HD23478', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 20.94, 0.10, 0.10)
         q.el('H2', 20.57, 0.10, 0.10)
         q.el('EBV', 0.20, 0.0, 0.0, f='d') #Jenkins2011
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 14.91, 0.05, 0.05)
         q.el('D', 252, 4, 4, f='d')
         q.el('Me_[O/H]', -0.15, 0.1, 0.1)  # Ritchey2023
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.00, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 53, 12, 8, f='d')
         co.el('T02', 78, 13, 9, f='d')
         co.el('H2', 20.57, 0.10, 0.10)
         co.el('T_h210', 55, 10, 10, f='d')
         co.el('T_h220', 79, 10, 10, f='d')
         co.el('T_h230', 101, 10, 10, f='d')
         co.el('T_h240', 171, 10, 10, f='d')
         co.el('H2j0', 20.42, 0.12, 0.12)
         co.el('H2j1', 19.97, 0.25, 0.25)
         co.el('H2j2', 18.27, 0.39, 0.39)
         co.el('H2j3', 17.33, 0.46, 0.46)
         co.el('H2j4', 17.08, 0.27, 0.27)
         co.el('CI', 14.79, 0.05, 0.05)
         co.el('CIj0', 14.66, 0.05)
         co.el('CIj1', 14.10, 0.05)
         co.el('CIj2', 13.59, 0.05)
         co.el('CO', 14.91, 0.05, 0.05)
         co.el('T_co',3.44, 0.8,0.3, f='d')
         co.el('T_co10', 3.4, 0.8, 0.8, f='d')
         co.el('T_co20', 3.6, 0.8, 0.8, f='d')
         co.el('T_co30', 4.2, 0.8, 0.8, f='d')
         co.el('COj0', 14.69, 0.06, 0.06)
         co.el('COj1', 14.45, 0.08, 0.08)
         co.el('COj2', 13.35, 0.31, 0.31)
         co.el('P_co', 3.81, 0.20, 0.48)
         co.el('n_co', 1.94, 0.26, 0.27)
         co.el('uv_co',0.18,0.18,0.17)
         co.el('P_ci', 4.08, 0.09, 0.09)
         co.el('n_ci', 1.81, 0.12, 0.11)
         co.el('uv_ci',0.13,0.19,0.14)
         co.el('PDRnH', 1.85, 0.11, 0.12)  # PDR fit
         co.el('PDRuv', 0.13, 0.51, 0.33)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 1.94, 0.20, 0.28)
         co.el('uv_co_pdr', -0.45, 0.2, 0.20)
         co.el('n_ci_pdr', 1.98, 0.08, 0.04)
         co.el('uv_ci_pdr', -0.37, 0.24, 0.22)
         co.el('n_co_goldsmith', 1.3, 0.4, 0.4)
         co.el('xco', 22.57, 0.14, 0.14)
         co.el('wco', -2, 0.17, 0.18)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.06, 0.12, 0.08)
         co.el('uv_ci_3dpdr', -0.35, 0.34, 0.30)
         co.el('n_co_3dpdr', 2.03, 0.26, 0.35)
         co.el('uv_co_3dpdr', -0.47, 0.26, 0.25)
         co.el('aG',-0.80,0.29,0.23)
         #
         co.el('pci-tripp', 3.62, 0.2, 0.2)
         co.el('uvci-tripp', 0.46, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 1.84, 0.07,0.07)
         co.el('ngas(co)', 1.77,0.24,0.64)
         co.el('pgas', 3.6, 0.34,0.38)
         co.el('tgas(ci)', 1.90, 0.23, 0.1)
         co.el('ngas(ci)', 1.85, 0.15, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 24190
         q = qso('HD24190', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.11, 0.10, 0.10)
         q.el('H2', 20.38, 0.10, 0.10)
         q.el('EBV', 0.23, 0.0, 0.0, f='d') #Jenkins
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 13.95, 0.05, 0.05)
         q.el('D', 380, 6, 6, f='d')
         q.el('Fstar', 1.013, 0.1, 0.1, f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.19, 0.08, 0.08)  # Ritchey2023
         q.el('Megas', -0.46, 0.1, 0.1)  # Rithcey2023
         q.el('Me', 0.07, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('H2', 20.38, 0.10, 0.10)
         co.el('T01', 66, 20, 12, f='d')
         co.el('T02', 85, 13, 10, f='d')
         co.el('T_h210', 66, 10, 10, f='d')
         co.el('T_h220', 86, 10, 10, f='d')
         co.el('T_h230', 119, 10, 10, f='d')
         co.el('T_h240', 193, 10, 10, f='d')
         co.el('H2j0', 20.15, 0.12, 0.12)
         co.el('H2j1', 19.96, 0.15, 0.15)
         co.el('H2j2', 18.24, 0.33, 0.33)
         co.el('H2j3', 17.73, 0.34, 0.34)
         co.el('H2j4', 17.31, 0.22, 0.22)
         co.el('CI', 14.68, 0.05, 0.05)
         co.el('CIj0', 14.54, 0.05)
         co.el('CIj1', 14.02, 0.05)
         co.el('CIj2', 13.41, 0.05)
         co.el('CO', 13.95, 0.05, 0.05)
         co.el('T_co', 3.2, 0.9, 0.3, f='d')
         co.el('T_co10', 3.1, 0.8, 0.8, f='d')
         co.el('T_co20', 3.5, 0.8, 0.8, f='d')
         co.el('COj0', 13.77, 0.06, 0.06)
         co.el('COj1', 13.44, 0.10, 0.10)
         co.el('COj2', 12.36, 0.32, 0.32)
         co.el('P_co', 3.62, 0.32, 0.74)
         co.el('n_co', 1.72, 0.30, 0.36)
         co.el('uv_co',0.18,0.25,0.21)
         co.el('P_ci', 4.07, 0.06, 0.06)
         co.el('n_ci', 1.72, 0.15, 0.15)
         co.el('uv_ci',0.18,0.25,0.21)
         co.el('PDRnH', 1.81, 0.11, 0.16)  # PDR fit
         co.el('PDRuv', 0.04, 0.53, 0.28)
         #co.el('PDRnHz1', 1.94, 0.06,0.10)
         #co.el('PDRuvz1', -0.28, 0.46,0.28)
         #co.el('PDRnHco', 1.81, 0.11, 0.16)  # PDR fit
         #co.el('PDRuvco', 1.81, 0.11, 0.16)  # PDR fit
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 1.66, 0.26, 0.3)
         co.el('uv_co_pdr', -0.84, 0.3, 0.20)
         co.el('n_ci_pdr', 1.98, 0.04, 0.08)
         co.el('uv_ci_pdr', -0.43, 0.19, 0.18)
         co.el('n_co_goldsmith', 1.3, 0.4, 0.4)
         co.el('xco', 23.51, 0.18, 0.17)
         co.el('wco', -3.13, 0.19, 0.21)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.03, 0.09, 0.12)
         co.el('uv_ci_3dpdr', -0.31, 0.33, 0.26)
         co.el('n_co_3dpdr', 1.64, 0.30, 0.4)
         co.el('uv_co_3dpdr', -0.45, 0.48, 0.34)
         co.el('aG', -0.83,0.23,0.19)
         #
         co.el('pci-tripp', 3.64, 0.2, 0.2)
         co.el('uvci-tripp', 0.62, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 2.03, 0.12, 0.12)
         co.el('ngas(co)', 1.2, 0.7, 0.4)
         co.el('pgas', 3.23, 0.6, 0.3)
         co.el('tgas(ci)', 1.94, 0.24, 0.07)
         co.el('ngas(ci)', 1.78, 0.19, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         # add HD HD 24398
         q = qso('HD24398', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.01, 0.10, 0.10)
         q.el('H2', 20.67, 0.10, 0.10)
         q.el('EBV', 0.34, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 15.26, 0.05, 0.05)
         q.el('D', 260, 30, 30, f='d')
         q.el('Fstar', 0.897, 0.197, 0.197, f='d')  # Ritchey2023
         q.el('Me', 0.033, 0.129, 0.129,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.21, 0.14, 0.14)  # Ritchey2023
         q.el('Megas', -0.53, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('H2', 20.67, 0.10, 0.10)
         co.el('T01', 58,10,10,f='d')
         co.el('TC2', 50, 10, 10,  f='d')
         co.el('CO', 15.26, 0.05, 0.05)
         co.el('T_co',3.9,1.0,0.1, f='d') #?????
         co.el('T_co10', 3.4, 0.8, 0.8, f='d')
         co.el('T_co20', 3.8, 0.8, 0.8, f='d')
         co.el('T_co20', 4.3, 0.8, 0.8, f='d')
         co.el('COj0', 15.05, 0.06, 0.06)
         co.el('COj1', 14.80, 0.08, 0.08)
         co.el('COj2', 14.05, 0.21, 0.21)
         co.el('P_co', 4.02, 0.18, 0.20)
         co.el('n_co', 1.99, 0.28, 0.21)
         co.el('uv_co',3.5,0.-4.5)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.22, 0.2, 0.3)
         #co.el('uv_co_pdr', -1, 1, 0)
         co.el('n_co_goldsmith', 1.5, 0.2, 0.3)
         co.el('xco', 22.19, 0.13, 0.13)
         co.el('wco', -1.52, 0.16, 0.16)
         co.el('wco_emission', 2.13, 0, 0, f='d')  # form Liszt+ 2008
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 2.27, 0.28, 0.31)
         co.el('aG', -0.56,0.33,0.29)
         #co.el('uv_co_3dpdr', 0.0, 1.0, 1.0,'l')
         # pdr estimate in CO region
         co.el('tgas', 1.74, 0.21, 0.13)
         co.el('ngas(co)', 1.86, 0.35, 0.2)
         co.el('pgas', 3.66, 0.3, 0.14)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 30122
         q = qso('HD30122', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.11, 0.10, 0.10)
         q.el('H2', 20.70, 0.10, 0.10)
         q.el('EBV', 0.40, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 14.85, 0.05, 0.05)
         q.el('D', 260, 2, 3, f='d')
         q.el('Me', -0.27, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.70, 0.10, 0.10)
         co.el('T01', 60, 20, 12, f='d')
         co.el('T02', 85, 14, 10, f='d')
         co.el('T_h210', 61, 10, 10, f='d')
         co.el('T_h220', 86, 10, 10, f='d')
         co.el('T_h230', 121, 10, 10, f='d')
         co.el('T_h240', 185, 10, 10, f='d')
         co.el('H2j0', 20.50, 0.12, 0.12)
         co.el('H2j1', 20.21, 0.19, 0.19)
         co.el('H2j2', 18.59, 0.34, 0.34)
         co.el('H2j3', 18.15, 0.33, 0.33)
         co.el('H2j4', 17.50, 0.24, 0.24)
         co.el('CO', 14.85, 0.05, 0.05)
         co.el('T_co', 3.8, 0.4, 0.3, f='d')
         co.el('T_co10', 3.8, 0.8, 0.8, f='d')
         co.el('T_co20', 4.0, 0.8, 0.8, f='d')
         co.el('COj0', 14.60, 0.06, 0.06)
         co.el('COj1', 14.43, 0.06, 0.06)
         co.el('COj2', 13.47, 0.24, 0.24)
         co.el('P_co', 4.01, 0.16, 0.22)
         co.el('n_co', 2.08, 0.19, 0.17)
         co.el('uv_co', 0.31,0.19,0.17)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.14, 0.23, 0.22)
         co.el('uv_co_pdr', -0.17, 0.27, 0.23)
         co.el('n_co_goldsmith', 1.7, 0.3, 0.3)
         co.el('xco', 22.61, 0.12, 0.12)
         co.el('wco', -1.91, 0.16, 0.16)
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 2.21, 0.18, 0.19)
         co.el('uv_co_3dpdr', -0.15, 0.36, 0.27)
         co.el('aG', -0.41,0.36,0.32)
         # pdr estimate in CO region
         co.el('tgas', 1.84, 0.05)
         co.el('ngas(co)', 2.0, 0.24, 0.27)
         co.el('pgas', 3.81, 0.25, 0.23)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 36841
         '''q = qso('HD36841', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.18, 0.10, 0.10)
         q.el('H2', 20.40, 0.10, 0.10)
         q.el('EBV', 0.35, 0.0, 0.0, f='d')
         q.el('Me', -0.2, 0.0, 0.0)
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 14.08, 0.05, 0.05)
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.40, 0.10, 0.10)
         #co.el('T01', 1.8,0,0)
         #co.el('T_h210', 61, 10, 10, f='d')
         #co.el('T_h220', 86, 10, 10, f='d')
         #co.el('T_h230', 121, 10, 10, f='d')
         #co.el('T_h240', 185, 10, 10, f='d')
         # co.el('H2j0', 20.0, 0.13, 0.13)
         # co.el('H2j1', 20.07, 0.13, 0.13)
         # co.el('CI', 14.88, 0.05, 0.05)
         # co.el('CIj0', 14.77, 0.05)
         # co.el('CIj1', 14.14, 0.05)
         # co.el('CIj2', 13.51, 0.05)
         co.el('CO', 14.08, 0.05, 0.05)
         co.el('T_co10', 2.7, 0.8, 0.8, f='d')
         co.el('T_co20', 3.0, 0.8, 0.8, f='d')
         co.el('COj0', 14.70, 0.07, 0.07)
         co.el('COj1', 14.25, 0.16, 0.16)
         co.el('COj2', 12.92, 0.46, 0.46)
         co.el('T_co', 2.76, 0.9, 0.4, f='d')
         #co.el('P_co', '<3.05')
         co.el('n_co', 1.27,0.4,0.6)
         co.el('uv_co',-1,5,0)
         # co.el('P_ci', 3.67, 0.07, 0.07)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)
        '''

         # add HD HD 96675
         q = qso('HD96675', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 20.46, 0.10, 0.10)
         q.el('H2', 20.86, 0.10, 0.10)
         q.el('EBV', 0.30, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 15.28, 0.05, 0.05)
         q.el('D', 160, 2, 2, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', 0.00, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.86, 0.10, 0.10)
         #co.el('T_h210', 61, 10, 10, f='d')
         #co.el('T_h220', 86, 10, 10, f='d')
         #co.el('T_h230', 121, 10, 10, f='d')
         #co.el('T_h240', 185, 10, 10, f='d')
         # co.el('H2j0', 20.0, 0.13, 0.13)
         # co.el('H2j1', 20.07, 0.13, 0.13)
         # co.el('CI', 14.88, 0.05, 0.05)
         # co.el('CIj0', 14.77, 0.05)
         # co.el('CIj1', 14.14, 0.05)
         # co.el('CIj2', 13.51, 0.05)
         co.el('CO', 15.33, 0.05, 0.05)
         co.el('T_co10', 3.7, 0.8, 0.8, f='d')
         co.el('T_co20', 5.9, 0.8, 0.8, f='d')
         co.el('COj0', 15.04, 0.06, 0.06)
         co.el('COj1', 14.85, 0.07, 0.07)
         co.el('COj2', 14.51, 0.12, 0.12)
         co.el('T_co', 5.5, 1.4, 0.6, f='d')
         co.el('n_co',2.44,0.23,0.24)
         #co.el('P_co', 3.89, 0.36, 0.62)
         # co.el('P_ci', 3.67, 0.07, 0.07)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.67, 0.19, 0.14)
         #co.el('n_co_goldsmith', 2.8, 0.2, 0.2)
         co.el('xco', 22.18, 0.12, 0.12)
         co.el('wco', -1.32, 0.16, 0.16)
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 2.82, 0.19, 0.18)
         co.el('aG', -0.74,0.33,0.3)
         #co.el('uv_co_3dpdr', -0.31, 0.30, 0.27)
         #co.el('uv_co_3dpdr', -0.90, 1.90, 0.10)
         # pdr estimate in CO region
         co.el('tgas', 1.85, 0.05, 0.33)
         co.el('ngas(co)', 2.47, 0.19, 0.23)
         co.el('pgas', 4.16, 0.2, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 99872
         q = qso('HD99872', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.15, 0.10, 0.10)
         q.el('H2', 20.52, 0.10, 0.10)
         q.el('EBV', 0.36, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         q.el('D',230,10,10,f='d')
         q.el('Me', 0.0, 0.1, 0.1) #Rithcey2023
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 14.65, 0.05, 0.05)
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.52, 0.10, 0.10)
         co.el('T01', 66, 12, 10, f='d')
         co.el('T02', 93, 15, 10, f='d')
         co.el('T_h210', 66, 10, 10, f='d')
         co.el('T_h220', 94, 10, 10, f='d')
         co.el('T_h230', 114, 10, 10, f='d')
         co.el('T_h240', 179, 10, 10, f='d')
         co.el('H2j0', 20.29, 0.11, 0.11)
         co.el('H2j1', 20.10, 0.15, 0.15)
         co.el('H2j2', 18.61, 0.29, 0.29)
         co.el('H2j3', 17.72, 0.37, 0.37)
         co.el('H2j4', 17.15, 0.26, 0.26)
         co.el('CI', 14.54, 0.05, 0.05)
         co.el('CIj0', 14.40, 0.05)
         co.el('CIj1', 13.87, 0.05)
         co.el('CIj2', 13.34, 0.05)
         co.el('CO', 14.65, 0.05, 0.05)
         co.el('T_co', 3.7, 0.7, 0.4, f='d')
         co.el('T_co10', 3.7, 0.8, 0.8, f='d')
         co.el('T_co20', 3.8, 0.8, 0.8, f='d')
         co.el('COj0', 14.41, 0.06, 0.06)
         co.el('COj1', 14.22, 0.07, 0.07)
         co.el('COj2', 13.17, 0.27, 0.27)
         co.el('P_co', 3.96, 0.21, 0.27)
         co.el('n_co', 1.99, 0.24, 0.26)
         co.el('uv_co',0.31,0.23,0.14)
         co.el('P_ci', 4.11, 0.07, 0.07)
         co.el('n_ci', 1.76, 0.14, 0.14)
         co.el('uv_ci',0.31,0.21,0.19)
         co.el('PDRnH', 1.81, 0.11, 0.19)  # PDR fit
         co.el('PDRuv', 0.40, 0.60, 0.35)
         co.el('pci-tripp', 3.6, 0.2, 0.2)
         co.el('uvci-tripp', 0.82, 0.2, 0.2)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.06, 0.15, 0.22)
         co.el('uv_co_pdr', -0.19, 0.24, 0.22)
         co.el('n_ci_pdr', 1.98, 0.06, 0.08)
         co.el('uv_ci_pdr', -0.15, 0.23, 0.18)
         co.el('n_co_goldsmith', 1.5, 0.4, 0.4)
         co.el('xco', 22.67, 0.13, 0.13)
         co.el('wco', -2.15, 0.16, 0.17)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.03, 0.10, 0.10)
         co.el('uv_ci_3dpdr', -0.11, 0.37, 0.26)
         co.el('n_co_3dpdr', 2.12, 0.25, 0.32)
         co.el('uv_co_3dpdr', -0.11, 0.23, 0.30)
         co.el('aG',-0.56,0.27,0.23)
         # pdr estimate in CO region
         co.el('tgas', 1.88, 0.09, 0.05)
         co.el('ngas(co)', 1.9, 0.34, 0.41)
         co.el('pgas', 3.74, 0.34, 0.32)
         co.el('tgas(ci)', 2.02, 0.23, 0.12)
         co.el('ngas(ci)', 1.80, 0.18, 0.09)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD137595
         q = qso('HD137595', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 20.78, 0.10, 0.10)
         q.el('H2', 20.62, 0.10, 0.10)
         q.el('EBV', 0.25, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         q.el('D', 760, 30, 22, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 13.89, 0.05, 0.05)
         q.el('Fstar', 0.92, 0.1, 0.1, f='d')  # Ritchey2023
         q.el('Megas', -0.39, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         q.el('Me_[O/H]', -0.15, 0.08, 0.08)  # Ritchey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.62, 0.10, 0.10)
         co.el('T01', 70, 20, 13, f='d')
         co.el('T02', 93, 13, 10, f='d')
         co.el('T_h210', 72, 10, 10, f='d')
         co.el('T_h220', 94, 10, 10, f='d')
         co.el('T_h230', 124, 10, 10, f='d')
         co.el('T_h240', 197, 10, 10, f='d')
         co.el('H2j0', 20.35, 0.11, 0.11)
         co.el('H2j1', 20.25, 0.13, 0.13)
         co.el('H2j2', 18.66, 0.29, 0.29)
         co.el('H2j3', 18.09, 0.31, 0.31)
         co.el('H2j4', 17.59, 0.22, 0.22)
         # co.el('CI', 14.88, 0.05, 0.05)
         # co.el('CIj0', 14.77, 0.05)
         # co.el('CIj1', 14.14, 0.05)
         # co.el('CIj2', 13.51, 0.05)
         co.el('CO', 13.89, 0.05, 0.05)
         co.el('T_co', 4.16, 0.9, 0.3, f='d')
         co.el('T_co10', 3.9, 0.8, 0.8, f='d')
         co.el('T_co20', 4.4, 0.8, 0.8, f='d')
         co.el('COj0', 13.63, 0.06, 0.06)
         co.el('COj1', 13.48, 0.06, 0.06)
         co.el('COj2', 12.68, 0.20, 0.20)
         co.el('P_co', 4.15, 0.16, 0.17)
         co.el('n_co', 2.13, 0.16, 0.18)
         co.el('uv_co',0.36,0.1,0.12)
         # co.el('P_ci', 3.67, 0.07, 0.07)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.14, 0.11, 0.16)
         co.el('uv_co_pdr', -0.07, 0.21, 0.18)
         co.el('n_co_goldsmith', 1.8, 0.2, 0.1)
         co.el('xco', 23.42, 0.12, 0.12)
         co.el('wco', -2.80, 0.15, 0.16)
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 2.24, 0.15, 0.19)
         co.el('uv_co_3dpdr', -0.03, 0.26, 0.25)
         co.el('aG', -0.36,0.32,0.29)
         # pdr estimate in CO region
         co.el('tgas', 1.97, 0.05, 0.05)
         co.el('ngas(co)', 2.0, 0.24, 0.25)
         co.el('pgas', 3.99, 0.23, 0.23)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD144965
         q = qso('HD144965', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 20.90, 0.10, 0.10)
         q.el('H2', 20.79, 0.10, 0.10)
         q.el('EBV', 0.35, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 15.28, 0.05, 0.05)
         q.el('D', 300, 40, 40, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.79, 0.10, 0.10)
         co.el('T01', 70, 15, 10, f='d')
         co.el('T02', 91, 13, 10, f='d')
         co.el('T_h210', 70, 10, 10, f='d')
         co.el('T_h220', 91, 10, 10, f='d')
         co.el('T_h230', 125, 10, 10, f='d')
         co.el('T_h240', 203, 10, 10, f='d')
         co.el('H2j0', 20.53, 0.12, 0.12)
         co.el('H2j1', 20.41, 0.14, 0.14)
         co.el('H2j2', 18.77, 0.30, 0.30)
         co.el('H2j3', 18.31, 0.31, 0.21)
         co.el('H2j4', 17.88, 0.21, 0.21)
         co.el('CI', 14.28, 0.05, 0.05)
         co.el('CIj0', 13.46, 0.05)
         co.el('CIj1', 13.20, 0.05)
         co.el('CIj2', 12.86, 0.05)
         co.el('CO', 15.28, 0.05, 0.05)
         co.el('T_co', 5.05, 0.9, 0.3, f='d')
         co.el('T_co10', 4.3, 0.8, 0.8, f='d')
         co.el('T_co20', 5.3, 0.8, 0.8, f='d')
         co.el('COj0', 14.99, 0.06, 0.06)
         co.el('COj1', 14.90, 0.06, 0.06)
         co.el('COj2', 14.32, 0.14, 0.14)
         co.el('P_co', 4.39, 0.11, 0.12)
         co.el('n_co', 2.40, 0.12, 0.12)
         co.el('uv_co', 0.40,0.13,0.10)
         co.el('P_ci', 4.64, 0.06, 0.06)
         co.el('n_ci', 2.26, 0.09, 0.11)
         co.el('uv_ci', 0.40, 0.11, 0.11)
         co.el('PDRnH', 2.31, 0.12, 0.25)  # PDR fit
         co.el('PDRuv', 0.54, 0.98, 0.34)
         co.el('pci-tripp', 3.8, 0.2, 0.2)
         co.el('uvci-tripp', 0.76, 0.2, 0.2)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.38, 0.12, 0.10)
         co.el('uv_co_pdr', 0.17, 0.12, 0.10)
         co.el('n_ci_pdr', 2.38, 0.21, 0.07)
         co.el('uv_ci_pdr', 0.25, 0.21, 0.18)
         co.el('n_co_goldsmith', 2.1, 0.1, 0)
         co.el('xco', 22.09, 0.12, 0.12)
         co.el('wco', -1.3, 0.15, 0.15)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.42, 0.11, 0.09)
         co.el('uv_ci_3dpdr', 0.15, 0.34, 0.22)
         co.el('n_co_3dpdr', 2.58, 0.12, 0.13)
         co.el('uv_co_3dpdr', 0.25, 0.26, 0.21)
         co.el('aG',-0.46,0.25,0.20)
         # pdr estimate in CO region
         co.el('tgas', 1.78, 0.07, 0.05)
         co.el('ngas(co)', 2.34, 0.2, 0.16)
         co.el('pgas', 4.15, 0.26, 0.24)
         co.el('tgas(ci)', 2.03, 0.16, 0.17)
         co.el('ngas(ci)', 2.19, 0.17, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD147683
         q = qso('HD147683', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.06, 0.10, 0.10)
         q.el('H2', 20.74, 0.10, 0.10)
         q.el('EBV', 0.39, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 15.95, 0.05, 0.05)
         q.el('D', 294, 3, 3, f='d')
         q.el('Fstar', 1.069, 0.11, 0.11, f='d')  # Ritchey2023
         q.el('Megas', -0.51, 0.1, 0.1) #Rithcey2023
         q.el('Me_[O/H]', -0.21, 0.13, 0.13)  # Ritchey2023
         q.el('Me', 0.07, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.74, 0.10, 0.10)
         co.el('T01', 56, 13, 8, f='d')
         co.el('T02', 84, 13, 9, f='d')
         co.el('T_h210', 58, 10, 10, f='d')
         co.el('T_h220', 85, 10, 10, f='d')
         co.el('T_h230', 116, 10, 10, f='d')
         co.el('T_h240', 185, 10, 10, f='d')
         co.el('H2j0', 20.57, 0.12, 0.12)
         co.el('H2j1', 20.20, 0.21, 0.21)
         co.el('H2j2', 18.62, 0.34, 0.34)
         co.el('H2j3', 18.06, 0.36, 0.36)
         co.el('H2j4', 17.56, 0.24, 0.24)
         co.el('CI', 14.81, 0.05, 0.05)
         co.el('CIj0', 14.59, 0.05)
         co.el('CIj1', 14.27, 0.05)
         co.el('CIj2', 13.83, 0.05)
         co.el('CO', 15.99, 0.05, 0.05)
         co.el('T_co', 7.5, 0.9, 0.5, f='d')
         co.el('T_co10', 5.2, 0.8, 0.8, f='d')
         co.el('T_co20', 6.5, 0.8, 0.8, f='d')
         co.el('T_co30', 6.9, 0.8, 0.8, f='d')
         co.el('T_co40', 7.7, 0.8, 0.8, f='d')
         co.el('T_co50', 8.5, 0.8, 0.8, f='d')
         co.el('COj0', 15.60, 0.05, 0.05)
         co.el('COj1', 15.61, 0.05, 0.05)
         co.el('COj2', 15.18, 0.10, 0.10)
         co.el('COj3', 14.34, 0.16, 0.16)
         co.el('COj4', 13.42, 0.21, 0.21)
         co.el('COj5', 11.94, 0.31, 0.31)
         co.el('P_co', 4.65, 0.07, 0.07)
         co.el('n_co', 2.71, 0.09, 0.10)
         co.el('uv_co', 0.36,0.11,0.13)
         co.el('P_ci', 4.38, 0.09, 0.09)
         co.el('n_ci', 2.17, 0.10, 0.11)
         co.el('uv_ci', 0.27,0.10,0.11)
         co.el('PDRnH', 2.17, 0.14, 0.14)  # PDR fit
         co.el('PDRuv', 0.31, 0.67, 0.29)
         co.el('pci-tripp', 3.9, 0.2, 0.2)
         co.el('uvci-tripp', 0.57, 0.2, 0.2)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.83, 0.09, 0.07)
         co.el('uv_co_pdr', 0.31, 0.12, 0.14)
         co.el('n_ci_pdr', 2.30, 0.10, 0.17)
         co.el('uv_ci_pdr', -0.05, 0.17, 0.17)
         co.el('n_co_goldsmith', 2.4, 0, 0)
         co.el('xco', 21.25, 0.11, 0.11)
         co.el('wco', -0.51, 0.15, 0.15)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.33, 0.11, 0.09)
         co.el('uv_ci_3dpdr', -0.05, 0.32, 0.22)
         co.el('n_co_3dpdr', 2.94, 0.10, 0.07)
         #co.el('n_co_3dpdr', 2.36, 0.07, 0.09)
         co.el('uv_co_3dpdr', 0.27, 0.20, 0.18)
         co.el('aG', -0.59, 0.22,0.21)
         # pdr estimate in CO region
         co.el('tgas', 1.67, 0.07, 0.05)
         co.el('ngas(co)', 2.69, 0.12, 0.10)
         co.el('pgas', 4.38, 0.14, 0.13)
         co.el('tgas(ci)', 1.94, 0.20, 0.10)
         co.el('ngas(ci)', 2.12, 0.16, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD148937
         q = qso('HD148937', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.20, 0.10, 0.20)
         q.el('H2', 20.90, 0.20, 0.20)
         q.el('EBV', 0.65, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 14.5, 0.05, 0.05)
         q.el('D', 1250, 180, 140, f='d')
         q.el('Fstar', 0.675, 0.127, 0.127, f='d')  # Ritchey2023
         q.el('Me', 0.109, 0.108, 0.108,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.06, 0.11, 0.11)  # Ritchey2023
         q.el('Megas', -0.21, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 69, 10, 10, f='d')
         co.el('T02', 97, 10, 10, f='d')
         co.el('H2', 20.90,0.2,0.2)
         co.el('T_h210', 69, 10, 10, f='d')
         co.el('T_h220', 97, 10, 10, f='d')
         co.el('T_h230', 132, 10, 10, f='d')
         co.el('T_h240', 228, 10, 10, f='d')
         # co.el('H2j0', 20.0, 0.13, 0.13)
         # co.el('H2j1', 20.07, 0.13, 0.13)
         co.el('CI', 15.14, 0.05, 0.05)
         co.el('CIj0', 14.97, 0.05)
         co.el('CIj1', 14.54, 0.05)
         co.el('CIj2', 14.00, 0.05)
         co.el('CO', 14.5, 0.05, 0.05)
         co.el('T_co10', 3.7, 0.8, 0.8, f='d')
         co.el('T_co20', 4.4, 0.8, 0.8, f='d')
         co.el('T_co30', 5.6, 0.8, 0.8, f='d')
         co.el('COj0', 14.26, 0.06, 0.06)
         co.el('COj1', 14.07, 0.07, 0.07)
         co.el('COj2', 13.30, 0.20, 0.20)
         co.el('P_co', 4.09, 0.14, 0.17)
         co.el('n_co', 1.94, 0.17, 0.16)
         co.el('T_co', 4.1, 1.0, 0.2, f='d')
         co.el('n_ci', 1.76, 0.16, 0.16)
         # co.el('P_ci', 3.67, 0.07, 0.07)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.10, 0.14, 0.20)
         co.el('n_ci_pdr', 2.06, 0.11, 0.07)
         co.el('n_co_goldsmith', 1.8, 0.1, 0)
         co.el('xco', 23.12, 0.21, 0.21)
         co.el('wco', -2.22, 0.29, 0.29)
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 2.36, 0.23, 0.24)
         #co.el('uv_co_3dpdr', 0.90, 0.10, 1.9)
         co.el('n_ci_3dpdr', 2.15, 0.30, 0.17)
         co.el('aG',-0.18,0.28,0.26)
         #co.el('uv_ci_3dpdr', '<1')
         # pdr estimate in CO region
         co.el('tgas', 1.81, 0.2, 0.06)
         co.el('ngas(co)', 2.06, 0.22, 0.27)
         co.el('pgas', 3.88, 0.40, 0.12)
         co.el('tgas(ci)', 2.17, 0.14, 0.95)
         co.el('ngas(ci)', 1.96, 1.76, 0.22)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD190918
         q = qso('HD190918', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.38, 0.10, 0.10)
         q.el('H2', 19.95, 0.10, 0.10)
         q.el('EBV', 0.45, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 13.18, 0.05, 0.05)
         q.el('D', 1670, 330, 240, f='d')
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 19.95, 0.10, 0.10)
         co.el('T01', 101, 25, 16, f='d')
         co.el('T02', 156, 22, 17, f='d')
         co.el('T_h210', 102, 10, 10, f='d')
         co.el('T_h220', 156, 10, 10, f='d')
         co.el('T_h230', 214, 10, 10, f='d')
         co.el('T_h240', 310, 10, 10, f='d')
         co.el('H2j0', 19.51, 0.10, 0.10)
         co.el('H2j1', 19.73, 0.10, 0.10)
         co.el('H2j2', 18.79, 0.14, 0.14)
         co.el('H2j3', 18.77, 0.15, 0.15)
         co.el('H2j4', 18.11, 0.13, 0.13)
         # co.el('CI', 14.88, 0.05, 0.05)
         # co.el('CIj0', 14.77, 0.05)
         # co.el('CIj1', 14.14, 0.05)
         # co.el('CIj2', 13.51, 0.05)
         co.el('CO', 13.18, 0.05, 0.05)
         co.el('T_co', 3.7, 1.1, 0.8, f='d')
         co.el('T_co10', 2.7, 0.8, 0.8, f='d')
         co.el('T_co20', 4.0, 0.8, 0.8, f='d')
         co.el('COj0', 13.03, 0.06, 0.07)
         co.el('COj1', 12.58, 0.16, 0.16)
         co.el('COj2', 11.90, 0.24, 0.24)
         co.el('P_co', 3.93, 0.26, 0.33)
         co.el('n_co', 1.99, 0.25, 0.44)
         co.el('uv_co', 0.63, 0.16,0.10)
         # co.el('P_ci', 3.67, 0.07, 0.07)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 1.86, 0.24, 0.35)
         co.el('uv_co_pdr', 1, 0.0, 0.7)
         co.el('xco', 23.75, 0.23, 0.22)
         co.el('wco', -3.80, 0.24, 0.26)
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 1.97, 0.20, 0.46)
         co.el('uv_co_3dpdr', 0.43, 0.20, 0.30)
         co.el('aG',-0.19,0.40,0.33)
         # pdr estimate in CO region
         co.el('tgas', 2.19, 0.06, 0.11)
         co.el('ngas(co)', 1.71, 0.29, 0.53)
         co.el('pgas', 3.82, 0.31, 0.42)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD192035
         q = qso('HD192035', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.07, 0.10, 0.10)
         q.el('H2', 20.68, 0.10, 0.10)
         q.el('EBV', 0.37, 0.0, 0.0, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 15.15, 0.05, 0.05)
         q.el('D', 1690, 60, 60, f='d')
         q.el('Fstar', 0.914, 0.128, 0.128, f='d')  # Ritchey2023
         q.el('Me', 0.131, 0.099, 0.099,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.11, 0.1, 0.1)  # Ritchey2023
         q.el('Megas', -0.35, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.68, 0.10, 0.10)
         co.el('T01', 66, 12, 9, f='d')
         co.el('T02', 91, 13, 10, f='d')
         co.el('T_h210', 68, 10, 10, f='d')
         co.el('T_h220', 92, 10, 10, f='d')
         co.el('T_h230', 126, 10, 10, f='d')
         co.el('T_h240', 205, 10, 10, f='d')
         co.el('H2j0', 20.44, 0.10, 0.10)
         co.el('H2j1', 20.28, 0.14, 0.14)
         co.el('H2j2', 18.70, 0.29, 0.29)
         co.el('H2j3', 18.24, 0.31, 0.31)
         co.el('H2j4', 17.82, 0.21, 0.21)
         co.el('CO', 15.15, 0.05, 0.05)
         co.el('T_co', 3.5, 1.0, 0.15, f='d')
         co.el('T_co10', 3.2, 0.8, 0.8, f='d')
         co.el('T_co20', 3.9, 0.8, 0.8, f='d')
         co.el('COj0', 14.96, 0.06, 0.06)
         co.el('COj1', 14.66, 0.09, 0.09)
         co.el('COj2', 13.77, 0.26, 0.26)
         co.el('P_co', 3.84, 0.25, 0.34)
         co.el('n_co', 1.94, 0.24, 0.30)
         co.el('uv_co', 0.31,0.27,0.16)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 1.98, 0.19, 0.19)
         co.el('uv_co_pdr', -0.2, 0.2, 0.2)
         co.el('n_co_goldsmith', 1.6, 0.1, 0.1)
         co.el('xco', 22.44, 0.14, 0.14)
         co.el('wco', -3.80, 0.24, 0.26)
         co.el('wco_emission', 2.13, 0, 0, f='d')  # form Liszt+ 2008
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 1.97, 0.27, 0.35)
         co.el('uv_co_3dpdr', -0.13, 0.39, 0.33)
         co.el('aG', -0.23,0.50,0.42)
         # pdr estimate in CO region
         co.el('tgas', 1.86, 0.08, 0.10)
         co.el('ngas(co)', 1.71, 0.39, 0.56)
         co.el('pgas', 3.54, 0.39, 0.43)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD198781
         q = qso('HD198781', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 21.11, 0.10, 0.10)
         q.el('H2', 20.56, 0.10, 0.10)
         q.el('EBV', 0.35, 0.04, 0.04, f='d')
         q.el('Av', 0.75, 0.13, 0.13, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Rv', 2.14, 0.3, 0.3, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 15.23, 0.05, 0.05)
         q.el('O/H', 501, 74, 74, f='d')  # O/H gas Zou 2021
         q.el('D', 943, 18, 18, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         # co.el('T01', 66, 10, 10, f='d')
         co.el('H2', 20.56, 0.10, 0.10)
         co.el('T01', 63, 12, 10, f='d')
         co.el('T02', 91, 13, 10, f='d')
         co.el('T_h210', 65, 10, 10, f='d')
         co.el('T_h220', 92, 10, 10, f='d')
         co.el('T_h230', 128, 10, 10, f='d')
         co.el('T_h240', 191, 10, 10, f='d')
         co.el('H2j0', 20.34, 0.12, 0.12)
         co.el('H2j1', 20.12, 0.16, 0.16)
         co.el('H2j2', 18.60, 0.29, 0.29)
         co.el('H2j3', 18.19, 0.29, 0.29)
         co.el('H2j4', 17.46, 0.23, 0.23)
         co.el('CI', 14.53, 0.05, 0.05)
         co.el('CIj0', 14.41, 0.05)
         co.el('CIj1', 13.78, 0.05)
         co.el('CIj2', 13.35, 0.05)
         co.el('CO', 15.23, 0.05, 0.05)
         co.el('T_co', 3.5, 0.8, 0.4, f='d')
         co.el('T_co10', 3.4, 0.8, 0.8, f='d')
         co.el('T_co20', 3.7, 0.8, 0.8, f='d')
         co.el('COj0', 15.01, 0.07, 0.07)
         co.el('COj1', 14.77, 0.08, 0.08)
         co.el('COj2', 13.73, 0.29, 0.29)
         co.el('P_co', 3.86, 0.28, 0.42)
         co.el('n_co', 2.08, 0.21, 0.253)
         co.el('uv_co', 0.27,0.13,0.09)
         co.el('P_ci', 4.08, 0.06, 0.07)
         co.el('n_ci', 1.76, 0.12, 0.09)
         co.el('uv_ci', 0.22,0.12,0.09 )
         co.el('PDRnH', 1.72, 0.12, 0.29)  # PDR fit
         co.el('PDRuv', 0.54, 0.90, 0.43)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.02, 0.19, 0.27)
         co.el('uv_co_pdr', -0.21, 0.24, 0.22)
         co.el('n_ci_pdr', 1.94, 0.06, 0.06)
         co.el('uv_ci_pdr', -0.11, 0.25, 0.20)
         co.el('n_co_goldsmith', 1.5, 0.3, 0.4)
         co.el('xco', 22.22, 0.15, 0.15)
         co.el('wco', -1.66, 0.17, 0.18)
         ##################################### results with 3D-PDR:
         co.el('pci-tripp', 3.7, 0.2, 0.2)
         co.el('uvci-tripp', 0.41, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.00, 0.10, 0.11)
         co.el('uv_ci_3dpdr', -0.13, 0.41, 0.26)
         co.el('n_co_3dpdr', 2.00, 0.27, 0.35)
         co.el('uv_co_3dpdr', -0.17, 0.38, 0.33)
         co.el('aG',-0.48,0.27,0.24)
         # pdr estimate in CO region
         co.el('tgas', 1.85, 0.07, 0.10)
         co.el('ngas(co)', 1.74, 0.44, 0.49)
         co.el('pgas', 3.57, 0.40, 0.40)
         co.el('tgas(ci)', 2.03, 0.23, 0.12)
         co.el('ngas(ci)', 1.75, 0.19, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD203532
         q = qso('HD203532', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 20.78, 0.10, 0.10)
         q.el('H2', 20.70, 0.10, 0.10)
         q.el('EBV', 0.28, 0.03, 0.03, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Av', 0.94, 0.11, 0.11, f='d')
         q.el('CI', 14.75, 0.05, 0.05)
         q.el('CO', 15.66, 0.05, 0.05)
         q.el('O/H', 257, 46, 46, f='d')  # O/H gas Zou 2021
         q.el('Rv', 3.37, 0.24, 0.24, f='d')
         q.el('D', 290, 2, 2, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('H2', 20.70, 0.10, 0.10)
         co.el('T01', 47, 18, 10, f='d')
         co.el('T02', 77, 13, 10, f='d')
         co.el('T_h210', 47, 10, 10, f='d')
         co.el('T_h220', 78, 10, 10, f='d')
         co.el('T_h230', 102, 10, 10, f='d')
         co.el('T_h240', 169, 10, 10, f='d')
         co.el('H2j0', 20.60, 0.12, 0.12)
         co.el('H2j1', 19.88, 0.48, 0.48)
         co.el('H2j2', 18.41, 0.40, 0.40)
         co.el('H2j3', 17.55, 0.45, 0.45)
         co.el('H2j4', 17.22, 0.27, 0.27)
         co.el('CI', 14.75, 0.05, 0.05)
         co.el('CIj0', 13.19, 0.05)
         co.el('CIj1', 13.27, 0.05)
         co.el('CIj2', 12.93, 0.05)
         co.el('CO', 15.66, 0.05, 0.05)
         co.el('T_co', 4.9, 0.5, 0.7, f='d')
         co.el('T_co10', 5.3, 0.8, 0.8, f='d')
         co.el('T_co20', 4.8, 0.8, 0.8, f='d')
         co.el('COj0', 15.30, 0.05, 0.05)
         co.el('COj1', 15.32, 0.05, 0.05)
         co.el('COj2', 14.49, 0.17, 0.17)
         co.el('P_co', 4.36, 0.10, 0.13)
         co.el('n_co', 2.44, 0.17, 0.16)
         co.el('uv_co', 0.18,0.14,0.11)
         co.el('P_ci', 4.96, 0.17, 0.12)
         co.el('n_ci', 2.76, 0.17, 0.16)
         co.el('uv_ci', 0.18,0.12,0.12)
         co.el('PDRnH', 2.80, 0.10, 0.13)  # PDR fit
         co.el('PDRuv', 0.40, 0.28, 0.21)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.46, 0.19, 0.2)
         co.el('uv_co_pdr', -0.15, 0.26, 0.23)
         co.el('n_ci_pdr', 2.83, 0.13, 0.14)
         co.el('uv_ci_pdr', 0.09, 0.19, 0.15)
         co.el('n_co_goldsmith', 1.9, 0.1, 0)
         co.el('xco', 21.59, 0.12, 0.11)
         co.el('wco', -0.89, 0.15, 0.15)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.85, 0.12, 0.09)
         co.el('uv_ci_3dpdr', 0.09, 0.26, 0.15)
         co.el('n_co_3dpdr', 2.58, 0.14, 0.13)
         co.el('uv_co_3dpdr', -0.07, 0.31, 0.13)
         co.el('aG', -0.94,0.22,0.19)
         #
         co.el('pci-tripp', 4.4, 0.2, 0.2)
         co.el('uvci-tripp', 1.26, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 1.65, 0.08, 0.05)
         co.el('ngas(co)', 2.34, 0.21, 0.19)
         co.el('pgas', 4.02, 0.2, 0.2)
         co.el('tgas(ci)', 1.88, 0.17, 0.09)
         co.el('ngas(ci)', 2.61, 0.23, 0.13)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD220057
         q = qso('HD220057', 0., 0.)
         q.telescope = 'HST'
         q.year = 2000
         q.ref.append('Sheffer2008')
         q.el('HI', 20.95, 0.10, 0.10)
         q.el('H2', 20.34, 0.10, 0.10)
         q.el('EBV', 0.23, 0.06, 0.06, f='d')
         q.el('Me', -0.25, 0.28, 0.28)
         q.el('Av', 0.62, 0.20, 0.20, f='d')
         q.el('Rv', 2.71, 0.49, 0.49, f='d')
         # q.el('CI', 14.88, 0.05, 0.05)
         q.el('CO', 14.63, 0.05, 0.05)
         q.el('O/H', 446, 114, 114, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 43, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('D', 389, 3, 3, f='d')
         q.el('Fstar', 0.908, 0.104, 0.104, f='d')  # Ritchey2023
         q.el('Me', 0.138, 0.128, 0.128,f='d')  # Ritchey2023
         q.el('Me_[O/H]', -0.102, 0.13, 0.13)  # Ritchey2023
         q.el('Megas', -0.33, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('H2', 20.34, 0.10, 0.10)
         co.el('T01', 63, 13, 10, f='d')
         co.el('T02', 86, 13, 10, f='d')
         co.el('T_h210', 65, 10, 10, f='d')
         co.el('T_h220', 87, 10, 10, f='d')
         co.el('T_h230', 122, 10, 10, f='d')
         co.el('T_h240', 192, 10, 10, f='d')
         co.el('H2j0', 20.12, 0.12, 0.12)
         co.el('H2j1', 19.90, 0.16, 0.16)
         co.el('H2j2', 18.24, 0.33, 0.33)
         co.el('H2j3', 17.80, 0.32, 0.32)
         co.el('H2j4', 17.25, 0.23, 0.23)
         co.el('CI', 14.71, 0.05, 0.05)
         co.el('CIj0', 14.34, 0.05)
         co.el('CIj1', 13.63, 0.05)
         co.el('CIj2', 13.11, 0.05)
         co.el('CO', 14.63, 0.05, 0.05)
         co.el('T_co', 3.42, 0.9, 0.2, f='d')
         co.el('T_co10', 3.0, 0.8, 0.8, f='d')
         co.el('T_co20', 3.8, 0.8, 0.8, f='d')
         co.el('COj0', 14.45, 0.06, 0.06)
         co.el('COj1', 14.10, 0.11, 0.11)
         co.el('COj2', 13.22, 0.27, 0.27)
         co.el('P_co', 3.75, 0.32, 0.42)
         co.el('n_co', 1.85, 0.15, 0.22)
         co.el('uv_co', 0.13,0.14,0.11)
         co.el('P_ci', 3.94, 0.07, 0.07)
         co.el('n_ci', 1.63, 0.09, 0.13)
         co.el('uv_ci', 0.09,0.10,0.14)
         co.el('PDRnH', 1.58, 0.12, 0.27)  # PDR fit
         co.el('PDRuv', 0.27, 0.73, 0.44)
         ##################################### results with new version PDR:
         co.el('n_co_pdr', 2.02, 0.23, 0.33)
         co.el('uv_co_pdr', -0.43, 0.26, 0.31)
         co.el('n_ci_pdr', 1.86, 0.07, 0.11)
         co.el('uv_ci_pdr', -0.56, 0.21, 0.19)
         co.el('xco', 22.69, 0.16, 0.15)
         co.el('wco', -2.35, 0.18, 0.19)
         ##################################### results with 3D-PDR:
         #co.el('n_ci_3dpdr', 2.79, 0.11, 0.10)
         #co.el('uv_ci_3dpdr', 0.25, 0.24, 0.25)
         co.el('n_ci_3dpdr', 1.88, 0.11, 0.10)
         co.el('uv_ci_3dpdr', -0.41, 0.39, 0.25)
         co.el('n_co_3dpdr', 1.91, 0.26, 0.40)
         co.el('uv_co_3dpdr', -0.41, 0.41, 0.27)
         co.el('aG', -0.86,0.21,0.19)
         #
         co.el('pci-tripp', 3.51, 0.2, 0.2)
         co.el('uvci-tripp', 0.35, 0.2, 0.2)
         # pdr estimate in CO region
         co.el('tgas', 1.89, 0.14, 0.05)
         co.el('ngas(co)', 1.62, 0.44, 0.58)
         co.el('pgas', 3.49, 0.42, 0.43)
         co.el('tgas(ci)', 1.96, 0.25, 0.10)
         co.el('ngas(ci)', 1.66, 0.14, 0.13)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         # SAMPLE Jensen2010

         # add HD HD 38087
         q = qso('HD38087', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '207.07-16.26']
         q.ref.append('Jensen2010')
         q.el('HI', 20.91, 0.30, 0.30)
         q.el('H2', 20.65, 0.03, 0.03)
         q.el('T01', 70, 6, 6, f='d')
         q.el('EBV', 0.29, 0.0, 0.0, f='d')
         q.el('Av', 1.61, 0.0, 0.0, f='d')
         q.el('Rv', 5.57, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 70, 6, 6, f='d')
         co.el('T02', 105, 4, 4, f='d')
         co.el('H2', 20.65, 0.03, 0.03)
         co.el('H2j0', 20.39, 0.08, 0.08, b=(2.4, 0.1, 0.1))
         co.el('H2j1', 20.29, 0.05, 0.05, b=(2.4, 0.1, 0.1))
         co.el('H2j2', 18.98, 0.02, 0.02, b=(2.4, 0.1, 0.1))
         co.el('H2j3', 18.24, 0.03, 0.02, b=(2.4, 0.1, 0.1))
         co.el('H2j4', 17.25, 0.05, 0.05, b=(2.4, 0.1, 0.1))
         co.el('H2j5', 16.96, 0.10, 0.09, b=(2.4, 0.1, 0.1))
         co.el('H2j6', 15.46, 0.16, 0.15, b=(2.4, 0.1, 0.1))
         co.el('H2j7', 14.89, 0.12, 0.11, b=(2.4, 0.1, 0.1))
         co.el('H2j8', 13.80, 0.16, 0.20, b=(2.4, 0.1, 0.1))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 40893
         q = qso('HD40893', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '180.09+4.34']
         q.ref.append('Jensen2010')
         q.ref.append('Burgh2010')
         q.el('HI', 21.50, 0.10, 0.10)
         q.el('H2', 20.58, 0.03, 0.03)
         q.el('T01', 78, 9, 8, f='d')
         q.el('EBV', 0.46, 0.0, 0.0, f='d')
         q.el('Av', 1.13, 0.0, 0.0, f='d')
         q.el('Rv', 2.46, 0.0, 0.0, f='d')
         #q.el('CI', 14.95, 0.05, 0.05)
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('O/H', 363, 41, 41, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 41, 104, 104, f='d')  # ppm [O/H]gas ref Zuo 2021
         q.el('MeZou', 0.04, 0.25, 0.21)
         q.el('CO', 14.18, 0.10)
         q.el('CI', 14.65, 0.05)
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 78, 9, 8, f='d')
         co.el('T02', 75, 2, 2, f='d')
         co.el('H2', 20.58, 0.03, 0.03)
         co.el('H2j0', 20.27, 0.05, 0.05, b=(9.6, 0.2, 0.2))
         co.el('H2j1', 20.28, 0.05, 0.05, b=(9.6, 0.2, 0.2))
         co.el('H2j2', 18.01, 0.05, 0.05, b=(9.6, 0.2, 0.2))
         co.el('H2j3', 17.34, 0.06, 0.07, b=(9.6, 0.2, 0.2))
         co.el('H2j4', 15.21, 0.03, 0.03, b=(9.6, 0.2, 0.2))
         co.el('H2j5', 14.52, 0.05, 0.04, b=(9.6, 0.2, 0.2))
         co.el('CI', 14.65, 0.02, 0.01)
         co.el('CIj0', 14.86, 0.02, 0.01)
         co.el('CIj1', 14.13, 0.01, 0.02)
         co.el('CIj2', 13.47, 0.01, 0.01)
         co.el('CO', 14.18, 0.2,0.2)
         co.el('PDRnH',1.76,0.09,0.08)
         co.el('PDRuv',-0.59,0.22, 0.16)
         co.el('pci-tripp', 3.4, 0.2, 0.2)
         co.el('uvci-tripp', 0.37, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.85, 0.20, 0.2)
         co.el('uv_ci_3dpdr', -0.58, 0.06, 0.09)
         co.el('aG', -0.36, 0.44, 0.30)
         co.el('tgas(ci)', 1.91, 0.12, 0.07)
         co.el('ngas(ci)', 1.63, 0.1, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 46056
         q = qso('HD46056', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '206.34-2.25']
         q.ref.append('Jensen2010')
         q.el('HI', 21.38, 0.15, 0.15)
         q.el('H2', 20.68, 0.03, 0.03)
         q.el('T01', 73, 7, 6, f='d')
         q.el('EBV', 0.50, 0.0, 0.0, f='d')
         q.el('Av', 1.30, 0.0, 0.0, f='d')
         q.el('Rv', 2.60, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('O/H', 467, 181, 181, f='d')  # O/H gas Zou 2021
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 73, 7, 6, f='d')
         co.el('T02', 78, 3, 3, f='d')
         co.el('H2', 20.68, 0.03, 0.03)
         co.el('H2j0', 20.40, 0.06, 0.06, b=(6.2, 0.1, 0.2))
         co.el('H2j1', 20.35, 0.06, 0.06, b=(6.2, 0.1, 0.2))
         co.el('H2j2', 18.28, 0.03, 0.04, b=(6.2, 0.1, 0.2))
         co.el('H2j3', 17.40, 0.06, 0.05, b=(6.2, 0.1, 0.2))
         co.el('H2j4', 15.75, 0.05, 0.06, b=(6.2, 0.1, 0.2))
         co.el('H2j5', 15.06, 0.04, 0.07, b=(6.2, 0.1, 0.2))
         co.el('H2j6', 14.11, 0.39, 1.12, b=(6.2, 0.1, 0.2))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 46202
         q = qso('HD46202', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '206.31-2.00']
         q.ref.append('Jensen2010')
         q.el('HI', 21.58, 0.15, 0.15)
         q.el('H2', 20.68, 0.03, 0.03)
         q.el('T01', 77, 9, 7, f='d')
         q.el('EBV', 0.49, 0.0, 0.0, f='d')
         q.el('Av', 1.39, 0.0, 0.0, f='d')
         q.el('Rv', 2.83, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('O/H', 363, 206, 206, f='d')  # O/H gas Zou 2021
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 77, 9, 7, f='d')
         co.el('T02', 83, 2, 3, f='d')
         co.el('H2', 20.68, 0.03, 0.03)
         co.el('H2j0', 20.38, 0.07, 0.07, b=(7.2, 0.1, 0.1))
         co.el('H2j1', 20.38, 0.07, 0.07, b=(7.2, 0.1, 0.1))
         co.el('H2j2', 18.40, 0.03, 0.04, b=(7.2, 0.1, 0.1))
         co.el('H2j3', 17.66, 0.05, 0.04, b=(7.2, 0.1, 0.1))
         co.el('H2j4', 15.72, 0.03, 0.03, b=(7.2, 0.1, 0.1))
         co.el('H2j5', 15.14, 0.04, 0.05, b=(7.2, 0.1, 0.1))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # add HD HD 53367
         q = qso('HD53367', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '223.71-1.90']
         q.ref.append('Jensen2010')
         q.el('HI', 21.32, 0.30, 0.30)
         q.el('H2', 20.68, 0.03, 0.03)
         q.el('T01', 74, 7, 6, f='d')
         q.el('EBV', 0.74, 0.0, 0.0, f='d')
         q.el('Av', 1.76, 0.0, 0.0, f='d')
         q.el('Rv', 2.38, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 74, 7, 6, f='d')
         co.el('T02', 79, 2, 2, f='d')
         co.el('H2', 20.68, 0.03, 0.03)
         co.el('H2j0', 20.40, 0.06, 0.06, b=(6.2, 0.1, 0.2))
         co.el('H2j1', 20.35, 0.06, 0.06, b=(6.2, 0.1, 0.2))
         co.el('H2j2', 18.28, 0.03, 0.04, b=(6.2, 0.1, 0.2))
         co.el('H2j3', 17.40, 0.06, 0.05, b=(6.2, 0.1, 0.2))
         co.el('H2j4', 15.75, 0.05, 0.06, b=(6.2, 0.1, 0.2))
         co.el('H2j5', 15.06, 0.04, 0.07, b=(6.2, 0.1, 0.2))
         co.el('H2j6', 14.11, 0.39, 1.12, b=(6.2, 0.1, 0.2))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)




         # add HD HD 149404
         q = qso('HD149404', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '340.54+3.01']
         q.ref.append('Jensen2010')
         q.el('HI', 21.40, 0.15, 0.15)
         q.el('H2', 20.79, 0.03, 0.03)
         q.el('T01', 61, 3, 3, f='d')
         q.el('EBV', 0.62, 0.0, 0.0, f='d')
         q.el('Av', 2.19, 0.0, 0.0, f='d')
         q.el('Rv', 3.53, 0.0, 0.0, f='d')
         q.el('Megas', -0.24, 0.1, 0.1) #Rithcey2023
         q.el('Fstar', 0.654, 0.113, 0.113, f='d')  # Ritchey2023
         q.el('Me', 0.085, 0.123, 0.123,f='d')  # Ritchey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 61, 3, 3, f='d')
         co.el('T02', 73, 1, 2, f='d')
         co.el('H2', 20.79, 0.03, 0.03)
         co.el('H2j0', 20.60, 0.03, 0.03, b=(8.3, 0.1, 0.1))
         co.el('H2j1', 20.34, 0.05, 0.05, b=(8.3, 0.1, 0.1))
         co.el('H2j2', 18.26, 0.04, 0.06, b=(8.3, 0.1, 0.1))
         co.el('H2j3', 17.13, 0.06, 0.06, b=(8.3, 0.1, 0.1))
         co.el('H2j4', 16.05, 0.04, 0.04, b=(8.3, 0.1, 0.1))
         co.el('H2j5', 15.49, 0.04, 0.04, b=(8.3, 0.1, 0.1))
         co.el('H2j6', 14.17, 0.04, 0.04, b=(8.3, 0.1, 0.1))
         co.el('H2j7', 14.03, 0.14, 0.18, b=(8.3, 0.1, 0.1))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 170740
         q = qso('HD170740', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '21.06-0.53']
         q.ref.append('Jensen2010')
         q.el('HI', 21.15, 0.15, 0.15)
         q.el('H2', 20.87, 0.03, 0.03)
         q.el('T01', 71, 9, 7, f='d')
         q.el('EBV', 0.48, 0.0, 0.0, f='d')
         q.el('Av', 1.30, 0.0, 0.0, f='d')
         q.el('Rv', 2.71, 0.0, 0.0, f='d')
         q.el('Me', -0.2, 0.0, 0.0)
         q.el('O/H', 396, 71, 71, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 60, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('Fstar', 0.986, 0.146, 0.146, f='d')  # Ritchey2023
         q.el('Me', 0.053, 0.109, 0.109,f='d')  # Ritchey2023
         q.el('Megas', -0.49, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 71, 9, 7, f='d')
         co.el('H2', 20.87, 0.03, 0.03)
         co.el('T02', 90, 2, 2, f='d')
         co.el('H2j0', 20.60, 0.05, 0.05, b=(2.4, 0.2, 0.1))
         co.el('H2j1', 20.52, 0.11, 0.11, b=(2.4, 0.2, 0.1))
         co.el('H2j2', 18.86, 0.02, 0.02, b=(2.4, 0.2, 0.1))
         co.el('H2j3', 17.73, 0.03, 0.02, b=(2.4, 0.2, 0.1))
         co.el('H2j4', 17.08, 0.05, 0.05, b=(2.4, 0.2, 0.1))
         co.el('H2j5', 15.91, 0.17, 0.17, b=(2.4, 0.2, 0.1))
         co.el('H2j6', 14.72, 0.26, 0.23, b=(2.4, 0.2, 0.1))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         # add HD HD 73882
         q = qso('HD73882', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '223.71-1.90']
         q.ref.append('Snow2000')
         q.el('HI', 21.11, 0.15, 0.15)
         q.el('H2', 21.08, 0.10, 0.10)
         q.el('T01', 58, 10, 10, f='d')
         q.el('EBV', 0.72, 0.0, 0.0, f='d')
         q.el('Av', 2.44, 0.0, 0.0, f='d')
         q.el('Rv', 3.9, 0.0, 0.0, f='d')
         q.el('Me', -0.2, 0.0, 0.0)
         q.el('O/H', 40, 216, 216, f='d')  # ppm [O/H]gas ref Zuo 2021
         q.el('Si/H', 62, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('Fstar', 1.120, 0.12, 0.12, f='d')  # Ritchey2023
         q.el('Megas', -0.55, 0.1, 0.1) #Rithcey2023
         q.el('Me', 0.054, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 58, 7, 7, f='d')
         co.el('T02', 88, 6, 5, f='d')
         co.el('H2', 20.8, 0.10, 0.10)
         co.el('H2j0', 20.91, 0.10)
         co.el('H2j1', 20.59, 0.10)
         co.el('H2j2', 19.10, 0.10)
         co.el('H2j3', 18.50, 0.20)
         co.el('H2j4', 17.50, 0.30)
         co.el('H2j5', 17.00, 0.30)
         co.el('H2j6', 14.50, 0.80)
         co.el('H2j7', 14.00, 0.50)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

         # add HD HD 179406
         q = qso('HD179406', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '28.23-8.31']
         q.ref.append('Jensen2010')
         q.el('HI', 21.23, 0.15, 0.15)
         q.el('H2', 20.73, 0.03, 0.03)
         q.el('T01', 59, 5, 5, f='d')
         q.el('EBV', 0.33, 0.0, 0.0, f='d')
         q.el('Av', 0.94, 0.0, 0.0, f='d')
         q.el('Rv', 2.86, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         q.el('O/H', 213, 63, 63, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 31, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 59, 5, 5, f='d')
         co.el('T02', 66, 2, 2, f='d')
         co.el('H2', 20.73, 0.03, 0.03)
         co.el('H2j0', 20.55, 0.07, 0.07, b=(6.3, 0.2, 0.2))
         co.el('H2j1', 20.26, 0.08, 0.08, b=(6.3, 0.2, 0.2))
         co.el('H2j2', 17.92, 0.04, 0.05, b=(6.3, 0.2, 0.2))
         co.el('H2j3', 16.68, 0.11, 0.11, b=(6.3, 0.2, 0.2))
         co.el('H2j4', 15.23, 0.07, 0.07, b=(6.3, 0.2, 0.2))
         co.el('H2j5', 14.44, 0.07, 0.08, b=(6.3, 0.2, 0.2))
         co.el('H2j6', 13.78, 0.13, 0.19, b=(6.3, 0.2, 0.2))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

         # add HD HD 186994
         q = qso('HD186994', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '78.62+10.06']
         q.ref.append('Jensen2010')
         q.el('HI', 20.90, 0.15, 0.15)
         q.el('H2', 19.59, 0.03, 0.03)
         q.el('T01', 96, 10, 8, f='d')
         q.el('EBV', 0.17, 0.0, 0.0, f='d')
         q.el('Av', 0.53, 0.0, 0.0, f='d')
         q.el('Rv', 3.10, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 96, 10, 8, f='d')
         co.el('T02', 94, 3, 3, f='d')
         co.el('H2', 19.59, 0.03, 0.03)
         co.el('H2j0', 19.18, 0.06, 0.06, b=(5.0, 0.1, 0.1))
         co.el('H2j1', 19.37, 0.03, 0.03, b=(5.0, 0.1, 0.1))
         co.el('H2j2', 17.53, 0.02, 0.02, b=(5.0, 0.1, 0.1))
         co.el('H2j3', 17.12, 0.02, 0.04, b=(5.0, 0.1, 0.1))
         co.el('H2j4', 15.02, 0.02, 0.03, b=(5.0, 0.1, 0.1))
         co.el('H2j5', 14.31, 0.05, 0.03, b=(5.0, 0.1, 0.1))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 195965
         q = qso('HD195965', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '85.71+5.00']
         q.ref.append('Jensen2010')
         q.el('HI', 20.95, 0.07, 0.07)
         q.el('H2', 20.37, 0.03, 0.03)
         q.el('T01', 110, 7, 7, f='d')
         q.el('EBV', 0.25, 0.0, 0.0, f='d')
         q.el('Av', 0.77, 0.0, 0.0, f='d')
         q.el('Rv', 3.08, 0.0, 0.0, f='d')
         q.el('CI', 14.67, 0.05, 0.05)
         q.el('Me', -0.2, 0.0, 0.0)
         q.el('Fstar', 0.50, 0.1, 0.1, f='d')  # Ritchey2023
         q.el('Me', 0.032, 0.08, 0.08,f='d')  # Ritchey2023
         q.el('Megas', -0.21, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 110, 7, 7, f='d')
         co.el('T02', 91, 2, 1.4, f='d')
         co.el('H2', 20.37, 0.03, 0.03)
         co.el('H2j0', 19.90, 0.03, 0.03, b=(6.4, 0.1, 0.1))
         co.el('H2j1', 20.18, 0.03, 0.03, b=(6.4, 0.1, 0.1))
         co.el('H2j2', 18.16, 0.04, 0.03, b=(6.4, 0.1, 0.1))
         co.el('H2j3', 17.20, 0.03, 0.04, b=(6.4, 0.1, 0.1))
         co.el('H2j4', 15.45, 0.02, 0.02, b=(6.4, 0.1, 0.1))
         co.el('H2j5', 14.57, 0.02, 0.02, b=(6.4, 0.1, 0.1))
         co.el('H2j6', 13.81, 0.08, 0.09, b=(6.4, 0.1, 0.1))
         co.el('CI', 14.67,0.05,0.05)
         co.el('CIj0', 14.55, 0.05,0.05)
         co.el('CIj1', 13.94, 0.05,0.05)
         co.el('CIj2', 13.38, 0.05,0.05)
         co.el('PDRnH',1.85,0.12,0.09)
         co.el('PDRuv',-0.32, 0.22, 0.22)
         co.el('pci-tripp', 3.56, 0.2, 0.2)
         co.el('uvci-tripp', 0.31, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.97, 0.10, 0.08)
         co.el('uv_ci_3dpdr', -0.49, 0.19, 0.19)
         co.el('tgas(ci)', 1.92, 0.11, 0.17)
         co.el('ngas(ci)', 1.77, 0.11, 0.12)
         #co.el('aG', 0.1, 0.51, 0.41)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 197512
         q = qso('HD197512', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '87.89+4.63']
         q.ref.append('Jensen2010')
         q.el('HI', 21.26, 0.15, 0.15)
         q.el('H2', 20.66, 0.03, 0.03)
         q.el('T01', 94, 10, 7, f='d')
         q.el('EBV', 0.32, 0.0, 0.0, f='d')
         q.el('Av', 0.75, 0.0, 0.0, f='d')
         q.el('Rv', 2.35, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         q.el('O/H', 158, 1888, 1888, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 33, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 94, 10, 7, f='d')
         co.el('H2', 20.66, 0.03, 0.03)
         co.el('T02', 80, 2, 2, f='d')
         co.el('H2j0', 20.27, 0.05, 0.05, b=(7.2, 0.1, 0.2))
         co.el('H2j1', 20.44, 0.05, 0.05, b=(7.2, 0.1, 0.2))
         co.el('H2j2', 18.19, 0.03, 0.04, b=(7.2, 0.1, 0.2))
         co.el('H2j3', 17.08, 0.07, 0.07, b=(7.2, 0.1, 0.2))
         co.el('H2j4', 15.63, 0.05, 0.05, b=(7.2, 0.1, 0.2))
         co.el('H2j5', 15.06, 0.05, 0.06, b=(7.2, 0.1, 0.2))
         co.el('H2j6', 14.38, 0.18, 0.26, b=(7.2, 0.1, 0.2))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 199579
         q = qso('HD199579', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '85.70-0.30']
         q.ref.append('Jensen2010')
         q.el('HI', 21.04, 0.11, 0.11)
         q.el('H2', 20.53, 0.03, 0.03)
         q.el('T01', 69, 3, 3, f='d')
         q.el('EBV', 0.37, 0.0, 0.0, f='d')
         q.el('Av', 1.09, 0.0, 0.0, f='d')
         q.el('Rv', 2.95, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         q.el('O/H', 46, 22, 22, f='d')  # O/H gas Zou 2021
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 69, 3, 3, f='d')
         co.el('T02', 76, 1, 2, f='d')
         co.el('H2', 20.53, 0.03, 0.03)
         co.el('H2j0', 20.28, 0.03, 0.03, b=(12.5, 0.1, 0.1))
         co.el('H2j1', 20.17, 0.03, 0.03, b=(12.5, 0.1, 0.1))
         co.el('H2j2', 18.08, 0.03, 0.06, b=(12.5, 0.1, 0.1))
         co.el('H2j3', 17.12, 0.01, 0.02, b=(12.5, 0.1, 0.1))
         co.el('H2j4', 15.87, 0.02, 0.02, b=(12.5, 0.1, 0.1))
         co.el('H2j5', 15.46, 0.03, 0.02, b=(12.5, 0.1, 0.1))
         co.el('H2j6', 14.31, 0.02, 0.03, b=(12.5, 0.1, 0.1))
         co.el('H2j7', 14.18, 0.05, 0.05, b=(12.5, 0.1, 0.1))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 203938
         q = qso('HD203938', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '90.56-2.23']
         q.ref.append('Jensen2010')
         q.el('HI', 21.48, 0.15)
         q.el('H2', 21.00, 0.03, 0.03)
         q.el('T01', 74, 8, 6, f='d')
         q.el('EBV', 0.74, 0.0, 0.0, f='d')
         q.el('Av', 2.15, 0.0, 0.0, f='d')
         q.el('Rv', 2.91, 0.0, 0.0, f='d')
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 74, 8, 6, f='d')
         co.el('T02', 91, 2, 3, f='d')
         co.el('H2', 21.00, 0.03, 0.03)
         co.el('H2j0', 20.72, 0.05, 0.05, b=(5.3, 0.2, 0.2))
         co.el('H2j1', 20.68, 0.08, 0.08, b=(5.3, 0.2, 0.2))
         co.el('H2j2', 18.98, 0.03, 0.04, b=(5.3, 0.2, 0.2))
         co.el('H2j3', 17.68, 0.07, 0.07, b=(5.3, 0.2, 0.2))
         co.el('H2j4', 16.08, 0.08, 0.07, b=(5.3, 0.2, 0.2))
         co.el('H2j5', 15.52, 0.09, 0.09, b=(5.3, 0.2, 0.2))
         co.el('H2j6', 14.17, 0.17, 0.22, b=(5.3, 0.2, 0.2))
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         # add HD HD 207538
         q = qso('HD207538', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '101.60+4.67']
         q.ref.append('Jensen2010')
         q.el('HI', 21.34, 0.12)
         q.el('H2', 20.91, 0.03, 0.03)
         q.el('T01', 73, 7, 6, f='d')
         q.el('EBV', 0.64, 0.0, 0.0, f='d')
         q.el('Av', 1.44, 0.0, 0.0, f='d')
         q.el('Rv', 2.25, 0.0, 0.0, f='d')
         q.el('Me', -0.2, 0.0, 0.0)
         q.el('Fstar', 0.980, 0.090, 0.090, f='d')  # Ritchey2023
         q.el('Me', 0.140, 0.08, 0.08,f='d')  # Ritchey2023
         q.el('Megas', -0.38, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 73, 7, 6, f='d')
         co.el('H2', 20.91, 0.03, 0.03)
         co.el('T02', 80, 2, 2, f='d')
         co.el('H2j0', 20.64, 0.07, 0.07, b=(4.6, 0.1, 0.1))
         co.el('H2j1', 20.58, 0.05, 0.05, b=(4.6, 0.1, 0.1))
         co.el('H2j2', 18.56, 0.02, 0.02, b=(4.6, 0.1, 0.1))
         co.el('H2j3', 17.62, 0.02, 0.03, b=(4.6, 0.1, 0.1))
         co.el('H2j4', 16.04, 0.04, 0.06, b=(4.6, 0.1, 0.1))
         co.el('H2j5', 14.82, 0.07, 0.07, b=(4.6, 0.1, 0.1))
         co.el('TC2', 35, 15, 15,  f='d')
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

     # Federman 2021+
     if 1:
         q = qso('HD28975', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '163.08-17.14']  # from table 1
         q.ref.append('Federman2021')
         q.el('Htot', 21.54, 0.1, 0.1)
         q.el('HI', 20.54, 0.1, 0.1)
         q.el('H2', 21.2, 0.1, 0.1)
         q.el('T01', 30, 3, 3, f='d')
         q.el('EBV', 0.60, 0.0, 0.0, f='d')
         q.el('Av', 1.9, 0.5, 0.5, f='d')  # Teixeira 1999
         q.el('Rv', 3.1, 0.5, 0.5, f='d')
         q.el('D',194,4,4,f='d')
         # q.el('CI', 14.98, 0.10, 0.10)  # ref 'Burgh2010'
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         q.el('Megas', -0.27, 0.1, 0.1)  # Rithcey2023
         q.el('Me_[O/H]', -0.15, 0.1, 0.1)  # Ritchey2023
         q.el('CO', 17.16, 0.02, 0.02)
         s = []
         co = sy(0, 0)
         #co.el('T01', 30, 10, 10, f='d')
         co.el('TC2', 30, 10, 10, f='d')
         co.el('H2', 21.2, 0.1, 0.1)
         co.el('H2j0', 21.18, 0.1, 0.1)
         co.el('H2j1', 19.59, 0.1, 0.1)
         # co.el('CI', '<16')  # ref 'Jenkins201'
         co.el('CO', 17.16, 0.02, 0.02)  # ref "Sonnentrucker2007
         co.el('COj0', 16.56, 0.11, 0.14)
         co.el('COj1', 16.83, 0.10, 0.14)
         co.el('COj2', 16.62, 0.11, 0.15)
         # co.el('T_co', 11.5, 0.6, 0.6, f='d')
         co.el('T_co', 11.3, 2.4, 2.4, f='d')
         # co.el('n_co_pdr', 3.60, 1.44, 0.37)
         co.el('n_co_pdr', 3.17, 0.21, 0.29)
         co.el('n_co_federman', 3.3, 0, 0)
         co.el('wco', 0.74, 0.17, 0.20)
         co.el('xco', 20.46, 0.17, 0.14)
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 3.12, 0.28, 0.29)
         co.el('uv_co_3dpdr', 0.25, 0.41, 0.43)
         co.el('aG', -0.70,0.51,0.51)
         # pdr estimate in CO region
         co.el('tgas', 1.21, 0.19, 0.03)
         co.el('ngas(co)', 2.93, 0.39, 0.44)
         co.el('pgas', 4.11, 0.61, 0.45)
         co.el('ngas(ci)', 2.12, 0.11, 0.09)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD29647', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '163.08-17.14']  # from table 1
         q.ref.append('Federman2021')
         # q.ref.append('Sheffer2008')
         q.el('Htot', 21.8, 0.1, 0.1)
         q.el('HI', 20.8, 0.1, 0.1)
         q.el('H2', 21.5, 0.1, 0.1)
         q.el('T01', 10, 3, 3, f='d')
         q.el('EBV', 1.09, 0.0, 0.0, f='d')
         q.el('Av', 3.8, 0.5, 0.5, f='d')  # Teixeira 1999
         q.el('Rv', 3.5, 0.5, 0.5, f='d')
         q.el('CO', 17.98, 0.02, 0.02)
         q.el('Megas', -0.27, 0.1, 0.1) #Rithcey2023
         q.el('Me', -0.0, 0.1, 0.1)  # Rithcey2023
         q.el('D',155,2,2,f='d')
         q.el('Me_[O/H]', -0.15, 0.1, 0.1)  # Ritchey2023
         s = []
         # v = 10      # if you dont know it write 0
         co = sy(0, 0)
         #co.el('T01', '<20', f='d')
         co.el('TC2', '<20', f='d')
         co.el('H2', 21.5, 0.1, 0.1)
         co.el('H2j0', 21.5, 0.1, 0.1)
         co.el('H2j1', 18.66, 0.6, 0.6)
         # co.el('CI', '<16')  #
         co.el('CO', 17.98, 0.02, 0.02)  #
         co.el('COj0', 17.40, 0.03, 0.03)
         co.el('COj1', 17.63, 0.02, 0.02)
         co.el('COj2', 17.35, 0.03, 0.03)
         co.el('COj3', 16.73, 0.09, 0.11)
         # co.el('T_co', 9.5, 0.5, 0.50, f='d')
         co.el('T_co', 9.6, 0.5, 0.70, f='d')
         # co.el('n_co_pdr', 3.14, 0.42, 0.05)
         co.el('n_co_pdr', 3.35, 0.31, 0.44)
         co.el('n_co_federman', 3.25, 0, 0)
         co.el('wco', 1.53, 0.02, 0.02)
         co.el('xco', 19.97, 0.10, 0.10)
         ##################################### results with 3D-PDR:
         co.el('n_co_3dpdr', 3.24, 0.40, 0.32)
         #co.el('uv_co_3dpdr', -1, 0.6, 0.0)
         co.el('uv_co_3dpdr', '<-0.4')
         co.el('aG', -2.45,0.3,-0.14)
         # pdr estimate in CO region
         co.el('tgas', 1.07, 0.05, 0.05)
         co.el('ngas(co)', 3.00, 0.78, 0.5)
         co.el('pgas', 4.05, 0.75, 0.43)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         ################################################33
         # Welty 2020+ sample (https://iopscience.iop.org/article/10.3847/1538-4357/ab8f8e/pdf)
         q = qso('HD62542', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '163.08-17.14']  # from table 1
         q.ref.append('Welty2020')
         # q.ref.append('Sheffer2008')
         q.el('HI', 20.73, 0.16, 0.16)
         q.el('H2', 20.80, 0.2, 0.2)
         q.el('T01', 43, 11, 11, f='d')
         q.el('EBV', 0.36, 0.02, 0.02, f='d')  # ref de Cia 2021
         q.el('Av', 1.02, 0.10, 0.10, f='d')  # ref de Cia 2021
         q.el('Rv', 2.82, 0.24, 0.24, f='d')  # ref de Cia 2021
         q.el('CI', 15.41, 0.10, 0.10)  # ref 'Burgh2010'
         q.el('Me', -0.69, 0.07, 0.07)  # ref de Cia 2021
         q.el('D', 385, 4, 4, f='d')
         q.el('CO', 16.42, 0.04, 0.04)
         q.el('O/H', 125, 355, 355, f='d')  # O/H gas Zou 2021
         q.el('Si/H', 41, 0, 0, f='d')  # ppm [Si/H]dust ref Zuo 2021
         q.el('MeZou', -0.2, 0.5, 0.5)
         q.el('Fstar', 1.406, 0.135, 0.135, f='d')  # Ritchey2023
         q.el('Me', 0.129, 0.198, 0.19,f='d')  # Ritchey2023
         q.el('Megas', -0.69, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10
         co = sy(0, 0)
         co.el('T01', 43, 11, 11, f='d')
         co.el('TC2', 30, 10, 10, f='d')
         # co.el('T02', 72, 1, 1, f='d')  # calc boot
         co.el('H2', 20.81, 0.21, 0.21)
         co.el('H2j0', 20.74, 0.21, 0.21)
         co.el('H2j1', 19.98, 0.14, 0.14)
         co.el('CI', 15.41, 0.02, 0.02)  #
         co.el('CIj0', 14.90, 0.04, 0.04)  #
         co.el('CIj1', 15.10, 0.03, 0.03)  #
         co.el('CIj2', 14.69, 0.03, 0.03)  #
         co.el('CO', 16.42, 0.04, 0.04)  #
         co.el('COj0', 15.78, 0.07, 0.08)
         co.el('COj1', 16.02, 0.08, 0.09)
         co.el('COj2', 15.85, 0.08, 0.10)
         co.el('COj3', 15.40, 0.08, 0.08)
         co.el('COj4', 14.54, 0.06, 0.07)
         co.el('COj5', 13.50, 0.06, 0.07)
         co.el('COj6', 12.70, 0.15, 0.22)
         # co.el('T_co', 11.7, 0.5, 0.50, f='d')
         co.el('T_co', 11.7, 0.3, 0.3, f='d')
         # co.el('PDRnH', 2.62, 0.10, 0.06)  # PDR fit
         # co.el('PDRuv', -0.05, 0.11, 0.11)
         co.el('n_co_pdr', 3.53, 0.04, 0.07)
         co.el('n_ci_pdr', 2.75, 0.12, 0.07)  # 0.11
         co.el('n_co_welty', 3.2, 0, 0)
         co.el('xco', 20.87, 0.23, 0.22)
         co.el('wco', -0.06, 0.08, 0.09)
         ##################################### results with 3D-PDR:
         co.el('n_ci_3dpdr', 2.88, 0.10, 0.08)
         co.el('uv_ci_3dpdr', 0.54, 0.10, 0.08)
         co.el('n_co_3dpdr', 3.36, 0.04, 0.05)
         # co.el('n_co_3dpdr', 2.91, 0.06, 0.06)
         co.el('uv_co_3dpdr', 1, 0., 0.10)
         co.el('aG', -0.71,0.14,0.14)
         # pdr estimate in CO region
         co.el('tgas', 1.7, 0.04, 0.04)
         co.el('ngas(co)', 3.14, 0.04, 0.08)
         co.el('pgas', 4.81, 0.07, 0.07)
         co.el('tgas(ci)', 2.03, 0.09, 0.15)
         co.el('ngas(ci)', 2.68, 0.10, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     # SAMPLE Jenkins2011
     if 1:
         q = qso('CPD-592603', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.15, 0.1)
         q.el('Me',-0.092,0.08)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.15, 0.1)
         co.el('H2j0', 19.85, 0.1)
         co.el('H2j1', 19.85, 0.1)
         co.el('CI', 14.82, 0.02, 0.02)  #
         co.el('CIj0', 14.67, 0.02, 0.02)  #
         co.el('CIj1', 14.15, 0.02, 0.02)  #
         co.el('CIj2', 13.67, 0.02, 0.02)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 77, 5, 5, f='d')
         co.el('pci-tripp', 3.63, 0.2, 0.2)
         co.el('uvci-tripp', 0.31, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.06,0.09,0.08)
         co.el('uv_ci_3dpdr', -0.2, 0.24, 0.3)
         co.el('aG', -0.33, 0.41, 0.60)
         co.el('tgas(ci)', 1.96, 0.2, 0.1)
         co.el('ngas(ci)', 1.86, 0.12, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD75309', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.2, 0.1)
         q.el('Me', -0.06, 0.07)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.2, 0.1)
         co.el('H2j0', 19.98, 0.1)
         co.el('H2j1', 19.79, 0.1)
         co.el('CI', 14.59, 0.02, 0.02)  #
         co.el('CIj0', 14.49, 0.05)  #
         co.el('CIj1', 13.80, 0.05)  #
         co.el('CIj2', 13.21, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 65, 5, 5, f='d')
         co.el('pci-tripp', 3.41, 0.2, 0.2)
         co.el('uvci-tripp', 0.46, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.95, 0.08, 0.1)
         #co.el('uv_ci_3dpdr', -0.98, 0.44, 0.02)
         co.el('uv_ci_3dpdr', '<-0.56')
         co.el('aG', -0.63, 0.41, 0.27)
         co.el('tgas(ci)', 1.92, 0.1, 0.1)
         co.el('ngas(ci)', 1.70, 0.15, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD88115', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 19.3, 0.1)
         q.el('Me', -0.037, 0.083)
         s = []
         co = sy(0, 0)
         co.el('H2', 19.3,0.1)
         co.el('H2j0', 18.72, 0.1)
         co.el('H2j1', 19.16, 0.1)
         #co.el('CI', 14.03, 0.05, 0.05)  #
         co.el('CI', 13.90, 0.05, 0.05)  #
         co.el('CIj0', 13.80, 0.05)  #
         co.el('CIj1', 13.10, 0.05)  #
         co.el('CIj2', 12.46, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 145, 5, 5, f='d')
         co.el('pci-tripp', 3.55, 0.2, 0.2)
         co.el('uvci-tripp', 0.51, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.65, 0.14, 0.36)
         co.el('uv_ci_3dpdr', 0.39, 0.4, 0.7)
         co.el('aG', 0.19, 0.61, 0.32)
         co.el('tgas(ci)', 2.55, 0.13, 0.6)
         co.el('ngas(ci)', 1.43, 0.2, 0.45)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)




         q = qso('HD91983', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 0, 0, 0)
         q.el('Me', 0.035, 0.1)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.23, 0.1, 0.1)
         co.el('H2j0', 20.03, 0.1)
         co.el('H2j1', 19.78, 0.1)
         #co.el('CI', 14.54, 0.01, 0.01)  #
         co.el('CI', 14.79, 0.01, 0.01)  #
         co.el('CIj0', 14.65, 0.05)  #
         co.el('CIj1', 14.10, 0.05)  #
         co.el('CIj2', 13.55, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 61, 5, 5, f='d')
         co.el('pci-tripp', 3.53, 0.2, 0.2)
         co.el('uvci-tripp', 0.51, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.06, 0.09, 0.1)
         co.el('uv_ci_3dpdr', -0.54, 0.17, 0.46)
         co.el('aG', -0.84, 0.32, 0.24)
         co.el('tgas(ci)', 1.90, 0.13, 0.1)
         co.el('ngas(ci)', 1.86, 0.12, 0.09)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD94454', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.7, 0.1)
         q.el('Me', -0.0, 0.08)  #mean Ritchey23
         s = []
         co = sy(0, 0)
         co.el('H2', 20.76, 0, 0)
         co.el('H2j0', 20.48, 0.1)
         co.el('H2j1', 20.43, 0.1)
         #co.el('CI', 14.29, 0.1, 0.1)  #
         co.el('CI', 14.10, 0.01, 0.01)  #
         co.el('CIj0', 13.94, 0.1)  #
         co.el('CIj1', 13.47, 0.1)  #
         co.el('CIj2', 12.98, 0.1)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 74, 5, 5, f='d')
         co.el('pci-tripp', 3.60, 0.2, 0.2)
         co.el('uvci-tripp', 0.68, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.15, 0.1, 0.1)
         co.el('uv_ci_3dpdr', 0.03, 0.32, 0.32)
         co.el('aG', -0.16, 0.60, 0.54)
         co.el('tgas(ci)', 2.0, 0.17, 0.13)
         co.el('ngas(ci)', 1.94, 0.16, 0.15)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD108002', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.34, 0.1)
         q.el('Me', -0.0, 0.1) #mean Ritchey23
         s = []
         co = sy(0, 0)
         co.el('H2', 20.34, 0.1)
         co.el('H2j0', 20.04, 0.1)
         co.el('H2j1', 20.03, 0.1)
         co.el('CI', 14.58, 0.1, 0.1)  #
         co.el('CIj0', 14.48,  0.05)  #
         co.el('CIj1', 13.79, 0.05)  #
         co.el('CIj2', 13.26, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 77, 5, 5, f='d')
         co.el('pci-tripp', 3.41, 0.2, 0.2)
         co.el('uvci-tripp', 0.34, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.85, 0.11, 0.14)
         co.el('uv_ci_3dpdr', -0.1, 1.1, 0.9)
         co.el('aG', -0.55, 0.32, 0.32)
         co.el('tgas(ci)', 2.3, 0.12, 0.45)
         co.el('ngas(ci)', 1.64, 0.15, 0.21)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD108639', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.04, 0.1)
         q.el('Me', -0.127, 0.08)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.04, 0, 0)
         co.el('H2j0', 19.67, 0.1)
         co.el('H2j1', 19.79, 0.1)
         co.el('CI', 14.53, 0.01, 0.01)  #
         co.el('CIj0', 14.41, 0.05)  #
         co.el('CIj1', 13.79, 0.05)  #
         co.el('CIj2', 13.22, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 88, 5, 5, f='d')
         co.el('pci-tripp', 3.49, 0.2, 0.2)
         co.el('uvci-tripp', 0.55, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.88, 0.1, 0.12)
         co.el('uv_ci_3dpdr', -0.17, 0.36, 0.48)
         co.el('aG', -0.38, 0.38, 0.38)
         co.el('tgas(ci)', 2.1, 0.2, 0.22)
         co.el('ngas(ci)', 1.86, 0.17, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD114886', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.34, 0.1)
         q.el('Me', -0.166, 0.08)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.34, 0.1)
         co.el('H2j0', 19.93, 0.1)
         co.el('H2j1', 20.12, 0.1)
         co.el('CI', 14.96, 0.01, 0.01)  #
         co.el('CIj0', 14.83, 0.05)  #
         co.el('CIj1', 14.27,0.05)  #
         co.el('CIj2', 13.74, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 96, 5, 5, f='d')
         co.el('pci-tripp', 3.62, 0.2, 0.2)
         co.el('uvci-tripp', 0.37, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.94, 0.12, 0.14)
         co.el('uv_ci_3dpdr', 0.66, 0.3, 0.83)
         co.el('aG', -0.06, 0.42, 0.50)
         co.el('tgas(ci)', 2.3, 0.12, 0.38)
         co.el('ngas(ci)', 1.78, 0.14, 0.21)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD115455', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.58, 0.1)
         q.el('Me', -0.0, 0.08) #mean Rithey23
         s = []
         co = sy(0, 0)
         co.el('H2', 20.58, 0.1)
         co.el('H2j0', 20.25, 0.1)
         co.el('H2j1', 20.3, 0.1)
         co.el('CI', 14.80, 0.01, 0.01)  #
         co.el('CIj0', 14.69, 0.1)  #
         co.el('CIj1', 14.05, 0.1)  #
         co.el('CIj2', 13.48, 0.1)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 81, 5, 5, f='d')
         co.el('pci-tripp', 3.52, 0.2, 0.2)
         co.el('uvci-tripp', 0.58, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.94, 0.11, 0.13)
         co.el('uv_ci_3dpdr', -0.19, 0.38, 0.36)
         co.el('aG', -0.32, 0.65, 0.65)
         co.el('tgas(ci)', 1.98, 0.23, 0.12)
         co.el('ngas(ci)', 1.7, 0.21, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD122879', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.36, 0.1)
         q.el('Me', -0.043, 0.08)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.36, 0.1)
         co.el('H2j0', 19.98, 0.1)
         co.el('H2j1', 20.11, 0.1)
         co.el('CI', 14.65, 0.01, 0.01)  #
         co.el('CIj0', 14.53, 0.05)  #
         co.el('CIj1', 13.92, 0.05)  #
         co.el('CIj2', 13.30, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 90, 5, 5, f='d')
         co.el('pci-tripp', 3.59, 0.2, 0.2)
         co.el('uvci-tripp', 0.49, 0.2, 0.2)
         co.el('n_ci_3dpdr',  1.91, 0.1, 0.12)
         co.el('uv_ci_3dpdr', -0.29, 0.48, 0.24)
         co.el('tgas(ci)', 2.1, 0.1, 0.2)
         co.el('ngas(ci)', 1.7, 0.15, 0.12)
         #co.el('aG', 0, 0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD202347', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 19.9, 0.1)
         q.el('Me', 0.244, 0.11)
         s = []
         co = sy(0, 0)
         co.el('H2', 19.9, 0.1)
         co.el('H2j0', 19.41, 0.1)
         co.el('H2j1', 19.72, 0.1)
         #co.el('CI', 14.61, 0.01, 0.01)  #
         co.el('CI', 14.86, 0.01, 0.01)  #
         co.el('CIj0', 14.71, 0.05)  #
         co.el('CIj1', 14.21, 0.05)  #
         co.el('CIj2', 13.69, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 116, 5, 5, f='d')
         co.el('pci-tripp', 3.76, 0.2, 0.2)
         co.el('uvci-tripp', 0.20, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.97, 0.1, 0.12)
         co.el('uv_ci_3dpdr', 0.25, 0.2, 0.47)
         co.el('aG', -0.08, 0.41, 0.40)
         co.el('tgas(ci)', 2.24, 0.1, 0.25)
         co.el('ngas(ci)', 1.78, 0.14, 0.16)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD206773', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.5, 0.1)
         q.el('Me', 0.133, 0.08)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.5, 0.1)
         co.el('H2j0', 20.10, 0.1)
         co.el('H2j1', 20.27, 0.1)
         #co.el('CI', 14.70, 0.01, 0.01)  #
         co.el('CI', 14.94, 0.01, 0.01)  #
         co.el('CIj0', 14.83, 0.05)  #
         co.el('CIj1', 14.18, 0.05)  #
         co.el('CIj2', 13.63, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 94, 5, 5, f='d')
         co.el('pci-tripp', 3.55, 0.2, 0.2)
         co.el('uvci-tripp', 0.19, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.91, 0.09, 0.12)
         co.el('uv_ci_3dpdr', -0.1, 0.42, 0.40)
         co.el('aG', -0.14, 0.43, 0.45)
         co.el('tgas(ci)', 2.06, 0.28, 0.15)
         co.el('ngas(ci)', 1.66, 0.17, 0.08)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD208440', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.34, 0.1)
         q.el('Me', 0.025, 0.08)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.34, 0.1)
         co.el('H2j0', 20.05, 0.1)
         co.el('H2j1', 20.02, 0.1)
         #co.el('CI', 14.84, 0.01, 0.01)  #
         co.el('CI', 15.1, 0.01, 0.01)  #
         co.el('CIj0', 14.95, 0.05)  #
         co.el('CIj1', 14.44, 0.05)  #
         co.el('CIj2', 13.92, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 75, 5, 5, f='d')
         co.el('pci-tripp', 3.66, 0.2, 0.2)
         co.el('uvci-tripp', 0.35, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.12, 0.11, 0.09)
         co.el('uv_ci_3dpdr', -0.29, 0.32, 0.30)
         co.el('aG', -0.46, 0.33, 0.41)
         co.el('tgas(ci)', 1.94, 0.2, 0.05)
         co.el('ngas(ci)', 1.89, 0.15, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD209339', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.25, 0.1)
         q.el('Me', -0.0, 0.07)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.25, 0.1)
         co.el('H2j0', 19.87, 0.1)
         co.el('H2j1', 20.00, 0.1)
         #co.el('CI', 14.76, 0.01, 0.01)  #
         co.el('CI', 15.03, 0.01, 0.01)  #
         co.el('CIj0', 14.89, 0.05)  #
         co.el('CIj1', 14.37, 0.05)  #
         co.el('CIj2', 13.80, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 90, 5, 5, f='d')
         co.el('pci-tripp', 3.69, 0.2, 0.2)
         co.el('uvci-tripp', 0.34, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.06, 0.13, 0.09)
         co.el('uv_ci_3dpdr', -0.07, 0.26, 0.24)
         co.el('aG', -0.2, 0.36, 0.43)
         co.el('tgas(ci)', 2.10, 0.12, 0.19)
         co.el('ngas(ci)', 1.89, 0.11, 0.21)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD210809', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20, 0.1)
         q.el('Me', -0.099, 0.123)
         s = []
         co = sy(0, 0)
         co.el('H2', 20, 0.1)
         co.el('H2j0', 19.64, 0.1)
         co.el('H2j1', 19.74, 0.1)
         #co.el('CI', 14.70, 0.01, 0.01)  #
         co.el('CI', 14.81, 0.01, 0.01)  #
         co.el('CIj0', 14.66, 0.05)  #
         co.el('CIj1', 14.14, 0.05)  #
         co.el('CIj2', 13.64, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 87, 5, 5, f='d')
         co.el('pci-tripp', 3.66, 0.2, 0.2)
         co.el('uvci-tripp', 0.29, 0.2, 0.2)
         co.el('n_ci_3dpdr', 2.03, 0.11, 0.12)
         co.el('uv_ci_3dpdr', -0.31, 0.5, 0.24)
         co.el('aG', -0.34, 0.36, 0.38)
         co.el('tgas(ci)', 2.01, 0.36, 0.09)
         co.el('ngas(ci)', 1.83, 0.19, 0.17)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HD219188', 0., 0.)
         q.telescope = 'HST,FUSE'
         q.ref.append('Jenkins2011')
         q.el('H2', 20.23, 0.1)
         q.el('Me', 0.157, 0.09)
         s = []
         co = sy(0, 0)
         co.el('H2', 20.23, 0.1)
         co.el('H2j0', 19.79, 0.1)
         co.el('H2j1', 20.03, 0.1)
         #co.el('CI', 13.92, 0.01, 0.01)  #
         co.el('CI', 14.56, 0.01, 0.01)
         co.el('CIj0', 14.50, 0.05)  #
         co.el('CIj1', 13.61, 0.05)  #
         co.el('CIj2', 13.01, 0.05)  #
         ##################################### results with 3D-PDR:
         co.el('T01', 103, 5, 5, f='d')
         co.el('pci-tripp', 2.97, 0.2, 0.2)
         co.el('uvci-tripp', 0.01, 0.2, 0.2)
         co.el('n_ci_3dpdr', 1.61, 0.46, 0.43)
         co.el('uv_ci_3dpdr', -0.31, 0.11, 0.17)
         co.el('aG', -0.17, 0.62, 0.62)
         co.el('tgas(ci)', 2.04, 0.36, 0.11)
         co.el('ngas(ci)', 1.31, 0.28, 0.11)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)
     #Low NH2 column density systems in MW Halo. Data by Gillmon 2006, select systems at NH2>18
     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     if 1:
         q = qso('3C249', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '101.60+4.67']
         q.ref.append('Gillmon2006')
         q.el('HI', 20.25, 0.2, 0.4)
         q.el('H2', 18.98, 0.14, 0.16)
         q.el('T01', 144, 184, 47, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 144, 143, 57, f='d')
         co.el('T02', 107, 73, 36, f='d')  # by bootstrap calc
         co.el('H2',  18.98, 0.14, 0.16)
         co.el('H2j0', 18.40, 0.22, 0.28)
         co.el('H2j1', 18.84, 0.14, 0.17)
         co.el('H2j2', 17.03, 0.80, 0.98)
         co.el('H2j3', 16.41, 1.19, 0.55)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('ESO141-G55', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '']
         q.ref.append('Gillmon2006')
         q.el('HI', 20.70, 0.1, 0.7)
         q.el('H2', 19.32, 0.07, 0.07)
         q.el('T01', 98, 22, 15, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 98, 21, 15, f='d')
         co.el('T02', 98, 18, 30, f='d')  # by bootstrap calc
         co.el('H2',  19.32, 0.07, 0.07)
         co.el('H2j0', 18.90, 0.11, 0.11)
         co.el('H2j1', 19.10, 0.08, 0.08)
         co.el('H2j2', 17.34, 0.44, 0.79)
         co.el('H2j3', 16.10, 1.27, 0.52)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('HS0624+6907', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '101.60+4.67']
         q.ref.append('Gillmon2006')
         q.el('H2', 19.82, 0.10, 0.10)
         q.el('HI', 20.82, 0.10, 1.06)
         q.el('T01', 100, 34, 18, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 100, 34, 18, f='d')
         co.el('H2',   19.82, 0.10, 0.10)
         co.el('H2j0', 19.39, 0.13, 0.14)
         co.el('H2j1', 19.60, 0.12, 0.12)
         co.el('H2j2', 17.95, 0.31, 1.56)
         co.el('H2j3', 17.13, 0.57, 1.14)
         co.el('H2j4', 15.49, 0.33, 0.37)
         co.el('H2j5', 14.63, 0.05, 0.09)
         co.el('T02', 104, 16, 50, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('Mrk9', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '']
         q.ref.append('Gillmon2006')
         q.el('H2', 19.36, 0.09, 0.08)
         q.el('HI', 20.64, 0.04, 1.14)
         q.el('T01', 115, 48, 25, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 115, 48, 25, f='d')
         co.el('H2', 19.36, 0.09, 0.08)
         co.el('H2j0', 18.87, 0.15, 0.14)
         co.el('H2j1', 19.18, 0.11, 0.11)
         co.el('H2j2', 17.13, 0.35, 0.46)
         co.el('T02', 90, 14, 16, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('Mrk116', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '']
         q.ref.append('Gillmon2006')
         q.el('H2', 19.08, 0.13, 0.13)
         q.el('HI', 20.41, 0.07, 0.42)
         q.el('T01', 71, 16, 12, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 71, 12, 10, f='d')
         co.el('H2', 19.08, 0.13, 0.13)
         co.el('H2j0', 18.82, 0.13, 0.13)
         co.el('H2j1', 18.73, 0.09, 0.11)
         co.el('H2j2', 17.12, 0.47, 0.85)
         co.el('T02', 92, 16, 26, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('Mrk335', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '']
         q.ref.append('Gillmon2006')
         q.el('H2', 18.83, 0.08, 0.08)
         q.el('HI', 20.43, 0.14, 1.30)
         q.el('T01', 92, 27, 14, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 92, 17, 14, f='d')
         co.el('H2', 18.83, 0.08, 0.08)
         co.el('H2j0', 18.44, 0.12, 0.12)
         co.el('H2j1', 18.59, 0.06, 0.07)
         co.el('H2j2', 16.78, 0.50, 0.44)
         co.el('T02', 94, 26, 14, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('Mrk1095', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '']
         q.ref.append('Gillmon2006')
         q.el('H2', 18.76, 0.21, 0.31)
         q.el('HI', 20.95, 0.02, 1.04)
         q.el('T01', 121, 107, 39, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 121, 130, 57, f='d')
         co.el('H2', 18.76, 0.21, 0.31)
         co.el('H2j0', 18.24, 0.25, 0.38)
         co.el('H2j1', 18.58, 0.20, 0.32)
         co.el('H2j2', 17.10, 0.77, 0.72)
         co.el('T02', 120, 114, 33, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('MS0700+6338', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '101.60+4.67']
         q.ref.append('Gillmon2006')
         q.el('H2', 18.75, 0.27, 0.68)
         q.el('HI', 20.43, 0.18, 0.50)
         q.el('T01', 82, 61, 21, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 82, 61, 21, f='d')
         #co.el('T01', '<380', f='d')
         co.el('H2', 18.75, 0.27, 0.68)
         co.el('H2j0', 18.41, 0.29, 0.30)
         co.el('H2j1', 18.46, 0.25, 0.35)
         co.el('H2j2', 17.27, 0.86, 1.57)
         co.el('T02', 120, 118, 63, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('NGC1068', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '101.60+4.67']
         q.ref.append('Gillmon2006')
         q.el('H2', 18.13, 0.13, 0.17)
         q.el('HI', 19.61, 0.81, 0.03)
         q.el('T01', 76, 23, 14, f='d')
         q.el('Me', -0.27, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 76, 10, 10, f='d')
         co.el('H2', 18.13, 0.13, 0.17)
         co.el('H2j0', 17.84, 0.08, 0.09)
         co.el('H2j1', 17.82, 0.08, 0.11)
         co.el('H2j2', 15.84, 0.66, 0.10)
         co.el('T02', 82, 41, 1, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('NGC7469', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '101.60+4.67']
         q.ref.append('Gillmon2006')
         q.el('H2', 19.67, 0.10, 0.10)
         q.el('HI', 20.59, 0.05, 1.46)
         q.el('T01', 71, 16, 11, f='d')
         q.el('Me', -0.27, 0.1, 0.1)  # Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 71, 8, 8, f='d')
         co.el('H2', 19.67, 0.10, 0.10)
         co.el('H2j0', 19.41, 0.10, 0.09)
         co.el('H2j1', 19.32, 0.08, 0.08)
         co.el('H2j2', 17.77, 0.24, 0.35)
         co.el('T02', 94, 9, 14, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('PG0804+761', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '101.60+4.67']
         q.ref.append('Gillmon2006')
         q.el('H2', 18.66, 0.14, 0.19)
         q.el('HI', 20.54, 0.04, 1.0)
         q.el('T01', 144, 162, 44, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 144, 100, 39, f='d')
         co.el('H2', 18.66, 0.14, 0.19)
         co.el('H2j0', 18.08, 0.16, 0.20)
         co.el('H2j1', 18.52, 0.08, 0.11)
         co.el('H2j2', 16.63, 0.54, 0.37)
         co.el('T02', 103, 50, 14, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('PG0844+349', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '101.60+4.67']
         q.ref.append('Gillmon2006')
         q.el('H2', 18.22, 0.18, 0.28)
         q.el('HI', 20.34, 0.03, 0.68)
         q.el('T01', 147, 312, 50, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 147, 138, 57, f='d')
         co.el('H2', 18.22, 0.18, 0.28)
         co.el('H2j0', 17.64, 0.21, 0.29)
         co.el('H2j1', 18.09, 0.11, 0.16)
         co.el('H2j2', 16.04, 0.78, 0.40)
         co.el('T02', 96, 78, 10, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('PG1211+143', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '']
         q.ref.append('Gillmon2006')
         q.el('H2', 18.38, 0.15, 0.14)
         q.el('HI', 20.25, 0.17, 1.03)
         q.el('T01', 142, 90, 38, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 141, 87, 29, f='d')
         co.el('H2', 18.38, 0.15, 0.14)
         co.el('H2j0', 17.80, 0.13, 0.16)
         co.el('H2j1', 18.23, 0.08, 0.08)
         co.el('H2j2', 16.72, 0.57, 0.59)
         co.el('T02', 124, 70, 32, f='d')  # by bootstrap calc
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         q = qso('VIIZw118', 0., 0.)
         q.telescope = 'FUSE'
         q.year = 2000
         q.coord = ['J0000', '']
         q.ref.append('Gillmon2006')
         q.el('H2', 18.84, 0.10, 0.12)
         q.el('HI', 20.56, 0.03, 0.80)
         q.el('T01', 108, 48, 24, f='d')
         q.el('Me', -0.27, 0.1, 0.1) #Rithcey2023
         s = []
         # v = 10      # velocity in km/s
         co = sy(0, 0)
         co.el('T01', 108, 48, 24, f='d')
         co.el('H2', 18.84, 0.10, 0.12)
         co.el('H2j0', 18.38, 0.11, 0.13)
         co.el('H2j1', 18.65, 0.07, 0.08)
         co.el('H2j2', 16.22, 0.59, 0.34)
         co.el('T02', 77, 26, 7, f='d')  # by bootstrap calc
         s.append(co)
         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         q.comp = s
         q.full = 'u'
         QSO.append(q)

    # dark clouds - CO emission form
     if 0:
         q = qso('L1450E', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.1, 0.2, 0.2, f='d')
         q.el('CO18', 0.6, 0,0, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('B5', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 22, 0.2, 0.2, f='d')
         q.el('CO18', 3.6, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1489', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 9.5, 0.2, 0.2, f='d')
         q.el('CO18', 1.4, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1498', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.7, 0.2, 0.2, f='d')
         q.el('CO18', 0.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1495C', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 16, 0.2, 0.2, f='d')
         q.el('CO18', 2.5, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1495', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 22, 0.2, 0.2, f='d')
         q.el('CO18', 3.6, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         #QSO.append(q)

         q = qso('L1495B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 12, 0.2, 0.2, f='d')
         q.el('CO18', 1.9, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1506', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.1, 0.2, 0.2, f='d')
         q.el('CO18', 0.6, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)


         q = qso('L1521C', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 10.1, 0.2, 0.2, f='d')
         q.el('CO18', 1.5, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1521B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 18, 0.2, 0.2, f='d')
         q.el('CO18', 3, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1521A', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 7.3, 0.2, 0.2, f='d')
         q.el('CO18', 1, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1521D', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.7, 0.2, 0.2, f='d')
         q.el('CO18', 0.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)



         q = qso('L1521E', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 11, 0.2, 0.2, f='d')
         q.el('CO18', 1.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1551A', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.1, 0.2, 0.2, f='d')
         q.el('CO18', 0.6, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1551C', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 10, 0.2, 0.2, f='d')
         q.el('CO18', 1.5, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1551T', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 12, 0.2, 0.2, f='d')
         q.el('CO18', 1.8, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1551B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 12, 0.2, 0.2, f='d')
         q.el('CO18', 1.9, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('TMC-2A', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 7.9, 0.2, 0.2, f='d')
         q.el('CO18', 1.1, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('TMC-2T', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 12, 0.2, 0.2, f='d')
         q.el('CO18', 1.8, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1536B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 7.3, 0.2, 0.2, f='d')
         q.el('CO18', 1, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('TMC-1B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 6.2, 0.2, 0.2, f='d')
         q.el('CO18', 0.8, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('TMC-1A', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 9, 0.2, 0.2, f='d')
         q.el('CO18', 1.3, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('TMC-1C', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 18, 0.2, 0.2, f='d')
         q.el('CO18', 3, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('TMC-1T', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 31, 0.2, 0.2, f='d')
         q.el('CO18', 5.2, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1517C', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 6.8, 0.2, 0.2, f='d')
         q.el('CO18', 0.9, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1517A', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 8.4, 0.2, 0.2, f='d')
         q.el('CO18', 1.2, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1517B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 6.2, 0.2, 0.2, f='d')
         q.el('CO18', 0.8, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1517D', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.7, 0.2, 0.2, f='d')
         q.el('CO18', 0.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1512', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.7, 0.2, 0.2, f='d')
         q.el('CO18', 0.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1544', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 13, 0.2, 0.2, f='d')
         q.el('CO18', 2.1, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1523', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 6.9, 0.2, 0.2, f='d')
         q.el('CO18', 1.1, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L134C', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 11, 0.2, 0.2, f='d')
         q.el('CO18', 1.6, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L183', 0., 0.)
         q.el('H2', 9.5, 0.2, 0.2, f='d')
         q.el('CO18', 1.4, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1719B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 9.5, 0.2, 0.2, f='d')
         q.el('CO18', 1.4, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1681B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 40, 0.2, 0.2, f='d')
         q.el('CO18', 7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1696A', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 11, 0.2, 0.2, f='d')
         q.el('CO18', 1.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1696B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 13, 0.2, 0.2, f='d')
         q.el('CO18', 2, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1709B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 17, 0.2, 0.2, f='d')
         q.el('CO18', 2.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1696V', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.7, 0.2, 0.2, f='d')
         q.el('CO18', 2, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1689B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 11, 0.2, 0.2, f='d')
         q.el('CO18', 1.6, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L43E', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 11, 0.2, 0.2, f='d')
         q.el('CO18', 1.6, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L255', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 7.9, 0.2, 0.2, f='d')
         q.el('CO18', 1.1, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1152', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.7, 0.2, 0.2, f='d')
         q.el('CO18', 0.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1155D', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.7, 0.2, 0.2, f='d')
         q.el('CO18', 0.7, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1174', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 8.4, 0.2, 0.2, f='d')
         q.el('CO18', 1.2, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1172D', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 5.1, 0.2, 0.2, f='d')
         q.el('CO18', 0.6, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L1172B', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2', 9.5, 0.2, 0.2, f='d')
         q.el('CO18', 1.4, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('L16262A', 0., 0.)
         q.ref.append('Myers1983')
         q.el('H2',9, 0.2, 0.2, f='d')
         q.el('CO18', 1.3, 184, 47, f='d')
         s = []
         co = sy(0, 0)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)
     if 1:
         q = qso('Taurus03', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 4.78, 0.5, 0.5, f='d') #1e14
         q.el('Av', 2.98, 0, 0, f='d')
         s = []
         co = sy(0, 0)
         #co.el('H2', 18.84, 0.10, 0.12)
         s.append(co)
         q.comp = s
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus04', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.93, 0.5, 0.5, f='d') #1e14
         q.el('Av', 2.35, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus06', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 3.05, 0.5, 0.5, f='d') #1e14
         q.el('Av', 2.76, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus07', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 1.90, 0.5, 0.5, f='d') #1e14
         q.el('Av', 1.70, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus08', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.77, 0.5, 0.5, f='d') #1e14
         q.el('Av', 1.39, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)


         q = qso('Taurus09', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 4.42, 0.5, 0.5, f='d') #1e14
         q.el('Av', 3.54, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus10', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 3.51, 0.5, 0.5, f='d') #1e14
         q.el('Av', 1.93, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus13', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.39, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 1.96, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus14', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 3.85, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.81, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus15', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 8.54, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 3.38, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus16', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.80, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 1.58, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus18', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 3.98, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.21, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus20', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 7.30, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.42, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus21', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 4.93, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.60, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus22', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.76, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.41, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus23', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 1.84, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 1.58, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus26', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.87, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.0, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus27', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 3.95, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.71, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus28', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 6.24, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.73, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus29', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 6.58, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 3.30, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus30', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 4.55, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 3.12, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus31', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 1.96, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 1.73, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus33', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.58, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.67, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus34', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 4.74, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 4.78, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus35', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.63, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.43, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus36', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 5.97, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 3.20, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus37', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 6.19, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.51, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus38', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 8.05, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 3.60, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus39', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 3.67, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 3.45, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Tauru42', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.99, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.55, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus43', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 5.39, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 3.44, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus44', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 2.70, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 2.73, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

         q = qso('Taurus46', 0., 0.)
         q.ref.append('Ungerer1985')
         q.el('CO18', 4.38, 0.5, 0.5, f='d')  # 1e14
         q.el('Av', 1.78, 0, 0, f='d')
         q.comp = [sy(0, 0)]
         q.full = 'u'
         QSO.append(q)

     return QSO


def load_Manga():
    global sy
    QSO = sample()
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0950+4309
    q = qso('J0950+4309', 0.3622, 0.0170)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-166736'
    q.zmanga = 0.01708
    q.vcorr = 12.95
    q.el('gal_rad_vel', -21.5, 0, 0, f='d')
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 8.98, f='l')
    q.el('sigMstar', 7.43, f='l')
    q.el('SFR', 0.08, f='d')
    # q.el('Dn4000', 1.295, f='d')
    q.el('Dn4000', 1.31, f='d')
    q.el('bpar', 23, 0, 0, f='d')
    q.el('b_rel', 6.9, f='d')
    q.el('drelow', 6.9, f='d')
    q.el('dreup', 9.2, f='d')
    q.el('Re', 3.41, f='d')
    q.el('Azi', 50, 8, 8, f='d')
    q.el('Azi_mean', 70, 34, 34, f='d')
    q.el('Azi_r_mean', 60, 23, 23, f='d')
    q.el('Azi_a_mean', 87, 17, 17, f='d')
    q.el('Azi_ra_mean', 80, 15, 15, f='d')
    q.el('Azi_h_mean', 60, 22, 22, f='d')
    q.el('Azi_NFW_6Re', 90 - 28, 25, 23, f='d')
    q.el('abszcen', -1.18, f='d')
    q.el('abszlow', -6.0, f='d')
    q.el('abszup', 3.17, f='d')
    q.el('r_dist', 9.0, 1.9, 1.9, f='d')
    q.el('h_dist', 0.2, 1.7, 1.7, f='d')
    # q.el('Azi_r_max', 50, f='d')
    # q.el('Azi_ra_max', 50, f='d')
    q.el('sini', 0.57, f='d')
    q.el('sSFR', -10.07, f='l')
    q.el('sigSFR', -2.63, f='l')
    # q.el('HI', 17.9, 0.3, 0.3)
    if 0:
        q.el('HI', 15.9, 0.8, 0.6)
        q.el('SiII', 13.0, 0.10, 0.10)
        q.el('SiIII', 13.56, 0.10, 0.10)
        # q.el('SiIII/SiII', 0.8,0.44,0.26)
        # q.el('Sitot', 13.66,0.19,0.09)
        q.el('SII', 14.10, 0.20, 0.20)
        q.el('CII', 14.05, 0.10, 0.10)
        q.el('Me_cloudy_5par', 0.1, 0.6, 0.3)
        q.el('q_cloudy_5par', -2.3, 0.6, 0.5)
        q.el('HI_5par', 15.5, 0.4, 0.7)
        q.el('Htot_5par', 18.2, 0.4, 0.3)
        q.el('depl_5par', 0.0, 0.2, 0.0, f='d')
    if 0:
        q.el('HI', 17.6, 0.3, 0.7)
        q.el('SiII', 13.0, 0.10, 0.10)
        q.el('SiIII', 13.56, 0.06, 0.06)
        # q.el('SiIII/SiII', 0.8,0.44,0.26)
        # q.el('Sitot', 13.66,0.19,0.09)
        q.el('SII', 14.10, 0.20, 0.20)
        q.el('CII', 14.05, 0.10, 0.10)
        q.el('Me_cloudy_5par', -0.6, 0.7, 0.55)
        # q.el('Me_cloudy_5par', -1.7, 0.5, 0.4)
        q.el('q_cloudy_5par', -2.5, 0.2, 0.2)
        q.el('HI_5par', 18, 0.2, 0.5)
        q.el('Htot_5par', 20.2, 0.4, 0.3)
        q.el('depl_5par', 0.0, 0.2, 0.0, f='d')
    if 1:
        q.el('HI', 17.6, 0.3, 0.7)
        q.el('SiII', 13.0, 0.10, 0.10)
        q.el('SiIII', 13.56, 0.06, 0.06)
        # q.el('SiIII/SiII', 0.8,0.44,0.26)
        # q.el('Sitot', 13.66,0.19,0.09)
        q.el('SII', 14.10, 0.20, 0.20)
        q.el('CII', 14.05, 0.10, 0.10)
        q.el('Me_cloudy_5par', -0.6, 0.2, 0.7)
        # q.el('Me_cloudy_5par', -1.7, 0.5, 0.4)
        q.el('q_cloudy_5par', -3.2, 0.2, 0.2)
        q.el('HI_5par', 16.8, 0.8, 0.3)
        q.el('Htot_5par', 19, 0.6, 0.2)
        q.el('depl_5par', 0.0, 0.2, 0.0, f='d')
    q.el('NI', 13.80, 0.40, 0.60)
    q.el('NII', '<16.20')
    q.el('NV', 14.00, 0.20, 0.40)
    q.el('OI', 13.60, 0.60, 0.60)
    q.el('FeII', 13.60, 0.30, 0.40)
    q.el('zabs', 0.0170, 0.0001, 0.0001, f='d')
    q.el('Me', -0.55, 0.20, 0.17)
    q.el('rSi', 0.80, 0.40, 0.26)
    q.el('Si/H', -5.04, 0.20, 0.20)
    q.el('deltaV', -23, 28, 28, f='d')
    q.el('V_mod_em_qso', 25.4, 3, 3, f='d')
    q.el('ewHI', 1440, 52, 52, f='d')
    q.el('ewSiII', 128, 22, 22, f='d')
    q.el('ewSiIII', 330, 33, 33, f='d')
    q.el('v90HI', 395, 38, 38, f='d')
    q.el('v90SiII', 91, 50, 50, f='d')
    q.el('v90SiIII', 116, 30, 30, f='d')
    q.manga_absmag = [-1.58E+01, -1.60E+01, -1.65E+01, -1.75E+01, -1.79E+01, -1.81E+01, -1.85E+01]
    # ***************** fit with depletion
    # q.el('Me_cloudy_4par', -0.9, 0.15, 0.15)
    # q.el('q_cloudy_4par', -2.8, 0.1, 0.1)
    # q.el('Htot_4par', 20.2, 0.2, 0.1)
    # q.el('depl_4par', 0.0, 0.2, 0.0,f='d')
    # q.el('nH_4par', -3.1, 0.1, 0.1)
    # q.el('fG_4par', -2.9, 0.2, 0.1)
    # *****************
    q.el('Mfuv', -1.58E+01, f='dec')
    q.el('Mhalo', 11.35, 0.3, 0.3)
    q.mangastatus = 'detection'
    q.el('[O/H]_manga', -0.12, 0.03, 0.03)
    q.el('O/H_manga', 8.56, 0.05, 0.05)
    q.el('q_manga', 7.04, 0.05, 0.05)
    q.el('q_re_manga', 6.85, 0.05, 0.05)
    q.el('q_cen_manga', 7.4, 0.05, 0.05)
    q.el('O/H_cen_manga', -0.1, 0.05, 0.05)
    q.comp = []
    co = sy(0.0171344, 40)
    if 0:
        co.el('HI', '<17.9', b=(25, 6, 6))
        co.el('CII', 14.01, 0.13, 0.10)
        co.el('SiII', 13.03, 0.10, 0.10)
        co.el('SiIII', 12.4, 0.4, 1.3)
        co.el('Me_cloudy_5par', 1.5, 0.3, 0.7)
        co.el('q_cloudy_5par', -2.2, 0.6, 0.7)
        co.el('HI_5par', 14.7, 0.4, 0.7)
        co.el('Htot_5par', 17, 0.5, 0.6)
        co.el('depl_5par', 0.0, 0.2, 0.0, f='d')

    if 1:
        co.el('HI', 17.6, 0.25, 0.84, b=(28, 6, 6))
        # co.el('HI', 17.6, 0.25, 0.5, b=(28, 6, 6))
        co.el('CII', 14.01, 0.13, 0.10)
        co.el('SiII', 13.03, 0.10, 0.14)
        co.el('SiIII', 13.0, 0.2, 0.3)
        co.el('Me_cloudy_5par', -1.1, 0.5, 0.3)
        co.el('q_cloudy_5par', -3.8, 0.3, 0.3)
        co.el('HI_5par', 17.6, 0.3, 0.7)
        co.el('Htot_5par', 18.9, 0.3, 0.5)
        co.el('depl_5par', 0.0, 0.2, 0.0, f='d')

    co.el('SII', '<14.7')
    co.el('Me_cloudy', -1.2, 0.3, 0.3)
    co.el('q_cloudy', -3.8, 0.5, 0.4)
    co.el('n/I_cloudy', 0.4, 0.5, 0.4)
    co.el('Htot', 19.1, 0.4, 0.3)
    co.el('fHI_cloudy', -1.2, 0.3, 0.4)
    co.el('Me_cloudy_4par', -1.2, 0.3, 0.3)
    co.el('q_cloudy_4par', -3.8, 0.5, 0.4)
    co.el('Htot_4par', 19.1, 0.4, 0.2)
    co.el('depl_4par', 0.0, 0.2, 0.0, f='d')
    co.el('FeII', '<14.2')
    # co.el('NII', 14.20,1.00,1.00)
    # co.el('PII', 13.00,0.50,0.50)
    co.el('NI', '<14.3')
    # co.el('OI', 13.60,0.60,0.60)
    co.el('NV', '<13.8')
    co.el('FeIII', '<14')
    q.comp.append(co)
    co = sy(0.01696162, 20)
    if 0:
        co.el('HI', 15.44, 0.50, 0.30, b=(70, 9, 9))
        co.el('SiIII', 13.57, 0.05, 0.10)
        co.el('CII', '<14.0')
        co.el('SiII', '<13.8')
    if 1:
        co.el('HI', 15.1, 0.7, 0.2, b=(70, 10, 10))
        co.el('CII', '<13.6')
        co.el('SiII', '<12.4')
        co.el('SiIII', 13.45, 0.1, 0.1)
    co.el('SII', '<14.6')
    co.el('NI', '<14.4')
    co.el('NII', '<16.3')
    # co.el('OI', '<14.0')
    co.el('NV', '<14.1')
    co.el('FeII', '<14.2')
    co.el('FeIII', 14., 0.3, 1.2)
    co.el('Me_cloudy', 0.0, 0.2, 0.5)
    co.el('q_cloudy', -1.1, 0.4, 0.4)
    co.el('n/I_cloudy', -1.7, 0.1, 0.4)
    co.el('Htot', 19.6, 0.5, 0.2)
    co.el('fHI_cloudy', -4.2, 0.2, 0.5)

    co.el('Me_cloudy_4par', 0.0, 0.2, 0.5)
    co.el('q_cloudy_4par', -1.1, 0.3, 0.3)
    co.el('Htot_4par', 19.6, 0.6, 0.8)
    co.el('depl_4par', 0.0, 0.5, 0.0, f='d')

    if 0:
        co.el('Me_cloudy_5par', -0.56, 0.45, 0.77)
        co.el('q_cloudy_5par', -1.47, 0.5, 0.4)
        co.el('HI_5par', 15.35, 0.4, 0.3)
        co.el('Htot_5par', 19.6, 0.7, 0.5)
        co.el('depl_5par', 0.0, 0.3, 0.0, f='d')
    if 1:
        co.el('Me_cloudy_5par', -0.47, 0.4, 0.50)
        co.el('q_cloudy_5par', -1.2, 0.4, 0.3)
        co.el('HI_5par', 14.89, 0.3, 0.3)
        co.el('Htot_5par', 19.42, 0.4, 0.5)
        co.el('depl_5par', 0.0, 0.24, 0., f='d')

    q.comp.append(co)
    co = sy(0.016305, 40)
    co.el('HI', 13.40, 0.20, 0.20)
    co.el('SiIII', '<12.7')
    co.el('CII', '<14.1')
    co.el('SiII', '<12.7')
    co.el('SII', '<14.6')
    # co.el('Me_cloudy', 0.4, 1.1, 0.7)
    # co.el('q_cloudy', -1.3, 0.5, 1.1)
    # co.el('Htot', 16.5, 1.5, 0.5)
    # co.el('Me_cloudy_4par', 0.4, 1.1, 0.7)
    # co.el('q_cloudy_4par', -1.3, 0.5, 1.1)
    # co.el('Htot_4par', 16.5, 1.5, 0.4)
    # co.el('depl_4par', 0.1, 0.5, 0.1, f='d')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0755+3911
    q = qso('J0755+3911', 0.03316, 0.0330)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-71974'
    #q.progID.append('085.A-0569(B)')
    q.zmanga = 0.0332 #0.0336121
    q.el('Mstar', 10.30, f='l')
    q.el('SFR', 1.72,f='d')
    q.el('sSFR', -10.06,f='l')
    #q.el('Dn4000', 1.345, f='d')
    q.el('Dn4000', 1.30, f='d')
    q.el('bpar', 2.5,2.5,2.3, f='d')
    q.el('b_rel', 1,1,0.9,f='d')
    q.el('Re', 4.90, f='d')
    q.el('drelow', 0, f='d')
    q.el('dreup', 1, f='d')
    q.el('Azi', 57,4,4, f='d')
    q.el('Azi_mean', 33, 4, 4, f='d')
    q.el('Azi_r_mean', 33, 2, 4, f='d')
    q.el('Azi_a_mean', 33, 2, 4, f='d')
    q.el('Azi_h_mean', 33, 2, 4, f='d')
    q.el('Azi_ra_mean', 33, 2, 4, f='d')
    q.el('Azi_NFW_6Re', 33, 4, 4, f='d')
    q.el('abszcen', 3.2, f='d')
    q.el('abszlow', -4.5, f='d')
    q.el('abszup', 10.8, f='d')
    #q.el('Azi_NFW_6Re', 26, 226, 6, f='d')
    q.el('sini', 0.59, f='d')
    q.el('HI', 13.58, 0.03, 0.03)
    q.el('SiII', 12.7,0.3,0.5)
    q.el('SiIII', 12.4,0.2,0.5)
    q.el('SiIII/SiII', -0.3,0.5,0.6)
    #q.el('Sitot', 12.65,0.19,0.29)
    q.el('SII', '<15')
    q.el('CII', 13.0,0.2,0.3)
    q.el('NI', 13.1,0.4,0.9)
    q.el('OI', 13.1, 0.4, 0.6)
    q.el('FeII', 13.2, 0.3, 0.9)
    q.el('zabs', 0.0330,0.0003,0.0003,f='d')
    q.el('deltaV', -45, 85, 85, f='d')
    q.el('V_mod_em_qso', 15, 10, 10, f='d')
    q.el('ewHI', 72, 5, 5, f='d')
    #q.el('ewSiII', 100, 0, 0, f='d')
    q.el('ewSiIII', 31, 10,10,f='d')
    q.el('v90HI', 82, 6,6,f='d')
    #q.el('v90SiII', 100, 0, 0, f='d')
    q.el('v90SiIII', 84, 24, 24, f='d')
    q.manga_absmag = [-1.91E+01, -1.93E+01, -1.96E+01, -2.04E+01, -2.08E+01, -2.10E+01, -2.11E+01]
    #Htot', 'Me_cloudy', 'q_cloudy', 'n/I_cloudy
    #q.el('Htot', 15.5,0.3,0.3)
    #q.el('Me_cloudy', 2, 0.3, 0.3)
    #q.el('q_cloudy', -2.1, 0.2, 0.2)
    #q.el('n/I_cloudy', -1.6, 0.2, 0.2)
    #q.el('fHI_cloudy', -1.4, 0.3, 0.3)
    # ***************** fit with depletion
    #q.el('Me_cloudy_4par', 2, 0.3, 0.3)
    #q.el('q_cloudy_4par', -2.1, 0.2, 0.2)
    #q.el('Htot_4par', 15.6, 0.3, 0.2)
    #q.el('depl_4par', 0.1, 0.2, 0.1,f='d')
    #q.el('nH_4par', -1.5, 0.9, 0.9)
    #q.el('fG_4par', 1, 0.0, 0.7)
    # *****************     ISM radiation
    q.el('Me_cloudy_5par', 1.2, 0.2, 0.2)
    q.el('q_cloudy_5par', -1.5, 0.3, 0.3)
    q.el('Htot_5par', 16.4, 0.3, 0.3)
    q.el('HI_5par', 13.57, 0.1, 0.1)
    q.el('depl_5par', 0.0, 0.3, 0.0,f='d')
    q.el('uv_to_nh_5par', 1.9, 0.3, 0.3)
    #*********************** AGN radiation in cloudy model
    q.el('Me_cloudy_5par_agn', 2.0, 0.0, 0.2)
    q.el('q_cloudy_5par_agn', -2.1, 0.3, 0.3)
    q.el('Htot_5par_agn', 15.5, 0.2, 0.2)
    q.el('HI_5par_agn', 13.57, 0.1, 0.1)
    q.el('depl_5par_agn', 0.0, 0.3, 0.0,f='d')
    q.el('uv_to_nh_5par_agn', 0.0, 0.2, 0.5)
    #******************
    q.el('Mfuv', -1.91E+01, f='dec')
    q.el('Mhalo', 13.23,0.3,0.3)
    q.mangastatus = 'detection'
    q.el('gal_rad_vel', 24.2, 0, 0, f='d')
    q.el('[O/H]_manga', 0.32,0.03,0.03)
    q.el('O/H_manga', 9.01, 0.02, 0.02)
    q.el('q_re_manga', 7.23, 0.05, 0.05)
    q.el('q_manga', 7.20, 0.05, 0.05)
    q.el('q_cen_manga', 7.2, 0.05, 0.05)
    q.el('O/H_cen_manga', 0.35, 0.05, 0.05)
    q.comp = []
    co = sy(0.032838, 5)
    co.el('HI', 13.23,0.05,0.05,b=(16,5,1))
    co.el('SII', '<14.71')
    co.el('SiII', '<12.8')
    co.el('SiIII', 12.4,0.4,1.0)
    co.el('SII', '<14.74')
    co.el('NI', '<13.5')
    co.el('CII', 12.7,0.5,1.1)
    co.el('OI', '<13.8')
    co.el('Me_cloudy_5par', 1.2, 0.2, 0.2)
    co.el('q_cloudy_5par', -1.5, 0.3, 0.3)
    q.comp.append(co)
    co = sy(0.03292, 7)
    co.el('SiIII', 12.0,0.3,0.8,b=(17,20,2))
    q.comp.append(co)
    co = sy(0.03315, 7)
    co.el('HI', '<12.2',b=(15,5,0))
    co.el('SiIII', 12.32, 0.30, 0.46)
    q.comp.append(co)
    co = sy(0.033420, 7)
    co.el('HI', 13.27, 0.05, 0.05,b=(24,5,5))
    co.el('SiII', '<12.68')
    co.el('SiIII', '<12.6')
    co.el('SII', '<14.5')
    co.el('CII', '<13.6')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1653+3945
    q = qso('J1653+3945', 0.0349, 0.03493)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-594755'
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 10.78, f='l')
    #q.el('SFR', '<0',f='d')
    q.el('SFR', 0.004, f='d')
    #q.el('Dn4000', 1.159, f='d')
    q.el('Dn4000', 1.23, f='d')
    q.el('bpar', 1.25,1.25,1.22, f= 'd')
    q.el('b_rel', 1,1,0.9, f= 'd')
    q.el('Re', 1.25, f='d')
    q.el('drelow', 0, f='d')
    q.el('dreup', 1, f='d')
    q.el('Azi',68,4,4, f='d')
    q.el('Azi_mean', 20, 2, 2, f='d')
    q.el('Azi_r_mean', 20, 2, 2, f='d')
    q.el('Azi_a_mean', 20, 2, 2, f='d')
    q.el('Azi_ra_mean', 20, 2, 2, f='d')
    q.el('Azi_a_mean', 20, 2, 2, f='d')
    q.el('Azi_NFW_6Re', 90-68, 4, 4, f='d')
    #q.el('Azi_NFW_6Re', 74, 85, 49, f='d')
    q.el('sini', 0.58, f='d')
    q.el('HI', '<12.8')
    q.el('SiII', '<13.77')
    q.el('SiIII', '<12.4')
    q.el('SII', '<14.72')
    q.el('NV', '<13.31')
    #q.el('zabs', 0.03410,0,0,f='d')
    q.el('ewHI', 40, 20, 20, f='d')
    q.el('ewSiII', 55, 14, 14, f='d')
    q.el('ewSiIII', 8, 14, 14, f='d')
    q.manga_absmag = [-1.95E+01, -1.99E+01, -2.04E+01, -2.12E+01, -2.19E+01, -2.22E+01, -2.25E+01]
    q.mangastatus = 'non-detection'
    q.el('gal_rad_vel', -47, 0, 0, f='d')
    q.el('O/H_manga', 8.76,0.04,0.04)
    q.el('Mfuv', -1.95E+01, f='dec')
    q.el('Mhalo', 13.91,0.3,0.3)
    q.comp = []
    co = sy(0.03410, 10)
    co.el('HI', '<13.27')
    co.el('SiII', '<13.77')
    co.el('SiIII', '<12.4')
    co.el('SII', '<14.72')
    co.el('NV', '<13.31')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1338+2620
    q = qso('J1338+2620', 0.0261,0.0255)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '12-192116'
    q.zmanga=0.0261
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 8.80, f='l')
    #q.el('SFR', 0.59,  f='d')
    q.el('SFR', 0.69, f='d')
    q.el('sSFR',  -9.03,  f='l')
    #q.el('Dn4000', 1.170, f='d')
    q.el('Dn4000', 1.22, f='d')
    q.el('bpar', 3.3,3.3,3.0, f='d')
    q.el('b_rel', 1,1,0.9, f='d')
    q.el('Re', 3.31, f='d')
    q.el('drelow', 0, f='d')
    q.el('dreup', 1, f='d')
    q.el('Azi', 54,1,1, f='d')
    q.el('Azi_mean', 90, 55, 55, f='d')
    q.el('Azi_r_mean', 90, 55, 55, f='d')
    q.el('Azi_a_mean', 35, 2, 2, f='d')
    q.el('Azi_ra_mean', 35, 2, 2, f='d')
    q.el('Azi_h_mean', 35, 2, 2, f='d')
    q.el('Azi_NFW_6Re', 90-54, 2, 2, f='d')
    q.el('abszcen', 0, f='d')
    q.el('abszlow', 0, f='d')
    q.el('abszup', 0, f='d')
    #q.el('Azi_NFW_6Re', 62, 83, 25, f='d')
    q.el('sini', 0.58, f='d')
    #q.el('HI', 19.2,0.1,0.1)
    q.el('HI', 20.2, 0.1, 0.1)
    q.el('SiII', 13.9,0.2,0.2)
    q.el('SiIII', 13.9,0.4,0.2)
    q.el('SII', 15.2,0.1,0.2)
    #q.el('NI', 14.7, 0.2, 0.2)
    #q.el('NII', 14.9, 0.3, 0.2)
    q.el('NI', '<14.3')
    q.el('NII', '<16')
    q.el('NV', '<13.8')
    q.el('OI', 15.5, 0.1, 0.1)
    q.el('FeII', '<14.6')
    q.el('zabs', 0.0255,0.0001,0.0001,f='d')
    #q.el('Me', '<3.31')
    #q.el('rSi', '<3.43')
    #q.el('Me', 0.05,0.54,0.28)
    #q.el('rSi', -0.33,0.77,0.55)
    #q.el('Si/H', -4.44,0.54,0.28)
    q.el('deltaV', -185, 28, 28, f='d')
    q.el('V_mod_em_qso', 8.75, 5, 5, f='d')
    q.el('ewHI', 1472, 140, 140, f='d')
    #q.el('ewSiII', 982, 65, 65, f='d')
    q.el('ewSiII', 735, 90, 90, f='d')
    q.el('ewSiIII', 680, 90, 90, f='d')
    q.el('v90HI', 437, 51, 51, f='d')
    #q.el('v90SiII', 320, 35, 35, f='d')
    q.el('v90SiII', 250, 35, 35, f='d')
    q.el('v90SiIII', 267, 85, 85, f='d')
    q.manga_absmag = [ -1.72E+01, -1.72E+01, -1.77E+01, -1.84E+01, -1.86E+01, -1.86E+01, -1.87E+01]
    #q.el('Me_cloudy', -0.9, 0.2, 0.2)
    #q.el('Me_cloudy', 0.1, 0.1, 0.1)
    #q.el('n/I_cloudy', -0.2, 0.2, 0.3)
    #q.el('q_cloudy', -3.3, 0.2, 0.1)
    #q.el('Htot', 19.7, 0.2, 0.1)
    #q.el('fHI_cloudy', -0.5, 0.1, 0.3)
    # ***************** fit with depletion
    #q.el('Me_cloudy_4par', 0.0, 0.2, 0.2)
    #q.el('q_cloudy_4par', -3.2, 0.3, 0.2)
    #q.el('Htot_4par', 19.8, 0.2, 0.1)
    #q.el('depl_4par', 0.14, 0.16, 0.14,f='d')
    #q.el('nH_4par', -2.4, 0.4, 0.3)
    #q.el('fG_4par', -2.2, 0.8, 0.5)
    # *****************
    q.el('Me_cloudy_5par', -0.4, 0.4, 0.1)
    q.el('q_cloudy_5par', -2.8, 0.3, 0.2)
    q.el('Htot_5par', 20.5, 0.2, 0.2)
    q.el('depl_5par', 0.8, 0.2, 0.2,f='d')
    q.el('HI_5par', 20.2, 0.1, 0.1)
    # *********************** AGN radiation in cloudy model
    q.el('Me_cloudy_5par_agn', -1.0, 0.3, 0.3)
    q.el('q_cloudy_5par_agn', -3.2, 0.2, 0.2)
    q.el('Htot_5par_agn', 20.3, 0.1, 0.1)
    q.el('HI_5par_agn', 20.1, 0.1, 0.1)
    q.el('depl_5par_agn', 0.0, 0.2, 0.0, f='d')
    q.el('uv_to_nh_5par_agn', -1.4, 0.4, 0.3)
    #************************
    q.el('Mfuv', -1.72E+01, f='dec')
    q.el('Mhalo', 11.45,0.3,0.3)
    q.mangastatus = 'detection'
    q.el('gal_rad_vel', -14.6, 0, 0, f='d')
    #q.el('O/H_manga', 8.58,0.03,0.03)
    q.el('[O/H]_manga', -0.22, 0.03, 0.03)
    q.el('O/H_manga', 8.46, 0.06, 0.06)
    q.el('q_re_manga', 6.96, 0.05, 0.05)
    q.el('q_manga', 7.28, 0.05, 0.05)
    q.comp = []
    #co = sy(0.0254290, 2)
    co = sy(0.0254291, 2)
    co.el('HI', 20.2,0.1,0.1)
    co.el('SiII', 12.9,0.6,1.6)
    co.el('SiIII', 13.0,1.1,1.7)
    co.el('SII', 10.1, 2.4, 0.1)
    #co.el('OI', 10.1, 2.4, 0.1)
    co.el('NV', 13.2,0.5,1.4)
    co.el('FeII', 13.2,0.7,0.7)
    co.el('NI', 14.1,0.4,0.9)
    co.el('NII', 13.3,1.6,2.4)
    co.el('Me_cloudy_5par', -1.0, 0.4, 0.4)
    co.el('q_cloudy_5par', -3.1, 0.4, 0.4)
    #co.el('Me_cloudy', -1.6, 0.3, 0.5)
    #co.el('n/I_cloudy', -0.1, 0.9, 1.1)
    #co.el('q_cloudy', -2.4, 0.5, 0.4)
    #co.el('Htot', 20.7, 0.5, 0.4)
    #co.el('fHI_cloudy', 0.0, 0.0, 1.0)
    #co.el('Me_cloudy_4par', -1.0, 0.4, 0.4)
    #co.el('q_cloudy_4par', -2.6, 0.4, 0.4)
    #co.el('Htot_4par', 20.7, 0.5, 0.4)
    #co.el('depl_4par', 0.9, 0.1, 0.5, f='d')
    q.comp.append(co)
    #co = sy(0.0256615, 1)
    co = sy(0.0256637, 1)
    co.el('HI', 20.2,0.50,0.7)
    co.el('SiII', 13.40,0.20,0.20)
    #co.el('SiIII', '<17.76')
    co.el('SiIII', 13.6, 0.3,0.5)
    co.el('SII', 10.1,2.5,0.1)
    co.el('NV', 10.3,1.1,0.3)
    co.el('FeII', 10.5, 2.0, 0.5)
    co.el('NI', 13.7, 0.6, 1.6)
    co.el('NII', 10.1, 4.6, 0.1)
    co.el('Me_cloudy_5par', -1.0, 0.3, 0.2)
    co.el('q_cloudy_5par', -3.1, 0.2, 0.3)
    #co.el('Me_cloudy', 1.0, 0.3, 0.2)
    #co.el('n/I_cloudy', -1.6, 0.3, 0.4)
    #co.el('q_cloudy', -1.6, 0.2, 0.3)
    #co.el('Htot', 17.6, 0.3, 0.4)
    #co.el('Me_cloudy_4par', 1.6, 0.3, 0.2)
    #co.el('q_cloudy_4par', -1.6, 0.2, 0.3)
    #co.el('Htot_4par', 17.4, 0.3, 0.3)
    #co.el('depl_4par', 0.5, 0.3, 0.3, f='d')
    q.comp.append(co)
    #co = sy(0.0260279, 1)
    co = sy(0.0260424, 1)
    co.el('HI', 20.20,0.30,0.50)
    co.el('SiII', 13.60,0.20,0.30)
    co.el('SiIII', 13.50,0.30,0.30)
    co.el('SII', 15.2,0.1,0.2)
    co.el('NV', 11.5, 0.8, 1.5)
    co.el('FeII', 14.6, 0.4, 1.0)
    co.el('NI', 14.2, 0.2, 0.6)
    co.el('NII', 14.9, 0.2, 0.2)
    co.el('Me_cloudy_5par', -1.0, 0.2, 0.2)
    co.el('q_cloudy_5par', -3.1, 0.2, 0.3)
    #co.el('Me_cloudy_4par', 2.0, 0.0, 0.0)
    #co.el('q_cloudy_4par', -1.1, 0.2, 0.3)
    #co.el('Htot_4par', 16.6, 0.2, 0.2)
    #co.el('depl_4par', 0.0, 0.2, 0.0, f='d')
    #co.el('Me_cloudy', 1.9, 0.0, 0.1)
    #co.el('n/I_cloudy', -2.3, 0.2, 0.2)
    #co.el('q_cloudy', -1.2, 0.2, 0.2)
    #co.el('Htot', 16.7, 0.3, 0.3)
    #co.el('fHI_cloudy', -3.0, 0.3, 0.3)
    q.comp.append(co)
    #co = sy(0.0263915, 1)
    co = sy(0.0265396, 1)
    co.el('HI', 12.3,0.4,0.5)
    co.el('SiII', 12.70, 0.40, 0.30)
    co.el('SiIII', 12.60, 0.30, 2.60)
    co.el('SII', 14.3, 0.4, 2.5)
    co.el('FeII', 13.6, 0.6, 1.6)
    co.el('NI', 14.2, 0.2, 0.3)
    co.el('NII', 13.9, 1.0, 2.8)
    #co.el('Me_cloudy', '<2')
    #co.el('n/I_cloudy', -2.0, 0.4, 0.3)
    #co.el('q_cloudy', -1.5, 0.4, 0.3)
    #co.el('Htot', 16.0, 0.4, 0.5)
    #co.el('fHI_cloudy', -3.3, 0.5, 0.4)
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0838+2453
    q = qso('J0838+2453A', 0.0287, 0.02831)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-385099'
    q.zmanga=0.02866
    q.zmanga2= 0.02825
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 10.69, f='l')
    #q.el('SFR', 0.20, f= 'd')
    q.el('SFR', 2.76, f='d')
    q.el('sSFR', -11.38, f='l')
    #q.el('Dn4000', 1.654, f='d')
    q.el('Dn4000', 1.54, f='d')
    q.el('bpar', 5.3,5.3,5, f='d')
    q.el('b_rel', 1,1,0.9,f= 'd')
    q.el('Re', 5.35,f='d')
    q.el('drelow', 0, f='d')
    q.el('dreup', 1, f='d')
    q.el('Azi', 90-32,2,2, f='d')
    q.el('Azi_mean', 32, 1, 1, f='d')
    q.el('Azi_r_mean', 32, 1, 1, f='d')
    q.el('Azi_a_mean', 32, 1, 1, f='d')
    q.el('Azi_ra_mean', 32, 1, 1, f='d')
    q.el('Azi_h_mean', 32, 2, 2, f='d')
    q.el('Azi_NFW_6Re', 32, 2, 2, f='d')
    q.el('abszcen', 0, f='d')
    q.el('abszlow', 0, f='d')
    q.el('abszup', 0, f='d')
    #q.el('Azi_NFW_6Re', 40, 107, 6, f='d')
    q.el('sini', 0.54, f='d')
    q.el('HI', 13.2, 0.1, 0.1)
    q.el('SiII', 13.4,0.2,0.2)
    q.el('SiIII', 13.8, 0.6, 0.5)
    q.el('SiIII/SiII', 0.4, 0.6, 0.5)
    q.el('SII', 14.8,0.4,1.0)
    q.el('NI', 13.80,0.4,0.8)
    q.el('OI', '<15')
    q.el('FeII', '<15')
    q.el('zabs', 0.02831, 0.0001, 0.0001,f='d')
    q.el('Me', '<4.36')
    q.el('Si/H', -0.21,0.26,0.59)
    q.el('deltaV', -99, 28, 28, f='d')
    q.el('V_mod_em_qso', -5.6e-01, 10, 10, f='d')
    q.el('ewHI', 215, 21, 21, f='d')
    q.el('ewSiII', 124, 60, 60, f='d')
    q.el('ewSiIII', 265, 70, 70, f='d')
    q.el('v90HI', 112, 16, 16, f='d')
    q.el('v90SiII', 77, 16, 16, f='d')
    q.el('v90SiIII', 165, 36, 36, f='d')
    q.manga_absmag = [-1.63E+01, -1.71E+01, -1.92E+01, -2.07E+01, -2.13E+01, -2.17E+01, -2.19E+01]
    #q.el('Htot', 15.7, 0.3, 0.3)
    #q.el('Me_cloudy', 1.9, 0.3, 0.3)
    #q.el('q_cloudy', -1.5, 0.3, 0.3)
    #q.el('n/I_cloudy', -2.5, 0.5, 0.5)
    #q.el('fHI_cloudy', -2.4, 0.7, 0.7)
    # ***************** fit with depletion
    #q.el('Me_cloudy_4par', 1.9, 0.3, 0.3)
    #q.el('q_cloudy_4par', -1.4, 0.2, 0.3)
    #q.el('Htot_4par', 15.9, 0.2, 0.3)
    #q.el('depl_4par', 0.0, 0.2, 0.0,f='d')
    #q.el('nH_4par', -1.5, 0.2, 1.4)
    #q.el('fG_4par', 0.9, 0.1, 1.3)
    q.el('Me_cloudy_5par', 1.9, 0.3, 0.3)
    q.el('q_cloudy_5par', -0.8, 0.2, 0.5)
    q.el('Htot_5par', 16.4, 0.3, 0.3)
    q.el('depl_5par', 0.0, 0.1, 0.0,f='d')
    q.el('HI_5par', 13.2, 0.2, 0.3)
    q.el('uv_to_nh_5par', 2.6, 0.2, 0.2)
    # *********************** AGN radiation in cloudy model
    q.el('Me_cloudy_5par_agn', 2.0, 0.0, 0.05)
    q.el('q_cloudy_5par_agn', -2.3, 0.2, 0.2)
    q.el('Htot_5par_agn', 15.2, 0.2, 0.2)
    q.el('HI_5par_agn', 13.5, 0.1, 0.1)
    q.el('depl_5par_agn', 0.0, 0.1, 0.0, f='d')
    q.el('uv_to_nh_5par_agn', -0.3, 0.2, 0.2)
    # ******************
    q.el('Mfuv', -1.63E+01, f='dec')
    q.el('Mhalo', 13.75,0.3,0.3)
    #q.el('O/H_manga', 'NA')
    #q.el('rSi', '>0.4')
    q.mangastatus = 'detection'
    q.el('gal_rad_vel', -12.3, 0, 0, f='d')
    q.comp = []
    #co = sy(0.028316, 10)
    co = sy(0.02834, 10)
    co.el('HI', 13.2, 0.1, 0.1,b=(41,10,10))
    co.el('SiII', '<13.5')
    co.el('SiIII', 12.5, 0.4,0.7)
    co.el('SII', 14.0,0.6,3.0)
    co.el('OI', '<14.7')
    co.el('NI', '<13.7')
    co.el('Me_cloudy_5par', 1.9, 0.3, 0.3)
    co.el('q_cloudy_5par', -0.8, 0.2, 0.5)
    q.comp.append(co)
    co = sy(0.02812, 13)
    co.el('SiIII', 12.8,0.3,0.3,b=(16,5,1))
    co.el('SiII', 12.3,0.5,1.0)
    co.el('SII', 14.7,0.5,2.1)
    co.el('OI', '<14.5')
    co.el('NI', '<13.6')
    q.comp.append(co)
    co = sy(0.02774, 18)
    co.el('SiIII', 12.8, 0.4, 0.3,b=(20,10,10))
    co.el('SiII', '<13.5')
    co.el('SII', '<15.5')
    co.el('OI', '<14.3')
    co.el('NI', '<13.7')
    q.comp.append(co)
    if 1:
        co = sy(0.02584, 3)
        co.el('HI', 14, 0.1, 0.1, b=(55, 5, 5))
        co.el('SiII', 12.8, 0.5, 1.3)
        co.el('SiIII', 13.2, 0.8, 1.7)
        co.el('SII', 14.2, 0.8, 1.5)
        co.el('Me_cloudy_5par', 0.5, 1.5, 0.5)
        co.el('q_cloudy_5par', -1.1, 0.5, 0.7)
        co.el('OI', '<14.5')
        co.el('NI', '<13.7')
        q.comp.append(co)
        co = sy(0.02556, 21)
        co.el('SiII', 13.4, 1.1, 0.7)
        co.el('SiIII', '<13.3')
        co.el('SII', 13.4, 0.5, 1.4)
        co.el('OI', '<13')
        co.el('NI', '<14.4')
        q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J0838+2453B', 0.0287, 0.02584)
    q.telescope = 'HST'
    q.year = 2021
    q.el('Mstar', 10.09, f='l')
    q.el('Aziangle', 33, f='d')
    if 0:
        q.mangaID = '1-585207'
        q.zmanga=0.02825
        #q.el('SFR', 0.20, f='d')
        #q.el('sSFR', -11.38, f='l')
        q.el('SFR', 0.00041, f='d')
        #q.el('Dn4000', 1.945, f='d')
        q.el('Dn4000', 1.93, f='d')
        q.el('bpar', 37, f= 'd')
        q.el('b_rel', 15.5,  f='d')
        q.el('Re', 2.45, f='d')
        q.el('drelow', 15.4, f='d')
        q.el('dreup', 18.7, f='d')
        q.el('Azi', 34,2,2, f='d')
        q.el('Azi_mean', 106, 35, 35, f='d')
        q.el('Azi_r_mean', 115, 25, 25, f='d')
        q.el('Azi_a_mean', 90, 15, 15, f='d')
        q.el('Azi_ra_mean', 95, 13, 13, f='d')
        q.el('Azi_h_mean', 108, 20, 19, f='d')
        q.el('Azi_NFW_6Re', 108, 21, 21, f='d')
        q.el('abszcen', 0, f='d')
        q.el('abszlow', 0, f='d')
        q.el('abszup', 0, f='d')
        q.el('abszcen', 3.2, f='d')
        q.el('abszlow', -4.5, f='d')
        q.el('abszup', 10.8, f='d')
    if 1:
        q.mangaID = '1-385099'
        q.zmanga = 0.02866
        q.zmanga2 = 0.02825
        # q.progID.append('085.A-0569(B)')
        q.el('Mstar', 10.69, f='l')
        q.el('SFR', 0.20, f='d')
        q.el('sSFR', -11.38, f='l')
        # q.el('Dn4000', 1.654, f='d')
        q.el('Dn4000', 1.54, f='d')
        q.el('bpar', 5.3, 5.3, 5, f='d')
        q.el('b_rel', 1, 1, 0.9, f='d')
        q.el('Re', 5.35, f='d')
        q.el('drelow', 0, f='d')
        q.el('dreup', 1, f='d')
        q.el('Azi', 90 - 32, 2, 2, f='d')
        q.el('Azi_mean', 32, 1, 1, f='d')
        q.el('Azi_r_mean', 32, 1, 1, f='d')
        q.el('Azi_a_mean', 32, 1, 1, f='d')
        q.el('Azi_ra_mean', 32, 1, 1, f='d')
        q.el('Azi_h_mean', 32, 2, 2, f='d')
        q.el('Azi_NFW_6Re', 32, 2, 2, f='d')
        q.el('abszcen', 0, f='d')
        q.el('abszlow', 0, f='d')
        q.el('abszup', 0, f='d')
        # q.el('Azi_NFW_6Re', 40, 107, 6, f='d')
        q.el('sini', 0.54, f='d')
    q.el('HI', 14, 0.1, 0.1)
    q.el('SiII', '<13.6')
    q.el('SiIII', 13.6, 0.5, 1.0)
    q.el('SII', 14.5,0.5,1.4)
    q.el('NI', '<14.4')
    q.el('OI', '<14.5')
    q.el('zabs', 0.02579, 0.0001, 0.0001, f='d')
    q.el('Me', '<4.36')
    q.el('deltaV', -99, 28, 28, f='d')
    q.el('V_mod_em_qso', -5.6e-01, 10, 10, f='d')
    q.el('ewHI', 215, 21, 21, f='d')
    q.el('ewSiII', 124, 60, 60, f='d')
    q.el('ewSiIII', 265, 70, 70, f='d')
    q.el('v90HI', 112, 16, 16, f='d')
    q.el('v90SiII', 77, 16, 16, f='d')
    q.el('v90SiIII', 165, 36, 36, f='d')
    q.manga_absmag = [-1.63E+01, -1.71E+01, -1.92E+01, -2.07E+01, -2.13E+01, -2.17E+01, -2.19E+01]
    # ***************** fit with depletion
    #q.el('Me_cloudy_4par', 2.0, 0.2, 0.1)
    #q.el('q_cloudy_4par', -3.1, 0.2, 0.3)
    #q.el('Htot_4par', 15.0, 0.2, 0.2)
    #q.el('depl_4par', '<1',f='d')
    #q.el('nH_4par', -0.7, 0.6, 0.8)
    #q.el('fG_4par', -0.3, 0.5, 1.0)
    # *****************
    q.el('Me_cloudy_5par', 0.5, 1.5, 0.5)
    q.el('q_cloudy_5par', -1.1, 0.5, 0.7)
    q.el('Htot_5par', 17.1, 1.5, 0.3)
    q.el('depl_5par', '<1')
    q.el('HI_5par', 13.8, 0.2, 0.3)
    # *********************** AGN radiation in cloudy model
    q.el('Me_cloudy_5par_agn', 2.0, 0.0, 0.3)
    q.el('q_cloudy_5par_agn', -2.1, 0.2, 0.2)
    q.el('Htot_5par_agn', 16.1, 0.4, 0.5)
    q.el('HI_5par_agn', 14, 0.1, 0.1)
    q.el('depl_5par_agn', 0.0, 0.4, 0.0, f='d')
    q.el('uv_to_nh_5par_agn', 0.2, 0.2, 0.2)
    # ******************
    q.manga_absmag = [-1.36E+01, -1.36E+01, -1.74E+01, -1.90E+01, -1.98E+01, -2.01E+01, -2.04E+01]
    q.el('Mfuv', -1.36E+01, f='dec')
    q.el('Mhalo', 11.84,0.3,0.3)
    q.mangastatus = 'detection'
    q.comp = []
    co = sy(0.02584, 3)
    co.el('HI', 14, 0.1, 0.1,b=(55,5,5))
    co.el('SiII', 12.8, 0.5, 1.3)
    co.el('SiIII', 13.2, 0.8, 1.7)
    co.el('SII', 14.2, 0.8, 1.5)
    co.el('Me_cloudy_5par', 0.5, 1.5, 0.5)
    co.el('q_cloudy_5par', -1.1, 0.5, 0.7)
    co.el('OI', '<14.5')
    co.el('NI', '<13.7')
    q.comp.append(co)
    co = sy(0.02556, 21)
    co.el('SiII', 13.4, 1.1, 0.7)
    co.el('SiIII', '<13.3')
    co.el('SII', 13.4, 0.5, 1.4)
    co.el('OI', '<13')
    co.el('NI', '<14.4')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1237+4447
    q = qso('J1237+4447A', 0.4612, 0.0597)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-575668'
    q.zmanga = 0.06018
    q.vcorr = -30.10
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 11.02, f='l')
    q.el('sigMstar', 8.5, f='l')
    #q.el('SFR', 0.0, f= 'd')
    q.el('SFR', 0.24, f='d')
    #q.el('Dn4000', 1.877, f='d')
    q.el('Dn4000', 1.87, f='d')
    q.el('bpar', 39, f='d')
    q.el('b_rel', 3.8,  f='d')
    q.el('Re', 10.57, f='d')
    q.el('drelow', 3.82, f='d')
    q.el('dreup', 3.88, f='d')
    q.el('Azi', 10, f='d')
    q.el('Azi_mean', 90, 60, 60, f='d')
    q.el('Azi_r_mean', 90, 30, 30, f='d')
    q.el('Azi_a_mean', 60, 40, 40, f='d')
    q.el('Azi_ra_mean', 90, 13, 13, f='d')
    q.el('Azi_h_mean', 90, 24, 24, f='d')
    q.el('Azi_NFW_6Re', 90, 27, 26, f='d')
    q.el('abszcen', 0.07, f='d')
    q.el('abszlow', -1.94, f='d')
    q.el('abszup', 2.03, f='d')
    q.el('r_dist', 3.7, 0.5, 0.5, f='d')
    q.el('h_dist', 0.0, 0.8, 0.8, f='d')
    q.el('gal_rad_vel', -66, 0, 0, f='d')
    if 0:
        q.el('HI', 16.3, 0.3, 0.6)
        q.el('SiII', 12.4,0.2,0.4)
        q.el('SiIII', 12.9,0.1,0.1)
        q.el('Sitot', '<13.5')
        q.el('CII', 14.3,0.2,0.2)
        q.el('SII', 14.3,0.4,0.8)
        q.el('NI', 12.6,0.5,1.3)
        q.el('NII', 13,0.3,1.5)
        q.el('NV', 13.2,0.3,0.5)
        q.el('FeII', 13.5,0.4,0.8)
        q.el('Me_cloudy_4par', -0.2, 0.2, 0.3)
        q.el('q_cloudy_4par', -2.9, 0.2, 0.2)
        q.el('Htot_4par', 18.9, 0.2, 0.2)
        q.el('depl_4par', 0.7, 0.2, 0.2, f='d')
        q.el('nH_4par', -3.0, 0.2, 0.2)
        q.el('fG_4par', -3, 0.3, 0.0)
    if 1:
        #case A
        q.el('HI', 17.2, 0.3, 0.7)
        q.el('SiII', '<13')
        q.el('SiIII', 12.9, 0.1, 0.1)
        q.el('CII', '<14.7')
        q.el('SII', 14.3, 0.5, 0.8)
        q.el('SIV', '<14.2')
        q.el('NI', 12.9, 0.5, 1.3)
        q.el('NII', '<13.5')
        q.el('NV', 13.2, 0.3, 0.5)
        q.el('OIV', '<14.3')
        q.el('FeII', 13.5, 0.5, 0.6)
        q.el('FeIII', '<11')
        q.el('ArI', '<13.5')

        q.el('Me_cloudy_5par', -1.8, 0.8, 0.8)
        q.el('q_cloudy_5par', -3, 0.6, 0.6)
        q.el('Htot_5par', 19.4, 0.8, 0.5)
        q.el('HI_5par', 17.2, 0.2, 0.7)
        q.el('depl_5par', 0.0, 0.2, 0.0, f='d')

    if 0:
        #case B
        q.el('HI', 15.2, 0.5, 0.3)
        q.el('SiII', 12.4, 0.4, 0.8)
        q.el('SiIII', 12.9, 0.1, 0.1)
        q.el('CII', 14.0, 0.5, 1.4)
        q.el('SII', 14.3, 0.3, 1.2)
        q.el('SIV', '<13.9')
        q.el('NI', 12.9, 0.3, 1.0)
        q.el('NII', '<13.5')
        q.el('NV', 13.2, 0.3, 0.8)
        q.el('OIV', 14.1, 0.5, 1.5)
        q.el('FeII', 13.7, 0.3, 0.9)
        q.el('FeIII', '<11')
        q.el('SII', '<13.2')

        q.el('Me_cloudy_5par', -0.2, 0.5, 1)
        q.el('q_cloudy_5par', -2.6, 0.6, 0.4)
        q.el('Htot_5par', 18.1, 0.9, 0.5)
        q.el('HI_5par', 15.2, 0.5, 0.3)
        q.el('depl_5par', 0.0, 0.2, 0.0, f='d')
    q.el('zabs', 0.0597, 0.0001, 0.0001,f= 'd')
    q.el('Me', '<-0.06')
    q.el('Si/H', '<-1.89')
    q.el('deltaV', -133, 28, 28, f='d')
    q.el('V_mod_em_qso', 115, 10, 10, f='d')
    q.el('ewHI', 780, 55, 55, f='d')
    q.el('ewSiIII', 74, 51, 51, f='d')
    q.el('v90HI', 320, 21, 21, f='d')
    q.el('v90SiIII', 114, 70, 70, f='d')
    q.manga_absmag = [ -1.71E+01, -1.86E+01, -1.98E+01, -2.14E+01, -2.22E+01, -2.26E+01, -2.28E+01]
    #q.el('Me_cloudy', -0.60,0.2,0.30)
    #q.el('q_cloudy', -2.9, 0.2, 0.2)
    #q.el('Htot', 18.8, 0.3, 0.3)
    # ***************** fit with depletion
    # *****************
    q.el('Mfuv', -1.71E+01, f='dec')
    q.el('Mhalo', 14.12,0.3,0.3)
    q.mangastatus = 'detection'
    q.comp = []
    co = sy(0.05960, 3)
    co.el('HI', 14.8, 0.6, 0.3)
    co.el('SiII', '<13')
    co.el('SiIII', 12.8,0.1,0.1)
    co.el('CII', 13.9, 0.3, 1.5)
    co.el('SII', 14.3, 0.5, 1.5)
    co.el('SIV', '<14.3')
    co.el('NI', '<13.3')
    co.el('NII', '<13.5')
    co.el('NV', 13.2,0.1,2.5)
    co.el('OVI', '<14.3')
    co.el('FeII', '<13.8')
    co.el('FeIII', '<13.8')
    co.el('ArI', '<13.4')
    if 0:
        co.el('Me_cloudy_4par', -0.3, 0.1, 0.1)
        co.el('q_cloudy_4par', -2.5, 0.1, 0.1)
        co.el('Htot_4par', 17.8, 0.2, 0.2)
        co.el('depl_4par', 0.0, 0.2, 0.0, f='d')
    if 1:
        co.el('Me_cloudy_5par', -0.2, 0.7, 1.1)
        co.el('q_cloudy_5par', -2.0, 0.9, 0.5)
        co.el('Htot_5par', 18.2, 1.3, 0.6)
        co.el('HI_5par', 14.9, 0.6, 0.3)
        co.el('depl_5par', '<1', f='d')
    q.comp.append(co)
    #component A
    co = sy(0.059963, 12)
    co.el('HI', 17.1, 0.3, 0.7,b=(16,4,1))
    co.el('SiII', '<12.5')
    co.el('SiIII', 12.3, 0.2, 0.3)
    co.el('CII', '<14.7')
    co.el('SII', '<14.3')
    co.el('SIV', '<14.1')
    co.el('NI', '<13.4')
    co.el('NII', '<13.8')
    co.el('NV', '<13.2')
    co.el('OVI', '<15.3')
    co.el('FeII', 13.5,0.4,1.5)
    co.el('FeIII', '<14.0')
    co.el('ArI', '<13.3')
    if 0:
        co.el('Me_cloudy_4par', -0.5, 0.3, 0.2)
        co.el('q_cloudy_4par', -3.3, 0.3, 0.3)
        co.el('Htot_4par', 18, 0.3, 0.3)
        co.el('depl_4par', 0.3, 0.3, 0.3, f='d')
    if 1:
        co.el('Me_cloudy_5par', -1.6,1,1)
        co.el('q_cloudy_5par', -3.1, 0.6, 0.5)
        co.el('Htot_5par', 19.3, 0.8, 0.9)
        co.el('HI_5par', 17.0, 0.4, 0.3)
        co.el('depl_5par', 0.0, 0.2, 0.0, f='d')
    q.comp.append(co)
    # component B
    if 0:
        co = sy(0.059963, 12)
        co.el('HI', 14.8, 0.8, 0.2, b=(28, 6, 6))
        co.el('SiII', '<12.6')
        co.el('SiIII', 12.4, 0.2, 0.2)
        co.el('CII', '<14.8')
        co.el('SII', '<14.2')
        co.el('SIV', '<14.1')
        co.el('NI', '<13.3')
        co.el('NII', '<13.6')
        co.el('NV', '<13.2')
        co.el('OVI', 14,0.2,1.5)
        co.el('FeII', 13.7,0.3,1.5)
        co.el('FeIII', '<14.2')
        co.el('ArI', '<13.2')
        if 0:
            co.el('Me_cloudy_4par', -0.5, 0.3, 0.2)
            co.el('q_cloudy_4par', -3.3, 0.3, 0.3)
            co.el('Htot_4par', 18, 0.3, 0.3)
            co.el('depl_4par', 0.3, 0.3, 0.3, f='d')
        if 1:
            co.el('Me_cloudy_5par', -0.7, 0.6, 1.4)
            co.el('q_cloudy_5par', -2.6, 1.1, 0.5)
            co.el('Htot_5par', 18.1, 1.4, 0.7)
            co.el('HI_5par', 14.8, 0.7, 0.3)
            co.el('depl_5par', 0.0, 0.2, 0.0, f='d')
        q.comp.append(co)

    co = sy(0.061187, 1)
    co.el('HI', 13.60, 0.1, 0.1, b=(20.0, 9.3, 9.3))
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J1237+4447B', 0.4612, 0.0597)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-575668'
    q.zmanga = 0.06018
    q.vcorr = -30.10
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 11.02, f='l')
    q.el('sigMstar', 8.5, f='l')
    #q.el('SFR', 0.0, f='d')
    q.el('SFR', 0.24, f='d')
    # q.el('Dn4000', 1.877, f='d')
    q.el('Dn4000', 1.87, f='d')
    q.el('bpar', 39, f='d')
    q.el('b_rel', 3.8, f='d')
    q.el('Re', 10.57, f='d')
    q.el('drelow', 3.82, f='d')
    q.el('dreup', 3.88, f='d')
    q.el('Azi', 10, f='d')
    q.el('Azi_mean', 90, 60, 60, f='d')
    q.el('Azi_r_mean', 90, 30, 30, f='d')
    q.el('Azi_a_mean', 60, 40, 40, f='d')
    q.el('Azi_ra_mean', 90, 13, 13, f='d')
    q.el('Azi_h_mean', 90, 24, 24, f='d')
    q.el('Azi_NFW_6Re', 90, 27, 26, f='d')
    q.el('abszcen', 0.07, f='d')
    q.el('abszlow', -1.94, f='d')
    q.el('abszup', 2.03, f='d')
    q.el('r_dist', 3.7, 0.5, 0.5, f='d')
    q.el('h_dist', 0.0, 0.8, 0.8, f='d')
    q.el('gal_rad_vel', -66, 0, 0, f='d')
    if 0:
        q.el('HI', 16.3, 0.3, 0.6)
        q.el('SiII', 12.4, 0.2, 0.4)
        q.el('SiIII', 12.9, 0.1, 0.1)
        q.el('Sitot', '<13.5')
        q.el('CII', 14.3, 0.2, 0.2)
        q.el('SII', 14.3, 0.4, 0.8)
        q.el('NI', 12.6, 0.5, 1.3)
        q.el('NII', 13, 0.3, 1.5)
        q.el('NV', 13.2, 0.3, 0.5)
        q.el('FeII', 13.5, 0.4, 0.8)
        q.el('Me_cloudy_4par', -0.2, 0.2, 0.3)
        q.el('q_cloudy_4par', -2.9, 0.2, 0.2)
        q.el('Htot_4par', 18.9, 0.2, 0.2)
        q.el('depl_4par', 0.7, 0.2, 0.2, f='d')
        q.el('nH_4par', -3.0, 0.2, 0.2)
        q.el('fG_4par', -3, 0.3, 0.0)
    if 0:
        # case A
        q.el('HI', 17.2, 0.3, 0.7)
        q.el('SiII', '<13')
        q.el('SiIII', 12.9, 0.1, 0.1)
        q.el('CII', '<14.7')
        q.el('SII', 14.3, 0.5, 0.8)
        q.el('SIV', '<14.2')
        q.el('NI', 12.9, 0.5, 1.3)
        q.el('NII', '<13.5')
        q.el('NV', 13.2, 0.3, 0.5)
        q.el('OIV', '<14.3')
        q.el('FeII', 13.5, 0.5, 0.6)
        q.el('FeIII', '<11')
        q.el('ArI', '<13.5')

        q.el('Me_cloudy_5par', -1.8, 0.8, 0.8)
        q.el('q_cloudy_5par', -3, 0.6, 0.6)
        q.el('Htot_5par', 19.4, 0.8, 0.5)
        q.el('HI_5par', 17.2, 0.2, 0.7)
        q.el('depl_5par', 0.0, 0.2, 0.0, f='d')

    if 1:
        # case B
        q.el('HI', 15.2, 0.5, 0.3)
        q.el('SiII', '<13')
        q.el('CII', '<14.7')
        q.el('SiIII', 12.9, 0.1, 0.1)
        q.el('SII', 14.3, 0.3, 1.2)
        q.el('SIV', '<13.9')
        q.el('NI', 12.9, 0.3, 1.0)
        q.el('NII', '<13.5')
        q.el('NV', 13.2, 0.3, 0.8)
        q.el('OIV', 14.1, 0.5, 1.5)
        q.el('FeII', 13.7, 0.3, 0.9)
        q.el('FeIII', '<11')
        q.el('SII', '<13.2')

        q.el('Me_cloudy_5par', -0.2, 0.5, 1)
        q.el('q_cloudy_5par', -2.6, 0.6, 0.4)
        q.el('Htot_5par', 18.1, 0.9, 0.5)
        q.el('HI_5par', 15.2, 0.5, 0.3)
        q.el('depl_5par', 0.0, 0.2, 0.0, f='d')
    q.el('zabs', 0.0597, 0.0001, 0.0001, f='d')
    q.el('Me', '<-0.06')
    q.el('Si/H', '<-1.89')
    q.el('deltaV', -133, 28, 28, f='d')
    q.el('V_mod_em_qso', 115, 10, 10, f='d')
    q.el('ewHI', 780, 55, 55, f='d')
    q.el('ewSiIII', 74, 51, 51, f='d')
    q.el('v90HI', 320, 21, 21, f='d')
    q.el('v90SiIII', 114, 70, 70, f='d')
    q.manga_absmag = [-1.71E+01, -1.86E+01, -1.98E+01, -2.14E+01, -2.22E+01, -2.26E+01, -2.28E+01]
    #q.el('Me_cloudy', -0.60, 0.2, 0.30)
    #q.el('q_cloudy', -2.9, 0.2, 0.2)
    #q.el('Htot', 18.8, 0.3, 0.3)
    # ***************** fit with depletion
    # *****************
    q.el('Mfuv', -1.71E+01, f='dec')
    q.el('Mhalo', 14.12, 0.3, 0.3)
    q.mangastatus = 'detection'
    q.comp = []
    co = sy(0.05960, 3)
    co.el('HI', 14.8, 0.6, 0.3)
    co.el('SiII', '<13')
    co.el('SiIII', 12.8, 0.1, 0.1)
    co.el('CII', 13.9, 0.3, 1.5)
    co.el('SII', 14.3, 0.5, 1.5)
    co.el('SIV', '<14.3')
    co.el('NI', '<13.3')
    co.el('NII', '<13.5')
    co.el('NV', 13.2, 0.1, 2.5)
    co.el('OVI', '<14.3')
    co.el('FeII', '<13.8')
    co.el('FeIII', '<13.8')
    co.el('ArI', '<13.4')
    if 0:
        co.el('Me_cloudy_4par', -0.3, 0.1, 0.1)
        co.el('q_cloudy_4par', -2.5, 0.1, 0.1)
        co.el('Htot_4par', 17.8, 0.2, 0.2)
        co.el('depl_4par', 0.0, 0.2, 0.0, f='d')
    if 1:
        co.el('Me_cloudy_5par', -0.2, 0.7, 1.1)
        co.el('q_cloudy_5par', -2.0, 0.9, 0.5)
        co.el('Htot_5par', 18.2, 1.3, 0.6)
        co.el('HI_5par', 14.8, 0.6, 0.3)
        co.el('depl_5par', '<1', f='d')
    q.comp.append(co)
    # component A
    if 0:
        co = sy(0.059963, 12)
        co.el('HI', 17.1, 0.3, 0.7 / 2, b=(16, 4, 1))
        co.el('SiII', '<12.5')
        co.el('SiIII', 12.3, 0.2, 0.3)
        co.el('CII', '<14.7')
        co.el('SII', '<14.3')
        co.el('SIV', '<14.1')
        co.el('NI', '<13.4')
        co.el('NII', '<13.8')
        co.el('NV', '<13.2')
        co.el('OVI', '<15.3')
        co.el('FeII', 13.5, 0.4, 1.5)
        co.el('FeIII', '<14.0')
        co.el('ArI', '<13.3')
        if 0:
            co.el('Me_cloudy_4par', -0.5, 0.3, 0.2)
            co.el('q_cloudy_4par', -3.3, 0.3, 0.3)
            co.el('Htot_4par', 18, 0.3, 0.3)
            co.el('depl_4par', 0.3, 0.3, 0.3, f='d')
        if 1:
            co.el('Me_cloudy_5par', -1.6, 1, 1)
            co.el('q_cloudy_5par', -3.1, 0.6, 0.5)
            co.el('Htot_5par', 19.3, 0.8, 0.9)
            co.el('HI_5par', 17.0, 0.4, 0.3)
            co.el('depl_5par', 0.5, 0.5, 0.5, f='d')
        # q.comp.append(co)
    # component B
    co = sy(0.059963, 12)
    co.el('HI', 14.8, 0.8, 0.2, b=(28, 6, 6))
    co.el('SiII', '<12.6')
    co.el('SiIII', 12.4, 0.2, 0.2)
    co.el('CII', '<14.8')
    co.el('SII', '<14.2')
    co.el('SIV', '<14.1')
    co.el('NI', '<13.3')
    co.el('NII', '<13.6')
    co.el('NV', '<13.2')
    co.el('OVI', 14, 0.2, 1.5)
    co.el('FeII', 13.7, 0.3, 1.5)
    co.el('FeIII', '<14.2')
    co.el('ArI', '<13.2')
    if 0:
        co.el('Me_cloudy_4par', -0.5, 0.3, 0.2)
        co.el('q_cloudy_4par', -3.3, 0.3, 0.3)
        co.el('Htot_4par', 18, 0.3, 0.3)
        co.el('depl_4par', 0.3, 0.3, 0.3, f='d')
    if 1:
        co.el('Me_cloudy_5par', -0.7, 0.6, 1.4)
        co.el('q_cloudy_5par', -2.6, 1.1, 0.5)
        co.el('Htot_5par', 18.1, 1.4, 0.7)
        co.el('HI_5par', 14.8, 0.7, 0.3)
        co.el('depl_5par', 0.0, 0.2, 0.0, f='d')
    q.comp.append(co)

    co = sy(0.061187, 1)
    co.el('HI', 13.60, 0.1, 0.1, b=(20.0, 9.3, 9.3))
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)



    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0950+4309
    q = qso('J0950+4309B', 0.3622,0.01708)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-166736'
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 8.98, f='l')
    q.el('sigMstar', 7.43, f='l')
    q.el('SFR', 0.08, f= 'd')
    q.el('bpar', 23, 0, 0,f= 'd')
    q.el('b_rel', 6.9,  f='d')
    q.el('Re', 3.41, f='d')
    q.el('sSFR', -10.07, f='l')
    q.el('sigSFR', -2.63, f='l')
    q.el('HI', 17.9,  0.3, 0.3)
    q.el('SiII', 12.8, 0.24, 0.4)
    q.el('SiIII', 13.60,0.20,0.10)
    q.el('SiIII/SiII', 0.8,0.44,0.26)
    q.el('Sitot', 13.66,0.19,0.09)
    q.el('SII', 14.20,0.30,0.60)
    q.el('CII', 14.13, 0.20, 0.10)
    q.el('NI', 13.80, 0.40, 0.60)
    q.el('NII', 14.20, 2.00, 1.80)
    q.el('NV', 14.00, 0.20, 0.40)
    q.el('OI', 13.60, 0.60, 0.60)
    q.el('FeII', 13.60, 0.30, 0.40)
    q.el('zabs', 0.0170, 0.0001, 0.0001, f='d')
    q.el('Me', -0.55, 0.20, 0.17)
    q.el('rSi', 0.80, 0.40, 0.26)
    q.el('Si/H', -5.04,0.20,0.20)
    q.el('deltaV', -23, 28, 28, f='d')
    q.el('V_mod_em_qso', 25.4, 3, 3, f='d')
    q.el('ewHI', 1440, 52, 52, f='d')
    q.el('ewSiII', 128, 22, 22, f='d')
    q.el('ewSiIII', 330, 33, 33, f='d')
    q.el('v90HI', 395, 38, 38, f='d')
    q.el('v90SiII', 91, 50, 50, f='d')
    q.el('v90SiIII', 116, 30, 30, f='d')
    q.manga_absmag = [-1.58E+01, -1.60E+01, -1.65E+01, -1.75E+01, -1.79E+01, -1.81E+01, -1.85E+01]
    #q.el('Me_cloudy', -2.0,0.3,0.8)
    #q.el('n/I_cloudy', -1, 0.4, 0.4)
    #q.el('q_cloudy', -2.4, 0.3, 0.3)
    #q.el('Htot', 20.7, 0.4, 0.4)
    #q.el('fHI_cloudy', -1.9, 0.4, 0.4)
    q.el('Mfuv', -1.58E+01, f='dec')
    q.el('Mhalo', 11.35,0.3,0.3)
    q.mangastatus = 'detection'
    q.el('[O/H]_manga', -0.12, 0.03, 0.03)
    q.el('O/H_manga', 8.56,0.05,0.05)
    q.el('q_manga', 7.04,0.05,0.05)
    q.el('q_re_manga', 6.85,0.05,0.05)
    q.el('q_cen_manga', 7.4, 0.05, 0.05)
    q.el('O/H_cen_manga', -0.1, 0.05, 0.05)
    q.comp = []
    co = sy(0.01710, 40)
    co.el('HI', 17.50, 0.50, 0.50)
    co.el('CII', 14.10, 0.30, 0.40)
    co.el('SiII', 12.74, 0.30,0.40)
    co.el('SiIII', 13.00,0.40,0.70)
    #co.el('SII', 14.20,0.30,0.60)
    co.el('SII', '<14.20')
    co.el('FeII', 13.60,0.30,0.40)
    co.el('NII', 14.20,1.00,1.00)
    co.el('PII', 13.00,0.50,0.50)
    co.el('NI', 13.80,0.40,0.80)
    co.el('OI', 13.60,0.60,0.60)
    co.el('NV', 14.00,0.20,0.40)
    co.el('FeIII', '<14.50')
    q.comp.append(co)
    co = sy(0.016930, 20)
    co.el('HI', 18.47, 0.10, 0.10)
    co.el('SiIII', 13.49,0.20,0.20)
    co.el('CII', '<14.3')
    co.el('SiII', '<13.0')
    co.el('SII', '<14.6')
    co.el('FeII', '<14.2')
    co.el('NII', '<16.0')
    co.el('PII', '<14.0')
    co.el('NI', '<14.5')
    co.el('OI', '<14.0')
    co.el('NV', '<14.0')
    co.el('FeIII', '<14.50')
    q.comp.append(co)
    co = sy(0.016240, 40)
    co.el('HI', 13.20, 0.20, 0.20)
    co.el('SiIII', 12.30,0.40,0.60)
    co.el('CII', '<14.0')
    co.el('SiII', '<12.7')
    co.el('SII', '<14.6')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    #QSO.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J2130-0025
    q = qso('J2130-0025', 0.4901, 0.0196675)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-180522'
    q.zmanga =  0.02014
    q.vcorr = -28.8
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 9.31, f='l')
    q.el('sigMstar', 7.60, f='l')
    q.el('SFR', 0.55, f='d')
    #q.el('Dn4000', 1.251, f='d')
    q.el('Dn4000', 1.26, f='d')
    q.el('bpar', 34, f='d')
    q.el('b_rel', 8.5,  f='d')
    q.el('drelow', 8.5, f='d')
    q.el('dreup', 10.0, f='d')
    q.el('Re', 4.15, f='d')
    q.el('Azi', 4,12,12, f='d')
    q.el('Azi_mean', 91, 13, 13, f='d')
    q.el('Azi_r_mean', 93, 9, 9, f='d')
    q.el('Azi_a_mean', 91, 12, 12, f='d')
    q.el('Azi_ra_mean', 93, 8, 8, f='d')
    q.el('Azi_h_mean', 93, 7, 7, f='d')
    q.el('Azi_NFW_6Re', 94, 17, 18, f='d')

    q.el('abszcen', 0.16, f='d')
    q.el('abszlow', -4.9, f='d')
    q.el('abszup', 5.3, f='d')
    q.el('r_dist', 8.7, 0.2, 0.2, f='d')
    q.el('h_dist', -0.4, 0.7, 0.7, f='d')
    q.el('gal_rad_vel', -137, 0, 0, f='d')
    q.el('sini', 0.99, f='d')
    q.el('sSFR', -9.56, f='l')
    q.el('sigSFR', -1.96, f='l')
    #q.el('HI', 18.88, 0.09, 0.07)
    q.el('HI', 18.83, 0.05, 0.05,b=(22,6,3))
    q.el('SiII', 13.77, 0.12, 0.19)
    q.el('SiIII', 14.6,0.9,0.5)
    #q.el('SiIII/SiII', 0.27,0.41,0.26)
    #q.el('Sitot', 13.98,0.30,0.12)
    q.el('CII', 15.3,0.9,0.5)
    #q.el('NV', 13.1,0.4,0.8)
    q.el('SII', 13.8, 0.1, 1.2)
    q.el('NI', 13.2, 0.3, 1.1)
    q.el('NII', '<14.7')
    q.el('NV', '<13.7')
    q.el('FeII', 13.6, 0.4, 1.2)
    q.el('OI', 14.6, 0.9, 0.5)

    q.el('Me', 1.32, 0.60, 0.77)
    q.el('rSi', 0.30, 0.15, 0.14)
    q.el('Si/H', -4.77,0.14,0.13)
    q.el('deltaV', -143, 28, 28, f='d')
    q.el('V_mod_em_qso', 100, 10, 10, f='d')
    q.el('ewHI', 1533, 64, 64, f='d')
    q.el('ewSiII', 275, 30, 30, f='d')
    q.el('ewSiIII', 447, 46, 46, f='d')
    q.el('v90HI', 450, 24, 24, f='d')
    q.el('v90SiII', 82, 13, 13, f='d')
    q.el('v90SiIII', 120, 11, 11, f='d')
    q.manga_absmag = [-1.54E+01, -1.59E+01, -1.70E+01, -1.80E+01, -1.85E+01, -1.87E+01, -1.89E+01]
    #q.el('Me_cloudy', -1.4, 0.1, 0.1)
    #q.el('q_cloudy', -2.5, 0.2, 0.2)
    #q.el('Htot', 20.7, 0.4, 0.3)
    #q.el('fHI_cloudy', -1.1, 0.3, 0.4)
    #***************** fit with depletion
    #q.el('Me_cloudy_4par', -1.1, 0.3, 0.3)
    #q.el('q_cloudy_4par', -2.6, 0.2, 0.3)
    #q.el('Htot_4par', 20.6, 0.2, 0.3)
    #q.el('depl_4par', 0.1, 0.3, 0.1,f='d')
    #q.el('nH_4par', -3.2, 0.3, 0.3)
    #q.el('fG_4par', -3, 0.6, 0.0)
    # ***************** cloudy fit with depletion and HI
    q.el('Me_cloudy_5par', -1.15, 0.2, 0.2)
    q.el('q_cloudy_5par', -2.1, 0.4, 0.5)
    q.el('Htot_5par', 21.1, 0.4, 0.6)
    q.el('depl_5par', 0.1, 0.3, 0.1, f='d')
    q.el('HI_5par', 18.83, 0.04, 0.06)
    #*****************
    q.el('Mfuv', -1.54E+01, f='dec')
    q.el('Mhalo', 11.63,0.3,0.3)
    q.mangastatus = 'detection'
    q.el('[O/H]_manga', 0.06, 0.05, 0.05)
    q.el('O/H_manga', 8.75, 0.05, 0.05)
    q.el('q_re_manga', 6.99, 0.05, 0.05)
    q.el('q_manga', 7.10, 0.05, 0.05)
    q.el('q_cen_manga', 7.4, 0.05, 0.05)
    q.el('O/H_cen_manga', 0.15, 0.05, 0.05)
    q.comp = []
    #co = sy(0.0195442, 15)
    #co.el('HI', '<18')
    #co.el('SiII', 12.86, 0.30,0.54)
    #co.el('SiIII', 13.10,0.4,0.4)
    #co.el('CII', 13.5, 0.5, 1.3)
    #q.comp.append(co)
    co = sy(0.0196675, 15)
    co.el('HI', 18.83, 0.05, 0.05)
    co.el('CII', 15.3, 0.90, 0.50)
    co.el('SiII', 13.77, 0.12, 0.19)
    co.el('SiIII', 14.6, 0.90, 0.50)
    co.el('OI', 14.6, 0.9, 0.5)
    #co.el('SII', 14.0, 0.40, 1.6)
    #co.el('Me_cloudy', -1.3, 0.3, 0.3)
    #co.el('n/I_cloudy', -0.3, 0.3, 0.6)
    #co.el('q_cloudy', -3.1, 0.6, 0.3)
    #co.el('Htot', 20.0, 0.6, 0.3)
    #co.el('fHI_cloudy', -1.1, 0.3, 0.6)
    #co.el('Me_cloudy_4par', -1.0, 0.3, 0.3)
    #co.el('q_cloudy_4par', -3.1, 0.4, 0.3)
    #co.el('Htot_4par', 20.0, 0.5, 0.3)
    #co.el('depl_4par', 0.0, 0.2, 0.0, f='d')
    co.el('Me_cloudy_5par', -1.15, 0.2, 0.2)
    co.el('q_cloudy_5par', -2.1, 0.4, 0.5)
    co.el('Htot_5par', 21.1, 0.4, 0.6)
    co.el('depl_5par', 0.1, 0.3, 0.1, f='d')
    co.el('HI_5par', 18.83, 0.04, 0.06)
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    # add J2130-0025
    q = qso('J2130-0025B', 0.4901, 0.0196675)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-635629'
    q.zmanga = 0.01989
    q.vcorr = 9.20
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 9.56, f='l')
    q.el('sigMstar', 8.59, f='l')
    q.el('SFR', 1.26, f='d')
    #q.el('Dn4000', 1.316, f='d')
    q.el('Dn4000', 1.34, f='d')
    q.el('bpar', 66, f='d')
    q.el('b_rel', 38.7, f='d')
    q.el('drelow', 38.7, f='d')
    q.el('dreup', 48.1, f='d')
    q.el('Azi', 28,2,2, f='d')
    q.el('Azi_mean', 100, 20, 20, f='d')
    q.el('Azi_r_mean', 114, 15, 15, f='d')
    q.el('Azi_a_mean', 96, 13, 13, f='d')
    q.el('Azi_ra_mean', 101, 12, 12, f='d')
    q.el('Azi_h_mean', 90, 12, 12, f='d')
    q.el('Azi_NFW_6Re', 117, 13, 16, f='d')
    q.el('abszcen', 6.97, f='d')
    q.el('abszlow', -14, f='d')
    q.el('abszup', 28, f='d')
    q.el('r_dist', 37, 2, 2, f='d')
    q.el('h_dist', -18, 1.5, 1.5, f='d')
    q.el('gal_rad_vel', 54.39, 0, 0, f='d')
    q.el('sini', 0.89, f='d')
    q.el('Re', 1.75, f='d')
    q.el('sSFR', -9.45, f='l')
    q.el('sigSFR', -0.86, f='l')
    q.el('HI', 18.90, 0.10, 0.10)
    # q.el('SiII', 13.87, 0.15,0.13)
    q.el('SiII', 13.52, 0.17, 0.07)
    # q.el('SiIII', 15.70,0.60,0.80)
    q.el('SiIII', 13.79, 0.40, 0.2)
    q.el('SiIII/SiII', 0.27, 0.41, 0.26)
    q.el('Sitot', 13.98, 0.30, 0.12)
    q.el('SII', '<14.78')
    q.el('NV', 13.1, 0.4, 0.8)
    q.el('SII', 14.00, 0.4, 1.2)
    q.el('CII', 15.02, 0.46, 0.24)
    # q.el('zabs',, f='d')
    q.el('Me', 1.32, 0.60, 0.77)
    q.el('rSi', 0.30, 0.15, 0.14)
    q.el('Si/H', -4.77, 0.14, 0.13)
    q.el('ewHI', 1533, 64, 64, f='d')
    q.el('ewSiII', 275, 30, 30, f='d')
    q.el('ewSiIII', 447, 46, 46, f='d')
    q.el('v90HI', 450, 24, 24, f='d')
    q.el('v90SiII', 82, 13, 13, f='d')
    q.el('v90SiIII', 120, 11, 11, f='d')
    q.manga_absmag = [-1.54E+01, -1.59E+01, -1.70E+01, -1.80E+01, -1.85E+01, -1.87E+01, -1.89E+01]
    q.el('Me_cloudy', -1.2, 0.2, 0.2)
    q.el('n/I_cloudy', -0.34, 0.3, 0.4)
    q.el('q_cloudy', -3.1, 0.4, 0.3)
    q.el('Htot', 20.0, 0.4, 0.3)
    q.el('fHI_cloudy', -1.1, 0.3, 0.4)
    # ***************** fit with depletion
    co.el('Me_cloudy_4par', -1.0, 0.3, 0.3)
    co.el('q_cloudy_4par', -3.1, 0.4, 0.3)
    co.el('Htot_4par', 20.0, 0.5, 0.3)
    co.el('depl_4par', 0.0, 0.2, 0.0, f='d')
    # *****************
    q.el('deltaV', -69, 28, 28, f='d')
    q.el('V_mod_em_qso', -1e4, 30, 30, f='d')
    q.el('Mfuv', -1.45E+01, f='dec')
    q.el('Mhalo', 11.63, 0.3, 0.3)
    q.mangastatus = 'detection'
    q.el('[O/H]_manga', 0.06, 0.05, 0.05)
    q.el('O/H_manga', 8.75, 0.05, 0.05)
    q.el('q_re_manga', 6.99, 0.05, 0.05)
    q.el('q_manga', 7.10, 0.05, 0.05)
    q.el('q_cen_manga', 7.4, 0.05, 0.05)
    q.el('O/H_cen_manga', 0.15, 0.05, 0.05)
    q.comp = []
    co = sy(0.019667, 15)
    co.el('HI', 18.83, 0.05, 0.05)
    co.el('CII', 15.3, 0.90, 0.50)
    co.el('SiII', 13.77, 0.12, 0.19)
    co.el('SiIII', 14.6, 0.90, 0.50)
    co.el('OI', 14.6, 0.9, 0.5)
    # co.el('SII', 14.0, 0.40, 1.6)
    # co.el('Me_cloudy', -1.3, 0.3, 0.3)
    # co.el('n/I_cloudy', -0.3, 0.3, 0.6)
    # co.el('q_cloudy', -3.1, 0.6, 0.3)
    # co.el('Htot', 20.0, 0.6, 0.3)
    # co.el('fHI_cloudy', -1.1, 0.3, 0.6)
    # co.el('Me_cloudy_4par', -1.0, 0.3, 0.3)
    # co.el('q_cloudy_4par', -3.1, 0.4, 0.3)
    # co.el('Htot_4par', 20.0, 0.5, 0.3)
    # co.el('depl_4par', 0.0, 0.2, 0.0, f='d')
    co.el('Me_cloudy_5par', -1.15, 0.2, 0.2)
    co.el('q_cloudy_5par', -2.1, 0.4, 0.5)
    co.el('Htot_5par', 21.1, 0.4, 0.6)
    co.el('depl_5par', 0.1, 0.3, 0.1, f='d')
    co.el('HI_5par', 18.83, 0.04, 0.06)
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    #QSO.append(q)



    q = qso('J2130-0025_1', 0.4901,0.01989)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-635629'
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 9.56, f='l')
    q.el('sigMstar', 8.59, f='l')
    q.el('SFR', 1.26,  f='d')
    q.el('bpar', 66, f='d')
    q.el('b_rel', 38.7, f='d')
    q.el('Azi', 66, f='d')
    q.el('Azi_mean', 67, 15, 15, f='d')
    q.el('Azi_r_mean', 63, 5, 5, f='d')
    q.el('Azi_a_mean', 77, 10, 10, f='d')
    q.el('Azi_ra_mean', 66, 5, 5, f='d')
    q.el('sini', 0.89, f='d')
    q.el('Re', 1.75, f='d')
    q.el('sSFR', -9.45, f='l')
    q.el('sigSFR', -0.86, f='l')
    q.el('HI', '<14.70')
    q.el('SiII', '<13.0')
    q.el('SiIII', '<13.2')
    q.el('SII', '<14.8')
    q.el('NV', '<14.0')
    q.el('CII', '<14.40')
    #q.el('zabs', 0.02014, 0.0001, 0.0001, f='d')
    q.el('deltaV', -69, 28, 28, f='d')
    q.el('V_mod_em_qso', -1e4, 30, 30, f='d')
    q.el('Mfuv', -1.45E+01, f='dec')
    q.el('Mhalo', 11.63,0.3,0.3)
    #q.el('Me', 1.47, 0.41, 0.77)
    #q.el('rSi', 1.92, 0.43, 0.81)
    q.manga_absmag = [-1.45E+01, -1.52E+01, -1.67E+01, -1.79E+01, -1.86E+01, -1.90E+01, -1.93E+01]
    #q.galex_absmag = -15.8
    q.mangastatus = 'detection'
    #q.el('O/H_manga', 8.52, 0.03, 0.03)
    q.el('[O/H]_manga', 0.06, 0.01, 0.01)
    q.el('O/H_manga', 9.06,0.01,0.01)
    q.el('q_re_manga', 7.08, 0.05, 0.05)
    q.el('q_manga', 7.10, 0.05, 0.05)
    q.el('q_cen_manga', 7.15, 0.05, 0.05)
    q.el('O/H_cen_manga', 0.4, 0.05, 0.05)
    q.comp = []
    co = sy(0.02014, 0)
    co.el('HI', '<14.7')
    co.el('CII', '<14.4')
    co.el('SiII', '<13')
    co.el('SiIII', '<13.2')
    co.el('NV', '<14.0')
    co.el('SII', '<14.82')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    #QSO.append(q)



    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1709+3421
    q = qso('J1709+3421', 0.3143,0.09008)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-561034'
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 10.86, f='l')
    q.el('SFR', 0.031, f='d')
    #q.el('Dn4000', 1.849, f='d')
    q.el('Dn4000', 1.86, f='d')
    q.el('sigMstar', 8.82,f= 'l')
    q.el('bpar', 75, f='d')
    q.el('b_rel', 12.8, f='d')
    q.el('Re', 6.04, f='d')
    q.el('Azi', 44,3,3, f='d')
    q.el('Azi_mean', 120, 40, 40, f='d')
    q.el('Azi_r_mean', 133, 30,30, f='d')
    q.el('Azi_a_mean', 90, 15, 15, f='d')
    q.el('Azi_ra_mean', 98, 13, 13, f='d')
    q.el('Azi_h_mean', 118, 27, 18, f='d')
    q.el('Azi_NFW_6Re', 118, 21, 18, f='d')
    q.el('drelow', 16.5, f='d')
    q.el('dreup', 12.7, f='d')
    q.el('abszcen', 4.75, f='d')
    q.el('abszlow', -1.72, f='d')
    q.el('abszup', 10.58, f='d')
    #q.el('sini', 0.59, f='d')
    q.el('HI', '<13.52')
    q.el('SiII', '<12.55')
    q.el('SiIII', '<13.02')
    q.el('SII', '<14.41')
    q.el('NV', '<13.60')
    q.el('zabs', 0.088, 0.001, 0.001, f='d')
    q.el('ewHI', 124, 40, 40, f='d')
    q.el('ewSiII', '<30', f='d')
    q.el('ewSiIII', 42, 35, 35, f='d')
    q.manga_absmag = [-1.38E+01, -1.61E+01, -1.94E+01, -2.11E+01, -2.17E+01, -2.20E+01, -2.23E+01]
    q.el('Mfuv', -1.38E+01, f='dec')
    q.el('Mhalo', 13.96,0.3,0.3)
    q.mangastatus = 'non-detection'
    q.comp = []
    co = sy(0.08830, 3)
    co.el('HI', '<13.52')
    #co.el('CII', 16.10, 0.60, 1.0)
    co.el('SiII', '<12.55')
    co.el('SiIII', '<13.01')
    co.el('SII', '<14.41')
    co.el('NV', '<13.60')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J2106+0909
    q = qso('J2106+0909', 0.3896, 0.04423)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-113242'
    q.zmanga = 0.04372
    q.vcorr = 21.3
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 10.88, f='l')
    q.el('sigMstar', 8.93,  f='l')
    #q.el('Dn4000', 2.017, f='d')
    #q.el('SFR','<0.01')
    q.el('SFR', 0.0419, f='d')
    q.el('Dn4000', 2.14, f='d')
    q.el('bpar', 116, f='d')
    q.el('b_rel', 21.9,  f='d')
    q.el('Re', 5.48, f='d')
    q.el('drelow', 21.9, f='d')
    q.el('dreup', 24.7, f='d')
    q.el('Azi', 17,2,2, f='d')
    q.el('Azi_mean', 82, 50, 50, f='d')
    q.el('Azi_r_mean', 82, 27, 27, f='d')
    q.el('Azi_a_mean', 82, 20, 20, f='d')
    q.el('Azi_ra_mean',88, 13, 13, f='d')
    q.el('Azi_h_mean', 92, 22, 20, f='d')
    q.el('Azi_NFW_6Re', 84, 22, 20, f='d')
    q.el('abszcen', -0.9, f='d')
    q.el('abszlow', -11.4, f='d')
    q.el('abszup', 9.43, f='d')
    q.el('r_dist', 21.8, 0.5, 0.5, f='d')
    q.el('h_dist', -0.8, 3, 3, f='d')
    q.el('gal_rad_vel', -66, 0, 0, f='d')
    q.el('sini', 0.543, f='d')
    q.el('HI', 13.75,0.18,0.30)
    q.el('SiII', '<13.03')
    q.el('SiIII', '<13.20')
    q.el('Sitot', '<13.42')
    q.el('Sitot', '<13.4')
    q.el('SII', '<14.8')

    q.el('NI', '<13.7')
    q.el('NII', '<15.1')
    q.el('NV', '<13.7')
    q.el('OI', '<14.4')
    q.el('FeII', '<13.9')

    q.el('zabs', 0.04423, 0.0002, 0.0002, f='d')
    q.el('Me', '<4.16')
    q.el('Si/H', '<-0.33')
    q.el('deltaV', 143, 56, 56, f='d')
    q.el('ewHI', 180, 40, 40, f='d')
    q.el('ewSiII', 35, 30, 30, f='d')
    q.el('ewSiIII', 60, 50, 50, f='d')
    q.el('v90HI', 80, 15, 15, f='d')
    q.el('v90SiII', '<60', f='d')
    q.el('v90SiIII', '<100', f='d')
    q.el('V_mod_em_qso', 237, 30, 30, f='d')
    q.el('Mfuv', -1.60E+01, f='dec')
    q.el('Mhalo', 13.07,0.3,0.3)
    q.manga_absmag = [-1.60E+01, -1.64E+01, -1.91E+01, -2.10E+01, -2.18E+01, -2.22E+01, -2.24E+01]
    #q.el('fG_cloudy', -2.8, 0.28, 0.10)
    # ***************** fit with depletion
    #q.el('q_cloudy_4par', -2.9, 1.4, 1.9)
    # *****************
    q.mangastatus = 'detection'
    q.comp = []
    co = sy(0.04423, 2)
    co.el('HI', 13.75,0.18,0.30)
    co.el('SiII', '<13.03')
    co.el('SiIII', '<13.20')
    co.el('SII', '<15.03')
    co.el('NV', '<14.08')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0758+4219
    q = qso('J0758+4219', 0.2112, 0.0320)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-44487'
    q.zmanga=0.0316
    q.vcorr = 41.6
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 10.47, f='l')
    q.el('sigMstar', 8.41, f='l')
    q.el('SFR', 1.34, f= 'd')
    #q.el('Dn4000', 1.654, f='d')
    q.el('Dn4000', 1.74, f='d')
    q.el('bpar', 136, f='d')
    q.el('b_rel', 22.6, f='d')
    q.el('Re', 6.22, f='d')
    q.el('drelow', 22.6, f='d')
    q.el('dreup', 26.4, f='d')
    q.el('Azi', 6,2,2, f='d')
    q.el('Azi_mean', 96, 10, 10, f='d')
    q.el('Azi_r_mean', 98, 6, 6, f='d')
    q.el('Azi_a_mean', 95, 10, 10, f='d')
    q.el('Azi_ra_mean', 97, 7, 7, f='d')
    q.el('Azi_h_mean', 101, 6,8, f='d')
    q.el('Azi_NFW_6Re', 90+11, 6, 8, f='d')

    q.el('abszcen', 0.6, f='d')
    q.el('abszlow', -12, f='d')
    q.el('abszup', 13.6, f='d')
    q.el('r_dist', 23, 0.3, 0.3, f='d')
    q.el('h_dist', -3.5, 1.2, 1.2, f='d')
    q.el('gal_rad_vel', 247, 0, 0, f='d')
    q.el('sini', 0.80, f='d')
    q.el('sSFR', -10.34, f='l')
    q.el('sigSFR', -1.92, f='l')
    q.el('HI', 15.22,0.3,0.2)
    q.el('SiII', 13.20,0.05,0.06)
    q.el('SiIII', 13.36,0.05,0.06)
    #q.el('SiIII/SiII', -0.16,0.08,0.08)
    #q.el('Sitot', 13.58,0.04,0.04)
    q.el('SII', 14.66, 0.12,0.20)
    q.el('NI', 13.2, 0.2, 2.0)
    q.el('NII', 13.5, 0.4, 0.4)
    q.el('NV', '<13.5')
    q.el('OI', '<14')
    q.el('FeII', '<13.9')

    q.el('zabs', 0.0320, 0.0002, 0.0002,f= 'd')
    q.el('Me', 2.77, 0.16, 0.11)
    q.el('rSi',0.16,0.10,0.10)
    q.el('Si/H', -1.72, 0.16,0.11)
    q.el('deltaV', 122, 21, 21, f='d')
    q.el('V_mod_em_qso', -111, 10, 10, f='d')
    q.el('ewHI', 860, 14, 14, f='d')
    q.el('ewSiII', 182, 22, 22, f='d')
    q.el('ewSiIII', 331, 23, 23, f='d')
    q.el('v90HI', 168, 4, 4, f='d')
    q.el('v90SiII', 147, 17, 17, f='d')
    q.el('v90SiIII', 166, 20, 20, f='d')
    q.manga_absmag = [-1.64E+01, -1.67E+01, -1.87E+01, -2.01E+01, -2.07E+01, -2.11E+01, -2.14E+01]
    #q.el('Me_cloudy', 0.8, 0.1, 0.1)
    #q.el('n/I_cloudy', -1, 0.4, 0.4)
    #.el('q_cloudy', -2.37, 0.07, 0.07)
    #q.el('Htot', 17.7, 0.1, 0.1)
    # ***************** fit with depletion
    #q.el('Me_cloudy_4par', 0.8, 0.1, 0.1)
    #q.el('q_cloudy_4par', -2.4, 0.1, 0.1)
    #q.el('Htot_4par', 17.7, 0.1, 0.1)
    #q.el('depl_4par', 0.0, 0.1, 0.0,f='d')
    #q.el('nH_4par', -3.1, 0.1, 0.1)
    #q.el('fG_4par', 1, 0., 0.1)
    # ***************** cloudy fit HI + depletion
    q.el('Me_cloudy_5par', 0.77, 0.3, 0.3)
    q.el('q_cloudy_5par', -2.0, 0.3, 0.4)
    q.el('HI_5par', 15.3, 0.3, 0.2)
    q.el('Htot_5par', 18.05, 0.3, 0.4)
    q.el('depl_5par', 0.1, 0.5, 0.1, f='d')
    # *****************
    q.el('Mfuv', -1.64E+01, f='dec')
    q.el('Mhalo', 10.47,0.3,0.3)
    q.mangastatus = 'detection'
    #q.el('O/H_manga', 8.55, 0.03, 0.03)
    q.el('[O/H]_manga', 0.34, 0.02, 0.02)
    q.el('O/H_manga', 9.04, 0.02, 0.02)
    q.el('q_re_manga', 7.05, 0.05, 0.05)
    q.el('q_manga', 7.10, 0.04, 0.04)
    q.el('q_cen_manga', 7.2, 0.05, 0.05)
    q.el('O/H_cen_manga', 0.32, 0.05, 0.05)
    q.comp = []
    co = sy(0.03138, 3)
    co.el('HI', 14.17, 0.25, 0.14, b = (42,7,7))
    co.el('SiII', 12.3,0.2,0.9)
    co.el('SiIII', 11.9,1.8,0.6)
    co.el('SII', '<14.44') #, 0.20,0.20)
    co.el('NI','<13.8')
    co.el('NII', '<14.3')
    co.el('NV', '<13.6')
    co.el('OI', '<14')
    co.el('FeII', '<13.7')
    co.el('FeIII', '<14')
    co.el('ArI','<14')
    #co.el('Me_cloudy', 1.1, 0.6, 0.5)
    #co.el('n/I_cloudy', -1.3, 0.6, 0.6)
    #co.el('q_cloudy', -2.2, 0.6, 0.5)
    #co.el('Htot', 16.26, 1.2, 0.5)
    #co.el('fHI_cloudy', -2.25, 0.5, 1.2)
    #co.el('Me_cloudy_4par', 0.8, 0.5, 0.7)
    #co.el('q_cloudy_4par', -2.1, 0.6, 0.5)
    #co.el('Htot_4par', 16.3, 1.2, 0.5)
    #co.el('depl_4par', '<1', f='d')
    co.el('Me_cloudy_5par', 0.7, 0.6, 0.8)
    co.el('q_cloudy_5par', -2.4, 1.0, 0.4)
    co.el('Htot_5par', 16.9, 1.0, 0.7)
    co.el('HI_5par', 14.2, 0.3, 0.2)
    co.el('depl_5par', '<1', f='d')
    q.comp.append(co)
    co = sy(0.03202, 3)
    co.el('HI', 14.84, 0.22, 0.21,b=(40.6,8,6))
    co.el('SiII', 12.69, 0.18, 0.20)
    co.el('SiIII', 13.15, 0.06, 0.06)
    co.el('SII', 14.16, 0.20, 0.50)
    co.el('NI', '<13.7')
    co.el('NII', '<14.')
    co.el('NV', '<13.3')
    co.el('OI', '<13.9')
    co.el('FeII', '<14')
    co.el('FeIII', '<14.3')
    co.el('ArI', '<14.3')
    #co.el('Me_cloudy', 0.33, 0.2, 0.2)
    #co.el('n/I_cloudy', -2.35, 1.3, 1.3)
    #co.el('q_cloudy', -2.2, 0.1, 0.1)
    #co.el('Htot', 17.9, 0.15, 0.15)
    #co.el('fHI_cloudy', -2.81, 0.14, 0.14)
    #co.el('Me_cloudy_4par', 0.3, 0.2, 0.2)
    #co.el('q_cloudy_4par', -2.2, 0.1, 0.1)
    #co.el('Htot_4par', 17.9, 0.2, 0.2)
    #co.el('depl_4par', '<1', f='d')
    co.el('Me_cloudy_5par', 0.6, 0.3, 0.3)
    co.el('q_cloudy_5par', -2.0, 0.2, 0.3)
    co.el('Htot_5par', 17.8, 0.2, 0.2)
    co.el('HI_5par', 14.2, 0.3, 0.1)
    co.el('depl_5par', 0,0.4,0, f='d')
    q.comp.append(co)
    co = sy(0.03226, 3)
    co.el('HI', 14.82, 0.45, 0.28,b=(27,8,4))
    co.el('SiII', 13.01,0.08,0.09)
    co.el('SiIII', 12.82,0.17,0.15)
    co.el('SII', 13.80,0.50,0.70)
    co.el('NI', '<13.6')
    co.el('NII', '<14.1')
    co.el('NV', '<13.3')
    co.el('OI', '<13.9')
    co.el('FeII', '<13.8')
    co.el('FeIII', '<14.4')
    co.el('ArI', '<14.3')
    #co.el('Me_cloudy', 0.7, 0.2, 0.2)
    #co.el('n/I_cloudy', -1.09, 0.1, 0.1)
    #co.el('q_cloudy', -2.2, 0.4, 0.1)
    #co.el('Htot', 17.51, 0.1, 0.1)
    #co.el('fHI_cloudy', -2.62, 0.1, 0.1)
    #co.el('Me_cloudy_4par', 0.7, 0.2, 0.2)
    #co.el('q_cloudy_4par', -2.2, 0.4, 0.1)
    #co.el('Htot_4par', 17.5, 0.2, 0.2)
    #co.el('depl_4par', 0,0.5,0, f='d')
    co.el('Me_cloudy_5par', 0.86, 0.35, 0.4)
    co.el('q_cloudy_5par', -2.1, 0.6, 0.6)
    co.el('Htot_5par', 17.6, 0.3, 0.3)
    co.el('HI_5par', 15.0, 0.4, 0.3)
    co.el('depl_5par', 0.2, 0.4, 0.2, f='d')
    q.comp.append(co)
    co = sy(0.032629, 4)
    co.el('HI', 13.07, 0.29, 0.20,b=(17,50,2))
    co.el('SiIII', '<12.3')
    #co.el('Me_cloudy_5par', 0.1, 1, -1)
    #co.el('q_cloudy_5par', -2.1, 0.6, 0.6)
    #co.el('Htot_5par', 14, 0.3, 0.3)
    #co.el('HI_5par', 15.0, 0.4, 0.3)
    #co.el('depl_5par', 0.2, 0.4, 0.2, f='d')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1629+4007
    q = qso('J1629+4007', 0.4901, 0.02588)
    q.telescope = 'HST'
    q.year = 2021
    q.mangaID = '1-564490'
    # q.progID.append('085.A-0569(B)')
    q.el('Mstar', 10.15, f='l')
    q.el('sigMstar', 8.16, f='l')
    q.el('SFR', 2.60, f= 'd')
    #q.el('Dn4000', 1.313, f='d')
    q.el('Dn4000', 1.27, f='d')
    q.el('bpar', 132, f='d')
    q.el('b_rel', 23.8,  f='d')
    q.el('Re', 5.71, f='d')
    q.el('sSFR', -9.73, f='l')
    q.el('sigSFR', -1.56, f='l')
    q.el('Azi', 56,2,2, f='d')
    q.el('Azi_mean', 103, 33, 3, f='d')
    q.el('Azi_r_mean', 111, 23, 23, f='d')
    q.el('Azi_a_mean', 90, 17, 17, f='d')
    q.el('Azi_ra_mean', 96, 13, 13, f='d')
    q.el('Azi_h_mean', 60, 25, 22, f='d')
    q.el('Azi_NFW_6Re', 60, 24, 21, f='d')
    q.el('drelow', 23.8, f='d')
    q.el('dreup', 30.2, f='d')
    q.el('abszcen', -3.4, f='d')
    q.el('abszlow', -18.5, f='d')
    q.el('abszup', 10.5, f='d')
    q.el('sini', 0.72, f='d')
    q.el('HI', '<13.2')
    q.el('SiII', '<12.5')
    q.el('SiIII', '<12.6')
    q.el('SII', '<14.3')
    q.el('NV', '<13.6')
    q.el('zabs', 0.0238, 0.001, 0.001, f='d')
    q.el('ewHI', 70, 20, 20, f='d')
    q.el('ewSiII', 30, 12, 12, f='d')
    q.el('ewSiIII', 59, 22, 2, f='d')
    #q.el('v90HI', '<200', f='d')
    #q.el('v90SiII', '<100', f='d')
    q.el('v90SiIII', 68, 70, 70, f='d')
    q.manga_absmag = [-1.70E+01, -1.76E+01, -1.88E+01, -1.99E+01, -2.03E+01, -2.06E+01, -2.08E+01]
    q.mangastatus = 'non-detection'
    q.el('Mfuv', -1.70E+01, f='dec')
    q.el('Mhalo', 12.19,0.3,0.3)
    #q.el('O/H_manga', 8.53, 0.03, 0.03)
    q.el('O/H_manga', 9.08, 0.02, 0.02)
    q.el('q_re_manga', 7.11, 0.05, 0.05)
    q.el('q_manga', 7.15, 0.04, 0.04)
    q.el('q_cen_manga', 7.4, 0.05, 0.05)
    q.el('O/H_cen_manga', 0.45, 0.05, 0.05)
    q.comp = []
    co = sy(0.02388, 0)
    co.el('HI', '<13.2')
    co.el('SiII', '<12.5')
    co.el('SiIII', '<12.6')
    co.el('SII', '<14.3')
    co.el('NV', '<13.6')
    q.comp.append(co)
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)


    return QSO


def load_QSO_pairs():
    global sy
    QSO = sample()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    q = qso('J0902+1414', 0.98, 0.0505)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.coord = ['J2000', '09:02:50.47, 14:14:08.29']
    #q.el('Mstar', 9.65, f='l')
    q.el('SFR', 0.03,f='d')
    q.el('bpar', 3.6, f='d')
    q.el('b_rel', 2.6, f='d')
    q.el('Mstar',9.265, f='l')
    q.el('Re', 1.38, f='d') #in kpc #Staka2015
    #q.assgalname = 'J090250.24+141409.9'
    #q.assgalz = 0.05062005
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    #QSO.append(q)

    q = qso('J1005+5302', 0.56, 0.1358)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.coord = ['J2000', '10:05:14.21, 53:02:40.04']
    q.el('Mstar', 9.6, f='l')
    q.el('SFR', 0.19, f='d')
    q.el('bpar', 3.6, f='d')
    q.el('HI', 20.08,0.08,0.07 )
    q.el('Re', 4.89, f='d')  # in kpc #ref York2012
    q.el('b_rel', 0.73, f='d')
    #q.el('Mstar', 10.17, f='l')
    #q.el('Re', 5.92, f='d')  # in kpc
    #q.el('b_rel', 0.60, f='d')
    #q.assgalname = 'J100559.42+524201.1'
    #q.assgalz = 0.13576391
    #q.galcoord([151.49767609, 52.70033269])
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J1010-0100', 0.23, 0.0213)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.coord = ['J2000', '10:10:15.74, −01:00:38.11']
    #q.el('Mstar', 8.88, f='l')
    q.el('SFR', 0.04, f='d')
    q.el('bpar', 1.2, f='d')
    q.el('HI', 21.38,0.07,0.07 )
    q.el('b_rel', 0.56, f='d')
    q.el('Mstar', 8.88, f='l')
    q.el('Re', 1.38, f='d')  # in kpc
    q.el('b_rel', 0.86, f='d')
    #q.el('Re', 1.59, f='d') # in kpc
    #q.el('b_rel', 0.75, f='d')
    #q.assgalname = 'J101015.58-010038.7'
    #q.assgalz = 0.0213
    #q.galcoord([152.564913, -1.01090325])
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J1130+6026', 0.37, 0.0604)
    q.telescope = 'HST'
    q.year = 2021
    q.iauname = 'J112956.10+602916.9'
    q.ref = 'Kulkarni2022'
    q.coord = ['J2000', '11:30:02.99, 60:26:28.62']
    q.el('Mstar', 7.31, f='l')
    q.el('SFR', 0.02, f='d')
    q.el('bpar', 2.0, f='d')
    q.el('HI', 20.57,0.04,0.04)
    q.el('Re', 2.81, f='d')  # in kpc #2.338'' ref:Straka13
    q.el('b_rel',0.71,f='d')
    #q.el('Mstar', 10.6, f='l')
    #q.el('Re', 6.02, f='d')
    #q.el('b_rel', 0.33, f='d')
    #q.assgalname = 'J112956.10+602916.9'
    #q.assgalz = 0.06033659
    #q.galcoord([172.48377423, 60.4880428])
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J1135+2414', 1.45, 0.0343)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.el('Mstar', 9.41, f='l')
    q.el('SFR', 0.01, f='d')
    q.el('bpar', 3.9, f='d')
    q.el('HI', 20.57,0.03,0.03)
    q.el('Re', 0.8, f='d')
    q.el('b_rel', 4.87, f='d')
    #q.assgalname = 'J113555.66+241438GAL'
    #q.assgalz = 0.0343
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J1328+2159', 0.33, 0.1352)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.el('Mstar', 9.06, f='l')
    q.el('SFR', 0.16, f='d')
    q.el('bpar', 5.7, f='d')
    q.el('HI', 21.01,0.11,0.11)
    q.el('Re', 2.75, f='d') #kpc Straka2015
    q.el('b_rel', 2.07, f='d') #kpc Straka2015
    #q.assgalname = 'J132824.33+215919GAL'
    #q.assgalz = 0.1352
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J1452+5443', 1.52, 0.1026)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.el('Mstar', 7.96, f='l')
    q.el('SFR', 0.25, f='d')
    q.el('bpar', 3.6, f='d')
    q.el('HI', 20.66,0.04,0.03)
    q.el('Re', 3.78, f='d')  # kpc ref:York2012
    q.el('b_rel', 0.95, f='d')  #
    #q.assgalname = 'J145240+544345 GAL'
    #q.assgalz = 0.1026
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J1457+5321', 1.2, 0.0660)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.el('Mstar', 9.28, f='l')
    q.el('SFR', 0.04, f='d')
    q.el('bpar', 4.2, f='d')
    q.el('Re',1.82,f='d') #kpc ref:Straka2015
    q.el('b_rel', 2.3, f='d')
    #q.assgalname = 'J145719.00+532159GAL'
    #q.assgalz = 0.0660
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J1659+6202', 0.23, 0.1103)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.el('Mstar', 9.78, f='l')
    q.el('SFR', 0.26, f='d')
    q.el('bpar', 7.2, f='d')
    q.el('HI', 21.20,0.07,0.05 )
    q.el('Re', 3.20)#kpc
    q.el('b_rel', 0.44, f='d')  # kpc
    q.assgalname = 'J165958.94+620218GAL'
    q.assgalz = 0.1103
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('J2117-0026', 1.137, 0.0580)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.el('Mstar', 9.64, f='l')
    q.el('SFR', 0.04, f='d')
    q.el('bpar', 5.7, f='d')
    q.el('HI', 21.35,0.07,0.06 )
    q.el('b_rel', 0.79, f='d')
    q.el('Re', 7.83, f='d') #kpc
    #q.el('Re', 7.14, f='d')
    #q.assgalname = 'WISEA J211701.27-002632.7'
    #q.assgalz = 0.0580
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)
####################################################################################################
# COS-HALOS Tumlinson 2013
    if 1:
        q = qso('J0226+0015', 0.227, 0.227)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.8, f='l')
        q.el('SFR', '<0.09', f='d')
        q.el('bpar', 80, f='d')
        q.el('bvir', 303, f='d')
        q.el('HI', 14.2, 0.03, 0.03)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0401-0540', 0.219, 0.219)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.2, f='l')
        q.el('SFR', 1.14, f='d')
        q.el('bpar', 85, f='d')
        q.el('bvir', 200, f='d')
        q.el('HI', 15.7, 0.8, 0.1)
        q.el('SiIII', 12.88, 0.06, 0.06)
        q.el('q_cloudy', -1.5, 0.2, 0.3)
        q.el('Me_cloudy', -0.8, 0.4, 0.4) #Werk2014
        q.el('Htot_cloudy', 19.6, 0.3, 0.3)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0803+4332', 0.25, 0.25)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11.3, f='l')
        q.el('SFR', '<0.21', f='d')
        q.el('bpar', 77, f='d')
        q.el('bvir', 581, f='d')
        q.el('HI', 14.78, 0.05, 0.05)
        q.el('q_cloudy', -1.9, 0.9)
        q.el('Me_cloudy', -0.8, 0.8) #Werk2014
        q.el('Htot_cloudy', 18.5, 1.0)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0910+1014A', 0.14, 0.14)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.6, f='l')
        q.el('SFR', 14.12, f='d')
        q.el('bpar', 112, f='d')
        q.el('bvir', 279, f='d')
        q.el('HI', 16.5, 2, 0.8)
        q.el('SiIII', '>13.28')
        q.el('q_cloudy', -2.7, 0.7)
        q.el('Me_cloudy', -0.8, 0.6) #Werk2014
        q.el('Htot_cloudy', 19.2, 0.8)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0910+1014B', 0.26, 0.26)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11.5, f='l')
        q.el('SFR', '<0.3', f='d')
        q.el('bpar', 139, f='d')
        q.el('bvir', 716, f='d')
        q.el('HI', 17., 0.5, 1.2)
        q.el('CII', 14.08, 0.13, 0.13)
        q.el('SiII', 13.22, 0.13, 0.13)
        q.el('q_cloudy', -3.7, 0.3)
        q.el('Me_cloudy', -0.9, 0.9) #Werk2014
        q.el('Htot_cloudy', 18.7, 0.3)

        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0914+2823', 0.24, 0.24)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 9.8, f='l')
        q.el('SFR', 2.83, f='d')
        q.el('bpar', 104, f='d')
        q.el('bvir', 169, f='d')
        q.el('HI', 15.45, 0.05, 0.05)
        q.el('SiIII', 12.63, 0.11, 0.11)
        q.el('q_cloudy', -2.4, 0.6)
        q.el('Me_cloudy', -0.9, 0.45) #Werk2014
        q.el('Htot_cloudy', 18.4, 0.7)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0925+4004', 0.24, 0.24)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11.3, f='l')
        q.el('SFR', '<0.57', f='d')
        q.el('bpar', 83, f='d')
        q.el('bvir', 569, f='d')
        q.el('HI', 19.55, 0.15, 0.15)
        q.el('SiII', '>14.55')
        q.el('SiIII', '>13.75')
        q.el('q_cloudy', -3.4, 0.4)
        q.el('Me_cloudy', -0.7, 0.2) #Werk2014
        q.el('Htot_cloudy', 20.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0928+6025', 0.15, 0.15)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.8, f='l')
        q.el('SFR', '<0.04', f='d')
        q.el('bpar', 93, f='d')
        q.el('bvir', 317, f='d')
        q.el('HI', 19.4, 0.15, 0.15)
        q.el('SiIII', '>13.7')
        q.el('q_cloudy', -3.2, 0.4)
        q.el('Me_cloudy', -0.4, 0.2) #Werk2014
        q.el('Htot_cloudy', 20.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0935+0204', 0.26, 0.26)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11, f='l')
        q.el('SFR', '<0.10', f='d')
        q.el('bpar', 113, f='d')
        q.el('bvir', 365, f='d')
        q.el('HI', '<12.6')
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0943+0531A', 0.35, 0.23)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.8, f='l')
        q.el('SFR', 4.52, f='d')
        q.el('bpar', 121, f='d')
        q.el('bvir', 355, f='d')
        q.el('HI', 15.3, 0.05, 0.05)
        q.el('SiIII', 12.89, 0.10)
        q.el('q_cloudy', -2.2, 0.7)
        q.el('Me_cloudy', -0.5, 0.5) #Werk2014
        q.el('Htot_cloudy', 18.5, 1,0.8)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0943+0531B', 0.35, 0.35)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 9.6, f='l')
        q.el('SFR', 0.47, f='d')
        q.el('bpar', 96, f='d')
        q.el('bvir', 141, f='d')
        q.el('HI', 16.3, 0.05, 0.05)
        q.el('CII', 14.42, 0.10)
        q.el('q_cloudy', -2.0, 1, 0.5)
        q.el('Me_cloudy', -1.6, 0.2)  # Werk2014
        q.el('Htot_cloudy', 20.2, 1, 0.8)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J0950+4831', 0.215, 0.215)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11.2, f='l')
        q.el('SFR', '<0.3', f='d')
        q.el('bpar', 93, f='d')
        q.el('bvir', 511, f='d')
        #q.el('HI', 16.2, 0.1, 0.1)
        q.el('HI', 18.5, 0.0, 1) #Werk
        q.el('SiII', 14.01,0.04,0.04)
        q.el('SiIII', '>13.7')
        q.el('q_cloudy', -3.0, 0.3)
        q.el('Me_cloudy', -0.7, 0.7) #Werk2014
        q.el('Htot_cloudy', 20.1, 0.3)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1009+0713A', 0.22, 0.22)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 9.9, f='l')
        q.el('SFR', 4.58, f='d')
        q.el('bpar', 63, f='d')
        q.el('bvir', 174, f='d')
        #q.el('HI', 15.25, 0.05, 0.05)
        q.el('HI', 16.2, 2, 0.1) #Werk
        q.el('CII', 14.37,0.05,0.05)
        q.el('SiII', '<13.6')
        q.el('SiIII', '>13.3')
        q.el('q_cloudy', -2.5, 0.5)
        q.el('Me_cloudy', -1, 1) #Werk2014
        q.el('Htot_cloudy', 19, 0.6)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1009+0713B', 0.35, 0.35)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.3, f='l')
        q.el('SFR', 3.04, f='d')
        q.el('bpar', 46, f='d')
        q.el('bvir', 189, f='d')
        q.el('HI', 18.5,0.5,0.5)
        q.el('SiII', 14.36,0.07,0.07)
        q.el('SiII', '>13.92')
        q.el('q_cloudy', -2.7, 0.3)
        q.el('Me_cloudy', -0.6,0.2)  # Werk2014
        q.el('Htot_cloudy', 20.2, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1016+4706A', 0.25, 0.25)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.2, f='l')
        q.el('SFR', 0.64, f='d')
        q.el('bpar', 23, f='d')
        q.el('bvir', 202, f='d')
        q.el('HI', 16.6, 2, 0.05)
        q.el('CII', 14.68,0.03,0.03)
        q.el('SiII', 13.81,0.05,0.05)
        q.el('SiIII', '>13.87')
        q.el('q_cloudy', -2.8, 0.1)
        q.el('Me_cloudy', '<-0.1')  # Werk2014
        q.el('Htot_cloudy', 19.0, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1016+4706B', 0.16, 0.16)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.5, f='l')
        q.el('SFR', 1.37, f='d')
        q.el('bpar', 45, f='d')
        q.el('bvir', 251, f='d')
        #q.el('HI', 15.5, 0.1, 0.1)
        q.el('HI', 16.4, 1.8, 0.1)
        q.el('CII', 14.27,0.07,0.07)
        q.el('SiII', 13.68,0.06,0.046)
        q.el('SiIII', '>13.74')
        q.el('q_cloudy', -3.2, 0.3)
        q.el('Me_cloudy', '<-0.1')  # Werk2014
        q.el('Htot_cloudy', 19.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1112+3539', 0.24, 0.24)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.3, f='l')
        q.el('SFR', 5.68, f='d')
        q.el('bpar', 53, f='d')
        q.el('bvir', 214, f='d')
        q.el('HI', 15.8, 1.7, 0.1)
        q.el('SiIII', 12.97,0.11,0.11)
        q.el('SiII', '<12.9')
        q.el('CII', '<13.91')
        q.el('q_cloudy', -3.2, 0.3)
        q.el('Me_cloudy', '<-0.3')  # Werk2014
        q.el('Htot_cloudy', 17.8, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1133+0327A', 0.23, 0.23)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11.2, f='l')
        q.el('SFR', '<0.3', f='d')
        q.el('bpar', 17, f='d')
        q.el('bvir',515, f='d')
        q.el('HI', 18.6, 0.1, 0.1)
        q.el('CII', '>14.81')
        q.el('SiII', 13.70, 0.06, 0.06)
        q.el('SiIII', '>13.6')
        q.el('q_cloudy', -3.4, 0.2)
        q.el('Me_cloudy', -1,0.2)  # Werk2014
        q.el('Htot_cloudy', 19.7, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1133+0327B', 0.15, 0.15)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.1, f='l')
        q.el('SFR', 1.83, f='d')
        q.el('bpar', 55, f='d')
        q.el('bvir', 206, f='d')
        q.el('HI', 16.1, 2, 1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1157-0022', 0.16, 0.16)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.9, f='l')
        q.el('SFR', '<0.1', f='d')
        q.el('bpar', 19, f='d')
        q.el('bvir', 334, f='d')
        q.el('HI', 15.12, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1220+3853', 0.27, 0.27)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.8, f='l')
        q.el('SFR', '<0.13', f='d')
        q.el('bpar', 156, f='d')
        q.el('bvir', 279, f='d')
        q.el('HI', 15.8, 0.1, 0.1)
        q.el('CII', '<13.6')
        q.el('SiII', '<13.0')
        q.el('SiIII', '>13.5')
        q.el('q_cloudy', -2.2, 0.3)
        q.el('Me_cloudy', -0.6,0.2)  # Werk2014
        q.el('Htot_cloudy', 19.1, 0.3)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1233+4758', 0.22, 0.22)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.8, f='l')
        q.el('SFR', 4.4, f='d')
        q.el('bpar', 135, f='d')
        q.el('bvir', 295, f='d')
        #q.el('HI', 16.3, 0.1, 0.1)
        q.el('HI', 16.7, 1.5, 0.5)
        q.el('SiII', 13.45,0.05,0.05)
        q.el('CII', '>14.5')
        q.el('SiIII', '>13.4')
        q.el('q_cloudy', -3.1, 0.2)
        q.el('Me_cloudy', '<-0.4')  # Werk2014
        q.el('Htot_cloudy', 19.0, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1233-0031', 0.31, 0.3)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.6, f='l')
        q.el('SFR', 3.42, f='d')
        q.el('bpar',29, f='d')
        q.el('bvir', 230, f='d')
        q.el('HI', 15.6, 0.1, 0.1)
        q.el('SiIII', 13.01,0.1,0.1)
        q.el('CII', '<13.6')
        q.el('SIII', '<13.3')
        q.el('Me_cloudy', -0.7, 0.7)  # Werk2014
        q.el('Htot_cloudy', 19.3, 0.7)
        q.el('q_cloudy', -1.9, 0.5)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1241+5721A', 0.22, 0.20)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.2, f='l')
        q.el('SFR', 4.32, f='d')
        q.el('bpar', 21, f='d')
        q.el('bvir', 205, f='d')
        q.el('CII', '>15.1')
        q.el('HI', 17.9, 0.6, 1)
        q.el('SiII', 14.3,0.06,0.06)
        q.el('SiIII', '>13.9')
        q.el('q_cloudy', -3.1, 0.5)
        q.el('Me_cloudy', -0.6,0.6)  # Werk2014
        q.el('Htot_cloudy', 19.9, 0.5)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1241+5721B', 0.22, 0.22)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.1, f='l')
        q.el('SFR', 1.06, f='d')
        q.el('bpar', 93, f='d')
        q.el('bvir', 192, f='d')
        q.el('HI', 15.3, 0.1, 0.1)
        q.el('CII', '<13.4')
        q.el('SiII', 13.45, 0.06, 0.06)
        q.el('SiIII', '>13.4')
        q.el('q_cloudy', -3.1, 0.2)
        q.el('Me_cloudy', 0.0, 0.2)  # Werk2014
        q.el('Htot_cloudy', 17.4, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1245+3356', 0.19, 0.19)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 9.9, f='l')
        q.el('SFR', 1.05, f='d')
        q.el('bpar',113, f='d')
        q.el('bvir', 178, f='d')
        q.el('HI', 14.7, 0.1, 0.1)
        q.el('CII', '<13.5')
        q.el('SiII', '<13.2')
        q.el('SiIII', '<12.3')
        q.el('q_cloudy', -1.9, 0.6)
        q.el('Me_cloudy', -1.5, 0.4)  # Werk2014
        q.el('Htot_cloudy', 18.7, 0.6)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1322+4645', 0.21, 0.21)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.8, f='l')
        q.el('SFR', 0.6, f='d')
        q.el('bpar', 38, f='d')
        q.el('bvir', 303, f='d')
        q.el('HI', 16.5, 1.8, 0.2)
        q.el('CII', 14.34,0.05,0.05)
        q.el('SiII', 13.21,0.06,0.06)
        q.el('SiIII', '>13.6')
        q.el('q_cloudy', -2.6, 0.4)
        q.el('Me_cloudy', -1.0, 0.8)  # Werk2014
        q.el('Htot_cloudy', 19, 0.5)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1330+2813', 0.19, 0.19)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.3, f='l')
        q.el('SFR', 1.99, f='d')
        q.el('bpar', 89, f='d')
        q.el('bvir', 225, f='d')
        #q.el('HI', 15.9, 0.1, 0.1)
        q.el('HI', 16.6, 1.9, 0.1)
        q.el('CII', 14.37,0.05,0.05)
        q.el('SiII', 13.49,0.04,0.04)
        q.el('SiIII', 13.19,0.04,0.04)
        q.el('q_cloudy', -3.2, 0.5)
        q.el('Me_cloudy', '<-0.2')  # Werk2014
        q.el('Htot_cloudy', 18.7, 0.5)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1342-0053A', 0.23, 0.23)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11, f='l')
        q.el('SFR', 6.04, f='d')
        q.el('bpar', 34, f='d')
        q.el('bvir', 345, f='d')
        q.el('HI', 19.0, 0.5, 0.7)
        q.el('CII', '>15.1')
        q.el('SiII', '>14.5')
        q.el('SiIII', '>14')
        q.el('q_cloudy', -3.2, 0.3)
        q.el('Me_cloudy', -0.3,0.3)  # Werk2014
        q.el('Htot_cloudy', 19.8, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1342-0053B', 0.2, 0.2)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.5, f='l')
        q.el('SFR', '<0.3', f='d')
        q.el('bpar', 31, f='d')
        q.el('bvir', 247, f='d')
        q.el('HI', '<12.4')
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1419+4207', 0.18, 0.18)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.6, f='l')
        q.el('SFR', 11.3, f='d')
        q.el('bpar', 90, f='d')
        q.el('bvir', 272, f='d')
        #q.el('HI', 15.4, 0.1, 0.1)
        q.el('HI', 17., 1, 1.6)
        q.el('CII', 14.06,0.08,0.08)
        q.el('SiII', 13.28,0.07,0.07)
        q.el('SiIII', '>13.3')
        q.el('q_cloudy', -3.8, 0.5)
        q.el('Me_cloudy', -1,1)  # Werk2014
        q.el('Htot_cloudy', 18.5, 0.5)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1435+3604A', 0.26, 0.26)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.4, f='l')
        q.el('SFR', 5.56, f='d')
        q.el('bpar', 84, f='d')
        q.el('bvir', 218, f='d')
        q.el('HI', 15.2, 0.1, 0.1)
        q.el('CII', '<13.6')
        q.el('SiII', '<12.7')
        q.el('SiIII', 12.8,0.1)
        q.el('q_cloudy', -2.8, 0.3)
        q.el('Me_cloudy', -0.4, 0.2)  # Werk2014
        q.el('Htot_cloudy', 18.0, 0.4)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1435+3604B', 0.20, 0.2)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11.1, f='l')
        q.el('SFR', 18.9, f='d')
        q.el('bpar', 37, f='d')
        q.el('bvir', 433, f='d')
        q.el('HI', 19.8, 0.1, 0.1)
        q.el('CII', '>14.3')
        q.el('SiII', '>14.2')
        q.el('SiIII', '>13.4')
        q.el('q_cloudy', -3.3, 0.3)
        q.el('Me_cloudy', -1.2, 0.2)  # Werk2014
        q.el('Htot_cloudy', 20.3, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1437+5045', 0.24, 0.24)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.2, f='l')
        q.el('SFR', 4.3, f='d')
        q.el('bpar', 144, f='d')
        q.el('bvir', 196, f='d')
        q.el('HI', 14.5, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1445+3428', 0.21, 0.21)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.4, f='l')
        q.el('SFR', 2.6, f='d')
        q.el('bpar', 113, f='d')
        q.el('bvir', 230, f='d')
        q.el('HI', 15.1, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1514+3619', 0.21, 0.21)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 9.7, f='l')
        q.el('SFR', 1.9, f='d')
        q.el('bpar', 47, f='d')
        q.el('bvir', 164, f='d')
        q.el('SiII', '<13.2')
        q.el('HI', 16.6, 1.9, 1)
        q.el('SiIII', 13.26,0.06,0.06)
        q.el('q_cloudy', -3.5, 0.5)
        q.el('Me_cloudy', '<-0.5')  # Werk2014
        q.el('Htot_cloudy', 18.4, 0.5)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1550+4001', 0.31, 0.31)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11.4, f='l')
        q.el('SFR', '<0.16', f='d')
        q.el('bpar', 106, f='d')
        q.el('bvir', 578, f='d')
        q.el('HI', 16.5, 0.1, 0.1)
        q.el('CII', 14.05,0.12,0.12)
        q.el('SiIII', 13.39,0.04,0.04)
        q.el('q_cloudy', -2.7, 0.3)
        q.el('Me_cloudy', -0.8, 0.2)  # Werk2014
        q.el('Htot_cloudy', 19.2, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1550+4001B', 0.32, 0.32)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.9, f='l')
        q.el('SFR', 7.42, f='d')
        q.el('bpar', 151, f='d')
        q.el('bvir', 313, f='d')
        q.el('HI', 13.8, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1555+3628', 0.18, 0.18)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.5, f='l')
        q.el('SFR', 4.2, f='d')
        q.el('bpar', 32, f='d')
        q.el('bvir', 254, f='d')
        #q.el('HI', 15.7, 0.1, 0.1)
        q.el('HI', 17.2, 1, 1)
        q.el('CII', 14.5,0.05,0.05)
        q.el('SiII', 13.41,0.05,0.05)
        q.el('SiII', '>13.5')
        q.el('q_cloudy', -3.0, 0.4)
        q.el('Me_cloudy', -0.8, 0.8)  # Werk2014
        q.el('Htot_cloudy', 19.5, 0.5)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1617+0638', 0.15, 0.15)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 11.5, f='l')
        q.el('SFR', '<0.2', f='d')
        q.el('bpar', 102, f='d')
        q.el('bvir', 912, f='d')
        q.el('HI', '<13.1')
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('1619+3342', 0.14, 0.14)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.1, f='l')
        q.el('SFR', 1.33, f='d')
        q.el('bpar', 98, f='d')
        q.el('bvir', 211, f='d')
        q.el('HI', 14.9, 0.1, 0.1)
        q.el('CII', 14.3, 0.1, 0.1)
        q.el('SiII', '<13.4')
        q.el('q_cloudy', -2.0, 0.4)
        q.el('Me_cloudy', '<-0.1')  # Werk2014
        q.el('Htot_cloudy', 18.4, 0.5)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('2345-0059', 0.25, 0.25)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Tumlinson2013'
        q.el('Mstar', 10.9, f='l')
        q.el('SFR', '<0.14', f='d')
        q.el('bpar', 47, f='d')
        q.el('bvir', 304, f='d')
        q.el('HI', 16, 0.1, 0.1)
        q.el('CII', 14.10,0.12,0.12)
        q.el('SiII', '<12.7')
        q.el('SiIII', 13.04,0.06,0.06)
        q.el('q_cloudy', -2.4, 0.2)
        q.el('Me_cloudy', -0.3, 0.2)  # Werk2014
        q.el('Htot_cloudy', 19.0, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)
    ################################## Lehner et al 2018  q = qso('J0226+0015', 0.227, 0.227)
    if 1:
        q = qso('J004222.29-103743.8', 1.0, 0.31)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.6, 0.1, 0.1)
        q.el('SiIII', 13.07, 0.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J004705.89+031954.9', 1.0, 0.44)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.55, 0.1, 0.1)
        q.el('CII', 13.36, 0.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J011013.14-021952.8', 1.0, 0.22)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.82, 0.1, 0.1)
        q.el('CII', 13.84, 0.1, 0.1)
        q.el('SiII', 13.00, 0.1, 0.1)
        q.el('SiIII', 12.84, 0.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J011623.04+142940.5', 1.0, 0.33)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 16, 0.1, 0.1)
        q.el('SiIII', 13.05, 0.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J011935.69-282131', 1.0, 0.34)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.67, 0.1, 0.1)
        q.el('SiIII', 12.55, 0.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J012236.76-284321.3', 1.0, 0.36)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.39, 0.1, 0.1)
        q.el('SiIII', 12.96, 0.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J020930.74-043826.2', 1.0, 0.39)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 19, 0.1, 0.1)
        q.el('SiIII', 14.1, 0.1, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J023507.38-040205.6', 1.0, 0.73)
        q.telescope = 'HST'
        q.year = 2013
        q.ref = 'Lehner2018'
        q.el('HI', 16.7, 0.1, 0.1)
        q.el('CII', 13.4, 0.2, 0.2)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J040148.98-054056.5', 1.0, 0.21)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.6, 0.1, 0.1)
        q.el('SiIII', 13.0, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J044011.90-524818.0', 1.0, 0.61)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.6, 0.1, 0.1)
        q.el('CII', 13.16, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J055224.49-640210.7', 1.0, 0.44)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.96, 0.1, 0.1)
        q.el('CII', 12.74, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J093518.19+020415.5', 1.0, 0.35)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.34, 0.1, 0.1)
        q.el('SiIII', 12.95, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J093603.88+320709.3', 1.0, 0.39)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 16.4, 0.1, 0.1)
        q.el('CII', 13.87, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J095123.92+354248.8', 1.0, 0.33)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 17.65, 0.1, 0.1)
        q.el('CII', 14.64, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J100102.64+594414.2', 1.0, 0.41)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.9, 0.1, 0.1)
        q.el('SiIII', 13.18, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J100102.64+594414.2', 1.0, 0.41)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 16.5, 0.1, 0.1)
        q.el('CII', 14.1, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J100535.25+013445.5', 1.0, 0.41)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 16.9, 0.1, 0.1)
        q.el('CII', 13.5, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J102056.37+100332.7', 1.0, 0.31)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 16.95, 0.1, 0.1)
        q.el('CII', 13.82, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J110539.79+342534.3', 1.0, 0.23)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.85, 0.1, 0.1)
        q.el('SiIII', 12.85, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J111132.20+554725.9', 1.0, 0.61)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.37, 0.1, 0.1)
        q.el('CII', 13.26, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J111754.23+263416.6', 1.0, 0.35)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.84, 0.1, 0.1)
        q.el('CII', 13.26, 0.1, 0.1)
        q.el('SiII', 12.57, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J112553.78+591021.', 1.0, 0.55)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.9, 0.1, 0.1)
        q.el('CII', 13.13, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J113457.71+255527.8', 1.0, 0.42)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 16.33, 0.1, 0.1)
        q.el('CII', 13.83, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J113910.70-135044.0', 1.0, 0.33)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 17, 0.1, 0.1)
        q.el('CII', 13.79, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J115120.46+543733.0', 1.0, 0.25)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.2, 0.1, 0.1)
        q.el('SiIII', 12.16, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J122317.79+092306.9', 1.0, 0.37)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.9, 0.1, 0.1)
        q.el('CII', 13.37, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J122454.44+212246.3', 1.0, 0.37)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.4, 0.1, 0.1)
        q.el('SiIII', 13.22, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J122454.44+212246.3B', 1.0, 0.42)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.84, 0.1, 0.1)
        q.el('SiIII', 12.32, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J123304.05-003134.1', 1.0, 0.31)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.57, 0.1, 0.1)
        q.el('SiIII', 12.89, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J134100.78+412314.01', 1.0, 0.46)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.30, 0.1, 0.1)
        q.el('SiIII', 12.67, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J140923.90+261820.9', 1.0, 0.59)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.87, 0.1, 0.1)
        q.el('CII', 13.86, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J155048.29+400144.9', 1.0, 0.49)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 16.52, 0.1, 0.1)
        q.el('CII', 14.11, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J155304.92+354828.6', 1.0, 0.21)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.47, 0.1, 0.1)
        q.el('SiIII', 13.20, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)

        q = qso('J155304.92+354828.6B', 1.0, 0.45)
        q.telescope = 'HST'
        q.year = 2018
        q.ref = 'Lehner2018'
        q.el('HI', 15.71, 0.1, 0.1)
        q.el('CII', 13.69, 0.1, 0.1)
        q.comp = []
        q.comment = 't'
        q.full = 'n'
        QSO.append(q)


    return QSO

def load_LLSs():
    global sy
    QSO = sample()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    q = qso('SDSSJ0212-0737', 1.0, 0.01603)
    q.coord = ['J2000', '33.07633, -7.62217']
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 18.27, 0.17)
    q.el('CII', '>15.50')
    q.el('SiII', 14.14, 0.14)
    q.el('SiIII', '>14.65')
    q.el('bpar', 53, f='d')
    q.el('Re', 2.90, f='d')
    q.el('b_rel', 18.3, f='d')
    q.el('Mstar', 8.67, f='l')
    q.assgalz = 0.01747127
    q.IAUNAME = 'J021213.92-073500.9'
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)


    q = qso('PG0003+158', 1.0, 0.16512)
    q.coord = ['J2000', '1.49683, 16.16361']
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 18.16, 0.03)
    q.el('CII',  13.60,0.03)
    q.el('SiII', 12.23, 0.09)
    q.el('SiIII', '<12.6')
    q.el('bpar',127,f='d')
    q.el('U',-3.1,f='l')
    q.el('X/H', -2.5, f='l')
    q.el('NHtot', 19.9, f='l')
    q.assgalname = 'SDSSJ000600.63+160908.1'
    q.assgalz = 0.16519
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('Q0107-025A', 1.0, 0.22722)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.92, 0.08)
    q.el('CII', 13.85 , 0.07)
    q.el('SiII', 13.04, 0.06)
    q.el('SiIII', 12.90, 0.05)
    q.el('bpar',166,f='d')
    q.el('U', -3.4, f='l')
    q.el('X/H', 0.2, f='l')
    q.el('NHtot', 17.7, f='l')
    q.assgalname = '17.56667, −2.32678  Crighton et al. (2010)'
    q.assgalz = 0.2272
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('HE0153-4520', 1.0, 0.22597)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.83 , 0.07)
    q.el('CII', 14.09 , 0.03)
    q.el('SiII', 13.04, 0.06)
    q.el('SiIII', 12.78, 0.04)
    q.el('U', -2.9, f='l')
    q.el('X/H', -0.7, f='l')
    q.el('NHtot', 19.2, f='l')
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('3C57', 1.0, 0.32338)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.14 , 0.02)
    q.el('CII', 14.05 , 0.02)
    q.el('SiII', 12.83, 0.11)
    q.el('SiIII', '>13.62')
    q.el('U', -2.7, f='l')
    q.el('X/H', 0, f='l')
    q.el('NHtot', 18.6, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)




    q = qso('SDSSJ0212-0737', 1.0, 0.13422)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 14.95, 0.08)
    q.el('CII', 13.91, 0.10)
    q.el('SiII', 12.81, 0.04)
    q.el('SiIII', 12.43, 0.15)
    q.el('bpar', 860, f='d')
    q.el('Re', 5.67, f='d')
    q.el('b_rel', 140, f='d')
    q.el('Mstar', 11.45, f='l')
    q.el('U', -3.5, f='l')
    q.el('X/H', 0.9, f='l')
    q.el('NHtot', 16.6, f='l')

    q.assgalz = 0.1374255
    q.IAUNAME = 'J021156.69-073901.2'
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('UKS-0242-724', 1.0,  0.06376)
    q.telescope = 'HST'
    q.year = 2021
    q.ref = 'Kulkarni2022'
    q.el('HI', 15.27,0.27)
    q.el('CII', 13.73, 0.09)
    q.el('SiII', 12.90, 0.05)
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('Q0349-146', 1.0,  0.07256)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.24, 0.53)
    q.el('CII',  13.91, 0.09)
    q.el('SiII', 13.18, 0.05)
    q.el('SiIII', 13.01, 0.09)
    q.el('U', -3.4, f='l')
    q.el('X/H', 0.0, f='l')
    q.el('NHtot', 18, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PKS0405-123', 1.0,  0.16710)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.45, 0.05)
    q.el('CII',  '>14.40')
    q.el('SiII', 13.33, 0.03)
    q.el('SiIII', '>13.40')
    q.el('bpar',101,f='d')
    q.el('U', -3.2, f='l')
    q.el('X/H', -0.1, f='l')
    q.el('NHtot', 18.4, f='l')

    q.assgalname = '61.95084, −12.1836 Prochaska et al. (2006)'
    q.assgalz = 0.1670
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('IRAS-F04250-5718', 1.0,  0.00369)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.85, 0.08)
    q.el('CII',  13.49, 0.04 )
    q.el('SiII', 12.25, 0.17)
    q.el('SiIII', 12.78, 0.06)
    q.el('U', -2.9, f='l')
    q.el('X/H', -0.4, f='l')
    q.el('NHtot', 18.1, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('FBQS-0751+2919', 1.0, 0.20399)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI',  17.77, 0.05)
    q.el('CII', 14.23, 0.06)
    q.el('SiII', 12.97, 0.07)
    q.el('SiIII', '>13.55')
    q.el('U', -3.2, f='l')
    q.el('X/H', -1.5, f='l')
    q.el('NHtot', 19.8, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('VV2006J0808+0514', 1.0, 0.02930)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 17.56, 0.43)
    q.el('CII', '<14.17')
    q.el('SiII',  13.46, 0.09)
    q.el('SiIII', '>14.11')
    q.el('U', -3.2, f='l')
    q.el('X/H', -0.8, f='l')
    q.el('NHtot', 19.7, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PG0832+251', 1.0,  0.02811)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI',  14.28, 0.14)
    q.el('CII', 13.56, 0.08)
    q.el('SiII', 12.79, 0.10)
    q.el('SiIII',12.51, 0.11)
    q.el('U', -3.5, f='l')
    q.el('X/H', 1.6, f='l')
    q.el('NHtot', 16.0, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('SDSSJ0929+4644', 1.0, 0.06498)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 14.67, 0.71)
    q.el('CII', 13.50, 0.11)
    q.el('SiII',  12.68, 0.06)
    q.el('SiIII',  12.50, 0.09)
    q.el('U', -3.4, f='l')
    q.el('X/H', 1.1, f='l')
    q.el('NHtot', 16.4, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PMNJ1103-2329', 1.0,  0.08352)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 17.74, 0.44)
    q.el('CII', 14.03, 0.10)
    q.el('SiII',  12.82, 0.13)
    q.el('SiIII', '>13.37')
    q.el('U', -3.2, f='l')
    q.el('X/H', -1.6, f='l')
    q.el('NHtot', 19.7, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PG1116+215', 1.0, 0.13850)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.20,0.05)
    q.el('CII', 13.84, 0.02)
    q.el('SiII', 13.84, 0.02)
    q.el('bpar',137,f='d')
    q.el('Mstar', 10.55, f='l') #'J111906.68+211828.7'
    q.el('Re', 4.22,f='d')
    q.el('Mstar2', 10.05, f='l') #'J111905.10+211502.9'
    q.el('Re2', 10.60, f='d')
    q.el('b_rel',32.46, f='d')
    q.el('b_rel2', 12.92, f='d')
    q.el('U', -3.3, f='l')
    q.el('X/H', -0.3, f='l')
    q.el('NHtot', 18.1, f='l')

    q.assgalname = '169.77779, 21.30785 Tripp, Lu & Savage (1998), SDSS'
    q.assgalz = 0.13814
    #q.el('SiIII', '>13.37')
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('SDSSJ1122+5755', 1.0, 0.05319)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.16, 1.43)
    q.el('CII', 13.56, 0.13)
    q.el('SiII', 12.80, 0.08 )
    q.el('SiIII', 12.85, 0.11 )
    q.el('U', -3.3, f='l')
    q.el('X/H', 0.7, f='l')
    q.el('NHtot', 17.1, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PG1121+422', 1.0, 0.19238)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.64, 0.05)
    q.el('CII', 14.12, 0.02)
    q.el('SiII', 13.07, 0.02)
    q.el('SiIII', 13.34, 0.06)
    q.el('U', -3.1, f='l')
    q.el('X/H', 0.5, f='l')
    q.el('NHtot', 17.7, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('3C263', 1.0, 0.06350)
    q.coord = ['J2000','174.98746, 65.79700']
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.40, 0.12)
    q.el('CII', 13.55, 0.06)
    q.el('SiII', 12.35, 0.05)
    q.el('SiIII', 13.09, 0.02)
    q.el('bpar',63,f='d')
    q.el('U', -2.8, f='l')
    q.el('X/H', 0.3, f='l')
    q.el('NHtot', 17.8, f='l')
    #q.assgalname = '175.02154, 65.80042 Tripp, Savage et al. (2012)'
    #q.assgalz = 0.06322
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PG1202+281', 1.0, 0.13988)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.73, 0.10)
    q.el('CII', 14.04, 0.17)
    q.el('SiII', 12.79, 0.09)
    q.el('SiIII', 13.41, 0.24)
    q.el('U', -2.9, f='l')
    q.el('X/H', 0.3, f='l')
    q.el('NHtot', 18.1, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PG-1206+459', 1.0, 0.21439)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.71, 0.06)
    q.el('CII', 14.24, 0.05)
    q.el('SiII', 13.13, 0.05)
    q.el('SiIII', '<13.65')
    q.el('U', -2.9, f='l')
    q.el('X/H', 0.6, f='l')
    q.el('NHtot', 18.0, f='l')

    q.el('bpar',31,f='d')
    q.assgalname = 'N/A Rosenwasser et al. (2018)'
    q.assgalz = 0.2144
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('SDSSJ1210+3157', 1.0, 0.05974)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 14.83, 0.27)
    q.el('CII', 14.06, 0.05)
    q.el('SiII',  13.10, 0.08)
    q.el('SiIII', 13.23, 0.11)
    q.el('U', -3.2, f='l')
    q.el('X/H', 1.3, f='l')
    q.el('NHtot', 16.8, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('SDSSJ1210+3157B', 1.0,  0.14964)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 17.28, 0.12)
    q.el('CII', 14.23, 0.11)
    q.el('SiII', 12.98, 0.12)
    q.el('SiIII', '>13.69')
    q.el('U', -3.2, f='l')
    q.el('X/H', -1.0, f='l')
    q.el('NHtot', 19.4, f='l')
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('SDSSJ1214+0825', 1.0, 0.07407)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.00, 0.40)
    q.el('CII', 14.13, 0.07)
    q.el('SiII', 13.05, 0.06 )
    #q.el('SiIII', '>13.69')
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('RXJ1230.8+0115', 1.0,  0.00575)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.42, 0.43)
    q.el('CII', 13.79, 0.01)
    q.el('SiII', 12.64, 0.02)
    q.el('SiIII',  13.47, 0.07)
    q.el('U', -2.7, f='l')
    q.el('X/H', 0.6, f='l')
    q.el('NHtot', 17.9, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PKS1302-102', 1.0, 0.09495)
    q.coord = ['J2000','196.38750, -10.55528']
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.88, 0.03)
    q.el('CII', 13.65, 0.09)
    q.el('SiII', 12.58, 0.11)
    q.el('SiIII', 13.02, 0.06)
    q.el('bpar',68,f='d')
    q.el('U', -3.4, f='l')
    q.el('X/H', -1.1, f='l')
    q.el('NHtot', 18.8, f='l')

    #q.assgalname = '196.38375, −10.56555  Cooksey et al. (2008)'
    #q.assgalz = 0.09358
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('SDSSJ1322+4645', 1.0, 0.21451)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 17.14, 0.05)
    q.el('CII',  14.40,0.05)
    q.el('SiII', 13.27, 0.09)
    q.el('SiIII', '>13.71')
    q.el('bpar',37,f='d')
    q.el('U', -3.4, f='l')
    q.el('X/H', -0.6, f='l')
    q.el('NHtot', 19, f='l')

    q.assgalname = 'NA Werk et al. (2014)'
    q.assgalz = 0.2142
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('SDSSJ1357+1704', 1.0, 0.09784)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.79, 0.52)
    q.el('CII',  13.97, 0.04)
    q.el('SiII', 12.76, 0.05)
    q.el('SiIII', '>13.64')
    q.el('U', -2.7, f='l')
    q.el('X/H', 0.4, f='l')
    q.el('NHtot', 18.3, f='l')

    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('SDSSJ1419+4207', 1.0, 0.17885)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.63, 0.30)
    q.el('CII', 14.32, 0.14)
    q.el('SiII', 13.37, 0.09)
    q.el('SiIII', '>14.18')
    q.el('bpar',88,f='d')
    q.el('U', -2.7, f='l')
    q.el('X/H', 0.1, f='l')
    q.el('NHtot', 19.2, f='l')

    q.assgalname = 'NA Werk et al. (2014)'
    q.assgalz = 0.1792
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PG1424+240', 1.0,  0.12126)
    q.coord = ['J2000','216.75163, 23.80000']
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 14.96, 0.08)
    q.el('CII', 13.22, 0.08)
    q.el('SiII', 12.36, 0.09)
    q.el('SiIII', 12.46, 0.10)
    q.el('bpar',205,f='d')
    q.el('Re', 3.69,f='d')
    q.el('b_rel',55.5, f='d')
    q.el('Mstar',10.42, f='l')
    q.el('U', -3.2, f='l')
    q.el('X/H', 0.4, f='l')
    q.el('NHtot', 16.9, f='l')

    q.IAUNAME= 'J142701.72+234630.9'
    q.assgalname = 'SDSSJ142701.72+234630.9'
    q.assgalz = 0.121177
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PG1424+240B', 1.0, 0.14683)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.56, 0.96)
    q.el('CII', 13.71, 0.06)
    q.el('SiII', 12.59, 0.07)
    q.el('SiIII', 13.04, 0.05)
    q.el('bpar',495,f='d')
    q.el('U', -3.4, f='l')
    q.el('X/H', -0.7, f='l')
    q.el('NHtot', 18.5, f='l')
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PG-1630+377', 1.0,  0.17388)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 15.83, 0.08)
    q.el('CII', 14.20, 0.03)
    q.el('SiII', 13.07, 0.02)
    q.el('SiIII', 13.14, 0.02)
    q.el('U', -3.2, f='l')
    q.el('X/H', 0.3, f='l')
    q.el('NHtot', 17.7, f='l')
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PHL1811', 1.0, 0.07774)
    q.coord = ['J2000','328.75623, -9.37361']
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 16.00, 0.05)
    q.el('CII', 13.26, 0.03)
    q.el('SiII', 12.35, 0.05)
    q.el('SiIII', 12.48, 0.04)
    q.el('bpar',237,f='d')
    q.el('Mstar', 9.46, f='l')
    q.el('U', -3.6, f='l')
    q.el('X/H', -0.4, f='l')
    q.el('NHtot', 17.6, f='l')
    q.assgalname = 'J215450.8-092235 Jenkins2005'
    q.assgalz = 0.078822
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    q = qso('PHL1811B', 1.0, 0.08091)
    q.telescope = 'HST/COS'
    q.year = 2018
    q.ref = 'Muzahid2018'
    q.el('HI', 17.94, 0.07)
    q.el('CII', '>14.55')
    q.el('SiII',  13.79, 0.04)
    q.el('SiIII', '<13.80')
    q.el('bpar',34,f='d')
    q.el('Re', 0.8, f='d')
    q.el('b_rel', 42.5, f='d')
    q.el('U', -3.5, f='l')
    q.el('X/H', -0.8, f='l')
    q.el('NHtot', 19.4, f='l')

    q.assgalname = 'J21545996-0922249 Jenkins et al. (2005)'
    q.assgalz = 0.0808
    q.comp = []
    q.comment = 't'
    q.full = 'n'
    QSO.append(q)

    return QSO


def load_MWH2CI():
    global sy
    QSO = load_MV()
    ex = load_MV()
    for q in QSO.values():
        if q.CI.col.val==0.0:
            ex.remove(q)

    return ex

def load_lowzDLAs():
    global sy
    QSO = load_QSO()
    ex = load_QSO()
    for q in QSO.values():
        if q.z_dla > 0.6:
            ex.remove(q)

    return ex

# select H2 data with NH2>17.3 and -1.25 <log z < -0.75
def load_ex1():
    global sy
    QSO=load_QSO()
    ex1 = load_QSO()
    for q in QSO.values():
        print(q.name)
        if q.e['H2'].col.val < 17.3:
            ex1.remove(q)
        #else:
        #    for comp in q.comp:
        #        if comp.e['H2'].col.val < 17:
        #            print(q.name, comp.e['H2'].col.val)
        #            ex1.q.remove(comp)
        elif q.e['Me'].col.val < (-1.55):
            ex1.remove(q)
            print(q.name, q.e['Me'].col.val)
        elif q.e['Me'].col.val > (-0.75):
            ex1.remove(q)
            print(q.name, q.e['Me'].col.val)
        elif q.z_dla < 1.7:
            ex1.remove(q)
            print(q.name, q.z_dla)
        elif q.name == 'B1331+0170':
            ex1.remove(q)
        else:
            if q.name == 'J0816+1446':
                del ex1[q.name].comp[1]
            if q.name == 'J0843+0221':
                del ex1[q.name].comp[2]
            for i,c in enumerate(q.comp):
                if 'H2' in c.e:
                    if c.e['H2'].col.val<17.3:
                        del ex1[q.name].comp[i]
                        #print('remove',q.name,i)
                else:
                    for i_c, c_local in enumerate(ex1[q.name].comp):
                        if c_local.z == c.z:
                            del ex1[q.name].comp[i_c]

    for q in ex1.values():
        print(q.name, q.z_dla,q.e['Me'].col.val,q.e['H2'].col.val)

    return ex1

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# select H2 data with NH2>17 and -0.75 <log z < -0.25
def load_ex2():
    global sy
    QSO=load_QSO()
    ex2 = load_QSO()
    for q in QSO.values():
        print(q.name)
        if q.e['H2'].col.val < 17.3:
            ex2.remove(q)
        #else:
        #    for comp in q.comp:
        #        if comp.e['H2'].col.val < 17:
        #            print(q.name, comp.e['H2'].col.val)
        #            ex1.q.remove(comp)
        elif q.e['Me'].col.val < (-0.75):
            ex2.remove(q)
            print(q.name, q.e['Me'].col.val)
        elif q.e['Me'].col.val > (-0.25):
            ex2.remove(q)
            print(q.name, q.e['Me'].col.val)
        elif q.z_dla < 1.7:
            ex2.remove(q)
            print(q.name, q.z_dla)
        elif q.name in ['0013-0029','J2340-0053']:
            ex2.remove(q)
        else:
            for i,c in enumerate(q.comp):
                if 'H2' in c.e:
                    if c.e['H2'].col.val<17.3:
                        for j, c_global in enumerate(ex2[q.name].comp):
                            if c_global.e['H2'].col.val == c.e['H2'].col.val:
                                del ex2[q.name].comp[j]
                                #print('remove', q.name, i)
                else:
                    for i_c, c_local in enumerate(ex2[q.name].comp):
                        if c_local.z == c.z:
                            del ex2[q.name].comp[i_c]

    for q in ex2.values():
        print(q.name, q.z_dla,q.e['Me'].col.val,q.e['H2'].col.val)

    return ex2


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# select H2 data with NH2>17 and -0.25 <log z < 0.25
def load_ex3():
    global sy
    QSO=load_QSO()
    ex3 = load_QSO()
    for q in QSO.values():
        print(q.name)
        if q.e['H2'].col.val < 17.3:
            ex3.remove(q)
        #else:
        #    for comp in q.comp:
        #        if comp.e['H2'].col.val < 17:
        #            print(q.name, comp.e['H2'].col.val)
        #            ex1.q.remove(comp)
        elif q.e['Me'].col.val < (-0.25):
            ex3.remove(q)
            print(q.name, q.e['Me'].col.val)
        elif q.e['Me'].col.val > (0.25):
            ex3.remove(q)
            print(q.name, q.e['Me'].col.val)
        elif q.z_dla < 1.7:
            ex3.remove(q)
            print(q.name, q.z_dla)
        else:
            for i,c in enumerate(q.comp):
                if 'H2' in c.e:
                    if c.e['H2'].col.val<17.3:
                        for j,c_global in enumerate(ex3[q.name].comp):
                            if c_global.e['H2'].col.val == c.e['H2'].col.val:
                                del ex3[q.name].comp[j]
                                #print('remove',q.name,i)
                else:
                    for i_c, c_global in enumerate(ex3[q.name].comp):
                        if c_global.z == c.z:
                            del ex3[q.name].comp[i_c]

    for q in ex3.values():
        print(q.name, q.z_dla,q.e['Me'].col.val,q.e['H2'].col.val)

    return ex3

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# select H2 data with NH2>17 and 0.25 <log z < 0.75
def load_ex4():
    global sy
    QSO=load_QSO()
    ex3 = load_QSO()
    for q in QSO.values():
        print(q.name)
        if q.e['H2'].col.val < 17.3:
            ex3.remove(q)
        #else:
        #    for comp in q.comp:
        #        if comp.e['H2'].col.val < 17:
        #            print(q.name, comp.e['H2'].col.val)
        #            ex1.q.remove(comp)
        elif q.e['Me'].col.val < (0.25):
            ex3.remove(q)
            print(q.name, q.e['Me'].col.val)
        elif q.e['Me'].col.val > (0.75):
            ex3.remove(q)
            print(q.name, q.e['Me'].col.val)
        elif q.z_dla < 1.7:
            ex3.remove(q)
            print(q.name, q.z_dla)
        else:
            for i,c in enumerate(q.comp):
                if 'H2' in c.e:
                    if c.e['H2'].col.val<17.3:
                        for j,c_global in enumerate(ex3[q.name].comp):
                            if c_global.e['H2'].col.val == c.e['H2'].col.val:
                                del ex3[q.name].comp[j]
                                #print('remove',q.name,i)
                else:
                    for i_c, c_global in enumerate(ex3[q.name].comp):
                        if c_global.z == c.z:
                            del ex3[q.name].comp[i_c]

    for q in ex3.values():
        print(q.name, q.z_dla,q.e['Me'].col.val,q.e['H2'].col.val)

    return ex3


#------------------------------------------------------------------------------
# The final total sample of high H2 column density systems to fot of H2 excitation (H2UV analysis)
def load_total():
    global sy
    QSO=load_QSO()
    sample= load_QSO()
    for q in QSO.values():
        print(q.name, q.z_dla, q.e['Me'].col.val, q.e['H2'].col.val)
        if q.e['H2'].col.val < 17.31:
            sample.remove(q)
            #print('remove_s_lowH2', q.name)
        elif q.z_dla < 1.7:
            sample.remove(q)
            #print('remove_s_lowz', q.name)
        elif q.name in ['0013-0029','J2340-0053','B1331+0170']:
            sample.remove(q)
        #elif q.e['Me'].col.val < (-1.25):
        #    sample.remove(q)
        #    print('remove_s_Me<-1.25',q.name)
        elif np.size(q.comp) == 0:
            sample.remove(q)
            #print('remove_no_comp_data',q.name)
        else:
            #print('len',np.size(q.comp))
            if q.name == 'J0816+1446':
                del sample[q.name].comp[1]
            for i,c in enumerate(q.comp):
                if 'H2' in c.e:
                    if c.e['H2'].col.val<17:
                        for j, c_global in enumerate(sample[q.name].comp):
                            if c_global.e['H2'].col.val == c.e['H2'].col.val:
                                del sample[q.name].comp[j]
                                #print('remove_c_lowH2', q.name, i)
                else:
                    for i_c, c_local in enumerate(sample[q.name].comp):
                        if c_local.z == c.z:
                            del sample[q.name].comp[i_c]
                            #print('remove_c_no_H2', q.name, i_c)

#    for q in QSO.values():
#        print(q.name, q.z_dla, q.e['HI'].col.val, q.e['H2'].col.val,q.e['CI'].col.val,
#              q.e['Me'].col.val,q.e['T01'].col.val,q.e['n'].col.val,q.e['UV'].col.val)

    return sample

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def load_fcov_sample():
    global sy
    QSO=load_QSO()
    sample= load_QSO()
    for q in QSO.values():
        if q.name in ['0013-0029', 'B0027-1836', 'B0347-3819','J0812+3208','J0843+0221','Q1232+0815',
                      'J1237+0647','J1439+1118','J1443+2724','B1444+0126','J2100-0641','J2123-0050','J2340-0053','B0528-2505']: #'B0405-4418','J0643-5041','B1331+0170',,'0551-3638']: #,'B0405-4418','B0528-2505','J0643-5041','J0816+1446','B2348-0108'
            for i,c in enumerate(q.comp):
                if 'H2' in c.e:
                    if c.e['H2'].col.val<14:
                        for j, c_global in enumerate(sample[q.name].comp):
                            if c_global.e['H2'].col.val == c.e['H2'].col.val:
                                del sample[q.name].comp[j]
                                print('remove_c_lowH2', q.name, i)
                else:
                    for i_c, c_local in enumerate(sample[q.name].comp):
                        if c_local.z == c.z:
                            if q.name == 'J2123-0050':
                                print('remove_c_no_H2', q.name, i_c)
                            del sample[q.name].comp[i_c]
                            print('remove_c_no_H2', q.name, i_c)
            #if q.name == 'J0843+0221':
            #    del sample[q.name].comp[1]
            #    del sample[q.name].comp[1]

        else:
            del sample[q.name]


    return sample

# select H2 data with NH2>17.3 and -1.25 <log z < -0.75
def load_co_sample():
    global sy
    MW = load_MV()
    MW.append(load_QSO())
    sample = load_MV()
    s = load_QSO()
    sample.append(s)
    for q in MW.values():
        for i, c in enumerate(q.comp):
            if 'CO' in c.e:
                print(q.name)
                if 'COj2' in c.e.keys():
                    print(q.name,'_',i)
                else:
                    for j, c_global in enumerate(sample[q.name].comp):
                        if c_global.e['CO'].col.val == c.e['CO'].col.val:
                            print(c.e.keys())
                            del sample[q.name].comp[j]
            else:
                if c.z > 0:
                    #print(q.name,i)
                    for j, c_global in enumerate(sample[q.name].comp):
                        if c_global.z == c.z:
                            #print('remove_noCO', q.name, i,j)
                            del sample[q.name].comp[j]
                elif 'CO18' in q.e.keys():
                    print(q.name)
                else:
                    del sample[q.name] #.comp[i]
    return sample


def load_JT():
    global sy
    MW = load_MV()
    sample = load_MV()
    for q in MW.values():
        for i, c in enumerate(q.comp):
            if 'pci-tripp' in c.e:
                print(q.name)
            else:
                del sample[q.name]
    return sample

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



if __name__ == '__main__':

    #ex1 = load_MWH2CI()
    #ex1.append(load_QSO())
    JT = load_JT()
    COsample = load_co_sample()
    MWsample = load_MV()
    Qall = load_QSO()
    #GRB = load_GRB()
    #P94 = load_P94()
    #H2UV = load_H2UV()
    #QSO.append(load_P94())
    #UN = load_Secret()1
    #H2UV = load_total()
    #H2fcov = load_fcov_sample()

    # >>> Print H2 systems within redshift range
    if 0:
        n = 1
        for q in QSO.values():
            if q.z_dla > q.z_em and q.full != 'r':
                n += 1
                print(q.name, q.z_dla, q.z_em)
        print(n)

    # >>> Print qso and references
    if 0:
        i = 0
        dates = []
        for q in QSO.values():
            dates.append(q.year)
        ind = np.argsort(dates)
        k = 0
        for i in ind:
            q = QSO[i]
            if q.z_dla > 1.7:
                k += 1
                print('{:d} {:12} {:.4f}  {:.2f}   {:.2f}    {!s:100}'.format(k, q.name, q.z_dla, q.e['HI'].col.val, q.e['H2'].col.val, q.ref))
        print('total number: ', k)


    # >>> Sum over the components
    if 0:
        species = 'H2j1'
        q = QSO.get('2340')
        n = 0
        for c in q.comp:
            try:
                n += c.e[species].col
            except:
                pass
        print(n.log())
        print('co.el(\'{:}\', {:.2f}, {:.2f}, {:.2f})'.format(species, n.val, n.plus, n.minus))
    
    # >>> Calc HD/2H2 ratio   
    if 0:
        for q in QSO.values():
            for c in q.comp:
                try:
                    ratio = c.e['HD'].col / c.e['H2'].col / 2
                    ratio.log()
                    print('co.el(\'HD2H2\', \'\', %.2f, %.2f, %.2f))' % (ratio.val, ratio.plus, ratio.minus))
                except:
                    pass

    # >>> Calc total H column density
    if 0:
        for q in P94.values():
            H = (q.e['H2'].col*2 + q.e['HI'].col).log()
            print('{:}  q.el(\'H\', {:.2f}, {:.2f}, {:.2f})'.format(q.name, H.val, H.plus, H.minus))
                    
    # >>> Calc molecular fraction
    if 0:
        for q in P94.values():
            try:
                Htot = q.e['H'].col
            except:
                Htot = q.e['H2'].col*2 + q.e['HI'].col
            ratio = q.e['H2'].col*2 / Htot
            ratio.log()
            print('{name}:  q.el(\'f\', {:.2f}, {:.2f}, {:.2f})'.format(ratio.val, ratio.plus, ratio.minus, name=q.name))
                    
    # >>> Calc T_01 ratio   
    if 0:
        q = COsample.get('J2331-0908')
        print(q.name)
        if 1:
            for c in q.comp:
                if 0:
                    if all([k in c.e.keys() for k in ['H2j0', 'H2j1', 'H2j2']]):
                        Temp = ExcitationTemp('H2')
                        n = []
                        n.append(c.e['H2j0'].col)
                        n.append(c.e['H2j1'].col)
                        n.append(c.e['H2j2'].col)
                        Temp.calcTemp(n, plot=1, verbose=1)
                        Temp.latex()
                        Temp.plot_temp()
                        plt.show()
                if 1:
                    if all([k in c.e.keys() for k in ['COj0', 'COj1','COj2','COj3','COj4']]):
                        Temp = ExcitationTemp('CO')
                        n = []
                        n.append(c.e['COj0'].col)
                        n.append(c.e['COj1'].col)
                        n.append(c.e['COj2'].col)
                        n.append(c.e['COj3'].col)
                        n.append(c.e['COj4'].col)
                        Temp.calcTemp(n, plot=1, verbose=1)
                        Temp.latex()
                        Temp.plot_temp()
                        plt.show()

    # >>> Print qso in Me raange
    if 0:
        ref = set()
        for q in QSO.values():
            print(q.name, q.e['Me'].col.val, q.e['H2'].col.val)

    # >>> Print qso in N_H2 range
    if 0:
        ref = set()
        for q in QSO.values():
            # print(q.H2.col.val)
            if q.z_dla > 1.7:
                if q.e['H2'].col.val > 17 and q.e['HD'].col.val == 0:
                    print(q.name, q.e['H2'].col, q.z_dla, q.e['HD'].col, q.ref[-1])

     # >>> Plot dates
    if 0:
        dates = []
        for q in QSO.values():
            if q.z_dla > 1.7 and q.year <= 2019:
                dates.append(q.year)
        print(len(dates))
        fig, ax = plt.subplots(figsize=(10, 5))
        n, bins, patches = ax.hist(dates, bins=max(dates)-min(dates)+1, color='dodgerblue', edgecolor='k')

        if 0:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)

            cm = plt.cm.get_cmap('RdYlBu')

            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))

        if 1:
            ax.annotate('KECK, VLT', xy=(1999, 3), xytext=(1999, 5), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=18, ha='center')
            ax.arrow(2002, 5.15, 1.5, 0, head_width=0.1, head_length=0.3, fc='k', ec='k', lw=4)
            ax.annotate('SDSS', xy=(2009, 4), xytext=(2009, 6), arrowprops=dict(facecolor='orangered', edgecolor='orangered', shrink=0.05),
                        fontsize=18, ha='center', color='orangered')
            ax.arrow(2010.7, 6.15, 1.5, 0, head_width=0.1, head_length=0.3, fc='orangered', ec='orangered', lw=4)

        ax.set_xlabel('Publication year')
        ax.set_ylabel('Number of H$_2$-bearing DLAs')
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.show()
        
    # >>> print prodIDs
    if 0:
        progIDs = set()
        for q in QSO.values():
            f = 0
            for c in q.comp:
                for el in c.el:
                    if el.name == 'Cl' and el.ion == 'I':
                        f = 1
            if q.e['ClI'].col.val != 0 or f == 1:
                print(q.name, q.progID)
                for p in q.progID:
                    progIDs.add(p)
        for s in progIDs:
            print(s)

    if 0:
        for q in QSO.values():
            for c in q.comp:
                if all([k in c.e.keys() for k in ['T01', 'H2j0', 'H2j1']]):
                    print(q.name, q.Me.col.latex(), c.z, c.e['T01'].col.latex(base=0), c.e['H2'].col.latex(), c.e['H2j0'].col.latex(), c.e['H2j1'].col.latex())

    # >>> print ZnII vs H2
    if 0:
        for q in QSO.values():
            if q.z_dla >1.5 and q.e['H2'].type == 'm':
                print(q.name)
                try:
                    if q.e['ZnII'].col.val != 0:
                        print (q.e['ZnII'].col/q.e['H2'].col.dec()/2)
                except KeyError:
                    if q.Me_ind == 'Zn':
                        print(q.Me.col.dec()/q.mol.col.dec()/10**(12-4.56))

    if 0:
        print(QSO.makelist(pars=['name', 'z_dla', 'Me__val', 'H2__val'], sys=QSO.all()[:10]))

    # >>> print data of MW list
    if 0:
        for qname in ex1:
            q = ex1[qname]
            print(q.name)
            plot = 0
            if 1:
                if q.name in ['J0843+0221']:
                    print('J0843+0221')
                else:
                    if q.name == 'J0816+1446':
                        plot = 1
                    for c in q.comp:
                        if all([k in c.e.keys() for k in ['H2j0', 'H2j1']]):
                            Temp = ExcitationTemp('H2')
                            n = []
                            n.append(c.e['H2j0'].col)
                            n.append(c.e['H2j1'].col)
                            # n.append(c.e['H2j2'].col)
                            Temp.calcTemp(n, plot=plot, verbose=0)
                            print(q.name, Temp.temp.dec())

    if 0:
        for q in QSO.values():
            if 'EBV' in q.m.keys():
                print(q.name, q.m['V'])
            elif 'r' in q.m.keys():
                print(q.name, q.m['r'])

    if 0:
        print(', '.join(["'"+q.name+"'" for q in P94]))



    if 0:
        for q in H2fcov.values():
            nh2list_corr.append(q.e['Me'].col.val)

# calc H2 and CO population based on exc_temps data
    if 0:
        H2energy = [0,170.5,509.8,1015.1,1681.6]
        COenergy = [2.766 * ((i+1) * (i) ) for i in range(6)]
        print(H2energy)
        for q in MWsample.values():
            if 'Gillmon2006' in q.ref:
                for c in q.comp:
                    if all([k in c.e.keys() for k in ['H2', 'T_h210']]):
                        print('................')
                        print(q.name, c.e['H2'].col.val,c.e['T_h210'].col.val)
                        size = 10000
                        Ntot = np.random.normal(c.e['H2'].col.val, 0.10, size)
                        T10 = np.random.normal(c.e['T_h210'].col.val, 10, size)
                        N0 = Ntot + np.log10(1 / (1 + 9 * np.exp(-170.5 / T10) + 5* np.exp(-511.0/ T10)))
                        N1 = N0 + np.log10(9 * np.exp(-170.5 / T10))
                        print('N0=', np.mean(N0), np.std(N0))
                        print('N1=', np.mean(N1), np.std(N1))
                        if 'T_h220' in c.e.keys():
                            T20 = np.random.normal(c.e['T_h220'].col.val, 10, size)
                            N2 = N0 +  np.log10(5 * np.exp(-H2energy[2]/ T20))
                            print('N2=', np.mean(N2), np.std(N2))
                        if 'T_h230' in c.e.keys():
                            T30 = np.random.normal(c.e['T_h230'].col.val, 10, size)
                            N3 = N0 + np.log10(21 * np.exp(-H2energy[3]/ T30))
                            print('N3=', np.mean(N3), np.std(N3))
                        if 'T_h240' in c.e.keys():
                            T40 = np.random.normal(c.e['T_h240'].col.val, 10, size)
                            N4 = N0 + np.log10(9 * np.exp(-H2energy[4]/ T40))
                            print('N4=', np.mean(N4), np.std(N4))
                        if 1:
                            Texc = np.mean(-170.5 / np.log(10 ** N1 / 10 ** N0 / 9))
                            print('test',Texc,c.e['T_h210'].col.val)
                            Texc = np.mean(-H2energy[2] / np.log(10 ** N2 / 10 ** N0 / 5))
                            print('test', Texc, c.e['T_h220'].col.val)

                    if all([k in c.e.keys() for k in ['CO', 'T_co10']]):
                        print(q.name, c.e['CO'].col.val,c.e['T_co10'].col.val)
                        size = 10000
                        Ntot = np.random.normal(c.e['CO'].col.val, 0.05, size)
                        T10 = np.random.normal(c.e['T_co10'].col.val, 0.5, size)
                        N0 = Ntot + np.log10(1 / (1 + 3 * np.exp(-COenergy[1]/ T10) + 5* np.exp(-COenergy[2]/ T10)))
                        N1 = N0 + np.log10(3 * np.exp(-COenergy[1] / T10))
                        print('CO')
                        print('N0=', np.mean(N0), np.std(N0))
                        print('N1=', np.mean(N1), np.std(N1))
                        if 'T_co20' in c.e.keys():
                            T20 = np.random.normal(c.e['T_co20'].col.val, 0.5, size)
                            N2 = N0 +  np.log10(5 * np.exp(-COenergy[2]/ T20))
                            print('N2=', np.mean(N2), np.std(N2))
                            if 1:
                                Texc = -COenergy[1] / np.log(10 ** N1 / 10 ** N0 / 3)
                                print('test', np.mean(Texc),np.std(Texc), c.e['T_co10'].col.val)
                                Texc = -COenergy[2] / np.log(10 ** N2 / 10 ** N0 / 5)
                                print('test', np.mean(Texc),np.std(Texc), c.e['T_co20'].col.val)
                        if 'T_co30' in c.e.keys():
                            T30 = np.random.normal(c.e['T_co30'].col.val, 0.5, size)
                            N3 = N0 + np.log10(7 * np.exp(-COenergy[3]/ T30))
                            print('N3=', np.mean(N3), np.std(N3))
                        if 'T_co40' in c.e.keys():
                            T40 = np.random.normal(c.e['T_co40'].col.val, 0.5, size)
                            N4 = N0 + np.log10(9 * np.exp(-COenergy[4]/ T40))
                            print('N4=', np.mean(N4), np.std(N4))
                        if 'T_co50' in c.e.keys():
                            T50 = np.random.normal(c.e['T_co40'].col.val, 0.5, size)
                            N5 = N0 + np.log10(11 * np.exp(-COenergy[5]/ T50))
                            print('N5=', np.mean(N5), np.std(N5))
# print CO exc temp and pressure
    if 0:
        i =0
        for q in COsample.values():
                for c in q.comp:
                    if 'COj2' in c.e.keys():
                        i +=1
                        print('**************')
                        print(i, q.name, q.ref)
                        if 0:
                            for el in ['H2','CI','CO','T01','T_co','P_co','n_co']:
                                if el in c.e.keys():
                                    print(el, c.e[el].col)
                            if 'H2' in c.e.keys():
                                x = c.e['CO'].col/c.e['H2'].col
                                x = x.log()
                                print('CO/H2', x)
                            if 'PDRnH' in c.e.keys():
                                print('nH', c.e['PDRnH'])
                                print('PCI', c.e['PDRnH'].col*c.e['T01'].col)

                        if 1:
                            if 'n_co_3dpdr' in c.e.keys():
                                print(c.e['n_co_3dpdr'])
                                if 'T01' in c.e.keys():
                                    p = c.e['n_co_3dpdr'].col*c.e['T01'].col
                                    print('T01', c.e['T01'].col.log())
                                    print('P_co_3dpdr', p)
                            if 'n_ci_3dpdr' in c.e.keys():
                                print(c.e['n_ci_3dpdr'])
                                if 'T01' in c.e.keys():
                                    p = c.e['n_ci_3dpdr'].col * c.e['T01'].col
                                    print('P_ci_3dpdr', p)

                        if 0:
                            if 'COj2' in c.e.keys():
                                Temp = ExcitationTemp('CO')
                                n = []
                                for el in ['COj0','COj1','COj2','COj3','COj4','COj5','COj6']:
                                    if el in c.e.keys():
                                        if c.e[el].col.type in ['m', 'f']:
                                            n.append(c.e[el].col)
                                print('CO temp')
                                Temp.calcTemp(n, plot=1, verbose=0)
                                Temp.latex()
                                print(Temp.temp.log())
                                plt.show()
                        if 0:
                            for el in ['CIj0','CIj1','CIj2','COj0','COj1','COj2','COj3','COj4','COj5']:
                                if el in c.e.keys():
                                    print(el, c.e[el].col)
                        if 0:
                            x = 0
                            for el in ['CIj0', 'CIj1', 'CIj2']:
                                if el in c.e.keys():
                                    x+=c.e[el].col
                            print('Ci',x)
                            x = 0
                            for el in ['COj0', 'COj1', 'COj2', 'COj3']:
                                if el in c.e.keys():
                                    x += c.e[el].col
                            print('CO', x)
                            x =0
                            for el in ['H2j0', 'H2j1', 'H2j2', 'H2j3','H2j4']:
                                if el in c.e.keys():
                                    x += c.e[el].col
                            print('H2', x)

    if 0:
        H2energy = [0, 170.5, 509.8, 1015.1, 1681.6]
        COenergy = [2.766 * ((i + 1) * (i)) for i in range(6)]
        print(H2energy)
        for q in Qall.values():
            #if q.name == 'MS0700+6338':
                         #'Gillmon2006' not in q.ref:
            for c in q.comp:
                print('**************')
                print(q.name, q.ref)
                if all([k in c.e.keys() for k in ['H2', 'H2j0', 'H2j1']]):
                    if c.e['H2'].col.val>17:
                        if 'T02' in c.e.keys():
                            print(c.e['T02'].col)
                        else:
                            print('T01')
                            Temp = ExcitationTemp('H2',levels=[0,1])
                            n = []
                            for el in ['H2j0', 'H2j1']:
                                if el in c.e.keys():
                                    if c.e[el].col.type in ['m', 'f']:
                                        n.append(c.e[el].col)
                            print('H2 temp')
                            Temp.calcTemp(n, plot=0, verbose=0)
                            Temp.latex()
                            print(Temp.temp.log())
                            if 'H2j2' in c.e.keys()and c.e['H2'].col.val>17:
                                print('T02')
                                Temp = ExcitationTemp('H2',levels=[0,2])
                                n = []
                                for el in ['H2j0', 'H2j2']:
                                    if el in c.e.keys():
                                        if c.e[el].col.type in ['m', 'f']:
                                            n.append(c.e[el].col)
                                print('H2 temp')
                                Temp.calcTemp(n, plot=0, verbose=0)
                                Temp.latex()
                                print(Temp.temp.log())

    if 0:
        print('start')
        for q in Qall.values():
            #print('name',q.name)
            #if q.name == 'MS0700+6338':
                         #'Gillmon2006' not in q.ref:
            for c in q.comp:
                if 'COj2' in c.e.keys():
                    if 1:
                        print('**************')
                        print('**************')
                        print('**************')
                        print(q.name, q.ref)
                        COtot = a(0, 0, 0, 'l')
                        for el in ['COj0', 'COj1', 'COj2', 'COj3', 'COj4', 'COj5']:
                            if el in c.e.keys():
                                COtot += c.e[el].col
                        print('COtot', COtot)
                        H2tot = a(0, 0, 0, 'l')
                        for el in ['H2j0', 'H2j1', 'H2j2','H2j3','H2j4','H2j5']:
                            if el in c.e.keys():
                                H2tot += c.e[el].col
                        print('H2tot',H2tot)
                        Citot = a(0,0,0,'l')
                        for el in ['CIj0', 'CIj1', 'CIj2']:
                            if el in c.e.keys():
                                Citot += c.e[el].col
                        print('CItot',Citot)
                        print('CO/H2', COtot/H2tot.log())
                        if 'T01' in c.e.keys():
                            T01 = c.e['T01'].col
                            print('T01', T01)
                        if 'T_co' in c.e.keys():
                            TCO = c.e['T_co'].col
                            print('Tco', TCO)
                        if 'n_co_pdr' in c.e.keys():
                            nCO = c.e['n_co_pdr'].col
                            print('nco', nCO)
                            if 'T01' in c.e.keys():
                                x=T01*nCO
                                print('pco', x.log())
                        if 'n_ci_pdr' in c.e.keys():
                            nCi = c.e['n_ci_pdr'].col
                            print('nci', nCi)
                            if 'T01' in c.e.keys():
                                x=T01*nCi
                                print('pci', x.log())

                    if 0:
                        print('**************')
                        if 'T01' in c.e.keys():
                            print('T01', c.e['T01'].col)
                        print('TCO', c.e['T_co'].col)
                        if 'n_ci' in c.e.keys():
                            print('n_ci',c.e['n_ci'].col)
                        print('nco',c.e['n_co'].col)
                        if 'T01' in c.e.keys():
                            if 'n_ci' in c.e.keys():
                                print('Pci', c.e['n_ci'].col*c.e['T01'].col)
                            print('Pco', c.e['n_co'].col*c.e['T01'].col)
                    if 0:
                        if 'Sonnetrucker2007' in q.ref:
                            print()
                            print('name', q.name)
                            print('CO/H2',c.e['CO'].col/c.e['H2'].col)
                            if 'EBV' in q.e.keys():
                                x = 5.8e21*q.e['EBV'].col.val - 2*10**c.e['H2'].col.val
                                print('ebv',q.e['EBV'].col.val)
                                print('NHI',np.log10(x))
                                if 'HI' in q.e.keys():
                                    print('NHI_w',q.e['HI'].col.val)
                            else:
                                print('no data')

    if 0:
        for q in MWsample.values():
            print('name',q.name)
            if 'O/H' in q.e.keys():
                Mesolar = a(483,0,0,'d')
                if 'Si/H' in q.e.keys():
                    dust = q.e['Si/H'].col*4
                    Me = (q.e['O/H'].col)/Mesolar
                    print('O/Hgas', q.e['O/H'].col, 'O/Hdust', q.e['Si/H'].col*4, 'Me', Me.log())
                else:
                    Me = q.e['O/H'].col/Mesolar
                    print('O/Hgas', q.e['O/H'].col, 'O/Hdust', 'Me', Me.log())
            if 'MeZou' in q.e.keys():
                print(q.e['Me'])
                print(q.e['MeZou'])
                print(q.e['CO'])
                print(q.e['H2'])


    if 0:
        fig,ax = plt.subplots(1,4,figsize=(12,3))
        Manga = load_Manga()
        for q in Manga.values():
            if 'Azi' in q.e.keys():
                print(q.name)
                if'Azi_h_mean' in q.e.keys():
                    s = q.e['Azi_h_mean'].col
                    s.val =90-s.val
                    print(q.mangaID,q.e['Azi'].col,s)
                else:
                    print(q.mangaID, q.e['Azi'].col)
                ax[0].errorbar(x=90-q.e['Azi_mean'].col.val,y=90-q.e['Azi'].col.val,xerr=[[q.e['Azi_mean'].col.minus],[q.e['Azi_mean'].col.plus]],fmt='o')
                ax[0].text(90-q.e['Azi_mean'].col.val,90-q.e['Azi'].col.val,q.mangaID)
                ax[1].errorbar(x=90 - q.e['Azi_r_mean'].col.val, y=90 - q.e['Azi'].col.val,
                               xerr=[[q.e['Azi_r_mean'].col.minus], [q.e['Azi_r_mean'].col.plus]], fmt='o')
                ax[2].errorbar(x=90 - q.e['Azi_a_mean'].col.val, y=90 - q.e['Azi'].col.val,
                           xerr=[[q.e['Azi_a_mean'].col.minus], [q.e['Azi_a_mean'].col.plus]], fmt='o')
                ax[3].errorbar(x=90 - q.e['Azi_ra_mean'].col.val, y=90 - q.e['Azi'].col.val,
                               xerr=[[q.e['Azi_ra_mean'].col.minus], [q.e['Azi_ra_mean'].col.plus]], fmt='o')
        for axs in ax[:]:
            axs.set_xlim(0,90)
            axs.set_ylim(0,90)

            #q.el('Azi', 50, f='d')
            #q.el('Azi_mean', 58, 22, 22, f='d')
            #q.el('Azi_r_mean', 63, 11, 11, f='d')
            #q.el('Azi_a_mean', 76, 10, 10, f='d')
            #q.el('Azi_ra_mean', 73, 9, 9, f='d')

    if 0:
        Manga = load_Manga()
        q = Manga['J1237+4447']
        for n in ['SiII','SiIII','CII','SII','NI','NII','NV','FeII','OI']:
            s = 0
            for c in q.comp:
                if n in c.e.keys():
                    s+=c.e[n].col
            print(n,s)

    if 0:
        print('QSO-gal_pairs Kulkarni 2022')
        QSO = load_QSO_pairs()
        for q in QSO.values():
            print('{0: <30}'.format(q.assgalname),q.assgalz)

        print('QSO-gal_pairs Muzahid 2018')
        QSO = load_LLSs()
        for q in QSO.values():
            if hasattr(q, 'assgalname'):
                print('{0: <55}'.format(q.assgalname),q.assgalz)

    if 0:
        print('QSO-gal_pairs Kulkarni 2022')
        QSO = load_MV()
        A = np.zeros((2,100))
        i=0
        for q in QSO.values():
            if 'CO' in q.e.keys():
                if q.e['EBV'].col.val*3.1>0.5:
                    print(q.name,q.e['CO'].col.val,q.e['H2'].col.val,q.e['EBV'].col.val)

    if 0:
        H2energy = [0, 170.5, 509.8, 1015.1, 1681.6]
        QSO = load_MV()
        for q in QSO.values():
            for i, c in enumerate(q.comp):
                if 'pci-tripp' in c.e.keys():
                    if 'H2j0' not in c.e.keys():
                        Texc = c.e['T01'].col.val
                        Ntot = 10**(c.e['H2'].col.val)
                        N0 = Ntot/(1+9*np.exp(-170.5/Texc))
                        N1 = Ntot - N0
                        print(q.name, np.log10(Ntot),np.log10(N0),np.log10(N1))

    #list Jenkins systems
    if 1:
        print('start')
        #from COexcitation.aG import alphaG
        import csv

        fig, ax = plt.subplots()
        list = []
        QSO = load_MV()
        x1,x2 = [],[]
        for q in QSO.values():
            for i,c in enumerate(q.comp):
                if 'CI' in c.e.keys() or 'CO' in  c.e.keys():
                    line = []
                    if 'HD' in q.name and len(q.name)==7:
                        q.name = q.name[:2]+'0'+q.name[2:]
                    line.append(q.name)
                    for el in ['HI', 'H2', 'CI', 'CO', 'T01', 'T_co', 'TC2', 'n_ci_3dpdr', 'n_co_3dpdr','uv_ci_3dpdr','uv_co_3dpdr']:
                        if el == 'HI':
                            if hasattr(q, 'HI'):
                                line.append(q.HI.col.val)
                                line.append((q.HI.col.plus+q.HI.col.minus)/2)
                            else:
                                line.append(-999)
                                line.append(-999)
                        elif el in c.e.keys():
                            line.append(c.e[el].col.val)
                            line.append((c.e[el].col.plus+c.e[el].col.minus)/2)
                        else:
                            line.append(-999)
                            line.append(-999)

                    list.append(line)
        with open('test.csv', 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(
                [['id', 'HI', 'HIer', 'H2', 'H2err', 'CI','CIerr', 'CO', 'COerr', 'T01','T01err', 'T_co','T_co_err',
                  'TC2', 'TC2err', 'n_ci_3dpdr','n_ci_3dpdr_err', 'n_co_3dpdr', 'n_co_3dpdr_err','uv_ci_3dpdr','uv_ci_3dpdr_err','uv_co_3dpdr','uv_co_3dpdr_err']])
            csv_writer.writerows(list)


    #print lines
    if 0:
        print('start')
        #from COexcitation.aG import alphaG
        import csv

        fig, ax = plt.subplots()
        list = []
        QSO = load_co_sample()
        for q in QSO.values():
            for i,c in enumerate(q.comp):
                if 'ngas(co)' in c.e.keys():
                    print(q.name)
                    line = []

                    if 'uv_ci_3dpdr' in c.e.keys():
                        line.append('uv_ci')
                        el = c.e['uv_ci_3dpdr']
                        line.append(el.col.val)
                        line.append(el.col.plus)
                        line.append(el.col.minus)
                    elif 'uv_co_3dpdr' in c.e.keys():
                        line.append('uv_co')
                        el = c.e['uv_co_3dpdr']
                        line.append(el.col.val)
                        line.append(el.col.plus)
                        line.append(el.col.minus)
                    print(line)
                    line = []
                    if 'ngas(co)' in c.e.keys():
                        el = c.e['n_co_3dpdr']
                        line.append('nH_co')
                        line.append(el.col.val)
                        line.append(el.col.plus)
                        line.append(el.col.minus)
                        if 'uv_co_3dpdr' in c.e.keys():
                            line.append('uv_co')
                            el = c.e['uv_co_3dpdr']
                            line.append(el.col.val)
                            line.append(el.col.plus)
                            line.append(el.col.minus)
                        el = c.e['tgas'].col.dec()
                        line.append('tgas')
                        line.append(el.val)
                        line.append(el.plus)
                        line.append(el.minus)
                        el = c.e['ngas(co)']
                        line.append('ngas')
                        line.append(el.col.val)
                        line.append(el.col.plus)
                        line.append(el.col.minus)
                        if 'tgas(ci)' in c.e.keys():
                            el = c.e['tgas(ci)'].col.dec()
                            line.append('tgas(ci)')
                            line.append(el.val)
                            line.append(el.plus)
                            line.append(el.minus)
                            el = c.e['ngas(ci)']
                            line.append(el.col.val)
                            line.append(el.col.plus)
                            line.append(el.col.minus)

                    #print(line)

    if 0:
        print('start')
        #from COexcitation.aG import alphaG
        import csv

        fig, ax = plt.subplots()
        list = []
        list_T01 = []
        list_nH = []
        list_UV = []

        QSO = load_JT()
        for q in QSO.values():
            for i,c in enumerate(q.comp):
                if 'pci-tripp' in c.e.keys():
                    print(q.name)
                    line = []
                    el = c.e['H2']
                    line.append('H2')
                    line.append(el.col.val)
                    line.append(el.col.plus)
                    line.append(el.col.minus)
                    el = c.e['CI']
                    line.append('CI')
                    line.append(el.col.val)
                    line.append(el.col.plus)
                    line.append(el.col.minus)
                    el = c.e['T01']
                    line.append('T01')
                    line.append(el.col.val)
                    line.append(el.col.plus)
                    line.append(el.col.minus)
                    list_T01.append(el.col.val)
                    print(line)
                    line = []
                    el = c.e['n_ci_3dpdr']
                    line.append('n_ci_3dpdr')
                    line.append(el.col.val)
                    line.append(el.col.plus)
                    line.append(el.col.minus)
                    list_nH.append(el.col.val)
                    el = c.e['uv_ci_3dpdr']
                    line.append('uv_ci_3dpdr')
                    line.append(el.col.val)
                    line.append(el.col.plus)
                    line.append(el.col.minus)
                    list_UV.append(el.col.val)
                    el = c.e['tgas(ci)'].col.dec()
                    line.append('tgas(ci)')
                    line.append(el.val)
                    line.append(el.plus)
                    line.append(el.minus)
                    el = c.e['ngas(ci)']
                    line.append('ngas(ci)')
                    line.append(el.col.val)
                    line.append(el.col.plus)
                    line.append(el.col.minus)
                    el = c.e['pci-tripp'].col/c.e['T01'].col
                    el=el.log()
                    line.append('ngas(JT)')
                    line.append(el.val)
                    el = c.e['uvci-tripp']
                    line.append('UV(JT)')
                    line.append(el.col.val)
                    print(line)


    if 0:
        print('start')
        #from COexcitation.aG import alphaG
        import csv

        fig, ax = plt.subplots()
        list = []
        list_T01 = []
        list_nH = []
        list_UV = []

        QSO = load_QSO()
        for q in QSO.values():
            for i,c in enumerate(q.comp):
                if 'CI' in c.e.keys():
                    print(q.name,q.z_dla,c.e['CI'],q.coord,q.m)

    #print(list_T01)
    #print(list_nH)
    #print(list_UV)


    plt.show()


