#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDR_systems.py

Classes for observed absorption systems:
    qso     an absorption system (quasar sightline, star, GRB, ...): name, redshifts, references, values (q.e)
            and components (q.comp)
    sy      a velocity component of a system, with its values (c.e)
    sample  an ordered collection of systems {name: qso}

Values are PDR_tools.e objects: e.g. q.e['H2'].col.val, .plus, .minus.
Samples are read from the FITS database with PDR_database.PDRdatabase.load(group).
"""
from collections import OrderedDict

import numpy as np

from PDR_tools import a, e


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
