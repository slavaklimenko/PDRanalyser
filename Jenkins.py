import numpy as np
from scipy.interpolate import Rbf
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from spectro.stats import distr2d
from H2_exc import H2_exc
from os import listdir
from os.path import isfile, join
from scipy import interpolate,integrate, optimize
import matplotlib.gridspec as gridspec
from scipy.stats import moment
from H2_exc import *
from H2_excitation import H2_summary
from spectro.atomic import e
matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 10
from spectro.a_unc import a
import pickle

class CI_sys:
    def __init__(self, name):
        self.name = name
        self.N1 = a(0,0,0,'d')
        self.N2 = a(0,0,0,'d')
        self.Ntot = a(0,0,0,'d')


database = 'JT'
H2 = H2_exc(H2database=database)
list_of_stars = []
if 0:
    CIdata = {}
    for q in H2.H2.values():
        list_of_stars.append(q.name)
        print(q.name)
        CIdata[q.name] = CI_sys(q.name)
    print(np.sort(list_of_stars))
    with open('data/J_ApJ_734_65_table4.dat.text', 'r') as f:
        data = f.readlines()
    ind = 0
    name = 'CPD-592603'
    CIdata = {}
    CIdata[name] = CI_sys(name)
    for d in data:
        words = d.split()
        #name = words[0]
        f1 = float(words[2])
        f1err = float(words[3])
        f2 = float(words[4])
        f2err = float(words[5])
        NCItot = float(words[6])
        #if name in list_of_stars:
        if 1:
            if words[0] != name:
                name = words[0]
                CIdata[name] = CI_sys(name)
                print(name)
            f =  a(f1,f1err,f1err,'d')
            CIdata[name].N1 += a(f1,f1err,f1err,'d')*NCItot
            CIdata[name].N2 += a(f2,f2err,f2err,'d')*NCItot
            CIdata[name].Ntot += a(0,0.05,0.05)*NCItot
            #print(NCItot,np.log10(NCItot))
            if 1:
                testf1 = (CIdata[name].N1 / CIdata[name].Ntot).dec()
                testf2 = (CIdata[name].N2 / CIdata[name].Ntot).dec()
                print(f1,f2,testf1,testf2)


    with open('MW_data_CI_test_all.pkl', 'wb') as f:
        pickle.dump(CIdata, f)

if 0:
    CIdata = {}
    with open('data/J_ApJ_734_65_table4.dat.text', 'r') as f:
        data = f.readlines()
    ind = 0
    CIdata['1']= CI_sys('1')
    CIdata['2']= CI_sys('2')
    for d in data:
        words = d.split()
        name = words[0]
        f1 = float(words[2])
        f1err = float(words[3])
        f2 = float(words[4])
        f2err = float(words[5])
        NCItot = float(words[6])
        if name == 'HD210839':
            if float(words[1])<-20:
                CIdata['1'].N1 += a(f1, f1err, f1err, 'd') * NCItot
                CIdata['1'].N2 += a(f2, f2err, f2err, 'd') * NCItot
                CIdata['1'].Ntot += a(1, 0.01, 0.01, 'd') * NCItot
            else:
                CIdata['2'].N1 += a(f1, f1err, f1err, 'd') * NCItot
                CIdata['2'].N2 += a(f2, f2err, f2err, 'd') * NCItot
                CIdata['2'].Ntot += a(1, 0.01, 0.01, 'd') * NCItot
    print('print 201839')
    n0 = CIdata['1'].Ntot - CIdata['1'].N1 - CIdata['1'].N2
    print(n0.log(),CIdata['1'].N1.log(), CIdata['1'].N2.log(), CIdata['1'].Ntot.log())
    n0 = CIdata['2'].Ntot - CIdata['2'].N1 - CIdata['2'].N2
    print(n0.log(), CIdata['2'].N1.log(),CIdata['2'].N2.log(),CIdata['2'].Ntot.log())

if 1:
    with open('MW_data_CI_test_all.pkl', 'rb') as f:
        CIdata = pickle.load(f)

    for q in CIdata:
        if CIdata[q].N1.val > 0:
            x = CIdata[q].Ntot - CIdata[q].N1 - CIdata[q].N2
            f1 = (CIdata[q].N1/CIdata[q].Ntot).dec()
            f2 = (CIdata[q].N2/CIdata[q].Ntot).dec()
            print(CIdata[q].name, CIdata[q].Ntot.log(), x.log(), CIdata[q].N1.log(),CIdata[q].N2.log(), f1.val, f2.val)


if 0:
    size=1000000
    Ntot = np.random.normal(13.36,0.10,size)
    T10 = np.random.normal(2.8,0.5,size)
    T20 = np.random.normal(3.3,0.5,size)
    #T30 = np.random.normal(5.0, 0.5, size)
    try:
        T20
    except NameError:
        T20 = T10
    try:
        T30
    except NameError:
        T30 = T10
    if T20 is not None:
        N0 = Ntot + np.log10(1 / (1 + 3 * np.exp(-5.53 / T10) + 5 * np.exp(-16.59 / T20) + 7* np.exp(-33.192 / T30) ))
    else:
        N0 = Ntot + np.log10(1 / (1+ 3 * np.exp(-5.53 / T10) ))
    N1 = N0 + np.log10(3*np.exp(-5.53/T10))
    N2 = N0 + np.log10(5*np.exp(-16.59/T20))
    N3 = N0 + np.log10(7*np.exp(-33.192/T30))
    Ntot_t = np.log10(10**N0 + 10**N1)
    Texc =  -5.53/np.log(10**N1/10**N0/3)
    Texc20 = -16.59 / np.log(10 ** N2 / 10 ** N0 / 5)
    print('N0=',np.mean(N0),np.std(N0))
    print('N1=', np.mean(N1), np.std(N1))
    print('N2=', np.mean(N2), np.std(N2))
    print('N3=', np.mean(N3), np.std(N3))
    print('Ntot=',np.mean(Ntot_t),np.std(Ntot_t))
    print('Texc=', np.mean(Texc), np.std(Texc))
    print('Texc20=', np.mean(Texc20), np.std(Texc20))

if 0:
    size=100000
    Ntot = np.random.normal(20.69,0.10,size)
    T10 = np.random.normal(86,17,size)
    N0 = Ntot + np.log10(1 / (1+ 9 * np.exp(-170.5 / T10) ))
    N1 =N0 + np.log10(9*np.exp(-170.5/T10))
    Ntot_t = np.mean(np.log10(10**N0 + 10**N1))
    Texc =  np.mean(-170.5/np.log(10**N1/10**N0/9))
    print('N0=',np.mean(N0),np.std(N0))
    print('N1=', np.mean(N1), np.std(N1))
    print('Ntot=',Ntot_t)
    print('Texc=', Texc)


print('end')


