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
from scipy.interpolate import interp2d, RectBivariateSpline, Rbf
import scipy
from scipy.optimize import curve_fit
from spectro.atomic import e
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 10
import numpy as np




H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/energy_X_H2.dat', dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                          unpack=True, skip_header=3, comments='#')
H2energy = np.zeros([max(H2_energy['nu']) + 1, max(H2_energy['j']) + 1])
for e in H2_energy:
    H2energy[e[0], e[1]] = e[2]
CIenergy = [0, 16.42, 43.41]

stat_H2 = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]
stat_CI = [(2 * i + 1) for i in range(3)]


def getatomic(species, levels=[0, 1, 2]):
    if species == 'H2':
        return [H2energy[0, i] for i in levels], [stat_H2[i] for i in levels]
    elif species == 'CI':
        return [CIenergy[i] for i in levels], [stat_CI[i] for i in levels]


case = 'fig02'
labelsize = 12
msize=5
lnLcolor= 'green'
filepath = "output/"
database = 'z=-1'
nonthermal_sys = ['B0405-4418_0','B0528-2505_1','J0643-5041_0','J1237+0647_0','J1443+2724_0','J1443+2724_1','0551-3638_0', 'J2123-0050_0',
                  'HD195965_0','HD40893_0']


# plot T01 vs T02
if case == 'fig01':
    fig01,ax = plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True)
    if 1:
        database = 'MW'
        H2 = H2_exc(H2database=database)
        x,y,z = [],[],[]
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if all([k in c.e.keys() for k in ['T01', 'T02', 'H2']]):
                        #ax[0].errorbar(y=c.e['T01'].col.val,x=c.e['T02'].col.val,yerr=[[c.e['T01'].col.minus],[c.e['T01'].col.plus]],
                        #               xerr=[[c.e['T02'].col.minus],[c.e['T02'].col.plus]],marker='.',ecolor='black',zorder=-5)
                        nh2 = c.e['H2'].col
                        t01 = c.e['T01'].col.log()
                        t02 = c.e['T02'].col.log()
                        #dt = (t01/t02).log()
                        ax[1].errorbar(x=nh2.val,y=t01.val, yerr=[[t01.minus],[t01.plus]], marker='o',markerfacecolor='black',ecolor='black',zorder=-5)
                                 #yerr=[[c.e['T01'].col.minus], [c.e['T01'].col.plus]], marker='o',markerfacecolor='black')
                        ax[1].errorbar(x=nh2.val, y=t02.val, yerr=[[t02.minus],[t02.plus]], marker='o',markerfacecolor='red',ecolor='black',zorder=-5)
                        #ax[1].errorbar(x=nh2.val, y=dt.val, yerr=[[dt.minus], [dt.plus]], marker='o',
                        #               markerfacecolor='red', ecolor='black', zorder=-5)
                        if nh2.val<20.5:
                            x.append(nh2.val)
                            y.append(t01.val)
                            z.append(t02.val)

                        #yerr=[[c.e['T02'].col.minus], [c.e['T02'].col.plus]], marker='o',markerfacecolor='red')
        #ax[0].scatter(x,y, 10, z, cmap='Reds', vmin=18, vmax=21,alpha=1)
        #ax[0].plot(np.linspace(0,200,10),np.linspace(0,200,10),ls='--')
    if 1:
        m, b = np.polyfit(x, y, 1)
        print('logn - logNH2 linear fit params:', 'm=,b=', m, b)
        x0 = np.linspace(18, 20.5, 10)
        ax[1].plot(x0, b + m * x0, '--', color='blue')
        m, b = np.polyfit(x, z, 1)
        print('logn - logNH2 linear fit params:', 'm=,b=', m, b)
        x0 = np.linspace(18, 20.5, 10)
        ax[1].plot(x0, b + m * x0, '--', color='red')
    if 1:
        database = 'all'
        H2 = H2_exc(H2database=database)
        x, y, z = [], [], []
        label = ['','']
        label[0] = "$T^{01}$"
        label[1] = "$T^{02}$"
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if c.z>1.5:
                    if all([k in c.e.keys() for k in ['T01', 'T02', 'H2']]):
                        #ax[0].errorbar(y=c.e['T01'].col.val, x=c.e['T02'].col.val,
                        #               yerr=[[c.e['T01'].col.minus], [c.e['T01'].col.plus]],
                        #               xerr=[[c.e['T02'].col.minus], [c.e['T02'].col.plus]], marker='D', ecolor='black',
                        #               zorder=-5)

                        nh2 = c.e['H2'].col
                        t01 = c.e['T01'].col.log()
                        t02 = c.e['T02'].col.log()
                        ax[0].errorbar(x=nh2.val, y=t01.val,
                                       yerr=[[t01.minus], [t01.plus]], marker='D',
                                       markerfacecolor='black', ecolor='black', zorder=-5,label=label[0])
                        # yerr=[[c.e['T01'].col.minus], [c.e['T01'].col.plus]], marker='o',markerfacecolor='black')
                        ax[0].errorbar(x=nh2.val, y=t02.val,
                                       yerr=[[t02.minus], [t02.plus]], marker='D',
                                       markerfacecolor='red', ecolor='black', zorder=-5,label=label[1])
                        label[0] = "_nolegend_"
                        label[1] = "_nolegend_"
                        if nh2.val < 20.5:
                            x.append(nh2.val)
                            y.append(t01.val)
                            z.append(t02.val)
                        # yerr=[[c.e['T02'].col.minus], [c.e['T02'].col.plus]], marker='o',markerfacecolor='red')
            #ax[0].scatter(x, y, 10, z, cmap='Reds', vmin=18, vmax=21, alpha=1)
            #ax[0].plot(np.linspace(0, 200, 10), np.linspace(0, 200, 10), ls='--')
        ax[0].legend()
    if 1:
        m, b = np.polyfit(x, y, 1)
        print('logn - logNH2 linear fit params:', 'm=,b=', m, b)
        x0 = np.linspace(17,20.5,10)
        ax[0].plot(x0, b + m*x0,'--',color='blue')
        m, b = np.polyfit(x, z, 1)
        print('logn - logNH2 linear fit params:', 'm=,b=', m, b)
        x0 = np.linspace(17, 20.5, 10)
        ax[0].plot(x0, b + m * x0, '--',color='red')
    if 1:
        for axs in ax:
            axs.xaxis.set_minor_locator(AutoMinorLocator(5))
            axs.xaxis.set_major_locator(MultipleLocator(1))
            axs.yaxis.set_minor_locator(AutoMinorLocator(5))
            axs.yaxis.set_major_locator(MultipleLocator(0.2))
            axs.tick_params(which='both', width=1, direction='in', right='True', top='True')
            axs.tick_params(which='major', length=4, direction='in', right='True', top='True')
            axs.tick_params(which='minor', length=3, direction='in')
            axs.tick_params(axis='both', which='major', labelsize=10, direction='in')
            axs.set_xlabel('$\\log N(H_2)$',fontsize=labelsize)
            axs.set_ylabel('$T^{\\rm H_2}_{01}$, $T^{\\rm H_2}_{02}$, K',fontsize=labelsize)
        ax[0].text(20.5,2.65,'DLAs', fontsize=labelsize)
        ax[1].text(20.5, 2.65, 'MW', fontsize=labelsize)

if case == 'fig02':
    fig02, ax = plt.subplots(2, 1, figsize=(4.5, 8),sharex=False)
    if 1:
        database = 'all'
        H2 = H2_exc(H2database=database)
        x, y, z, dt = [], [], [], []
        n0,n1,n2 = [],[],[]
        n0h, n1h, n2h = [], [], []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if all([k in c.e.keys() for k in ['T01', 'T02', 'H2']]):
                    # ax[0].errorbar(y=c.e['T01'].col.val,x=c.e['T02'].col.val,yerr=[[c.e['T01'].col.minus],[c.e['T01'].col.plus]],
                    #               xerr=[[c.e['T02'].col.minus],[c.e['T02'].col.plus]],marker='.',ecolor='black',zorder=-5)
                    nh2 = c.e['H2'].col
                    nh = q.e['HI'].col
                    #nht = nh + nh2 +0.3
                    t01 = c.e['T01'].col.log()
                    t02 = c.e['T02'].col.log()
                    dt.append(t01.val - t02.val)
                    x.append(nh2.val)
                    # dt = (t01/t02).log()
                    ax[1].errorbar(x=nh2.val, y=t01.val-t02.val, yerr=[[t01.minus], [t01.plus]], marker='o',
                                   markerfacecolor='black', ecolor='black', zorder=-5)
                    # yerr=[[c.e['T01'].col.minus], [c.e['T01'].col.plus]], marker='o',markerfacecolor='black')
                    #ax[1].errorbar(x=nh2.val, y=t02.val, yerr=[[t02.minus], [t02.plus]], marker='o',
                    #               markerfacecolor='red', ecolor='black', zorder=-5)
                    # ax[1].errorbar(x=nh2.val, y=dt.val, yerr=[[dt.minus], [dt.plus]], marker='o',
                    #               markerfacecolor='red', ecolor='black', zorder=-5)
                    #ax[0].plot(nh.val,nh2.val,'o',color='blue')
                    mod = [c.e[k].col.val - c.e['H2j0'].col.val for k in ['H2j0','H2j1','H2j2']]
                    Eh2, Sh2 = getatomic('H2')
                    Eh2 =1.428*np.array(Eh2)
                    if nh2.val< 19.:
                        ax[0].plot(Eh2,np.array(mod)-np.log10(Sh2),'o',color='red',alpha=0.2)
                        n0.append(np.array(mod[0]) - np.log10(Sh2[0]))
                        n1.append(np.array(mod[1]) - np.log10(Sh2[1]))
                        n2.append(np.array(mod[2]) - np.log10(Sh2[2]))
                    if nh2.val> 19:
                        ax[0].plot(Eh2, np.array(mod) - np.log10(Sh2), 'o', color='blue',alpha=0.2)
                        n0h.append(np.array(mod[0]) - np.log10(Sh2[0]))
                        n1h.append(np.array(mod[1]) - np.log10(Sh2[1]))
                        n2h.append(np.array(mod[2]) - np.log10(Sh2[2]))
        ax[0].plot(Eh2,[np.mean(n0),np.mean(n1),np.mean(n2)],'s',color='red',markersize=10)
        ax[0].plot(Eh2, [np.mean(n0h), np.mean(n1h), np.mean(n2h)], 's', color='blue', markersize=10)
        #ax[0].plot(Eh2, np.mean(n0) + np.log10(np.exp(-Eh2/100)))
        #ax[0].plot(Eh2, np.mean(n0h) + np.log10(np.exp(-Eh2/80)), '--')

        if 1:
            m, b = np.polyfit(x, dt, 1)
            print('logn - logNH2 linear fit params:', 'm=,b=', m, b)
            x0 = np.linspace(18, 21, 10)
            ax[1].plot(x0, b + m * x0, '--', color='blue')
#        ax[0].plot(Eh2, np.log10(np.exp(-Eh2/70)))


save = 1
if save:

    figname='Figs/th2_qso.pdf'
    if case == 'fig5':
        figname = "lnL_z-1.pdf"
    if case == 'fig6':
        figname = "fit_H2_z-1.pdf"
    if case == 'fig7':
        figname = "fit_CI_z-1.pdf"
    fig02.savefig("".join((figname)), bbox_inches='tight')

plt.show()
