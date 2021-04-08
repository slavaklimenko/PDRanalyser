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


case = 'fig11'
labelsize = 10
msize=5
lnLcolor= 'green'
filepath = "output/"
database = 'z=-1'
nonthermal_sys = ['B0405-4418_0','B0528-2505_1','J0643-5041_0','J1237+0647_0','J1443+2724_0','J1443+2724_1','0551-3638_0', 'J2123-0050_0',
                  'HD195965_0','HD40893_0']
if case == 'fig1_3':

    fig01, ax = plt.subplots(2,2,figsize=(9,4))
    fig01.subplots_adjust(wspace=0.4,hspace=0.3)
    H2 = H2_exc(folder='./data/sample/1_5_4/av0_5_cmb2_5_z0_1_n_uv/')
    H2.readmodel(filename='pdr_grid_z0_1_n2e2_uv0_7_cmb2_5_pah_av0_5_s_25.hdf5')
    for m in H2.listofmodels():
        #ax[0].plot(np.log10(m.h2),np.log10(m.tgas))
        ax[0,0].plot(np.log10(m.h2),np.log10(m.sp['H']),label='H',color='royalblue')
        ax[0,0].plot(np.log10(m.h2), np.log10(m.sp['H2']),label='H$_2$',color='orange')
        ax[0,0].plot(np.log10(m.h2), np.log10(m.sp['H+']),label='H$^+$',color='green')
        if 1:
            axi = ax[0,0].twinx()
            axi.plot(np.log10(m.h2),m.tgas,color='red',ls='--')
            axi.set_ylabel('Temperature, K', color='red', fontsize=labelsize)
            axi.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True',
                            color='red')
            axi.tick_params(which='major', length=5, color='red')
            axi.tick_params(which='minor', length=2, color='red')
            axi.set_ylim(20, 80)
            axi.yaxis.set_minor_locator(AutoMinorLocator(4))
            axi.yaxis.set_major_locator(MultipleLocator(20))
            for t in axi.get_yticklabels():
                t.set_color('red')

        for s in ['NCj1', 'NCj2']:
            ax[1,1].plot(np.log10(m.h2), np.log10(m.sp[s] / m.sp['NCj0']),label="".join(['CIJ', s[3]]))
        if 1:
            # add J0812 CI/CIJ=0 observed ratio
            #ax[0,1].axhline(-0.28, ls = '--',color='black')
            ax[1,1].axhline(-0.83, ls = '--',color='black')
            ax[1,1].axhspan(-0.48,-0.08,facecolor = 'black',alpha = 0.2)
            ax[1,1].plot(np.linspace(13,22,10), -0.28 + np.zeros(10), color = 'black',label='J0812$+$3208$_0$',ls = '--')
            ax[1,1].axhspan(-0.83-0.21, -0.83+0.21, facecolor='black', alpha=0.2)

        for k,s in enumerate(['NH2j0','NH2j1','NH2j2','NH2j3','NH2j4']):
            ax[0,1].plot(np.log10(m.h2), np.log10(m.sp[s]/m.sp['NH2']),label="".join(['J', s[4]]))

        name = 'J0812+3208_0'
        h1 = H2.plot_objects(objects=name, ax=ax[1,0], syst=0.3,label='J0812$+$3208$_0$')
        h2 = H2.plot_models(ax=ax[1,0], models='all', logN=16,species='<5',color='red',labelsize=labelsize,label='$\log~N({\\rm{H_2}})=16$',legend=False)
        h3 = H2.plot_models(ax=ax[1,0], models='all', logN=19.6,species='<5',color='purple',labelsize=labelsize,label='$\log~N({\\rm{H_2}})=19.6$',legend=False)
        h4 = H2.plot_models(ax=ax[1,0], models='all', logN=21,species='<5',color='orange',labelsize=labelsize,label='$\log~N({\\rm{H_2}})=21$',legend=False)

        for axs in [ax[0,0], ax[0,1],ax[1,1]]:
            axs.axvline(16,ls='--',color='red',lw=1)
            axs.axvline(19.6,ls='--',color='purple',lw=1)
            axs.axvline(21,ls='--',color='orange',lw=1)


    #tune axes
    if 1:
        ax[0,1].set_ylabel('$\log~N({\\rm{H_2}})_{\\rm J}/N({\\rm{H_2}})_{\\rm tot}$', fontsize=labelsize)
        ax[0,1].set_ylim(-7, 0.5)
        ax[1,1].set_ylabel('$\log~N({\\rm{CI}})_{\\rm J}/N({\\rm{CI}})_{\\rm 0}$', fontsize=labelsize)
        ax[1,1].set_ylim(-1.5, 0.3)
        for axs in [ax[0,0],ax[0,1], ax[1,0],ax[1,1]]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)
        #ax[1,1].tick_params(which='left',labelsize=labelsize-2)
        ax[0,0].tick_params(
            which='both',  # both major and minor ticks are affected
            right=False,  # ticks along the bottom edge are off
            )

        ax[0,0].set_ylabel('$\log~n({\\rm{X}}), {\\rm cm}^{-3}$', fontsize=labelsize)
        for axs in  [ax[0,0],ax[1,1], ax[0,1]]:
            axs.set_xlabel('$\log~N({\\rm{H_2}}), {\\rm{cm}}^{-2}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(4))
            axs.xaxis.set_major_locator(MultipleLocator(2))
            axs.set_xlim(14,np.log10(m.h2)[-1])
        ax[0,0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0,0].yaxis.set_major_locator(MultipleLocator(1))
        ax[0,1].yaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1,1].yaxis.set_major_locator(MultipleLocator(2))
        ax[1,1].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1,1].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[1,0].yaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1,0].yaxis.set_major_locator(MultipleLocator(2))
        ax[1,0].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1,0].xaxis.set_major_locator(MultipleLocator(400))

        #set legends
        ax[0,0].text(14.5,-2.5,'H$+$',fontsize=labelsize,color='green')
        ax[0,0].text(14.5,-0.8,'H$_2$',fontsize=labelsize,color='orange')
        ax[0,0].text(14.5,1.5,'H',fontsize=labelsize,color='royalblue')
        ax[0,0].set_ylim(-3,2.8)
        #ax[1].text(18, -6, 'J=4', fontsize=labelsize, color='purple')
        #ax[1].text(18, -5, 'J=3', fontsize=labelsize, color='red')
        ax[0,1].legend(bbox_to_anchor=(0.1, 0.35, 0.1, 0.2), frameon=True,fontsize=labelsize-4)
        ax[1,1].legend(bbox_to_anchor=(0.16, 0.17, 0.2, 0.2), frameon=True,fontsize=labelsize-4)
        handles, labels = ax[1,0].get_legend_handles_labels()
        order = [3,0,1,2]
        ax[1,0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(0.23, 0.24, 0.17, 0.2),
                       frameon=False, fontsize=labelsize - 4)
        ax[1,0].text(400,-1, 'Rotational excitation of H$_2$',fontsize=labelsize)

if case == 'fig1_2':

    fig01, ax = plt.subplots(1,4,figsize=(9,2))
    fig01.subplots_adjust(wspace=0.4)
    H2 = H2_exc(folder='./data/sample/1_5_4/av0_5_cmb2_5_z0_1_n_uv/')
    H2.readmodel(filename='pdr_grid_z0_1_n1e2_uv01_cmb2_5_pah_s_25.hdf5')
    for m in H2.listofmodels():
        #ax[0].plot(np.log10(m.h2),np.log10(m.tgas))
        ax[0].plot(np.log10(m.h2),np.log10(m.sp['H']),label='H',color='royalblue')
        ax[0].plot(np.log10(m.h2), np.log10(m.sp['H2']),label='H$_2$',color='orange')
        ax[0].plot(np.log10(m.h2), np.log10(m.sp['H+']),label='H$^+$',color='green')
        if 0:
            axi = ax[0].twinx()
            axi.plot(np.log10(m.h2),np.log10(m.tgas),color='red')
            # axi.set_ylabel('$\log T_{\\rm{kin}},K$', color='red', fontsize=labelsize)
            axi.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True',
                            color='red')
            axi.tick_params(which='major', length=5, color='red')
            axi.tick_params(which='minor', length=2, color='red')
            axi.set_ylim(1, 3)
            axi.yaxis.set_minor_locator(AutoMinorLocator(5))
            axi.yaxis.set_major_locator(MultipleLocator(0.5))
            for t in axi.get_yticklabels():
                t.set_color('red')

        for s in ['H2j0','H2j1','H2j2','H2j3','H2j4']:
            ax[1].plot(np.log10(m.h2), np.log10(m.sp[s]/m.sp['H2']),label="".join(['J', s[3]]))

        for s in ['CIj0', 'CIj1', 'CIj2']:
            ax[2].plot(np.log10(m.h2), np.log10(m.sp[s] / m.sp['C']),label="".join(['CIJ', s[3]]))

        H2.plot_models(ax=ax[3], models='all', logN=17,species='<5',color='red',labelsize=labelsize,label='$\log~N=16$',legend=False)
        H2.plot_models(ax=ax[3], models='all', logN=19.9,species='<5',color='purple',labelsize=labelsize,label='$\log~N=19.9$',legend=False)
        H2.plot_models(ax=ax[3], models='all', logN=21,species='<5',color='orange',labelsize=labelsize,label='$\log~N=21$',legend=False)
        name = 'J0812+3208_0'
        H2.plot_objects(objects=name, ax=ax[3], syst=0.3,label='J0812$+$3208$_0$')

        for axs in [ax[0],ax[1], ax[2]]:
            axs.axvline(17,ls='--',color='red',lw=1)
            axs.axvline(19.9,ls='--',color='purple',lw=1)
            axs.axvline(21,ls='--',color='orange',lw=1)


    #tune axes
    if 1:
        ax[1].set_ylabel('$\log~n({\\rm{H_2(J)}})/n({\\rm{H_2}})$', fontsize=labelsize)
        ax[1].set_ylim(-7, 0.5)
        ax[2].set_ylabel('$\log~n({\\rm{CI(J)}})/n({\\rm{CI}})$', fontsize=labelsize)
        ax[2].set_ylim(-1, 0.0)
        for axs in ax[:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)
        ax[2].tick_params(which='Left',labelsize=labelsize-2)

        ax[0].set_ylabel('$\log~n(X), {\\rm cm}^{-3}$', fontsize=labelsize)
        for axs in  [ax[0],ax[1], ax[2]]:
            axs.set_xlabel('$\log~N({\\rm{H_2}}), {\\rm{cm}}^{-2}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(4))
            axs.xaxis.set_major_locator(MultipleLocator(2))
            axs.set_xlim(14,np.log10(m.h2)[-1])
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1].yaxis.set_major_locator(MultipleLocator(2))
        ax[2].yaxis.set_minor_locator(AutoMinorLocator(4))
        ax[2].yaxis.set_major_locator(MultipleLocator(0.2))

        #set legends
        ax[0].text(15,-1.2,'H$+$',fontsize=labelsize,color='green')
        ax[0].text(15,1.2,'H$_2$',fontsize=labelsize,color='orange')
        ax[0].text(15,2.5,'H',fontsize=labelsize,color='royalblue')
        #ax[1].text(18, -6, 'J=4', fontsize=labelsize, color='purple')
        #ax[1].text(18, -5, 'J=3', fontsize=labelsize, color='red')
        ax[1].legend(bbox_to_anchor=(0.3, 0.25, 0.1, 0.2), frameon=True,fontsize=labelsize-4)
        ax[2].legend(bbox_to_anchor=(0.4, 0.1, 0.1, 0.2), frameon=True,fontsize=labelsize-4)
        ax[3].legend(bbox_to_anchor=(0.31, 0.8, 0.2, 0.2), frameon=True, fontsize=labelsize - 4)

if case == 'fig1':
    H2 = H2_exc(folder='./data/test/')
    H2.readfolder()
    for m in H2.listofmodels():
        if 0:
            H2.red_hdf_file(filename=m.filename)

    m.plot_profiles_test(species=['H', 'H2','H+'], ax=ax[0], logx=True, logy=True, label='$\log~{\\rm{n(X)}}$, cm$^{-3}$', parx='av',
                         normed=False,fontsize=labelsize)

    # add top axis label
    if 1:
        x = np.log10(m.av)
        xx = np.log10(m.h2)

        def h2toav(b):
            return np.interp(b, xx, x)

        def avtoh2(a):
            return np.interp(a, x, xx)

        secax = ax[0].secondary_xaxis('top', functions=(avtoh2, h2toav), color='red')
        secax.set_xlabel('$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)
        secax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        secax.tick_params(which='major', length=5, direction='in', right='True', top='True')
        secax.tick_params(which='minor', length=4, direction='in')
        secax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        secax.xaxis.set_major_locator(MultipleLocator(1))
        secax.xaxis.set_minor_locator(AutoMinorLocator(1))

    y = h2toav(np.linspace(12,20,10))
    print(np.linspace(10,20,11),y)

    ax[0].set_xlim(-4, -0.2)
    ax[0].axvline(x=-1.491, ls='--', color='red',zorder=-5)
    ax[0].axvline(x=-0.3, ls='--', color='black',zorder=-5)
    ax[0].xaxis.set_major_locator(MultipleLocator(1))
    ax[0].yaxis.set_major_locator(MultipleLocator(1))
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))

    if 1:
        m.plot_profiles_test(species=['NH', 'NH2j0','NH2j1','NH2j2','NH2j3','NH2j4','NH2j5'], ax=ax[1], logx=True, logy=True,
                             label='$\log~{\\rm{n(X)}}$, cm$^{-3}$', parx='av',
                             normed=False, fontsize=labelsize)
        ax[1].set_xlim(-4, -0.2)
        ax[1].set_ylim(12, 22)
        ax[1].axvline(x=-1.491, ls='--', color='red',zorder=-5)
        ax[1].axvline(x=-0.3, ls='--', color='black',zorder=-5)
        ax[1].xaxis.set_major_locator(MultipleLocator(1))
        ax[1].yaxis.set_major_locator(MultipleLocator(2))
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))

    if 1:
        species = ['H2j0', 'H2j1', 'H2j2','H2j3', 'H2j4', 'H2j5']
        m.calc_cols(species=species, logN={'H2': 19.3})
        ax[2].set_xlabel('Energy level H$_2$(J), K', fontsize=labelsize)
        ax[2].set_ylabel('$\log N({\\rm{H_2,J})/g({\\rm{J}})$', fontsize=labelsize)
        print([H2energy[0,i] for i in range(len(species))], [m.cols[s] for s in species])
        ax[2].plot([H2energy[0,i] for i in range(len(species))], [m.cols[s] - np.log10(stat_H2[k]) for k,s in enumerate(species)],'o',
                   markeredgecolor = 'black',markerfacecolor = 'royalblue',markersize=8)
        ax[2].set_ylim(12,19.5)
        ax[2].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[2].xaxis.set_major_locator(MultipleLocator(1000))
        ax[2].yaxis.set_minor_locator(AutoMinorLocator(1))
        ax[2].yaxis.set_major_locator(MultipleLocator(1))
        ax[2].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[2].tick_params(which='major', length=5)
        ax[2].tick_params(which='minor', length=4)

if case == 'fig2_1':
    fig02, ax = plt.subplots(2, 3, figsize=(9,4))
    fig02.subplots_adjust(hspace=0.4)
    with open('Figs/fig02/lnL_H2_nodes.pkl', 'rb') as f:
        x1, y1, z1 = pickle.load(f)
    with open('Figs/fig02/lnL_H2.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        vmin = -20
        vmax = 0
        cmap = 'Purples'
        zmax = np.amax(z)
        ax[1,2].pcolor(X, Y, z-zmax, cmap=cmap, vmin=vmin, vmax=vmax)
        #ax[1,2].scatter(x1, y1, 15, z1-zmax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='black')
        ax[1,2].title.set_text('$\log{\\rm{\,Likelihood}}$')
        d = distr2d(x=x, y=y, z=np.exp(z))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        cmap_lnL = None
        d.plot_contour(ax=ax[1,2], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)
        xlnL,ylnL,zlnL = x,y,z

    cmap = 'Oranges'
    msize= 15
    if 1:
        with open('Figs/fig02/H2j0_nodes', 'rb') as f:
            x1, y1, z1 = pickle.load(f)
        with open('Figs/fig02/H2j0', 'rb') as f:
            x, y, z = pickle.load(f)
            X, Y = np.meshgrid(x, y)
            vmin = 12
            vmax = 21
            ax[0,0].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
            #ax[0,0].scatter(x1, y1, msize, z1, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='black')
            ax[0,0].title.set_text('$\log N({\\rm{H_2, J=0}})$')
            ax[0,0].contour(x, y, z, levels=[19.53, 19.83, 20.13], colors='black', linewidths=1.0, alpha=1,linestyles =['--','-','--'])
            #ax[0,1].contour(xlnL, ylnL, zlnL - zmax, levels=[-0.16], colors=lnLcolor, linewidths=2.0, alpha=1,
            #              linestyles=['-'])
            d.plot_contour(ax=ax[0,0], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)

    if 1:
        with open('Figs/fig02/H2j1_nodes', 'rb') as f:
            x1, y1, z1 = pickle.load(f)
        with open('Figs/fig02/H2j1', 'rb') as f:
            x, y, z = pickle.load(f)
            X, Y = np.meshgrid(x, y)
            vmin = 12
            vmax = 21
            #cmap = 'Reds'
            ax[0,1].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
            #ax[0,1].scatter(x1, y1, msize, z1, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='black')
            ax[0,1].title.set_text('$\log N({\\rm{H_2, J=1}})$')
            #ax[1].contourf(x, y, z, levels=[19.5,19.83,20.1], cmap='Greens',alpha=0.2)
            ax[0,1].contour(x,y, z, levels=[18.95,19.25,19.55], colors='black', alpha=1,linestyles =['--','-','--'], linewidths=1.0)
            #ax[0,2].contour(xlnL, ylnL, zlnL - zmax, levels=[-0.16], colors=lnLcolor, linewidths=2.0, alpha=1,
            #              linestyles=['-'])
            d.plot_contour(ax=ax[0,1], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)

    if 1:
        with open('Figs/fig02/H2j2_nodes', 'rb') as f:
            x1, y1, z1 = pickle.load(f)
        with open('Figs/fig02/H2j2', 'rb') as f:
            x, y, z = pickle.load(f)
            X, Y = np.meshgrid(x, y)
            #vmin = 12
            #vmax = 19
            #cmap = 'Reds'
            ax[0,2].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
            #ax[0,2].scatter(x1, y1, msize, z1, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='black')
            ax[0,2].title.set_text('$\log N({\\rm{H_2, J=2}})$')
            ax[0,2].contour(x, y, z, levels=[16.2,16.5,16.8], colors='black', alpha=1,linestyles =['--','-','--'],linewidths=1.0)
            #ax[1,0].contour(xlnL, ylnL, zlnL - zmax, levels=[-0.16], colors=lnLcolor, linewidths=2.0, alpha=1,
            #              linestyles=['-'])
            d.plot_contour(ax=ax[0,2], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)

    if 1:
        with open('Figs/fig02/H2j3_nodes', 'rb') as f:
            x1, y1, z1 = pickle.load(f)
        with open('Figs/fig02/H2j3', 'rb') as f:
            x, y, z = pickle.load(f)
            X, Y = np.meshgrid(x, y)
            #vmin = 12
            #vmax = 19
            ax[1,0].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
            #ax[1,1].scatter(x1, y1, msize, z1, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='black')
            ax[1,0].title.set_text('$\log N({\\rm{H_2, J=3}})$')
            ax[1,0].contour(x, y, z, levels=[14.85,15.15,15.45], colors='black', linewidths=1.0, alpha=1,linestyles =['--','-','--'])
            #ax[1,1].contour(xlnL, ylnL, zlnL - zmax, levels=[-0.16], colors=lnLcolor, linewidths=2.0, alpha=1,
            #              linestyles=['-'])
            d.plot_contour(ax=ax[1, 0], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)

    if 1:
        with open('Figs/fig02/H2j4_nodes', 'rb') as f:
            x1, y1, z1 = pickle.load(f)
        with open('Figs/fig02/H2j4', 'rb') as f:
            x, y, z = pickle.load(f)
            X, Y = np.meshgrid(x, y)
            #vmin = 12
            #vmax = 17
            c = ax[1,1].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
            #c = ax[1,2].scatter(x1, y1, msize, z1, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='black')
            ax[1,1].title.set_text('$\log N({\\rm{H_2, J=4}})$')
            ax[1,1].contour(x, y, z, levels=[13.65, 13.95, 14.25], colors='black', linewidths=1.0, alpha=1,linestyles =['--','-','--'])
            #ax[1,2].contour(xlnL, ylnL, zlnL - zmax, levels=[-0.16], colors=lnLcolor, linewidths=2.0, alpha=1,
            #              linestyles=['-'])
            d.plot_contour(ax=ax[1, 1], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)

    if 1:
        for axs in ax[0,:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=4)
        for axs in ax[1,:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=4)

    ax[0,0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
    ax[1,0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
    ax[1,0].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
    ax[1,1].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
    ax[1,2].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)

    if 1:
        cax = fig02.add_axes([0.93, 0.27, 0.01, 0.47])
        fig02.colorbar(c, cax=cax, orientation='vertical', ticks=[12,15,18,21])
        cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        ax[1,2].text(7, 2.0, '$\log N{\\rm(H_2, J)}$, cm$^{-2}$', fontsize=labelsize+2,rotation = 90)

if case == 'fig2_2':
    fig02, ax = plt.subplots(2, 4, figsize=(9,4))
    fig02.subplots_adjust(hspace=0.4,wspace=0.5)
    name = 'J0812+3208_0' #J0812+3208_0_H2_lnL_j02
    with open('output/{:s}_H2_lnL_{:s}.pkl'.format(name,'j02'), 'rb') as f:
        x, y, z = pickle.load(f)
    X, Y = np.meshgrid(x, y)
    vmin = -40
    vmax = 0
    zmax = np.amax(z)
    ax[0,0].title.set_text('${\\rm{H_2{(J=0\\,to\\,J=2)}}}$')
    d = distr2d(x=x, y=y, z=np.exp(z))
    dx, dy = d.marginalize('y'), d.marginalize('x')
    dy.stats(latex=-1, name='log UV')
    dx.stats(latex=-1, name='log n')
    cmap_lnL = 'Purples'
    lnLcolor = 'purple'
    d.plot_contour(ax=ax[0,0], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0)
    d.plot_contour(ax=ax[0,3], color='blue', color_point=None, cmap=None, alpha=0, lw=1.0,zorder=-5)

    # plot H2 excitaiton
    with open('output/{:s}_H2_cols.pkl'.format(name), 'rb') as f:
        xH2, yH2, cols_H2 = pickle.load(f)
    database = 'H2UV'
    species = 'H2'
    H2 = H2_exc(H2database=database)
    H2.plot_objects(objects=name, ax=ax[1,0], syst=0.3, label='Obs. data')
    d.dopoint()

    q = H2.comp(name)
    sp = [s for s in q.e.keys() if species + 'j' in s]
    j = np.sort([0,1,2]) #[int(s[3:]) for s in sp if 'v' not in s])
    x, stat = getatomic(species, levels=j)
    mod = [cols_H2['H2j' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
    ax[1,0].plot([H2energy[0,i] for i in j], mod,color='purple')

    with open('output/{:s}_H2_lnL_{:s}.pkl'.format(name,'j03'), 'rb') as f:
        x, y, z = pickle.load(f)
    X, Y = np.meshgrid(x, y)
    vmin = -40
    vmax = 0
    zmax = np.amax(z)
    ax[0,1].title.set_text('${\\rm{H_2{(J=0\\,to\\,J=3)}}}$')
    d = distr2d(x=x, y=y, z=np.exp(z))
    dx, dy = d.marginalize('y'), d.marginalize('x')
    dy.stats(latex=-1, name='log UV')
    dx.stats(latex=-1, name='log n')
    cmap_lnL = 'Purples'
    lnLcolor = 'purple'
    d.plot_contour(ax=ax[0,1], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0)

        # plot H2 excitaiton
    database = 'H2UV'
    species = 'H2'
    H2 = H2_exc(H2database=database)
    H2.plot_objects(objects=name, ax=ax[1, 1], syst=0.3, label='Obs. data')
    d.dopoint()

    q = H2.comp(name)
    sp = [s for s in q.e.keys() if species + 'j' in s]
    j = np.sort([0, 1, 2,3])  # [int(s[3:]) for s in sp if 'v' not in s])
    x, stat = getatomic(species, levels=j)
    mod = [cols_H2['H2j' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
    ax[1, 1].plot([H2energy[0, i] for i in j], mod, color='purple')

    with open('output/{:s}_H2_lnL.pkl'.format(name), 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        vmin = -40
        vmax = 0
        zmax = np.amax(z)
        ax[0,2].title.set_text('${\\rm{H_2{(J=0\\,to\\,J=4)}}}$')
        d = distr2d(x=x, y=y, z=np.exp(z))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        cmap_lnL = 'Purples'
        lnLcolor = 'purple'
        d.plot_contour(ax=ax[0,2], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0)
        d.plot_contour(ax=ax[0, 3], color=lnLcolor, color_point=None, cmap=None, alpha=0, lw=1.0,zorder=5,ls='--')

    # plot H2 excitaiton
    database = 'H2UV'
    species = 'H2'
    H2 = H2_exc(H2database=database)
    H2.plot_objects(objects=name, ax=ax[1, 2], syst=0.3, label='Obs. data')
    d.dopoint()

    q = H2.comp(name)
    sp = [s for s in q.e.keys() if species + 'j' in s]
    j = np.sort([0, 1, 2,3,4])  # [int(s[3:]) for s in sp if 'v' not in s])
    x, stat = getatomic(species, levels=j)
    mod = [cols_H2['H2j' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
    ax[1, 2].plot([H2energy[0, i] for i in j], mod, color='purple')


    if 1:
         with open('output/{:s}_CI_lnL.pkl'.format(name), 'rb') as f:

             xCi, yCi, zCi = pickle.load(f)
             d = distr2d(x=xCi, y=yCi, z=np.exp(zCi))
             dx, dy = d.marginalize('y'), d.marginalize('x')
             dy.stats(latex=-1, name='log UV')
             dx.stats(latex=-1, name='log n')
             cmap_lnL = 'Greens'
             lnLcolor = 'green'
             print('plot_ci_contours')
             d.plot_contour(ax=ax[0,3], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0,conf_levels=[0.3,0.68]) #color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)
             ax[0, 3].title.set_text('${\\rm{CI(J=1\\,and\\,J=2)}}$')

             with open('output/{:s}_CI_cols.pkl'.format(name), 'rb') as f:
                 xH2, yH2, cols_CI = pickle.load(f)
             database = 'H2UV'
             species = 'CI'
             H2 = H2_exc(H2database=database)
             H2.plot_objects(objects=name, ax=ax[1, 3], syst=0.0, species='CI', label='Observation')
             d.dopoint()

             q = H2.comp(name)
             sp = [s for s in q.e.keys() if species + 'j' in s]
             j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
             x, stat = getatomic(species, levels=j)
             mod = [cols_CI['CIj' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
             ax[1, 3].plot([CIenergy[i] for i in j], mod-mod[0], color='green')

    #tune axes
    if 1:
        for axs in ax[0, :]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)
            axs.set_ylim(-1,3)
        for axs in ax[1, :]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)

        ax[0, 0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        for axs in [ax[0, 0],ax[0, 1], ax[0, 2], ax[0, 3]]:
            axs.set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(5))
            axs.xaxis.set_major_locator(MultipleLocator(1))
            axs.yaxis.set_minor_locator(AutoMinorLocator(5))
            axs.yaxis.set_major_locator(MultipleLocator(1))

        for axs in [ax[1,0],ax[1,1],ax[1,2]]:
            axs.set_xlabel('Energy of H$_2$ levels, cm$^{-1}$', fontsize=labelsize)
            axs.set_ylabel('$\log N{\\rm(H_2)}_{\\rm{J}}/g_{\\rm{J}}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(5))
            axs.xaxis.set_major_locator(MultipleLocator(500))
            axs.yaxis.set_minor_locator(AutoMinorLocator(4))
            axs.yaxis.set_major_locator(MultipleLocator(2))
        ax[1,3].set_xlabel('Energy of C\,{\\sc i} levels, cm$^{-1}$', fontsize=labelsize)
        ax[1,3].set_ylabel('$\log N{\\rm(CI)}_{\\rm{J}}/g_{\\rm{J}} - \log N{\\rm(CI)}_{\\rm{0}}$', fontsize=labelsize)
        ax[1, 3].tick_params(width=1, direction='in', labelsize=labelsize, left='True')
        ax[1,3].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1,3].xaxis.set_major_locator(MultipleLocator(20))
        ax[1,3].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1,3].yaxis.set_major_locator(MultipleLocator(0.5))

if case == 'fig2_3':
    fig02, ax = plt.subplots(1, 4, figsize=(9,2))
    fig02.subplots_adjust(wspace=0.5)
    name = 'J0812+3208_0' #J0812+3208_0_H2_lnL_j02
    with open('output/{:s}_H2_lnL_{:s}.pkl'.format(name,'j02'), 'rb') as f:
        x, y, z = pickle.load(f)
    X, Y = np.meshgrid(x, y)
    vmin = -40
    vmax = 0
    zmax = np.amax(z)
    ax[1].title.set_text('${\\rm{H_2{(J=0\\,to\\,J=2)}}}$')
    d = distr2d(x=x, y=y, z=np.exp(z))
    dx, dy = d.marginalize('y'), d.marginalize('x')
    dy.stats(latex=-1, name='log UV')
    dx.stats(latex=-1, name='log n')
    cmap_lnL = 'Purples'
    lnLcolor = 'purple'
    d.plot_contour(ax=ax[1], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0)
    d.plot_contour(ax=ax[3], color='blue', color_point=None, cmap=None, alpha=0, lw=1.0,zorder=-5)

    # plot H2 excitaiton
    with open('output/{:s}_H2_cols.pkl'.format(name), 'rb') as f:
        xH2, yH2, cols_H2 = pickle.load(f)
    database = 'H2UV'
    species = 'H2'
    H2 = H2_exc(H2database=database)
    H2.plot_objects(objects=name, ax=ax[0], syst=0.3, label='Obs. data')
    d.dopoint()

    q = H2.comp(name)
    sp = [s for s in q.e.keys() if species + 'j' in s]
    j = np.sort([0,1,2]) #[int(s[3:]) for s in sp if 'v' not in s])
    x, stat = getatomic(species, levels=j)
    mod = [cols_H2['H2j' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
    ax[0].plot([H2energy[0,i] for i in j], mod,color='purple')




    if 1:
         with open('output/{:s}_CI_lnL.pkl'.format(name), 'rb') as f:

             xCi, yCi, zCi = pickle.load(f)
             d = distr2d(x=xCi, y=yCi, z=np.exp(zCi))
             dx, dy = d.marginalize('y'), d.marginalize('x')
             dy.stats(latex=-1, name='log UV')
             dx.stats(latex=-1, name='log n')
             cmap_lnL = 'Greens'
             lnLcolor = 'green'
             print('plot_ci_contours')
             d.plot_contour(ax=ax[3], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0,conf_levels=[0.3,0.68]) #color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)
             ax[3].title.set_text('${\\rm{CI(J=1\\,and\\,J=2)}}$')

             with open('output/{:s}_CI_cols.pkl'.format(name), 'rb') as f:
                 xH2, yH2, cols_CI = pickle.load(f)
             database = 'H2UV'
             species = 'CI'
             H2 = H2_exc(H2database=database)
             H2.plot_objects(objects=name, ax=ax[2], syst=0.0, species='CI', label='Observation')
             d.dopoint()

             q = H2.comp(name)
             sp = [s for s in q.e.keys() if species + 'j' in s]
             j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
             x, stat = getatomic(species, levels=j)
             mod = [cols_CI['CIj' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
             ax[2].plot([CIenergy[i] for i in j], mod-mod[0], color='green')


             if 1:
                with open('output/{:s}_H2_lnL_{:s}.pkl'.format(name, 'j02'), 'rb') as f:
                    x, y, z = pickle.load(f)
                X, Y = np.meshgrid(x, y)
                vmin = -40
                vmax = 0
                zmax = np.amax(z)
                d = distr2d(x=x, y=y, z=np.exp(z+zCi))
                dx, dy = d.marginalize('y'), d.marginalize('x')
                dy.stats(latex=-1, name='log UV')
                dx.stats(latex=-1, name='log n')
                cmap_lnL = 'Reds'
                lnLcolor = 'red'
                d.plot_contour(ax=ax[3], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0,
                            conf_levels=[0.3, 0.68])  # color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)

    #tune axes
    if 1:
        for axs in ax[:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)
        ax[1].set_ylim(-1, 3)
        ax[3].set_ylim(-1, 3)

        ax[1].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[3].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)

        for axs in [ax[1],ax[3]]:
            axs.set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(5))
            axs.xaxis.set_major_locator(MultipleLocator(1))
            axs.yaxis.set_minor_locator(AutoMinorLocator(5))
            axs.yaxis.set_major_locator(MultipleLocator(1))

        for axs in [ax[0]]:
            axs.set_xlabel('Energy of H$_2$ levels, cm$^{-1}$', fontsize=labelsize)
            axs.set_ylabel('$\log N{\\rm(H_2)}_{\\rm{J}}/g_{\\rm{J}}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(5))
            axs.xaxis.set_major_locator(MultipleLocator(500))
            axs.yaxis.set_minor_locator(AutoMinorLocator(4))
            axs.yaxis.set_major_locator(MultipleLocator(2))
        ax[2].set_xlabel('Energy of C\,{\\sc i} levels, cm$^{-1}$', fontsize=labelsize)
        ax[2].set_ylabel('$\log N{\\rm(CI)}_{\\rm{J}}/g_{\\rm{J}} - \log N{\\rm(CI)}_{\\rm{0}}$', fontsize=labelsize)
        ax[2].tick_params(width=1, direction='in', labelsize=labelsize, left='True')
        ax[2].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[2].xaxis.set_major_locator(MultipleLocator(20))
        ax[2].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[2].yaxis.set_major_locator(MultipleLocator(0.5))

if case == 'fig2_4':
    fig02, ax = plt.subplots(2, 2, figsize=(4.5,4.5))
    fig02.subplots_adjust(wspace=0.5,hspace=0.4)
    name = 'J0812+3208_0' #J0812+3208_0_H2_lnL_j02
    with open('output/{:s}_H2_lnL_{:s}.pkl'.format(name,'j02'), 'rb') as f:
        x, y, z = pickle.load(f)
    X, Y = np.meshgrid(x, y)
    vmin = -40
    vmax = 0
    zmax = np.amax(z)
    ax[0,0].title.set_text('${\\rm{H_2{(J=0\\,to\\,J=2)}}}$')
    ax[0, 1].title.set_text('${\\rm{CI{(J=1\\,and\\,J=2)}}}$')
    d = distr2d(x=x, y=y, z=np.exp(z))
    dx, dy = d.marginalize('y'), d.marginalize('x')
    dy.stats(latex=-1, name='log UV')
    dx.stats(latex=-1, name='log n')
    cmap_lnL = 'Purples'
    lnLcolor = 'purple'
    d.plot_contour(ax=ax[1,0], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0)
    d.plot_contour(ax=ax[1,1], color='purple', color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0,zorder=-5)

    # plot H2 excitaiton
    with open('output/{:s}_H2_cols.pkl'.format(name), 'rb') as f:
        xH2, yH2, cols_H2 = pickle.load(f)
    database = 'H2UV'
    species = 'H2'
    H2 = H2_exc(H2database=database)
    H2.plot_objects(objects=name, ax=ax[0,0], syst=0.3, label='Obs. data')
    d.dopoint()

    q = H2.comp(name)
    sp = [s for s in q.e.keys() if species + 'j' in s]
    j = np.sort([0,1,2]) #[int(s[3:]) for s in sp if 'v' not in s])
    x, stat = getatomic(species, levels=j)
    mod = [cols_H2['H2j' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
    ax[0,0].plot([H2energy[0,i] for i in j], mod,color='purple')




    if 1:
         with open('output/{:s}_CI_lnL.pkl'.format(name), 'rb') as f:

             xCi, yCi, zCi = pickle.load(f)
             d = distr2d(x=xCi, y=yCi, z=np.exp(zCi))
             dx, dy = d.marginalize('y'), d.marginalize('x')
             dy.stats(latex=-1, name='log UV')
             dx.stats(latex=-1, name='log n')
             cmap_lnL = 'Greens'
             lnLcolor = 'green'
             print('plot_ci_contours')
             d.plot_contour(ax=ax[1,1], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0,conf_levels=[0.3,0.68]) #color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)
             #ax[1,1].title.set_text('${\\rm{CI(J=1\\,and\\,J=2)}}$')

             with open('output/{:s}_CI_cols.pkl'.format(name), 'rb') as f:
                 xH2, yH2, cols_CI = pickle.load(f)
             database = 'H2UV'
             species = 'CI'
             H2 = H2_exc(H2database=database)
             H2.plot_objects(objects=name, ax=ax[0,1], syst=0.0, species='CI', label='Observation')
             d.dopoint()

             q = H2.comp(name)
             sp = [s for s in q.e.keys() if species + 'j' in s]
             j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
             x, stat = getatomic(species, levels=j)
             mod = [cols_CI['CIj' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
             ax[0,1].plot([CIenergy[i] for i in j], mod-mod[0], color='green')


             if 1:
                with open('output/{:s}_H2_lnL_{:s}.pkl'.format(name, 'j02'), 'rb') as f:
                    x, y, z = pickle.load(f)
                X, Y = np.meshgrid(x, y)
                vmin = -40
                vmax = 0
                zmax = np.amax(z)
                d = distr2d(x=x, y=y, z=np.exp(z+zCi))
                dx, dy = d.marginalize('y'), d.marginalize('x')
                dy.stats(latex=-1, name='log UV')
                dx.stats(latex=-1, name='log n')
                cmap_lnL = 'Reds'
                lnLcolor = 'red'
                d.plot_contour(ax=ax[1,1], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=1.0,
                            conf_levels=[0.3, 0.68])  # color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)

    #tune axes
    if 1:
        for axs in ax[0,:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)
        for axs in ax[1,:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)
        ax[1,0].set_ylim(-1, 3)
        ax[1,1].set_ylim(-1, 3)

        ax[1,0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[1,1].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)

        for axs in [ax[1,0],ax[1,1]]:
            axs.set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(5))
            axs.xaxis.set_major_locator(MultipleLocator(1))
            axs.yaxis.set_minor_locator(AutoMinorLocator(5))
            axs.yaxis.set_major_locator(MultipleLocator(1))

        for axs in [ax[0,0]]:
            axs.set_xlabel('Energy of H$_2$ levels, cm$^{-1}$', fontsize=labelsize)
            axs.set_ylabel('$\log N{\\rm(H_2)}_{\\rm{J}}/g_{\\rm{J}}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(5))
            axs.xaxis.set_major_locator(MultipleLocator(500))
            axs.yaxis.set_minor_locator(AutoMinorLocator(4))
            axs.yaxis.set_major_locator(MultipleLocator(2))
        ax[0,1].set_xlabel('Energy of C\,{\\sc i} levels, cm$^{-1}$', fontsize=labelsize)
        ax[0,1].set_ylabel('$\log N{\\rm(CI)}_{\\rm{J}}/g_{\\rm{J}} - \log N{\\rm(CI)}_{\\rm{0}}$', fontsize=labelsize)
        ax[0,1].tick_params(width=1, direction='in', labelsize=labelsize, left='True')
        ax[0,1].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[0,1].xaxis.set_major_locator(MultipleLocator(20))
        ax[0,1].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0,1].yaxis.set_major_locator(MultipleLocator(0.5))


if case == 'fig3':
    fig03, ax = plt.subplots(1, 3, figsize=(9, 2))
    #fig03.subplots_adjust(hspace=0.4)
    with open('Figs/fig03/lnL_CI.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        vmin = -13
        vmax = 0
        cmap = 'Greens'
        zmax = np.amax(z)
        ax[2].pcolor(X, Y, z - zmax, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[2].title.set_text('$\log{\\rm{\,Likelihood}}$')
        d = distr2d(x=x, y=y, z=np.exp(z))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        cmap_lnL = None
        d.plot_contour(ax=ax[2], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)
        xlnL, ylnL, zlnL = x, y, z

    cmap = 'Oranges'
    msize = 15
    if 1:
        with open('Figs/fig03/CIj1', 'rb') as f:
            x, y, z = pickle.load(f)
            X, Y = np.meshgrid(x, y)
            vmin =-2.5
            vmax = 0.6
            ax[0].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
            ax[0].title.set_text('$\log N({\\rm{CI, J=1}})/N({\\rm{CI, J=0}})$')
            ax[0].contour(x, y, z, levels=[-0.58, -0.28, 0.02], colors='black', linewidths=1.0, alpha=1,
                             linestyles=['--', '-', '--'])
            d.plot_contour(ax=ax[0], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)

    if 1:
        with open('Figs/fig03/CIj2', 'rb') as f:
            x, y, z = pickle.load(f)
            X, Y = np.meshgrid(x, y)
            c = ax[1].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
            ax[1].title.set_text('$\log N({\\rm{CI, J=2}})/N({\\rm{CI, J=0}})$')
            ax[1].contour(x, y, z, levels=[-1.18, -0.83, -0.53], colors='black', alpha=1,
                             linestyles=['--', '-', '--'], linewidths=1.0)
            d.plot_contour(ax=ax[1], color=lnLcolor, color_point=None, cmap=cmap_lnL, alpha=0, lw=2.0)

    # add analytic CI grid
    if 1:
        f_name = 'C:/slava/Science/program/fortran/MCMC/emcee/grid_ci.value'
        num = 101
        x = np.linspace(0, 5, num)
        y = np.linspace(-2, 3, num)
        z = np.zeros([num, num])
        with open(f_name) as f:
            for k, line in enumerate(f):
                values = [float(s) for s in line.split()]
                z[k, :] = -np.array(values)
        zmax = np.amax(z)
        d = distr2d(x=x, y=y, z=np.exp(z))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        d.plot_contour(ax=ax[2], color='blue', color_point=None, cmap=None, alpha=0.0, lw=2.0, label='CI, T=const')
        #ax[2].text(0.6, 0.9, 'J0812+3208 A', fontsize=labelsize, color='black', transform=ax[1].transAxes)


    if 1:
        for axs in ax[:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=4)

    ax[0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
    ax[0].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
    ax[1].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
    ax[2].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)


    if 1:
        cax = fig03.add_axes([0.93, 0.27, 0.01, 0.47])
        fig03.colorbar(c, cax=cax, orientation='vertical', ticks=[-2,-1,0,1.5])
        cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        ax[2].text(7, -1.5, '$\log N_{\\rm{CI}}{\\rm{(J)}}/N_{\\rm{CI}}{\\rm{(J=0)}}$', fontsize=labelsize, rotation=90)

if case == 'fig4_2':
    fig04, ax = plt.subplots(2, 3, figsize=(9,4))
    fig04.subplots_adjust(wspace=0.4,hspace=0.4)
    #add lnL CI
    if 1:
        with open('Figs/fig02/lnL_CI.pkl', 'rb') as f:
            xCi, yCi, zCi = pickle.load(f)
            d = distr2d(x=xCi, y=yCi, z=np.exp(zCi))
            dx, dy = d.marginalize('y'), d.marginalize('x')
            dy.stats(latex=-1, name='log UV')
            dx.stats(latex=-1, name='log n')
            d.plot_contour(ax=ax[0,0], color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)
    # add lnL H2
    with open('Figs/fig02/lnL_H2.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        d = distr2d(x=x, y=y, z=np.exp(z))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        d.plot_contour(ax=ax[0,0], color='purple', color_point=None, cmap='Purples', alpha=0, lw=1.0)
    #plot join constraint
    if 1:
        d = distr2d(x=x, y=y, z=np.exp(z+zCi))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        d.plot_contour(ax=ax[0,0], color='Red', color_point=None, cmap='Reds', alpha=0, lw=1.0)


    # add Ci excitation
    if 1:
        name = 'J0812+3208_0'
        with open('result/cr_fixed/CI/BF/{:s}_CI.pkl'.format(name), 'rb') as f:
            EjCI, logNCI = pickle.load(f)
        ax[0,2].plot(EjCI, logNCI - logNCI[0], color='red',label='Fit')
        H2exc = H2_exc(H2database='H2UV')
        H2exc.plot_objects(objects=name, ax=ax[0,2], syst=0.0, species='CI',label='Observation')

    # add H2 excitation
    if 1:
        with open('result/cr_fixed/bf_H2/J0812+3208_0_H2.pkl', 'rb') as f:
            eH2, NH2 = pickle.load(f)
        species = ['H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5']
        ax[0,1].plot(eH2, NH2,
                markeredgecolor='black', markerfacecolor='green', markersize=5, color='red', label='Fit',
                alpha=0.7)
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        H2.plot_objects(objects='J0812+3208_0', ax=ax[0,1], syst=0.3, label='Observation')
        left, width = .6, .5
        bottom, height = .25, .6
        right = left + width
        top = bottom + height

    # tune labels and ticks
    if 1:
        ax[0,0].text(2.3,2,'Q 0812$+$3208$_0$')
        for axs in ax[0,:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)
        ax[0,0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[0,0].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
        ax[0,0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0,0].xaxis.set_major_locator(MultipleLocator(1))
        ax[0,0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0,0].yaxis.set_major_locator(MultipleLocator(1))
        #ax[0,0].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)

        ax[0,1].set_xlabel('Energy level H$_2$(J), K', fontsize=labelsize)
        ax[0,1].set_ylabel('$\log N{\\rm(H_2, J)}/g({\\rm{J}})$', fontsize=labelsize)
        ax[0,1].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0,1].xaxis.set_major_locator(MultipleLocator(500))
        ax[0,1].yaxis.set_minor_locator(AutoMinorLocator(4))
        ax[0,1].yaxis.set_major_locator(MultipleLocator(2))
        ax[0, 1].set_ylim(12.0, 20.5)
        #ax[1].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)

        ax[0,2].set_xlabel('Energy level CI(J), K', fontsize=labelsize)
        ax[0,2].set_ylabel('$\log N{\\rm(CI, J)}/N{\\rm(CI, 0)}/g({\\rm{J}})$', fontsize=labelsize)
        ax[0,2].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[0,2].xaxis.set_major_locator(MultipleLocator(20))
        ax[0,2].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0,2].yaxis.set_major_locator(MultipleLocator(0.5))
        #ax[0,2].set_xlim(-5,50)
        ax[0,2].set_ylim(-2.0, 0.5)
##############################
    qname = 'Q1232+0815_0'
    with open('output/{:s}_CI_lnL.pkl'.format(qname), 'rb') as f:
        xCi, yCi, zCi = pickle.load(f)
    with open('output/{:s}_CI_cols.pkl'.format(qname), 'rb') as f:
        xCi, yCi, cols_Ci = pickle.load(f)
    # add lnL and cols for H2 for J=0-5
    with open('output/{:s}_H2_lnL_J05.pkl'.format(qname), 'rb') as f:
        xH2, yH2, zH2 = pickle.load(f)
    with open('output/{:s}_H2_cols_J05.pkl'.format(qname), 'rb') as f:
        xH2, yH2, cols_H2 = pickle.load(f)
    # add lnL for H2 for J=0-2
    with open('output/{:s}_H2_lnL_J02.pkl'.format(qname), 'rb') as f:
        xH2, yH2, zH2J02 = pickle.load(f)

    # plot lnL countours
    if 1:
        d = distr2d(x=xCi, y=yCi, z=np.exp(zCi))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        d.plot_contour(ax=ax[1,0], color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)

        dH2 = distr2d(x=xH2, y=yH2, z=np.exp(zH2))
        dx, dy = dH2.marginalize('y'), dH2.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        dH2.plot_contour(ax=ax[1,0], color='purple', color_point=None, cmap='Purples', alpha=0, lw=1.0)
        dH2.dopoint()

        dH2J02 = distr2d(x=xH2, y=yH2, z=np.exp(zH2J02))
        dx, dy = dH2J02.marginalize('y'), dH2J02.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        dH2J02.plot_contour(ax=ax[1,0], color='royalblue', color_point=None, cmap='Blues', alpha=0, lw=1.0)

        djoin = distr2d(x=xH2, y=yH2, z=np.exp(zH2J02+zCi))
        djoin.dopoint()
        dx, dy = djoin.marginalize('y'), djoin.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        djoin.plot_contour(ax=ax[1,0], color='red', color_point=None, cmap='Reds', alpha=0, lw=1.0)

    # plot H2 excitaiton
    if 1:
        species = 'H2'
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        H2.plot_objects(qname, ax=ax[1,1], syst=0.3, label='Data')

        q = H2.comp(qname)
        sp = [s for s in q.e.keys() if species + 'j' in s]
        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
        x, stat = getatomic(species, levels=j)
        mod = [cols_H2['H2j' + str(i)](dH2.point[0], dH2.point[1]) - np.log10(stat[i]) for i in j]
        ax[1,1].plot([H2energy[0,i] for i in j], mod,color='purple')
        mod = [cols_H2['H2j' + str(i)](djoin.point[0], djoin.point[1]) - np.log10(stat[i]) for i in j]
        ax[1,1].plot([H2energy[0,i] for i in j], mod,color='red')

    # plot CI excitation
    if 1:
        H2.plot_objects(objects=qname, ax=ax[1,2], syst=0.0, species='CI', label='Data')
        species = 'CI'
        syst = 0.1
        q = H2.comp(qname)
        sp = [s for s in q.e.keys() if species + 'j' in s]
        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
        x, stat = getatomic(species, levels=j)
        mod = [cols_Ci['CIj' + str(i)](dH2.point[0], dH2.point[1]) - np.log10(stat[i]) for i in j]
        print('mod',mod)
        ax[1,2].plot([CIenergy[i] for i in j], mod-mod[0], color='purple')
        mod = [cols_Ci['CIj' + str(i)](djoin.point[0], djoin.point[1]) - np.log10(stat[i]) for i in j]
        ax[1,2].plot([CIenergy[i] for i in j], mod-mod[0], color='red')

    # tuneaxis
    if 1:
        ax[1, 0].text(2.3, 2, 'Q 1232$+$0815')
        ax[1,0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[1,0].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
        ax[1,0].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[1,0].tick_params(which='major', length=5)
        ax[1,0].tick_params(which='minor', length=2)
        #ax[1,0].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)

        ax[1,1].set_xlabel('Energy level H$_2$, K', fontsize=labelsize)
        ax[1,1].set_ylabel('$\log N{\\rm(H_2)_J}/g_{\\rm{J}}$', fontsize=labelsize)
        ax[1,1].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1,1].xaxis.set_major_locator(MultipleLocator(1000))
        ax[1,1].yaxis.set_minor_locator(AutoMinorLocator(1))
        ax[1,1].yaxis.set_major_locator(MultipleLocator(1))
        ax[1,1].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[1,1].tick_params(which='major', length=5)
        ax[1,1].tick_params(which='minor', length=2)
        ax[1,1].set_ylim(12.0, 20.5)
        #ax[1].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)

        ax[1,2].set_xlabel('Energy level CI, K', fontsize=labelsize)
        ax[1,2].set_ylabel('$\log N{\\rm(CI)}_{\\rm{J}}/g_{\\rm{J}} - \log N_{\\rm{0}}/g_{\\rm{0}}$', fontsize=labelsize)
        ax[1,2].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1,2].xaxis.set_major_locator(MultipleLocator(20))
        ax[1,2].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1,2].yaxis.set_major_locator(MultipleLocator(1))
        ax[1,2].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[1,2].tick_params(which='major', length=5)
        ax[1,2].tick_params(which='minor', length=2)
        #ax[2].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)
        #ax[1,2].set_xlim(-5,50)
        ax[1,2].set_ylim(-2.0, 0.5)

if case == 'fig4_3':
    fig04, ax = plt.subplots(1, 3, figsize=(9,2))
    fig04.subplots_adjust(wspace=0.4,hspace=0.4)
##############################
    qname = 'Q1232+0815_0'
    with open('output/{:s}_CI_lnL.pkl'.format(qname), 'rb') as f:
        xCi, yCi, zCi = pickle.load(f)
    with open('output/{:s}_CI_cols.pkl'.format(qname), 'rb') as f:
        xCi, yCi, cols_Ci = pickle.load(f)
    # add lnL and cols for H2 for J=0-5
    with open('output/{:s}_H2_lnL_allJ.pkl'.format(qname), 'rb') as f:
        xH2, yH2, zH2 = pickle.load(f)
    with open('output/{:s}_H2_cols.pkl'.format(qname), 'rb') as f:
        xH2, yH2, cols_H2 = pickle.load(f)
    # add lnL for H2 for J=0-2
    with open('output/{:s}_H2_lnL.pkl'.format(qname), 'rb') as f:
        xH2, yH2, zH2J02 = pickle.load(f)

    # plot lnL countours
    if 1:
        d = distr2d(x=xCi, y=yCi, z=np.exp(zCi))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        d.plot_contour(ax=ax[0], color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)

        dH2 = distr2d(x=xH2, y=yH2, z=np.exp(zH2))
        dx, dy = dH2.marginalize('y'), dH2.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        dH2.plot_contour(ax=ax[0], color='purple', color_point=None, cmap='Purples', alpha=0, lw=1.0)
        dH2.dopoint()

        dH2J02 = distr2d(x=xH2, y=yH2, z=np.exp(zH2J02))
        dx, dy = dH2J02.marginalize('y'), dH2J02.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        dH2J02.plot_contour(ax=ax[0], color='royalblue', color_point=None, cmap='Blues', alpha=0, lw=1.0)

        djoin = distr2d(x=xH2, y=yH2, z=np.exp(zH2J02+zCi))
        djoin.dopoint()
        dx, dy = djoin.marginalize('y'), djoin.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        djoin.plot_contour(ax=ax[0], color='red', color_point=None, cmap='Reds', alpha=0, lw=1.0)

    # plot H2 excitaiton
    if 1:
        species = 'H2'
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        H2.plot_objects(qname, ax=ax[1], syst=0.3, label='Data')

        q = H2.comp(qname)
        sp = [s for s in q.e.keys() if species + 'j' in s]
        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
        x, stat = getatomic(species, levels=j)
        mod = [cols_H2['H2j' + str(i)](dH2.point[0], dH2.point[1]) - np.log10(stat[i]) for i in j]
        ax[1].plot([H2energy[0,i] for i in j], mod,color='purple')
        mod = [cols_H2['H2j' + str(i)](djoin.point[0], djoin.point[1]) - np.log10(stat[i]) for i in j]
        ax[1].plot([H2energy[0,i] for i in j], mod,color='red')

    # plot CI excitation
    if 1:
        H2.plot_objects(objects=qname, ax=ax[2], syst=0.0, species='CI', label='Data')
        species = 'CI'
        syst = 0.1
        q = H2.comp(qname)
        sp = [s for s in q.e.keys() if species + 'j' in s]
        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
        x, stat = getatomic(species, levels=j)
        mod = [cols_Ci['CIj' + str(i)](dH2.point[0], dH2.point[1]) - np.log10(stat[i]) for i in j]
        print('mod',mod)
        ax[2].plot([CIenergy[i] for i in j], mod-mod[0], color='purple')
        mod = [cols_Ci['CIj' + str(i)](djoin.point[0], djoin.point[1]) - np.log10(stat[i]) for i in j]
        ax[2].plot([CIenergy[i] for i in j], mod-mod[0], color='red')

    # tuneaxis
    if 1:
        for axs in ax[:]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)

        ax[0].text(2, 2.2, 'Q 1232$+$0815')
        ax[0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[0].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
        ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].xaxis.set_major_locator(MultipleLocator(1))
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].yaxis.set_major_locator(MultipleLocator(1))
        ax[0].set_ylim(-1, 3)
        ax[0].set_xlim(0, 4)
        #ax[1,0].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)

        ax[1].set_xlabel('Energy of H$_2$ levels, cm$^{-1}$', fontsize=labelsize)
        ax[1].set_ylabel('$\log N{\\rm{(H_2)_J}}/g_{\\rm{J}}$', fontsize=labelsize)
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1].xaxis.set_major_locator(MultipleLocator(500))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1].yaxis.set_major_locator(MultipleLocator(2))
        ax[1].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[1].tick_params(which='major', length=5)
        ax[1].tick_params(which='minor', length=2)
        ax[1].set_ylim(12.0, 20.5)
        #ax[1].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)

        ax[2].set_xlabel('Energy of C\,{\\sc{i}} levels, cm$^{-1}$', fontsize=labelsize)
        ax[2].set_ylabel('$\log N{\\rm(CI)}_{\\rm{J}}/g_{\\rm{J}} - \log N{\\rm(CI)}_{\\rm{0}}$', fontsize=labelsize)
        ax[2].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[2].xaxis.set_major_locator(MultipleLocator(20))
        ax[2].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[2].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[2].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[2].tick_params(which='major', length=5)
        ax[2].tick_params(which='minor', length=2)
        #ax[2].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)
        #ax[1,2].set_xlim(-5,50)
        ax[2].set_ylim(-2.0, 0.5)


#plot fits for all qso systems
if case == 'fig8':
    nfigs = 5
    fig08, ax = plt.subplots(nfigs, 3, figsize=(9, 1.9*nfigs)) #1.9
    fig08.subplots_adjust(wspace=0.4,hspace=0.6)

    list =['J0000+0048_0','B0528-2505_0','J0812+3208_0','J0812+3208_1','J0816+1446_0',
           'J0843+0221_0','J0843+0221_0','Q1232+0815_0','J1237+0647_1', 'J1439+1118_0',
           'B1444+0126_0', 'J1513+0352_0','J2100-0641_0','J2140-0321_0']
    list = list[0:5]
    #list =[]
    if 0:
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        list = H2.listofPDR()[0:5]
        #database = 'z=-0.5'
        #H2 = H2_exc(H2database=database)
        #for q in H2.listofPDR()[0:4]:
        #    list.append(q)
        #database = 'z=0'
        #H2 = H2_exc(H2database=database)
        #for q in H2.listofPDR()[0:3]:
        #    list.append(q)
        #database = 'z=-1'
        #H2 = H2_exc(H2database=database)
        #for q in H2.listofPDR()[10:14]:
        #    list.append(q)
    if 0:
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        for q in H2.listofPDR()[5:11]:
            list.append(q)
    if 0:
        database = 'Magellan'
        H2 = H2_exc(H2database=database)
        for q in H2.listofPDR()[0:5]:
            list.append(q)


    print('list:',list)
    database = 'H2UV'
    H2 = H2_exc(H2database=database)

    H2Jall_names = ['J0000+0048_0','J0816+1446_0','J1513+0352_0','Q1232+0815_0',
                    'J2123-0050_0','J1439+1118_0','J2100-0641_0',
                    'HD24534_0','HD27778_0','HD40893_0','HD147888_0','HD185418_0','HD192639_0',
                    'HD195965_0','HD206267_0','HD207198_0','HD210839_0','HD210839_1',
                    'SK-6705_0','SK-6705_1','SK-70115_0',
                    'SK-70115_1','SK-70115_2',
                    'SK13_0','SK13_1','SK13_2','SK13_3']

    for num,qname in enumerate(list):
        if num < nfigs:
            print(num, qname)
            q = H2.comp(qname)
            # plot lnL
            if 1:
                col = ax[num,0]
                sp = [s for s in q.e.keys()]
                with open('{:s}{:s}_H2_lnL.pkl'.format(filepath, qname), 'rb') as f:
                    x, y, z_H2 = pickle.load(f)
                if 'CIj1' in sp:
                    with open('{:s}{:s}_CI_lnL.pkl'.format(filepath, qname), 'rb') as f:
                        x, y, z_CI = pickle.load(f)
                else:
                    z_CI = np.zeros([len(y), len(x)])
                with open('{:s}{:s}_join_lnL.pkl'.format(filepath, qname), 'rb') as f:
                    x, y, z_join = pickle.load(f)

                if 1:
                    if qname in H2Jall_names:
                        #>>>> fit to H2 all J levels
                        if 1:
                            print('fit to all J of H2:')
                            with open('{:s}{:s}_H2_lnL_allJ.pkl'.format(filepath, qname), 'rb') as f:
                                x, y, z_H2_all = pickle.load(f)
                            d_all = distr2d(x=x, y=y, z=np.exp(z_H2_all))
                            dx, dy = d_all.marginalize('y'), d_all.marginalize('x')
                            dx.stats(latex=2, name='nH')
                            dy.stats(latex=2, name='UV')
                            d_all.plot_contour(ax=col, color='purple', color_point=None, cmap='Purples', alpha=0, lw=1.0)



                        print('min_lnL for CI')
                        d = distr2d(x=x, y=y, z=np.exp(z_CI))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name='nH')
                        dy.stats(latex=2, name='UV')
                        if 'CIj1' in sp:
                            d.plot_contour(ax=col, color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)

                        print('min_lnL for H2')
                        d = distr2d(x=x, y=y, z=np.exp(z_H2))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name='nH')
                        dy.stats(latex=2, name='UV')
                        d.plot_contour(ax=col, color='blue', color_point=None, cmap='Blues', alpha=0, lw=1.0,zorder=-5)


                    else:
                        print('min_lnL for H2')
                        d = distr2d(x=x, y=y, z=np.exp(z_H2))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name='nH')
                        dy.stats(latex=2, name='UV')
                        d.plot_contour(ax=col, color='purple', color_point=None, cmap='Purples', alpha=0, lw=1.0)

                        print('min_lnL for CI')
                        d = distr2d(x=x, y=y, z=np.exp(z_CI))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dx.stats(latex=2, name='nH')
                        dy.stats(latex=2, name='UV')
                        if 'CIj1' in sp:
                            d.plot_contour(ax=col, color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0)

                print('min_lnL for the join')
                d = distr2d(x=x, y=y, z=np.exp(z_join))
                dx, dy = d.marginalize('y'), d.marginalize('x')
                dx.stats(latex=2, name='nH')
                dy.stats(latex=2, name='UV')
                d.dopoint()
                if 'CIj1' in sp:
                    d.plot_contour(ax=col, color='red', color_point=None, cmap='Reds', alpha=0, lw=1.0)

                col.title.set_text("".join(['$', qname, '$']))
                col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
                col.tick_params(which='major', length=5)
                col.tick_params(which='minor', length=3)
                col.set_xlabel('$\log~{\\rm{n_{H}}}$', fontsize=labelsize)
                col.set_ylabel('$\log~{\\rm{I_{UV}}}$', fontsize=labelsize)
                col.set_ylim(-1,3)
                col.set_xlim(0, 4)
                col.xaxis.set_minor_locator(AutoMinorLocator(5))
                col.xaxis.set_major_locator(MultipleLocator(1))
                col.yaxis.set_minor_locator(AutoMinorLocator(2))
                col.yaxis.set_major_locator(MultipleLocator(1))
            #plot H2_exc
            if 1:
                col = ax[num,1]
                species = 'H2'
                H2.plot_objects(objects=qname, ax=col, syst=0.3)
                with open('{:s}{:s}_H2_cols.pkl'.format(filepath, qname), 'rb') as f:
                    xH2, yH2, cols_H2 = pickle.load(f)

                if qname in H2Jall_names:
                    if 1:
                        color = 'purple'
                        d_all.dopoint()
                        sp = [s for s in cols_H2.keys() if species + 'j' in s]
                        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                        x, stat = getatomic(species, levels=j)
                        mod = [cols_H2['H2j' + str(i)](d_all.point[0], d_all.point[1]) - np.log10(stat[i]) for i in j]
                        ax[num, 1].plot([H2energy[0, i] for i in j], mod, color=color)


                color='red'
                sp = [s for s in cols_H2.keys() if species + 'j' in s]
                j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                x, stat = getatomic(species, levels=j)
                mod = [cols_H2['H2j' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
                col.plot([H2energy[0, i] for i in j], mod, color=color)

                col.title.set_text("".join(['$', qname, '$']))
                col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
                col.tick_params(which='major', length=5)
                col.tick_params(which='minor', length=3)
                col.set_xbound(lower=-100, upper=2500)
                col.set_ybound(lower=11.5, upper=21.5)
                col.xaxis.set_minor_locator(AutoMinorLocator(5))
                col.xaxis.set_major_locator(MultipleLocator(1000))
                col.yaxis.set_minor_locator(AutoMinorLocator(2))
                col.yaxis.set_major_locator(MultipleLocator(2))
                col.set_xlabel('Energy of H$_2$ levels, cm$^{-1}$', fontsize=labelsize)
                col.set_ylabel('$\log~N_{\\rm{J}}/g_{\\rm{J}}$', fontsize=labelsize)
            #plot CI exc
            if 1:
                col = ax[num, 2]
                species = 'CI'
                syst = 0.1
                q = H2.comp(qname)
                sp = [s for s in q.e.keys() if species + 'j' in s]
                col.title.set_text("".join(['$', qname, '$']))
                color='red'
                msize=4
                if 'CIj1' in sp:
                    if 0:
                        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                        x, stat = getatomic(species, levels=j)
                        y = [q.e[species + 'j' + str(i)].col * a(0.0, syst, syst) / stat[i] for i in j]
                        yerr = [y[i].minus for i in j]
                        col.errorbar(x=x, y=[y[i].val - y[0].val for i in j],
                                     yerr=[[y[i].minus for i in j], [y[i].plus for i in j]], fmt='o',
                                     color='black', markeredgecolor='black', markeredgewidth=1, markersize=msize, capsize=1,
                                     ecolor='black')
                    H2.plot_objects(objects=qname, ax=col, syst=0.0, species='CI')
                    with open('{:s}{:s}_CI_cols.pkl'.format(filepath, qname), 'rb') as f:
                        xCI, yCI, cols_CI = pickle.load(f)
                    sp = [s for s in cols_CI.keys() if species + 'j' in s]
                    j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
                    x, stat = getatomic(species, levels=j)
                    mod = [cols_CI['CIj' + str(i)](d.point[0], d.point[1]) - np.log10(stat[i]) for i in j]
                    col.plot([CIenergy[i] for i in j], mod-mod[0], color=color)
                    #with open('{:s}bf/{:s}_CI.pkl'.format(filepath, qname), 'rb') as f:
                    #    CIe, CIn = pickle.load(f)
                    #col.plot(CIe, CIn - CIn[0],color=color)

                col.title.set_text("".join(['$', qname, '$']))
                col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
                col.tick_params(which='major', length=5)
                col.tick_params(which='minor', length=3)
                col.xaxis.set_minor_locator(AutoMinorLocator(5))
                col.xaxis.set_major_locator(MultipleLocator(20))
                col.yaxis.set_minor_locator(AutoMinorLocator(2))
                col.yaxis.set_major_locator(MultipleLocator(1))
                col.set_xbound(lower=-5, upper=50)
                col.set_ybound(lower=-3, upper=0.5)
                col.set_xlabel('Energy of C\,{\\sc i} levels, cm$^{-1}$', fontsize=labelsize)
                col.set_ylabel('$\log~N_{\\rm{J}}/g_{\\rm{J}} - \log N_{\\rm{0}}/g_{\\rm{0}}$', fontsize=labelsize)

# plot results - statistic - 'nH-uv' plane and others

if case == 'fig9':

    labx, laby = 17.7,2.6
    colbar_x, colbar_y = 20.1, 2.1
    printresult = 0
    if printresult:
        print('print_summary****************')
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if 'PDRnH' in c.e:
                    print('sysname:',name)
                    print('nH=',c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                    print('uv=',c.e['PDRuv'].col.val,c.e['PDRuv'].col.plus,c.e['PDRuv'].col.minus)
    # with hists on axis
    if 0:
        fig09, ax = plt.subplots(1, 2, figsize=(9, 4))
        fig09.subplots_adjust(wspace=0.3)

        add_DLAs = 1
        if add_DLAs:
            database = 'H2UV'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if 'PDRnH' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val,c.e['H2'].col.plus,c.e['H2'].col.minus)
                            y = t #a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax[0].errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='black', zorder=-10, capsize=2, ecolor='black',markersize= 1)
                            x = a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            y = a(c.e['PDRuv'].col.val,c.e['PDRuv'].col.plus,c.e['PDRuv'].col.minus)
                            ax[1].errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='black', zorder=-10, capsize=2, ecolor='black',markersize= 1)

            c = ax[1].scatter(np.array(nH), uv, 50, me, cmap='hot', vmin=-1.5, vmax=0.5,
                        edgecolors='black')

            if 1:
                cax = fig09.add_axes([0.6, 0.63, 0.01, 0.2])
                fig09.colorbar(c, cax=cax,ticks=[-1, -0.5, 0, 0.5])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[1].text(0.15, 1.32, '[X/H]', fontsize=labelsize)

                adax = fig09.add_axes([0.563, 0.11 + 0.77, 0.337, 0.1])
                adax.hist(nH, color='gold')
                adax.axis('off')  # get_xaxis().set_visible(False)
                adax.set_xlim(-0.1, 3)

                aday = fig09.add_axes([0.9, 0.11, 0.04, 0.77])
                aday.hist(uv, color='darkred', orientation="horizontal")
                aday.set_ylim(-1, 3)
                aday.axis('off')  # get_xaxis().set_visible(False)

            c = ax[0].scatter(NH2, t01, 50, uv, cmap='hot', vmin=-1, vmax=2.0,
                        edgecolors='black',label='DLAs')

            if 1:
                cax = fig09.add_axes([0.41, 0.63, 0.01, 0.2])
                fig09.colorbar(c, cax=cax,ticks=[ 0, 1,2])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[0].text(20.45, 2.55, '$\log I_{\\rm{{UV}}}$', fontsize=labelsize)



            print('DLAs')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(10**np.array(t01)), np.std(10**np.array(t01)))
            print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

        add_P94 = 0
        if add_P94:
            database = 'P94'
            H2 = H2_exc(H2database=database)
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if (c.e['H2'].col.val > 17):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()

                        t01.append(t.val)
                        NH2.append(c.e['H2'].col.val)
                        #ax[0].errorbar(x=c.e['H2'].col.val, y=t.val,
                        #               yerr=[[t.minus], [t.plus]],
                        #               xerr=[[c.e['H2'].col.minus], [c.e['H2'].col.plus]],
                        #                   fmt='^', color='black', zorder=-10, capsize=2, ecolor='blue',markersize = 1)
            ax[0].scatter(NH2, t01, 30, color='blue', edgecolors='black',alpha=0.8,label='P94')
            print('P94')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:',np.mean(10**np.array(t01)), np.std(10**np.array(t01)))
            print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

        add_LMC = 1
        if add_LMC:
            database = 'Magellan'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if 'PDRnH' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val, c.e['H2'].col.plus, c.e['H2'].col.minus)
                            y = t  # a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax[0].errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='black', zorder=-10, capsize=2, ecolor='black', markersize=1)
                            x = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus)
                            ax[1].errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='black', zorder=-10, capsize=2, ecolor='black', markersize=1)

            c = ax[1].scatter(np.array(nH), uv, 50, me, cmap='hot', vmin=-1.5, vmax=0.5,
                              edgecolors='black')

            if 1:
                cax = fig09.add_axes([0.6, 0.63, 0.01, 0.2])
                fig09.colorbar(c, cax=cax, ticks=[-1, -0.5, 0, 0.5])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[1].text(0.15, 1.32, '[X/H]', fontsize=labelsize)

                adax = fig09.add_axes([0.563, 0.11 + 0.77, 0.337, 0.1])
                adax.hist(nH, color='gold')
                adax.axis('off')  # get_xaxis().set_visible(False)
                adax.set_xlim(-0.1, 3)

                aday = fig09.add_axes([0.9, 0.11, 0.04, 0.77])
                aday.hist(uv, color='darkred', orientation="horizontal")
                aday.set_ylim(-1, 3)
                aday.axis('off')  # get_xaxis().set_visible(False)

            c = ax[0].scatter(NH2, t01, 50, uv, cmap='hot', vmin=-1, vmax=2.0,
                              edgecolors='black', label='LMC+SMC')

            if 1:
                cax = fig09.add_axes([0.41, 0.63, 0.01, 0.2])
                fig09.colorbar(c, cax=cax, ticks=[0, 1, 2])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[0].text(20.45, 2.55, '$\log I_{\\rm{{UV}}}$', fontsize=labelsize)

            print('LMC')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(10 ** np.array(t01)), np.std(10 ** np.array(t01)))
            print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

        add_MW = 0
        if add_MW:
            database = 'Galaxy'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if 'PDRnH' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val, c.e['H2'].col.plus, c.e['H2'].col.minus)
                            y = t  # a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax[0].errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='black', zorder=-10, capsize=2, ecolor='black', markersize=1)
                            x = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus)
                            ax[1].errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='black', zorder=-10, capsize=2, ecolor='black', markersize=1)

            c = ax[1].scatter(np.array(nH), uv, 50, me, cmap='hot', vmin=-1.5, vmax=0.5,
                              edgecolors='black')

            if 1:
                cax = fig09.add_axes([0.6, 0.63, 0.01, 0.2])
                fig09.colorbar(c, cax=cax, ticks=[-1, -0.5, 0, 0.5])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[1].text(0.15, 1.32, '[X/H]', fontsize=labelsize)

                adax = fig09.add_axes([0.563, 0.11 + 0.77, 0.337, 0.1])
                adax.hist(nH, color='gold')
                adax.axis('off')  # get_xaxis().set_visible(False)
                adax.set_xlim(-0.1, 3)

                aday = fig09.add_axes([0.9, 0.11, 0.04, 0.77])
                aday.hist(uv, color='darkred', orientation="horizontal")
                aday.set_ylim(-1, 3)
                aday.axis('off')  # get_xaxis().set_visible(False)

            c = ax[0].scatter(NH2, t01, 50, uv, cmap='hot', vmin=-1, vmax=2.0,
                              edgecolors='black', label='Galaxy')

            if 1:
                cax = fig09.add_axes([0.41, 0.63, 0.01, 0.2])
                fig09.colorbar(c, cax=cax, ticks=[0, 1, 2])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[0].text(20.45, 2.55, '$\log I_{\\rm{{UV}}}$', fontsize=labelsize)

            print('MW')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(10**np.array(t01)), np.std(10**np.array(t01)))
            print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

        add_lowzDLA = 0
        if add_lowzDLA:
            database = 'lowzDLAs'
            H2 = H2_exc(H2database=database)
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if 'T01' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()

                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            #ax[0].errorbar(x=c.e['H2'].col.val, y=t.val,
                            #               yerr=[[t.minus], [t.plus]],
                            #               xerr=[[c.e['H2'].col.minus], [c.e['H2'].col.plus]],
                            #               fmt='^', color='black', zorder=-10, capsize=2, ecolor='magenta', markersize=1)
            ax[0].scatter(NH2, t01, 30, color='magenta', edgecolors='black', marker='^', label='low-$z$ DLAs', zorder=-5)
            print('lowz_DLAs')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(t01), np.std(t01))

        tune =1
        if tune:
            ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[0].xaxis.set_major_locator(MultipleLocator(1))
            ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
            ax[0].yaxis.set_major_locator(MultipleLocator(0.5))
            ax[0].set_xlim(17, 21.5)
            ax[0].set_ylim(1.3, 3.5)
            ax[0].set_xlabel('$\log N({\\rm{H_2}})$, cm$^{-2}$', fontsize=labelsize)
            ax[0].set_ylabel('$\log T_{\\rm{01}}$, K',
                             fontsize=labelsize)

            ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[1].xaxis.set_major_locator(MultipleLocator(1))
            ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
            ax[1].yaxis.set_major_locator(MultipleLocator(1))
            ax[1].set_xlim(-0.1, 3)
            ax[1].set_ylim(-1, 3)
            ax[1].set_xlabel('$\log n_{\\rm{H}}$, cm$^{-3}$', fontsize=labelsize)
            ax[1].set_ylabel('$\log I_{\\rm{UV}}$, Mathis unit',
                          fontsize=labelsize)
            ax[0].legend(fontsize=labelsize, bbox_to_anchor=(0.25, 0.7, 0.2, 0.3), frameon=True)

            for col in ax[:]:
                col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
                col.tick_params(which='major', length=5)
                col.tick_params(which='minor', length=4)

    if 0:
        fig09, ax = plt.subplots(1,2,figsize=(9, 4))
        fig09.subplots_adjust(wspace=0.3)
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        names = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join(['$',q.name, '_', str(i_c),'$'])
                print(name)
                if 'PDRnH' in c.e:
                    if (c.e['H2'].col.val > 17):
                        names.append(name)
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        p = a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus,'l')*t
                        nH.append(c.e['PDRnH'].col.val)
                        uv.append(c.e['PDRuv'].col.val)
                        me.append(q.e['Me'].col.val)
                        t01.append(t.log().val)
                        NH2.append(c.e['H2'].col.val)
                        xNH = a(c.e['H2'].col.val+0.3,c.e['H2'].col.plus,c.e['H2'].col.minus) + a(q.e['HI'].col.val,q.e['HI'].col.plus,q.e['HI'].col.minus)
                        NH.append(xNH.val)
                        x = a(c.e['PDRuv'].col.val,c.e['PDRuv'].col.plus,c.e['PDRuv'].col.minus,'l')*a(2,0,0,'l')/a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus,'l')

                        ax[0].errorbar(x=xNH.val, y=p.val,
                                       yerr=[[p.minus], [p.plus]],
                                       xerr=[[xNH.minus], [xNH.plus]],
                                       fmt='^', color='black', zorder=-10, capsize=1, ecolor='black',markersize=4)

                        ax[1].errorbar(x=x.val, y=t.log().val,
                                     yerr=[[t.log().minus],[t.log().plus]],
                                     xerr=[[x.minus], [x.plus]],
                                     fmt='^', color='black', zorder=-10,capsize=1,ecolor='black',markersize=4,markeredgewidth=2)
                        ax[1].text(x.val,t.log().val+0.0,name,fontsize=5)
        c = ax[0].scatter(np.array(NH), np.array(nH) + t01, 50, NH2, cmap='hot', edgecolors='black',vmin=19, vmax=22)
        ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].xaxis.set_major_locator(MultipleLocator(1))
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[0].set_xlim(18.5, 23)
        ax[0].set_ylim(2.5, 5.0)
        ax[0].set_xlabel('$\log N({\\rm{H}})_{\\rm{tot}}$, cm$^{-2}$', fontsize=labelsize)
        ax[0].set_ylabel('$\log n_{\\rm{H}} T_{\\rm{01}}$, K\,cm$^{-3}$',
                         fontsize=labelsize)
        if 1:
            cax = fig09.add_axes([0.16, 0.64, 0.01, 0.2])
            fig09.colorbar(c, cax=cax, ticks=[19,20,21])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax[0].text(18.8, 4, '$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)
            ax[0].text(20, 4.76, '$\log\\,N({\\rm{H_2}})>19$', fontsize=labelsize)

        c = ax[1].scatter(np.array(uv) - nH + 2, np.array(t01), 50 ,me, cmap='hot', edgecolors='black',vmin=-1.5, vmax=0.5)
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1].xaxis.set_major_locator(MultipleLocator(1))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(10))
        ax[1].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[1].set_xlim(-1.5, 2.5)
        ax[1].set_ylim(1.5, 2.5)
        ax[1].set_xlabel('$\log I_{\\rm{UV}} ({100\\mbox{cm}^{-3}}/{n_{\\rm{H}}})$', fontsize=labelsize)
        ax[1].set_ylabel('$\log T_{\\rm{01}}$, K',
                      fontsize=labelsize)
        if 1:
            cax = fig09.add_axes([0.6, 0.64, 0.01, 0.2])
            fig09.colorbar(c, cax=cax, ticks=[-1,0])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax[1].text(-1.2, 2.1, '$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)
            ax[1].text(0, 2.4, '$\log\\,N({\\rm{H_2}})>19$', fontsize=labelsize)





        for col in ax[:]:
            col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=4)
    if 0:
        fig09, ax = plt.subplots(figsize=(4.5, 4))
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                print(name)
                if 'PDRnH' in c.e:
                    if (c.e['H2'].col.val > 17):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        p = a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus,'l')*t
                        nH.append(c.e['PDRnH'].col.val)
                        uv.append(c.e['PDRuv'].col.val)
                        me.append(q.e['Me'].col.val)
                        t01.append(t.log().val)
                        NH2.append(c.e['H2'].col.val)
                        xNH = a(c.e['H2'].col.val+0.3,c.e['H2'].col.plus,c.e['H2'].col.minus) + a(q.e['HI'].col.val,q.e['HI'].col.plus,q.e['HI'].col.minus)
                        NH.append(xNH.val)
                        x = a(c.e['PDRuv'].col.val,c.e['PDRuv'].col.plus,c.e['PDRuv'].col.minus,'l')*a(2,0,0,'l')/a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus,'l')

                        ax.errorbar(x=xNH.val, y=q.e['Me'].col.val,
                                       yerr=[[q.e['Me'].col.minus], [q.e['Me'].col.plus]],
                                       xerr=[[xNH.minus], [xNH.plus]],
                                       fmt='^', color='black', zorder=-10, capsize=1, ecolor='black',markersize=3)


        c = ax.scatter(np.array(NH), me, 50, NH2, cmap='hot', edgecolors='black',vmin=17, vmax=21)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_xlim(18, 23)
        ax.set_ylim(-2, 1)
        ax.set_xlabel('$\log N({\\rm{H}})_{\\rm{tot}}$, cm$^{-2}$', fontsize=labelsize)
        ax.set_ylabel('$[X/H]$',
                         fontsize=labelsize)
        if 1:
            cax = fig09.add_axes([0.16, 0.64, 0.01, 0.2])
            fig09.colorbar(c, cax=cax, ticks=[18,19,20])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax.text(18.2, -0.15, '$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)







        ax.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax.tick_params(which='major', length=5)
        ax.tick_params(which='minor', length=4)
    if 0:
        fig09, ax = plt.subplots(figsize=(9, 4))
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        names = []
        for q in H2.H2.values():
            if q.name in ['J0000+0048', 'J1513+0352', 'J1443+2724', 'Q1232+0815', 'J1439+1118', 'J2100-0641',
                          'J2123-0050', 'J1439+1118']:
                print('bas system')
            else:
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if 'PDRnH' in c.e:
                        if(c.e['H2'].col.val >17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()

                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            NH.append(np.log10(2*10**c.e['H2'].col.val + 10**q.e['HI'].col.val))
                            names.append(name)
                            y = c.e['T01'].col.val
                            x = c.e['PDRuv'].col.val + 2.0 - 2*c.e['PDRnH'].col.val
                            ax.text(x,np.log10(y), q.name, fontsize=10)
                            #ax.errorbar(x=x.val, y=y.val , xerr=[[x.minus], [x.plus]],yerr=[[y.minus], [y.plus]],
                            #                                        fmt='o',
                            #                                        markeredgecolor='black', markeredgewidth=2, markersize=1)



        c = ax.scatter(np.array(uv) +2 - 2*np.array(nH),np.array(t01), 50, NH2, cmap='hot', edgecolors='black', vmin=14, vmax=22)
        ax.set_xlabel('nh', fontsize=labelsize)
        ax.set_ylabel('UV',fontsize=labelsize)


        if 1:
            cax = fig09.add_axes([0.6, 0.6, 0.01, 0.2])
            fig09.colorbar(c, cax=cax)
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
    # temp_ compile 3 figs to 1
    if 1:
        fig09, ax = plt.subplots(3, 2, figsize=(9, 8))
        fig09.subplots_adjust(wspace=0.3,hspace=0.3)
        add_DLAs = 1
        k = 0
        if add_DLAs:
            database = 'H2UV'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if name in nonthermal_sys:
                        print('remove sys:',name)
                    else:
                        if 'PDRnH' in c.e:
                            k+=1
                            if (c.e['H2'].col.val > 18):
                                print(k,name)
                                t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                me.append(q.e['Me'].col.val)
                                t01.append(t.val)
                                NH2.append(c.e['H2'].col.val)
                                x = a(c.e['H2'].col.val,c.e['H2'].col.plus,c.e['H2'].col.minus)
                                y = t #a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                                ax[0,0].errorbar(x=x.val, y=y.val,
                                               yerr=[[y.minus], [y.plus]],
                                               xerr=[[x.minus], [x.plus]],
                                               fmt='^', color='black', zorder=-10, capsize=2, ecolor='black',markersize= 1)
                                x = a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                                y = a(c.e['PDRuv'].col.val,c.e['PDRuv'].col.plus,c.e['PDRuv'].col.minus)
                                ax[0,1].errorbar(x=x.val, y=y.val,
                                               yerr=[[y.minus], [y.plus]],
                                               xerr=[[x.minus], [x.plus]],
                                               fmt='^', color='black', zorder=-10, capsize=2, ecolor='black',markersize= 1)

            c = ax[0,1].scatter(np.array(nH), uv, 50, me, cmap='hot', vmin=-1.5, vmax=0.5,
                        edgecolors='black')
            ax[0,0].text(labx,laby,'{\\bf{QSO DLAs}}',fontsize=labelsize)

            print('t01_DLAs', np.mean(t01),np.std(t01), np.amin(t01),np.amax(t01))
            p = np.array(nH) + t01
            print('pth_DLAs', np.mean(p), np.std(p), np.amin(p), np.amax(p))


            if 1:
                cax = fig09.add_axes([0.58, 0.79, 0.01, 0.07])
                fig09.colorbar(c, cax=cax,ticks=[-1, -0.5, 0, 0.5])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[0,1].text(0.15, 0.84, '[X/H]', fontsize=labelsize)

                if 0:
                    adax = fig09.add_axes([0.563, 0.11 + 0.77, 0.337, 0.1])
                    adax.hist(nH, color='gold')
                    adax.axis('off')  # get_xaxis().set_visible(False)
                    adax.set_xlim(-0.1, 3)

                    aday = fig09.add_axes([0.9, 0.11, 0.04, 0.77])
                    aday.hist(uv, color='darkred', orientation="horizontal")
                    aday.set_ylim(-1, 3)
                    aday.axis('off')  # get_xaxis().set_visible(False)
            z = np.array(uv) - np.array(nH)
            c = ax[0,0].scatter(NH2, t01, 50, z, cmap='hot', vmin=-3, vmax=0,
                        edgecolors='black',label='DLAs')

            if 1:
                cax = fig09.add_axes([0.41, 0.79, 0.01, 0.07])
                fig09.colorbar(c, cax=cax,ticks=[ -3, -2,-1])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[0,0].text(colbar_x, colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)



            print('DLAs')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(10**np.array(t01)), np.std(10**np.array(t01)))
            print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

        add_P94 = 0
        if add_P94:
            database = 'P94'
            H2 = H2_exc(H2database=database)
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if (c.e['H2'].col.val > 17):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()

                        t01.append(t.val)
                        NH2.append(c.e['H2'].col.val)
                        #ax[0].errorbar(x=c.e['H2'].col.val, y=t.val,
                        #               yerr=[[t.minus], [t.plus]],
                        #               xerr=[[c.e['H2'].col.minus], [c.e['H2'].col.plus]],
                        #                   fmt='^', color='black', zorder=-10, capsize=2, ecolor='blue',markersize = 1)
            ax[0].scatter(NH2, t01, 30, color='blue', edgecolors='black',alpha=0.8,label='P94')
            print('P94')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:',np.mean(10**np.array(t01)), np.std(10**np.array(t01)))
            print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

        add_MC = 1
        if add_MC:
            database = 'Magellan'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if name in nonthermal_sys:
                        print('remove sys:',name)
                    else:
                        if 'PDRnH' in c.e:
                            if (c.e['H2'].col.val > 17):
                                t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                me.append(q.e['Me'].col.val)
                                t01.append(t.val)
                                NH2.append(c.e['H2'].col.val)
                                x = a(c.e['H2'].col.val, c.e['H2'].col.plus, c.e['H2'].col.minus)
                                y = t  # a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                                ax[2,0].errorbar(x=x.val, y=y.val,
                                               yerr=[[y.minus], [y.plus]],
                                               xerr=[[x.minus], [x.plus]],
                                               fmt='^', color='black', zorder=-10, capsize=2, ecolor='black', markersize=1)
                                x = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus)
                                y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus)
                                ax[2,1].errorbar(x=x.val, y=y.val,
                                               yerr=[[y.minus], [y.plus]],
                                               xerr=[[x.minus], [x.plus]],
                                               fmt='^', color='black', zorder=-10, capsize=2, ecolor='black', markersize=1)

            print('t01_MCs', np.mean(t01), np.std(t01), np.amin(t01), np.amax(t01))
            p = np.array(nH) + t01
            print('pth_MCs', np.mean(p), np.std(p), np.amin(p), np.amax(p))
            c = ax[2,1].scatter(np.array(nH), uv, 50, me, cmap='hot', vmin=-1.5, vmax=0.5,
                              edgecolors='black')

            if 1:
                cax = fig09.add_axes([0.58, 0.51, 0.01, 0.07])
                fig09.colorbar(c, cax=cax, ticks=[-1, -0.5, 0, 0.5])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[2, 1].text(0.15, 0.84, '[X/H]', fontsize=labelsize)

                if 0:
                    adax = fig09.add_axes([0.563, 0.11 + 0.77, 0.337, 0.1])
                    adax.hist(nH, color='gold')
                    adax.axis('off')  # get_xaxis().set_visible(False)
                    adax.set_xlim(-0.1, 3)

                    aday = fig09.add_axes([0.9, 0.11, 0.04, 0.77])
                    aday.hist(uv, color='darkred', orientation="horizontal")
                    aday.set_ylim(-1, 3)
                    aday.axis('off')  # get_xaxis().set_visible(False)
            z = np.array(uv) - np.array(nH)
            c = ax[2,0].scatter(NH2, t01, 50, z, cmap='hot', vmin=-3, vmax=0,
                              edgecolors='black', label='LMC+SMC')
            ax[2, 0].text(labx,laby, '{\\bf{LMC+SMC}}', fontsize=labelsize)

            if 1:
                cax = fig09.add_axes([0.41, 0.51, 0.01, 0.07])
                fig09.colorbar(c, cax=cax, ticks=[-3, -2, -1])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[2, 0].text(colbar_x, colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)

            print('LMC')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(10 ** np.array(t01)), np.std(10 ** np.array(t01)))
            print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

        add_MW = 1
        if add_MW:
            database = 'Galaxy'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if name in nonthermal_sys:
                        print('remove sys:',name)
                    else:
                        if 'PDRnH' in c.e:
                            if (c.e['H2'].col.val > 17):
                                t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                me.append(q.e['Me'].col.val)
                                t01.append(t.val)
                                NH2.append(c.e['H2'].col.val)
                                x = a(c.e['H2'].col.val, c.e['H2'].col.plus, c.e['H2'].col.minus)
                                y = t  # a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                                ax[1,0].errorbar(x=x.val, y=y.val,
                                               yerr=[[y.minus], [y.plus]],
                                               xerr=[[x.minus], [x.plus]],
                                               fmt='^', color='black', zorder=-10, capsize=2, ecolor='black', markersize=1)
                                x = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus)
                                y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus)
                                ax[1,1].errorbar(x=x.val, y=y.val,
                                               yerr=[[y.minus], [y.plus]],
                                               xerr=[[x.minus], [x.plus]],
                                               fmt='^', color='black', zorder=-10, capsize=2, ecolor='black', markersize=1)

            print('t01_MWs', np.mean(t01), np.std(t01), np.amin(t01), np.amax(t01))
            p = np.array(nH) + t01
            print('pth_MWs', np.mean(p), np.std(p), np.amin(p), np.amax(p))
            c = ax[1,1].scatter(np.array(nH), uv, 50, me, cmap='hot', vmin=-1.5, vmax=0.5,
                              edgecolors='black')

            if 1:
                cax = fig09.add_axes([0.58, 0.23, 0.01, 0.07])
                fig09.colorbar(c, cax=cax,ticks=[-1, -0.5, 0, 0.5])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[1,1].text(0.15, 0.84, '[X/H]', fontsize=labelsize)

                if 0:
                    adax = fig09.add_axes([0.563, 0.11 + 0.77, 0.337, 0.1])
                    adax.hist(nH, color='gold')
                    adax.axis('off')  # get_xaxis().set_visible(False)
                    adax.set_xlim(-0.1, 3)

                    aday = fig09.add_axes([0.9, 0.11, 0.04, 0.77])
                    aday.hist(uv, color='darkred', orientation="horizontal")
                    aday.set_ylim(-1, 3)
                    aday.axis('off')  # get_xaxis().set_visible(False)

            z = np.array(uv) - np.array(nH)
            c = ax[1,0].scatter(NH2, t01, 50, z, cmap='hot', vmin=-3, vmax=0,
                              edgecolors='black', label='Galaxy')
            ax[1, 0].text(labx,laby, '{\\bf{MW}}', fontsize=labelsize)

            if 1:
                cax = fig09.add_axes([0.41, 0.23, 0.01, 0.07])
                fig09.colorbar(c, cax=cax, ticks=[-3, -2, -1])
                cax.tick_params(labelsize=labelsize)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                ax[1, 0].text(colbar_x,colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)

            print('MW')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(10**np.array(t01)), np.std(10**np.array(t01)))
            print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

        add_lowzDLA = 0
        if add_lowzDLA:
            database = 'lowzDLAs'
            H2 = H2_exc(H2database=database)
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if name in nonthermal_sys:
                        print('remove sys:',name)
                    else:
                        if 'T01' in c.e:
                            if (c.e['H2'].col.val > 17):
                                t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()

                                t01.append(t.val)
                                NH2.append(c.e['H2'].col.val)
                                #ax[0].errorbar(x=c.e['H2'].col.val, y=t.val,
                                #               yerr=[[t.minus], [t.plus]],
                                #               xerr=[[c.e['H2'].col.minus], [c.e['H2'].col.plus]],
                                #               fmt='^', color='black', zorder=-10, capsize=2, ecolor='magenta', markersize=1)
                ax[0].scatter(NH2, t01, 30, color='magenta', edgecolors='black', marker='^', label='low-$z$ DLAs', zorder=-5)
                print('lowz_DLAs')
                print('nH_stat:', np.mean(nH), np.std(nH))
                print('UV_stat:', np.mean(uv), np.std(uv))
                print('t01_stat:', np.mean(t01), np.std(t01))

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        add_LMC = 1
        if add_LMC:
            database = 'LMC'
            H2 = H2_exc(H2database=database)
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if name in nonthermal_sys:
                        print('remove sys:',name)
                    else:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)

            ax[2, 0].scatter(NH2, t01, s=30, edgecolors='green', marker="D", facecolors='none',zorder=-5)
        add_SMC = 1
        if add_SMC:
            database = 'SMC'
            H2 = H2_exc(H2database=database)
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if name in nonthermal_sys:
                        print('remove sys:',name)
                    else:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)

            ax[2, 0].scatter(NH2, t01, s=30, edgecolors='purple', marker="D", facecolors='none',zorder=-5)
        add_MWall = 1

        if add_MWall:
            database = 'MW'
            H2 = H2_exc(H2database=database)
            t01 = []
            NH2 = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if name in nonthermal_sys:
                        print('remove sys:',name)
                    else:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)

            ax[1, 0].scatter(NH2, t01,  s=30, edgecolors='black', marker="s",facecolors='none',zorder=-5)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        tune =1
        if tune:
            for axs in ax[:,0]:
                axs.xaxis.set_minor_locator(AutoMinorLocator(5))
                axs.xaxis.set_major_locator(MultipleLocator(1))
                axs.yaxis.set_minor_locator(AutoMinorLocator(5))
                axs.yaxis.set_major_locator(MultipleLocator(0.5))
                axs.set_xlim(17.5, 21.5)
                axs.set_ylim(1.5, 2.8)
                axs.set_xlabel('$\log N({\\rm{H_2}})$, cm$^{-2}$', fontsize=labelsize)
                axs.set_ylabel('$\log T_{\\rm{01}}$, K',
                                 fontsize=labelsize)
            for axs in ax[:,1]:
                axs.xaxis.set_minor_locator(AutoMinorLocator(5))
                axs.xaxis.set_major_locator(MultipleLocator(1))
                axs.yaxis.set_minor_locator(AutoMinorLocator(2))
                axs.yaxis.set_major_locator(MultipleLocator(1))
                axs.set_xlim(0, 4)
                axs.set_ylim(-1, 3)
                axs.set_xlabel('$\log n_{\\rm{H}}$, cm$^{-3}$', fontsize=labelsize)
                axs.set_ylabel('$\log I_{\\rm{UV}}$, Mathis unit',
                              fontsize=labelsize)
                #axs.legend(fontsize=labelsize, bbox_to_anchor=(0.25, 0.7, 0.2, 0.3), frameon=True)

            for col in fig09.axes:
                col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
                col.tick_params(which='major', length=5)
                col.tick_params(which='minor', length=3)

if case == 'fig9_2':

    labx, laby = 17.7,2.4
    colbar_x, colbar_y = 20.5, 2.58
    printresult = 0
    msize=80
    if printresult:
        print('print_summary****************')
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if 'PDRnH' in c.e:
                    print('sysname:',name)
                    print('nH=',c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                    print('uv=',c.e['PDRuv'].col.val,c.e['PDRuv'].col.plus,c.e['PDRuv'].col.minus)
    # with hists on axis

    fig09, ax = plt.subplots(1, 1, figsize=(9, 5))
    #fig09.subplots_adjust(wspace=0.3,hspace=0.3)
    add_DLAs = 1
    k = 0
    if add_DLAs:
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'PDRnH' in c.e:
                        k+=1
                        if (c.e['H2'].col.val > 18):
                            print(k,name)
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val,c.e['H2'].col.plus,c.e['H2'].col.minus)
                            y = t #a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax.errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='black', zorder=-10, capsize=2, ecolor='black',markersize= 1)



        z = np.array(uv) - np.array(nH)
        c = ax.scatter(NH2, t01, msize, z, cmap='Reds', vmin=-3, vmax=0,
                    edgecolors='black',label='DLAs',marker='o')

        if 1:
            cax = fig09.add_axes([0.7, 0.72, 0.015, 0.13])
            fig09.colorbar(c, cax=cax,ticks=[ -3, -2,-1,0])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax.text(colbar_x, colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)

    add_MW = 1
    if add_MW:
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:', name)
                else:
                    if 'PDRnH' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val, c.e['H2'].col.plus, c.e['H2'].col.minus)
                            y = t  # a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax.errorbar(x=x.val, y=y.val,
                                        yerr=[[y.minus], [y.plus]],
                                        xerr=[[x.minus], [x.plus]],
                                        fmt='.', color='green', zorder=-10, capsize=2, ecolor='green', markersize=1)

        z = np.array(uv) - np.array(nH)
        c = ax.scatter(NH2, t01, msize, z, cmap='Reds', vmin=-3, vmax=0,
                       edgecolors='green', label='MW', marker='D', linewidths=1.5)

    add_MC = 1
    if add_MC:
        database = 'Magellan'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'PDRnH' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val, c.e['H2'].col.plus, c.e['H2'].col.minus)
                            y = t  # a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax.errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='purple', zorder=-10, capsize=2, ecolor='purple', markersize=1)



        z = np.array(uv) - np.array(nH)
        c = ax.scatter(NH2, t01, msize*1.1, z, cmap='Reds', vmin=-3, vmax=0,
                          edgecolors='purple', label='MC',marker='v',linewidths =1.5)

        if 0:
            cax = fig09.add_axes([0.55, 0.72, 0.015, 0.13])
            fig09.colorbar(c, cax=cax, ticks=[-3, -2, -1,0])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax.text(colbar_x, colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)

        print('LMC')
        print('nH_stat:', np.mean(nH), np.std(nH))
        print('UV_stat:', np.mean(uv), np.std(uv))
        print('t01_stat:', np.mean(10 ** np.array(t01)), np.std(10 ** np.array(t01)))
        print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))


        if 0:
            cax = fig09.add_axes([0.41, 0.23, 0.01, 0.07])
            fig09.colorbar(c, cax=cax, ticks=[-3, -2, -1])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax.text(colbar_x,colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)

        print('MW')
        print('nH_stat:', np.mean(nH), np.std(nH))
        print('UV_stat:', np.mean(uv), np.std(uv))
        print('t01_stat:', np.mean(10**np.array(t01)), np.std(10**np.array(t01)))
        print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

    add_lowzDLA = 0
    if add_lowzDLA:
        database = 'lowzDLAs'
        H2 = H2_exc(H2database=database)
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'T01' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()

                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            #ax[0].errorbar(x=c.e['H2'].col.val, y=t.val,
                            #               yerr=[[t.minus], [t.plus]],
                            #               xerr=[[c.e['H2'].col.minus], [c.e['H2'].col.plus]],
                            #               fmt='^', color='black', zorder=-10, capsize=2, ecolor='magenta', markersize=1)
            ax[0].scatter(NH2, t01, 30, color='magenta', edgecolors='black', marker='^', label='low-$z$ DLAs', zorder=-5)
            print('lowz_DLAs')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(t01), np.std(t01))

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    alpha = 0.5
    msize= 30
    add_LMC = 1
    if add_LMC:
        database = 'LMC'
        H2 = H2_exc(H2database=database)
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if (c.e['H2'].col.val > 17):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        t01.append(t.val)
                        NH2.append(c.e['H2'].col.val)

        ax.scatter(NH2, t01, s=msize, edgecolors='purple', marker="v", facecolors='purple',zorder=-5,alpha=alpha)
    add_SMC = 1
    if add_SMC:
        database = 'SMC'
        H2 = H2_exc(H2database=database)
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'PDRnH' in c.e:
                        print()
                    else:
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        t01.append(t.val)
                        NH2.append(c.e['H2'].col.val)

        ax.scatter(NH2, t01, s=msize, edgecolors='purple', marker="v", facecolors='purple',zorder=-5,alpha=alpha)
    add_MWall = 1

    if add_MWall:
        database = 'MW'
        H2 = H2_exc(H2database=database)
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'PDRnH' in c.e:
                        print()
                    else:
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        t01.append(t.val)
                        NH2.append(c.e['H2'].col.val)

        ax.scatter(NH2, t01,  s=msize, edgecolors='green', marker="D",facecolors='green',zorder=-5,alpha=alpha)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    tune =1
    if tune:
        axs = ax
        axs.xaxis.set_minor_locator(AutoMinorLocator(5))
        axs.xaxis.set_major_locator(MultipleLocator(1))
        axs.yaxis.set_minor_locator(AutoMinorLocator(5))
        axs.yaxis.set_major_locator(MultipleLocator(0.5))
        axs.set_xlim(17.5, 21.6)
        axs.set_ylim(1.5, 3.0)
        axs.set_xlabel('$\log N({\\rm{H_2}})$, cm$^{-2}$', fontsize=labelsize)
        axs.set_ylabel('$\log T_{\\rm{01}}$, K',
                         fontsize=labelsize)
        axs.legend(fontsize=labelsize, bbox_to_anchor=(0.78, 0.79, 0.2, 0.2), frameon=False,markerscale=1.0,labelspacing=0.8)
        legend = ax.get_legend()
        legend.legendHandles[0].set_color('red')
        legend.legendHandles[2].set_color('purple')
        legend.legendHandles[1].set_color('green')

        for col in fig09.axes:
            col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)

if case == 'fig9_3':

    labx, laby = 17.7,2.6
    colbar_x, colbar_y = 20.5, 2.73
    printresult = 0
    if printresult:
        print('print_summary****************')
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if 'PDRnH' in c.e:
                    print('sysname:',name)
                    print('nH=',c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                    print('uv=',c.e['PDRuv'].col.val,c.e['PDRuv'].col.plus,c.e['PDRuv'].col.minus)
    # with hists on axis

    fig09, ax = plt.subplots(1, 1, figsize=(9, 5))
    #fig09.subplots_adjust(wspace=0.3,hspace=0.3)
    add_DLAs = 1
    k = 0
    if add_DLAs:
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'PDRnH' in c.e:
                        k+=1
                        if (c.e['H2'].col.val > 18):
                            print(k,name)
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val,c.e['H2'].col.plus,c.e['H2'].col.minus)
                            y = t #a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax.errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='black', zorder=-10, capsize=2, ecolor='black',markersize= 1)
                            ax.text(x.val, y.val, ''.join(['$', name, '$']))

        #ax[0,0].text(labx,laby,'{\\bf{QSO DLAs}}',fontsize=labelsize)
        if 0:
            print('t01_DLAs', np.mean(t01),np.std(t01), np.amin(t01),np.amax(t01))
        # add histogram to axes
        if 0:
            adax = fig09.add_axes([0.563, 0.11 + 0.77, 0.337, 0.1])
            adax.hist(nH, color='gold')
            adax.axis('off')  # get_xaxis().set_visible(False)
            adax.set_xlim(-0.1, 3)

            aday = fig09.add_axes([0.9, 0.11, 0.04, 0.77])
            aday.hist(uv, color='darkred', orientation="horizontal")
            aday.set_ylim(-1, 3)
            aday.axis('off')  # get_xaxis().set_visible(False)

        z = np.array(uv) - np.array(nH)
        c = ax.scatter(NH2, t01, 100, z, cmap='Reds', vmin=-3, vmax=0,
                    edgecolors='black',label='DLAs',marker='o')

        if 1:
            cax = fig09.add_axes([0.7, 0.72, 0.015, 0.13])
            fig09.colorbar(c, cax=cax,ticks=[ -3, -2,-1,0])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax.text(colbar_x, colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)

    add_MW = 1
    if add_MW:
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:', name)
                else:
                    if 'PDRnH' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val, c.e['H2'].col.plus, c.e['H2'].col.minus)
                            y = t  # a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax.errorbar(x=x.val, y=y.val,
                                        yerr=[[y.minus], [y.plus]],
                                        xerr=[[x.minus], [x.plus]],
                                        fmt='.', color='green', zorder=-10, capsize=2, ecolor='green', markersize=1)


        z = np.array(uv) - np.array(nH)
        c = ax.scatter(NH2, t01, 100, z, cmap='Reds', vmin=-3, vmax=0,
                       edgecolors='green', label='Galaxy', marker='D', linewidths=1.5)

    add_MC = 1
    if add_MC:
        database = 'Magellan'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'PDRnH' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            x = a(c.e['H2'].col.val, c.e['H2'].col.plus, c.e['H2'].col.minus)
                            y = t  # a(c.e['PDRnH'].col.val,c.e['PDRnH'].col.plus,c.e['PDRnH'].col.minus)
                            ax.errorbar(x=x.val, y=y.val,
                                           yerr=[[y.minus], [y.plus]],
                                           xerr=[[x.minus], [x.plus]],
                                           fmt='^', color='purple', zorder=-10, capsize=2, ecolor='purple', markersize=1)



        z = np.array(uv) - np.array(nH)
        c = ax.scatter(NH2, t01, 120, z, cmap='Reds', vmin=-3, vmax=0,
                          edgecolors='purple', label='MC',marker='v',linewidths =1.5)

        if 0:
            cax = fig09.add_axes([0.55, 0.72, 0.015, 0.13])
            fig09.colorbar(c, cax=cax, ticks=[-3, -2, -1,0])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax.text(colbar_x, colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)

        print('LMC')
        print('nH_stat:', np.mean(nH), np.std(nH))
        print('UV_stat:', np.mean(uv), np.std(uv))
        print('t01_stat:', np.mean(10 ** np.array(t01)), np.std(10 ** np.array(t01)))
        print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))


        if 0:
            cax = fig09.add_axes([0.41, 0.23, 0.01, 0.07])
            fig09.colorbar(c, cax=cax, ticks=[-3, -2, -1])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax.text(colbar_x,colbar_y, '$\log I_{\\rm{{UV}}}/n_{\\rm{H}}$', fontsize=labelsize)

        print('MW')
        print('nH_stat:', np.mean(nH), np.std(nH))
        print('UV_stat:', np.mean(uv), np.std(uv))
        print('t01_stat:', np.mean(10**np.array(t01)), np.std(10**np.array(t01)))
        print('t01_stat:', np.amin(10 ** np.array(t01)), np.amax(10 ** np.array(t01)))

    add_lowzDLA = 0
    if add_lowzDLA:
        database = 'lowzDLAs'
        H2 = H2_exc(H2database=database)
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'T01' in c.e:
                        if (c.e['H2'].col.val > 17):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()

                            t01.append(t.val)
                            NH2.append(c.e['H2'].col.val)
                            #ax[0].errorbar(x=c.e['H2'].col.val, y=t.val,
                            #               yerr=[[t.minus], [t.plus]],
                            #               xerr=[[c.e['H2'].col.minus], [c.e['H2'].col.plus]],
                            #               fmt='^', color='black', zorder=-10, capsize=2, ecolor='magenta', markersize=1)
            ax[0].scatter(NH2, t01, 30, color='magenta', edgecolors='black', marker='^', label='low-$z$ DLAs', zorder=-5)
            print('lowz_DLAs')
            print('nH_stat:', np.mean(nH), np.std(nH))
            print('UV_stat:', np.mean(uv), np.std(uv))
            print('t01_stat:', np.mean(t01), np.std(t01))

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    alpha = 0.5
    size = 30
    add_LMC = 1
    if add_LMC:
        database = 'LMC'
        H2 = H2_exc(H2database=database)
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if (c.e['H2'].col.val > 17):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        t01.append(t.val)
                        NH2.append(c.e['H2'].col.val)

        ax.scatter(NH2, t01, s=size, edgecolors='purple', marker="v", facecolors='purple',zorder=-5,alpha=alpha)
    add_SMC = 1
    if add_SMC:
        database = 'SMC'
        H2 = H2_exc(H2database=database)
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'PDRnH' in c.e:
                        print()
                    else:
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        t01.append(t.val)
                        NH2.append(c.e['H2'].col.val)

        ax.scatter(NH2, t01, s=size, edgecolors='purple', marker="v", facecolors='purple',zorder=-5, alpha = alpha)
    add_MWall = 1

    if add_MWall:
        database = 'MW'
        H2 = H2_exc(H2database=database)
        t01 = []
        NH2 = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in nonthermal_sys:
                    print('remove sys:',name)
                else:
                    if 'PDRnH' in c.e:
                        print()
                    else:
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        t01.append(t.val)
                        NH2.append(c.e['H2'].col.val)

        ax.scatter(NH2, t01,  s=size, edgecolors='green', marker="D",facecolors='green',zorder=-5, alpha = alpha)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    tune =1
    if tune:
        axs = ax
        axs.xaxis.set_minor_locator(AutoMinorLocator(5))
        axs.xaxis.set_major_locator(MultipleLocator(1))
        axs.yaxis.set_minor_locator(AutoMinorLocator(5))
        axs.yaxis.set_major_locator(MultipleLocator(0.5))
        axs.set_xlim(17.5, 21.6)
        axs.set_ylim(1.5, 2.8)
        axs.set_xlabel('$\log N({\\rm{H_2}})$, cm$^{-2}$', fontsize=labelsize)
        axs.set_ylabel('$\log T_{\\rm{01}}$, K',
                         fontsize=labelsize)
        axs.legend(fontsize=labelsize, bbox_to_anchor=(0.78, 0.77, 0.2, 0.2), frameon=False,markerscale=1.0,labelspacing=0.8)
        legend = ax.get_legend()
        legend.legendHandles[0].set_color('red')
        legend.legendHandles[2].set_color('purple')
        legend.legendHandles[1].set_color('green')

        for col in fig09.axes:
            col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)


#plot exc of H2 and Ci for q1232+082
if case == 'fig10':
    fig10, ax = plt.subplots(1, 3, figsize=(9, 2))
    fig10.subplots_adjust(wspace=0.4)
    qname = 'Q1232+0815_0'
    with open('output/{:s}_CI_lnL.pkl'.format(qname), 'rb') as f:
        xCi, yCi, zCi = pickle.load(f)
    with open('output/{:s}_CI_cols.pkl'.format(qname), 'rb') as f:
        xCi, yCi, cols_Ci = pickle.load(f)
    # add lnL and cols for H2 for J=0-5
    with open('output/{:s}_H2_lnL_J05.pkl'.format(qname), 'rb') as f:
        xH2, yH2, zH2 = pickle.load(f)
    with open('output/{:s}_H2_cols_J05.pkl'.format(qname), 'rb') as f:
        xH2, yH2, cols_H2 = pickle.load(f)
    # add lnL for H2 for J=0-2
    with open('output/{:s}_H2_lnL_J02.pkl'.format(qname), 'rb') as f:
        xH2, yH2, zH2J02 = pickle.load(f)

    # plot lnL countours
    if 1:
        d = distr2d(x=xCi, y=yCi, z=np.exp(zCi))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        d.plot_contour(ax=ax[0], color='green', color_point=None, cmap='Greens', alpha=0, lw=1.0,label='CI')

        dH2 = distr2d(x=xH2, y=yH2, z=np.exp(zH2))
        dx, dy = dH2.marginalize('y'), dH2.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        dH2.plot_contour(ax=ax[0], color='purple', color_point=None, cmap='Purples', alpha=0, lw=1.0, label='CI')
        dH2.dopoint()

        dH2J02 = distr2d(x=xH2, y=yH2, z=np.exp(zH2J02))
        dx, dy = dH2J02.marginalize('y'), dH2J02.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        dH2J02.plot_contour(ax=ax[0], color='royalblue', color_point=None, cmap='Blues', alpha=0, lw=1.0, label='CI')

        djoin = distr2d(x=xH2, y=yH2, z=np.exp(zH2J02+zCi))
        djoin.dopoint()
        dx, dy = djoin.marginalize('y'), djoin.marginalize('x')
        dy.stats(latex=-1, name='log UV')
        dx.stats(latex=-1, name='log n')
        djoin.plot_contour(ax=ax[0], color='red', color_point=None, cmap='Reds', alpha=0, lw=1.0, label='CI')

    # plot H2 excitaiton
    if 1:
        species = 'H2'
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        H2.plot_objects(qname, ax=ax[1], syst=0.3, label='Data')

        q = H2.comp(qname)
        sp = [s for s in q.e.keys() if species + 'j' in s]
        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
        x, stat = getatomic(species, levels=j)
        mod = [cols_H2['H2j' + str(i)](dH2.point[0], dH2.point[1]) - np.log10(stat[i]) for i in j]
        ax[1].plot([H2energy[0,i] for i in j], mod,color='purple')
        mod = [cols_H2['H2j' + str(i)](djoin.point[0], djoin.point[1]) - np.log10(stat[i]) for i in j]
        ax[1].plot([H2energy[0,i] for i in j], mod,color='red')

    # plot CI excitation
    if 1:
        H2.plot_objects(objects=qname, ax=ax[2], syst=0.1, species='CI', label='Data')
        species = 'CI'
        syst = 0.1
        q = H2.comp(qname)
        sp = [s for s in q.e.keys() if species + 'j' in s]
        j = np.sort([int(s[3:]) for s in sp if 'v' not in s])
        x, stat = getatomic(species, levels=j)
        mod = [cols_Ci['CIj' + str(i)](dH2.point[0], dH2.point[1]) - np.log10(stat[i]) for i in j]
        print('mod',mod)
        ax[2].plot([CIenergy[i] for i in j], mod-mod[0], color='purple')
        mod = [cols_Ci['CIj' + str(i)](djoin.point[0], djoin.point[1]) - np.log10(stat[i]) for i in j]
        ax[2].plot([CIenergy[i] for i in j], mod-mod[0], color='red')

    # tuneaxis
    if 1:
        ax[0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[0].set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
        ax[0].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[0].tick_params(which='major', length=5)
        ax[0].tick_params(which='minor', length=4)
        ax[0].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)

        ax[1].set_xlabel('Energy level H$_2$, K', fontsize=labelsize)
        ax[1].set_ylabel('$\log N{\\rm(H_2)_J}/g_{\\rm{J}}$', fontsize=labelsize)
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1].xaxis.set_major_locator(MultipleLocator(1000))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(1))
        ax[1].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[1].tick_params(which='major', length=5)
        ax[1].tick_params(which='minor', length=4)
        #ax[1].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)

        ax[2].set_xlabel('Energy level CI, K', fontsize=labelsize)
        ax[2].set_ylabel('$\log N{\\rm(CI)}_{\\rm{J}}/g_{\\rm{J}} - \log N_{\\rm{0}}/g_{\\rm{0}}$', fontsize=labelsize)
        ax[2].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[2].xaxis.set_major_locator(MultipleLocator(20))
        ax[2].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[2].yaxis.set_major_locator(MultipleLocator(1))
        ax[2].tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax[2].tick_params(which='major', length=5)
        ax[2].tick_params(which='minor', length=4)
        #ax[2].legend(fontsize=labelsize, bbox_to_anchor=(0.8, 0.65, 0.2, 0.3), frameon=True)
        ax[2].set_xlim(-5,50)
        ax[2].set_ylim(-2.5, 1)
# plot thermalization figure
if case == 'fig11':
    fig11, ax = plt.subplots(2, 2, figsize=(9, 8))
    fig11.subplots_adjust(wspace=0.3,hspace=0.4)

    vmin = 17
    vmax = 21
    sys = 0.6 #0.8
    xmin, xmax, ymin, ymax = 0, 4, -1, 3
    levels = [18,19,20]
    cmap = 'Greens'
    if 1:
        with open('temp/Nh2th_t01_z0_1_c1.pkl', 'rb') as f:
        #with open('temp/thermal_limit_z0_1_c1_fr50.pkl', 'rb') as f:
        #with open('temp/thermal_limit_z0_1_c1_fr50_t01.pkl', 'rb') as f:
            x, y, z = pickle.load(f)
        if sys>0:
            z = z +sys
        # interpolate
        z_rbf = Rbf(x, y, np.asarray([c for c in z]), function='multiquadric', smooth=0.1)
        # refine
        num = 100
        x, y = np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X)
        for i, xi in enumerate(x):
            for k, yi in enumerate(y):
                z[k, i] = z_rbf(xi, yi)
        # plot
        c = ax[0,0].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[0,0].contour(x, y, z, levels=levels, colors='black', alpha=1,
                   linestyles=['-'], linewidths=1.0)
        ax[0,0].text(0.2,2.4,'$\log Z=-1$',fontsize=labelsize,color='white')

    if 1:
        #with open('temp/Nh2th_t01_z0_31_c1.pkl', 'rb') as f:
        #with open('temp/thermal_limit_z0_3_c1_fr50.pkl', 'rb') as f:
        with open('temp/thermal_limit_z0_3_c1_fr50_t01.pkl', 'rb') as f:
            x, y, z = pickle.load(f)
        if sys>0:
            z = z +sys
        # interpolate
        z_rbf = Rbf(x, y, np.asarray([c for c in z]), function='multiquadric', smooth=0.1)
        # refine
        num = 100
        x, y = np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X)
        for i, xi in enumerate(x):
            for k, yi in enumerate(y):
                z[k, i] = z_rbf(xi, yi)
        # plot
        ax[0,1].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[0,1].contour(x, y, z, levels=levels, colors='black', alpha=1,
                   linestyles=['-'], linewidths=1.0)
        ax[0, 1].text(0.2, 2.4, '$\log Z=-0.5$', fontsize=labelsize, color='white')
    if 1:
        #with open('temp/Nh2th_t01_z1_0_c1.pkl', 'rb') as f:
        #with open('temp/thermal_limit_z1_0_c1_fr50.pkl', 'rb') as f:
        with open('temp/thermal_limit_z1_0_c1_fr50_t01.pkl', 'rb') as f:
            x, y, z = pickle.load(f)
        if sys>0:
            z = z +sys
        # interpolate
        z_rbf = Rbf(x, y, np.asarray([c for c in z]), function='multiquadric', smooth=0.1)
        # refine
        num = 100
        x, y = np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X)
        for i, xi in enumerate(x):
            for k, yi in enumerate(y):
                z[k, i] = z_rbf(xi, yi)
        # plot
        ax[1,0].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[1,0].contour(x, y, z, levels=levels, colors='black', alpha=1,
                   linestyles=['-'], linewidths=1.0)
        ax[1, 0].text(0.2, 2.4, '$\log Z=0$', fontsize=labelsize, color='white')
    if 1:
        #with open('temp/Nh2th_t01_z1_0_c1.pkl', 'rb') as f:
        #with open('temp/thermal_limit_z3_0_c1_fr50.pkl', 'rb') as f:
        with open('temp/thermal_limit_z3_0_c1_fr50_t01.pkl', 'rb') as f:
            x, y, z = pickle.load(f)
        if sys>0:
            z = z +sys
        # interpolate
        z_rbf = Rbf(x, y, np.asarray([c for c in z]), function='multiquadric', smooth=0.1)
        # refine
        num = 100
        x, y = np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X)
        for i, xi in enumerate(x):
            for k, yi in enumerate(y):
                z[k, i] = z_rbf(xi, yi)
        # plot
        ax[1,1].pcolor(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[1,1].contour(x, y, z, levels=levels, colors='black', alpha=1,
                   linestyles=['-'], linewidths=1.0)
        ax[1, 1].text(0.2, 2.4, '$\log Z=0.5$', fontsize=labelsize, color='white')
    #plot colorbar
    if 1:
        cax = fig11.add_axes([0.93, 0.27, 0.015, 0.47])
        fig11.colorbar(c, cax=cax, orientation='vertical')
        cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        ax[1, 1].text(4.15, 0.5, '$\log N{\\rm(H_2)}$', fontsize=labelsize)
    add_sys = 1
    if add_sys:

        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        NH2 = []
        names = []
        me = []
        for q in H2.H2.values():
            if q.name in []:
                print('bas system')
            else:
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    if name in nonthermal_sys:
                        print('remove sys',name)
                    else:
                        if 'PDRnH' in c.e:
                            if (c.e['H2'].col.val > 15):
                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                NH2.append(c.e['H2'].col.val)
                                me.append(q.e['Me'].col.val)
                                names.append(name)
                                x = c.e['PDRnH'].col
                                y = c.e['PDRuv'].col
        uv = np.array(uv)
        nH= np.array(nH)
        NH2 = np.array(NH2)
        me = np.array(me)
        for k,name in enumerate(names):
            if me[k]<-0.75:
                axs = ax[0,0]
            elif me[k]>-0.75 and me[k]<-0.25:
                axs = ax[0,1]
            elif me[k]>-0.25 and me[k]<0.25:
                axs = ax[1,0]
            else:
                axs = ax[1,1]
            axs.text(nH[k], uv[k], "${:s}$".format(name), fontsize=labelsize,color='black')
        mask = np.array(me)<-0.75
        ax[0,0].scatter(np.array(nH)[mask], np.array(uv)[mask], 40, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax)
        mask = (np.array(me) > -0.75)*(np.array(me) < -0.25)
        ax[0, 1].scatter(np.array(nH)[mask], np.array(uv)[mask], 40, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax)
        mask = (np.array(me) > -0.25)*(np.array(me) < 0.25)
        ax[1, 0].scatter(np.array(nH)[mask], np.array(uv)[mask], 40, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax)
        mask = (np.array(me) > 0.25)
        ax[1, 1].scatter(np.array(nH)[mask], np.array(uv)[mask], 40, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax)

        #>>>>>>>>>>>>>>>>>>>>>>>
        if 0:
            database = 'Galaxy'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            NH2 = []
            names = []
            me = []
            for q in H2.H2.values():
                if q.name in []:
                    print('bas system')
                else:
                    for i_c, c in enumerate(q.comp):
                        name = "".join([q.name, '_', str(i_c)])
                        if 'PDRnH' in c.e:
                            if (c.e['H2'].col.val > 17):
                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                NH2.append(c.e['H2'].col.val)
                                me.append(q.e['Me'].col.val)
                                names.append(name)
                                x = c.e['PDRnH'].col
                                y = c.e['PDRuv'].col
            uv = np.array(uv)
            nH = np.array(nH)
            NH2 = np.array(NH2)
            me = np.array(me)
            for k, name in enumerate(names):
                if me[k] < -0.75:
                    axs = ax[0, 0]
                elif me[k] > -0.75 and me[k] < -0.25:
                    axs = ax[0, 1]
                else:
                    axs = ax[1, 0]
                axs.text(nH[k], uv[k], "${:s}$".format(name), fontsize=5,color='blue')
            mask = np.array(me) < -0.75
            ax[0, 0].scatter(np.array(nH)[mask], np.array(uv)[mask], 20, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax,marker='^')
            mask = (np.array(me) > -0.75) * (np.array(me) < -0.25)
            ax[0, 1].scatter(np.array(nH)[mask], np.array(uv)[mask], 20, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax,marker='^')
            mask = (np.array(me) > -0.25)
            ax[1, 0].scatter(np.array(nH)[mask], np.array(uv)[mask], 20, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax,marker='^')

        #>>>>>>>>>>>>>>>>>>>>>>>
        if 0:
            database = 'Magellan'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            NH2 = []
            names = []
            me = []
            for q in H2.H2.values():
                if q.name in []:
                    print('bas system')
                else:
                    for i_c, c in enumerate(q.comp):
                        name = "".join([q.name, '_', str(i_c)])
                        if 'PDRnH' in c.e:
                            if (c.e['H2'].col.val > 17):
                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                NH2.append(c.e['H2'].col.val)
                                me.append(q.e['Me'].col.val)
                                names.append(name)
                                x = c.e['PDRnH'].col
                                y = c.e['PDRuv'].col
            uv = np.array(uv)
            nH = np.array(nH)
            NH2 = np.array(NH2)
            me = np.array(me)
            for k, name in enumerate(names):
                if me[k] < -0.75:
                    axs = ax[0, 0]
                elif me[k] > -0.75 and me[k] < -0.25:
                    axs = ax[0, 1]
                else:
                    axs = ax[1, 0]
                axs.text(nH[k], uv[k], "${:s}$".format(name), fontsize=5,color='green')
            mask = np.array(me) < -0.75
            ax[0, 0].scatter(np.array(nH)[mask], np.array(uv)[mask], 20, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax,marker='s')
            mask = (np.array(me) > -0.75) * (np.array(me) < -0.25)
            ax[0, 1].scatter(np.array(nH)[mask], np.array(uv)[mask], 20, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax,marker='s')
            mask = (np.array(me) > -0.25)
            ax[1, 0].scatter(np.array(nH)[mask], np.array(uv)[mask], 20, NH2[mask], cmap=cmap, vmin=vmin, vmax=vmax,marker='s')

    #tune
    if 1:
        for axs in ax[0, :]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)
        for axs in ax[1, :]:
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)

        ax[0, 0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[1, 0].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[0, 1].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        ax[1, 1].set_ylabel('$\log~{\\rm{I_{UV}}}$, Mathis unit', fontsize=labelsize)
        for axs in [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]:
            axs.set_xlabel('$\log~{\\rm{n_H}}$, cm$^{-3}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(5))
            axs.xaxis.set_major_locator(MultipleLocator(1))
            axs.yaxis.set_minor_locator(AutoMinorLocator(5))
            axs.yaxis.set_major_locator(MultipleLocator(1))
            axs.set_xlim(0,4)
            axs.set_ylim(-1, 3)
#plot thermalization procedure
if case == 'fig12':
    fig12, ax = plt.subplots(1, 4, figsize=(9, 2))
    fig12.subplots_adjust(wspace=0.4)

    H2 = H2_exc(folder='./data/sample/uv_n_z/test/')
    H2.readmodel(filename='h2uv_uv1e1_av0_5_z0_1_n1e2_s_25.hdf5')
    models = H2.listofmodels()
    m = models[0]
    parx = 'h2'
    #pars = ['OPR', 'OPR_logNJ1/J02', 'OPR_logNJ1/J0'], ['tgas', 'Nh2t01', 'T01']],
    #species = [['H2j0/H2', 'H2j1/H2', 'H2j2/H2', 'H2j3/H2', 'H2j4/H2'], ['H', 'H2', 'H+']],
    logx = True
    logy = True
    limit = {'H2': 21}
    legend = False
    NH2therm = m.calc_T01_limit(limith2={'H2': 21.5}, case=2)

    mask = getattr(m, parx) > 0
    if limit is not None:
        if hasattr(m, list(limit.keys())[0]):
            v = getattr(m, list(limit.keys())[0])
            mask *= v < list(limit.values())[0]
        elif list(limit.keys())[0] in m.sp.keys():
            # v = self.sp[list(limit.keys())[0]]
            m.set_mask(logN=limit)
            mask *= m.mask
    mask *=np.log10(getattr(m, 'h2')) > 12
    x = np.log10(getattr(m, parx))[mask]


    # plot OPR H2 ratio profile
    y = m.OPR[mask] #, 'OPR_logNJ1/J02', 'OPR_logNJ1/J0']
    #m.plot_phys_cond(pars=pars, logx=logx, ax=ax[0], legend=legend, parx=parx, limit=limit)
    ax[0].plot(x, y, label='$n_O/n_P$')
    y = (m.sp['NH2j1'] / m.sp['NH2j0'])[mask]
    ax[0].plot(x,y,label='N1/N0')
    y = ((m.sp['NH2j1']+m.sp['NH2j3']) / \
                    (m.sp['NH2j0'] + m.sp['NH2j2'] + m.sp['NH2j4']))[mask]
    ax[0].plot(x, y,label='$N_O/N_P$')

    #plot Tkin and T01 prodiles
    y = m.tgas[mask]
    ax[1].plot(x, np.log10(y),label='T$_{\\rm{kin}}$')
    y = 85.3 * 2 / np.log(9. / (m.sp['NH2j1']/ m.sp['NH2j0']))[mask]
    mask1 = y>0
    ax[1].plot(x[mask1], np.log10(y)[mask1],label='T$_{01}$')
    ax[1].set_ylim(1.5,3)

    species = ['H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4']
    for s in species:
        ax[2].plot(x,np.log10(m.sp[s] / m.sp['H2'])[mask], label="".join(['J', s[3]]))


    species = ['H', 'H2', 'H+']
    labels = ['H', 'H$_2$', 'H$+$']
    for k,s in enumerate(species):
        ax[3].plot(x, np.log10(m.sp[s])[mask], label=labels[k])

    # tune
    if 1:
        for axs in ax[:]:
            axs.axvline(x=NH2therm,ls='--',lw=1.0, color='black')
            axs.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=2)


            axs.set_xlabel('$\log~N({\\rm{H_2}}), {\\rm{cm}}^{-2}$', fontsize=labelsize)
            axs.xaxis.set_minor_locator(AutoMinorLocator(4))
            axs.xaxis.set_major_locator(MultipleLocator(2))
            axs.set_xlim(12, x[-1])
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(1))
        ax[0].yaxis.set_major_locator(MultipleLocator(3))
        ax[0].set_ylabel('OPR of H$_2$', fontsize=labelsize)

        ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[1].set_ylabel('$\log T$, K', fontsize=labelsize)


        ax[2].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[2].yaxis.set_major_locator(MultipleLocator(1))
        ax[2].set_ylabel('$n({\\rm{H_2,J}})/n({\\rm{H_2}})$', fontsize=labelsize)
        ax[2].set_ylim(-6, 0.5)

        ax[3].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[3].yaxis.set_major_locator(MultipleLocator(1))
        ax[3].set_ylabel('$\log n$, cm$^{-3}$', fontsize=labelsize)
        ax[3].set_ylim(-3, 2.5)
        #legend
        ax[0].legend(bbox_to_anchor=(0.45, 0.7, 0.1, 0.2), frameon=True, fontsize=labelsize - 4)
        ax[1].legend(bbox_to_anchor=(0.35, 0.15, 0.1, 0.2), frameon=True, fontsize=labelsize - 4)
        ax[2].legend(bbox_to_anchor=(0.3, 0.3, 0.1, 0.2),frameon=True, fontsize=labelsize - 4)
        ax[3].legend(bbox_to_anchor=(0.35, 0.6, 0.1, 0.2), frameon=True, fontsize=labelsize - 4)

if case == 'figtest':
    vmin = 16
    vmax = 20

    with open('temp/Nh2th_t01_z1_0_c2.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
    #z[32]=17.7
    #z = z +0.3 + 0.5
    if 1:
        fig1, ax1 = plt.subplots()
        for v1, v2, l in zip(x, y, z):
            #v1, v2 = np.log10(v1), np.log10(v2)
            ax1.scatter(v1, v2, 0)
            ax1.text(v1, v2, '{:.1f}'.format(l), size=20)

    if 1:
        # interpolate
        z_rbf = Rbf(x, y, np.asarray([c for c in z]), function='multiquadric', smooth=0.1)

        num = 100
        x1, y1, z1 = x, y, z  # copy for save
        xmin, xmax, ymin, ymax = 0, 4, -1, 3
        x, y = np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X)
        for i, xi in enumerate(x):
            for k, yi in enumerate(y):
                z[k, i] = z_rbf(xi, yi)
        if 1:
            fig, ax = plt.subplots()
            c = ax.pcolor(X, Y, z, cmap='Greens', vmin=vmin, vmax=vmax)
            ax.contour(x, y, z, levels=[16,17,17.5], colors='black', alpha=1,
                             linestyles=['-'], linewidths=1.0)

            #ax.scatter(x1, y1, 100, z1, cmap='hot_r', vmin=14, vmax=21 ,edgecolors='black')
            if 1:
                cax = fig.add_axes([0.93, 0.27, 0.01, 0.47])
                fig.colorbar(c, cax=cax, orientation='vertical')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
                #ax.text(7, 2.0, '$\log N{\\rm(H_2, J)}$, cm$^{-2}$', fontsize=labelsize + 2, rotation=90)

        if 0:
            database = 'H2UV'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            NH = []
            names = []
            for q in H2.H2.values():
                if q.name in ['J0000+0048', 'J1513+0352', 'J1443+2724', 'Q1232+0815', 'J1439+1118', 'J2100-0641',
                              'J2123-0050', 'J1439+1118']:
                    print('bas system')
                else:
                    for i_c, c in enumerate(q.comp):
                        name = "".join([q.name, '_', str(i_c)])
                        if 'PDRnH' in c.e:
                            if (c.e['H2'].col.val > 17):
                                t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()

                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                me.append(q.e['Me'].col.val)
                                t01.append(t.val)
                                NH2.append(c.e['H2'].col.val)
                                NH.append(np.log10(2 * 10 ** c.e['H2'].col.val + 10 ** q.e['HI'].col.val))
                                names.append(name)
                                x = c.e['PDRnH'].col
                                y = c.e['PDRuv'].col
                                ax.text(x.val, y.val, q.name, fontsize=8)
                                #ax.errorbar(x=x.val, y=y.val, xerr=[[x.minus], [x.plus]], yerr=[[y.minus], [y.plus]], markersize=1)

            c = ax.scatter(np.array(nH), np.array(uv), 50, NH2, cmap='Greens', vmin=vmin, vmax=vmax)


if case == 'fig13':
    f_name = './data/tmp/pdr_grid_z03_n1e3_u101_cmb2_5_pah_iforh2_0_bth_13.dat'
    av = []
    nh = []
    tk = []
    chotot = []
    reftot = []
    chophg = []
    with open(f_name) as f:
        for k, line in enumerate(f):
            if k > 0:
                values = [s for s in line.split()]
                av.append(float(values[2]))
                nh.append(float(values[3]))
                tk.append(float(values[4]))
                chotot.append(float(values[5]))
                chophg.append(float(values[6]))
                reftot.append(float(values[3]))
    av = np.array(av)
    nh = np.array(nh)
    tk = np.array(tk)
    chotot = np.array(chotot)
    reftot = np.array(reftot)
    chophg = np.array(chophg)
    fig,ax = plt.subplots()
    ax.plot(np.log10(av), (chotot))
    ax.plot(np.log10(av), (chophg))
    ax.plot(np.log10(av), (chotot) - chophg)
    axi = ax.twinx()
    axi.plot(np.log10(av),tk,color='red')

if case == 'fig14':
    fig14, ax = plt.subplots(1, 2, figsize=(9, 4))
    fig14.subplots_adjust(wspace=0.3)
    database = 'H2UV'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    names = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if name in nonthermal_sys:
                    print('remove sys:', name)
                else:
                    if (c.e['H2'].col.val > 18):
                        names.append(name)
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                        nH.append(c.e['PDRnH'].col.val)
                        uv.append(c.e['PDRuv'].col.val)
                        me.append(q.e['Me'].col.val)
                        t01.append(t.log().val)
                        NH2.append(c.e['H2'].col.val)
                        xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                      q.e['HI'].col.plus,
                                                                                                      q.e['HI'].col.minus)
                        NH.append(xNH.val)
                        x = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')  / a(
                            c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l')

                        ax[0].errorbar(x=t.val, y=x.val,
                                       yerr=[[x.minus], [x.plus]],
                                       xerr=[[t.minus], [t.plus]],
                                       fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=4)
                        ax[0].text(t.val,x.val,''.join(['$',name,'$']))
    c = ax[0].scatter( np.array(t01),np.array(uv) -nH, 50, NH2, cmap='Reds', edgecolors='black', vmin=18, vmax=21)


    if 1:
        cax = fig14.add_axes([0.15, 0.63, 0.01, 0.2])
        fig14.colorbar(c, cax=cax, ticks=[18, 19,20, 21])
        cax.tick_params(labelsize=labelsize)
        cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        cax.tick_params(which='minor', length=3, direction='in')
        cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        ax[0].text( 1.35, -1.1,'$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)
        #ax[0].text(20, 4.76, '$\log\\,N({\\rm{H_2}})>19$', fontsize=labelsize)


    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    names = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if (c.e['H2'].col.val > 19):
                    names.append(name)
                    t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                    p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                    nH.append(c.e['PDRnH'].col.val)
                    uv.append(c.e['PDRuv'].col.val)
                    me.append(q.e['Me'].col.val)
                    t01.append(t.log().val)
                    NH2.append(c.e['H2'].col.val)
                    xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                  q.e['HI'].col.plus,
                                                                                                  q.e['HI'].col.minus)
                    NH.append(xNH.val)
                    x = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')  / a(
                        c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l')

                    ax[1].errorbar(x=t.val, y=x.val,
                                   xerr=[[t.minus], [t.plus]],
                                   yerr=[[x.minus], [x.plus]],
                                   fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=4)
    ax[1].plot(np.array(t01),np.array(uv) -nH, 'o', markersize=8, markerfacecolor='red', markeredgecolor='black', label='DLAs $\\log N({\\rm{H_2}})>19$')
    if 1:
        print('fit:', np.polyfit(np.array(t01),np.array(uv) -nH,1))
        print('correlation:', scipy.stats.pearsonr(np.array(t01),np.array(uv) -nH))
    if 1:
        def linearFunc(x, intercept, slope):
            y = intercept + slope * x
            return y
        a_fit, cov = curve_fit(linearFunc, np.array(t01), np.array(uv) -nH)
        print('afit',a_fit,cov)
        #b_fit, bcov = curve_fit(linearFunc, np.array(uv) - nH, np.array(t01))

    database = 'Galaxy'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    names = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if (c.e['H2'].col.val > 19):
                    names.append(name)
                    t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                    p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                    nH.append(c.e['PDRnH'].col.val)
                    uv.append(c.e['PDRuv'].col.val)
                    me.append(q.e['Me'].col.val)
                    t01.append(t.log().val)
                    NH2.append(c.e['H2'].col.val)
                    xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                  q.e['HI'].col.plus,
                                                                                                  q.e['HI'].col.minus)
                    NH.append(xNH.val)
                    x = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l') / a(
                        c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l')

                    ax[1].errorbar(x=t.val, y=x.val,
                                   xerr=[[t.minus], [t.plus]],
                                   yerr=[[x.minus], [x.plus]],
                                   fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=4)
    ax[1].plot(np.array(t01),np.array(uv) - nH, 'D', markersize=7, markerfacecolor='green', markeredgecolor='black', label ='MW')
    if 1:
        print('fit_MW:', np.polyfit(np.array(t01),np.array(uv) -nH,1))
        print('correlation:', scipy.stats.pearsonr(np.array(t01),np.array(uv) -nH))
    if 1:
        def linearFunc(x, intercept, slope):
            y = intercept + slope * x
            return y
        a_fit, cov = curve_fit(linearFunc, np.array(t01), np.array(uv) -nH)
        print('afit', a_fit, cov)


    database = 'Magellan'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    names = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if (c.e['H2'].col.val > 19):
                    names.append(name)
                    t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                    p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                    nH.append(c.e['PDRnH'].col.val)
                    uv.append(c.e['PDRuv'].col.val)
                    me.append(q.e['Me'].col.val)
                    t01.append(t.log().val)
                    NH2.append(c.e['H2'].col.val)
                    xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                  q.e['HI'].col.plus,
                                                                                                  q.e['HI'].col.minus)
                    NH.append(xNH.val)
                    x = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l') / a(
                        c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l')

                    ax[1].errorbar(x=t.val, y=x.val,
                                   xerr=[[t.minus], [t.plus]],
                                   yerr=[[x.minus], [x.plus]],
                                   fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=4)
    ax[1].plot(np.array(t01), np.array(uv) - nH, 'v', markersize=8, markerfacecolor='purple',
               markeredgecolor='black', label = 'MC')


    if 1:
        x =np.linspace(1.6,2.1,10)
        ax[1].plot(x,-11.03+5.23*(x),'--',color='red')
        x = np.linspace(1.55,2.15,10)
        ax[1].plot(x, -4.78 + 1.29 * (x), '--', color='green',zorder = 10)

    ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[0].yaxis.set_major_locator(MultipleLocator(1))
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[0].xaxis.set_major_locator(MultipleLocator(0.2))
    ax[0].set_ylim(-3.4, 0.4)
    ax[0].set_xlim(1.3, 2.4)
    ax[0].set_ylabel('$\log I_{\\rm{UV}}/n_{\\rm{H}}$', fontsize=labelsize)
    ax[0].set_xlabel('$\log T_{\\rm{01}}$, K', fontsize=labelsize)
    ax[1].legend(bbox_to_anchor=(0.4, 0.73, 0.25, 0.2), frameon=False, fontsize=labelsize)
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[1].yaxis.set_major_locator(MultipleLocator(1))
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].xaxis.set_major_locator(MultipleLocator(0.2))
    ax[1].set_ylim(-3.4, 0.4)
    ax[1].set_xlim(1.3, 2.4)
    ax[1].set_ylabel('$\log I_{\\rm{UV}}/n_{\\rm{H}}$', fontsize=labelsize)
    ax[1].set_xlabel('$\log T_{\\rm{01}}$, K',   fontsize=labelsize)


    for col in ax[:]:
        col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        col.tick_params(which='major', length=5)
        col.tick_params(which='minor', length=4)

if case == 'fig15':
    fig15, ax = plt.subplots(1, 2, figsize=(9, 4))
    fig15.subplots_adjust(wspace=0.3)
    database = 'H2UV'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    names = []
    pth = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if name in nonthermal_sys:
                print('remove sys:', name)
            else:
                if 'PDRnH' in c.e:
                    if (c.e['H2'].col.val > 18):
                        names.append(name)
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                        nH.append(c.e['PDRnH'].col.val)
                        uv.append(c.e['PDRuv'].col.val)
                        me.append(q.e['Me'].col.val)
                        t01.append(t.log().val)
                        NH2.append(c.e['H2'].col.val)
                        pth.append(p.val)
                        xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                      q.e['HI'].col.plus,
                                                                                                      q.e['HI'].col.minus)
                        NH.append(xNH.val)
                        x = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l') * a(2, 0, 0,
                                                                                                            'l') / a(
                            c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l')

                        ax[0].errorbar(x=xNH.val, y=p.val,
                                       yerr=[[p.minus], [p.plus]],
                                       xerr=[[xNH.minus], [xNH.plus]],
                                       fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=4)
                        #ax[0].text(xNH.val, p.val, ''.join(['$',name,'$']))
    c = ax[0].scatter(np.array(NH), np.array(t01) + nH, 50, NH2, cmap='Reds', edgecolors='black', vmin=18, vmax=21)
    #c = ax[0].scatter(np.array(NH2), np.array(nH), 50, NH2, cmap='Reds', edgecolors='black', vmin=18, vmax=21)
    print('meanpressure for DLAs:',np.mean(pth),np.mean(nH))

    if 1:
        cax = fig15.add_axes([0.16, 0.64, 0.01, 0.2])
        fig15.colorbar(c, cax=cax, ticks=[18, 19, 20, 21])
        cax.tick_params(labelsize=labelsize)
        cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        cax.tick_params(which='minor', length=3, direction='in')
        cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        ax[0].text(18.8, 4.4, '$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)
        #ax[0].text(20, 4.76, '$\log\\,N({\\rm{H_2}})>19$', fontsize=labelsize)


    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    names = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if (c.e['H2'].col.val > 19):
                    names.append(name)
                    t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                    p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                    nH.append(c.e['PDRnH'].col.val)
                    uv.append(c.e['PDRuv'].col.val)
                    me.append(q.e['Me'].col.val)
                    t01.append(t.log().val)
                    NH2.append(c.e['H2'].col.val)
                    xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                  q.e['HI'].col.plus,
                                                                                                  q.e['HI'].col.minus)
                    NH.append(xNH.val)
                    x = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l') * a(2, 0, 0,
                                                                                                        'l') / a(
                        c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l')

                    ax[1].errorbar(x=xNH.val, y=p.val,
                                   yerr=[[p.minus], [p.plus]],
                                   xerr=[[xNH.minus], [xNH.plus]],
                                   fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=4)
    ax[1].plot(np.array(NH), np.array(t01) + nH, 'o', markersize=8, markerfacecolor='red', markeredgecolor='black', label = 'DLAs $\\log N({\\rm{H_2}})>19$',zorder=10)
    if 1:
        print('fit:', np.polyfit(NH, np.array(t01) + nH, 1))
        print('correlation:', scipy.stats.pearsonr(np.array(NH), np.array(t01) + nH))

    database = 'Galaxy'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    names = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if (c.e['H2'].col.val > 19):
                    names.append(name)
                    t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                    p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                    nH.append(c.e['PDRnH'].col.val)
                    uv.append(c.e['PDRuv'].col.val)
                    me.append(q.e['Me'].col.val)
                    t01.append(t.log().val)
                    NH2.append(c.e['H2'].col.val)
                    xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                  q.e['HI'].col.plus,
                                                                                                  q.e['HI'].col.minus)
                    NH.append(xNH.val)
                    x = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l') * a(2, 0, 0,
                                                                                                        'l') / a(
                        c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l')

                    ax[1].errorbar(x=xNH.val, y=p.val,
                                   yerr=[[p.minus], [p.plus]],
                                   xerr=[[xNH.minus], [xNH.plus]],
                                   fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=4)
                    ax[1].text(xNH.val, p.val,q.name)
    ax[1].plot(np.array(NH), np.array(t01) + nH, 'D', markersize=7, markerfacecolor='green', markeredgecolor='black', label = 'MW')

    database = 'Magellan'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    names = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if (c.e['H2'].col.val > 19):
                    names.append(name)
                    t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                    p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                    nH.append(c.e['PDRnH'].col.val)
                    uv.append(c.e['PDRuv'].col.val)
                    me.append(q.e['Me'].col.val)
                    t01.append(t.log().val)
                    NH2.append(c.e['H2'].col.val)
                    xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                  q.e['HI'].col.plus,
                                                                                                  q.e['HI'].col.minus)
                    NH.append(xNH.val)
                    x = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l') * a(2, 0, 0,
                                                                                                        'l') / a(
                        c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l')

                    ax[1].errorbar(x=xNH.val, y=p.val,
                                   yerr=[[p.minus], [p.plus]],
                                   xerr=[[xNH.minus], [xNH.plus]],
                                   fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=4)
    ax[1].plot(np.array(NH), np.array(t01) + nH, 'v', markersize=8, markerfacecolor='purple', markeredgecolor='black', label='MC')

    if 1:
        x =np.linspace(20,22.5,10)
        ax[1].plot(x,-7.26+0.51*(x),'--',color='red')

    ax[0].set_xlim(18.5, 23)
    ax[0].set_ylim(2, 6)
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
    ax[0].xaxis.set_major_locator(MultipleLocator(1))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[0].yaxis.set_major_locator(MultipleLocator(1))
    ax[0].set_xlabel('$\log N({\\rm{H}}_{\\rm{tot}})$, cm$^{-2}$', fontsize=labelsize)
    #ax[0].set_ylabel('$\log n_{\\rm{H}} T_{\\rm{01}}$, K\,cm$^{-3}$', fontsize=labelsize)
    ax[0].set_ylabel('$\log P/k$, K\,cm$^{-3}$', fontsize=labelsize)
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[1].yaxis.set_major_locator(MultipleLocator(1))
    ax[1].set_xlim(18.5, 23)
    ax[1].set_ylim(2, 6)
    ax[1].set_xlabel('$\log N({\\rm{H}}_{\\rm{tot}})$, cm$^{-2}$', fontsize=labelsize)
    #ax[1].set_ylabel('$\log n_{\\rm{H}} T_{\\rm{01}}$, K\,cm$^{-3}$', fontsize=labelsize)
    ax[1].set_ylabel('$\log P/k$, K\,cm$^{-3}$', fontsize=labelsize)
    #ax[1].legend(bbox_to_anchor=(0.4, 0.7, 0.25, 0.2), frameon=False, fontsize=labelsize)
    ax[1].legend(bbox_to_anchor=(0.4, 0.71, 0.25, 0.2), frameon=False, fontsize=labelsize)

    for col in ax[:]:
        col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        col.tick_params(which='major', length=5)
        col.tick_params(which='minor', length=4)

if case == 'fig16':
    fig16, ax = plt.subplots(figsize=(4.5, 5))
    database = 'H2UV'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if name in nonthermal_sys:
                    print('remove sys:', name)
                else:
                    if (c.e['H2'].col.val > 19):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                        nH.append(c.e['PDRnH'].col.val)
                        uv.append(c.e['PDRuv'].col.val)
                        me.append(q.e['Me'].col.val)
                        t01.append(t.log().val)
                        NH2.append(c.e['H2'].col.val)
                        xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                      q.e['HI'].col.plus,
                                                                                                      q.e['HI'].col.minus)
                        NH.append(xNH.val)
                        y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')  / a(
                            q.e['Me'].col.val, q.e['Me'].col.plus, q.e['Me'].col.minus, 'l')

                        if 1:
                            ax.errorbar(y=c.e['PDRnH'].col.val, x=y.val,
                                    xerr=[[y.minus], [y.plus]],
                                    yerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                    fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

    c3 = ax.scatter( np.array(uv)-me, np.array(nH), 50, NH2, cmap='Reds', edgecolors='black', vmin=18, vmax=21,label='DLAs')
    #ax.scatter(np.array(nH), np.array(uv), 50, NH2, cmap='Greens', edgecolors='black', vmin=18, vmax=21)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylim(-0.1, 3.2)
    ax.set_xlim(-1.2, 4.2)
    ax.set_xlabel('$\log I_{\\rm{UV}}/Z$', fontsize=labelsize)
    ax.set_ylabel('$\log n_{\\rm{H}}$, cm$^{-3}$',
                  fontsize=labelsize)
    if 1:
        cax = fig16.add_axes([0.18, 0.64, 0.01, 0.2])
        fig16.colorbar(c3, cax=cax, ticks=[18, 19, 20,21])
        cax.tick_params(labelsize=labelsize)
        cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        cax.tick_params(which='minor', length=3, direction='in')
        cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        ax.text(0.1, 2.1, '$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)

    ax.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=4)


    if 1:
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                print(name)
                if 'PDRnH' in c.e:
                    if name in nonthermal_sys:
                        print('remove sys:', name)
                    else:
                        if (c.e['H2'].col.val > 18):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.log().val)
                            NH2.append(c.e['H2'].col.val)
                            xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(
                                q.e['HI'].col.val,
                                q.e['HI'].col.plus,
                                q.e['HI'].col.minus)
                            NH.append(xNH.val)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l') / a(
                                q.e['Me'].col.val, q.e['Me'].col.plus, q.e['Me'].col.minus, 'l')

                            if 1:
                                ax.errorbar(y=c.e['PDRnH'].col.val, x=y.val,
                                            xerr=[[y.minus], [y.plus]],
                                            yerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                            fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

        c2 = ax.scatter( np.array(uv) - me,np.array(nH), 40, NH2, cmap='Greens',marker='D', edgecolors='black', vmin=18, vmax=21,label='MW')

        if 1:
            cax = fig16.add_axes([0.24, 0.64, 0.01, 0.2])
            fig16.colorbar(c2, cax=cax, ticks=[18, 19, 20, 21])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize,direction='in')

    if 0:
        database = 'Magellan'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                print(name)
                if 'PDRnH' in c.e:
                    if name in nonthermal_sys:
                        print('remove sys:', name)
                    else:
                        if (c.e['H2'].col.val > 18):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.log().val)
                            NH2.append(c.e['H2'].col.val)
                            xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(
                                q.e['HI'].col.val,
                                q.e['HI'].col.plus,
                                q.e['HI'].col.minus)
                            NH.append(xNH.val)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l') / a(
                                q.e['Me'].col.val, q.e['Me'].col.plus, q.e['Me'].col.minus, 'l')

                            if 1:
                                ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                            yerr=[[y.minus], [y.plus]],
                                            xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                            fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

        c = ax.scatter(np.array(nH), np.array(uv) - me, 50, NH2, cmap='Purples', marker='v' ,edgecolors='black', vmin=18, vmax=21,label='MC')
        if 1:
            cax = fig16.add_axes([0.3, 0.64, 0.01, 0.2])
            fig16.colorbar(c, cax=cax, ticks=[18, 19, 20, 21]) #,boundaries=[18,21])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize,direction='in')

        #ax.plot(100, 100, 'o', color='red', label='DLAs',markeredgecolor='black',markersize=6)
        #ax.plot(100, 100, 'o', color='green', label='MW',markeredgecolor='black',markersize=6,marker='D')
        #ax.plot(100, 100, 'o', color='purple', label='MC',markeredgecolor='black',markersize=6,marker='v')

        ax.legend(bbox_to_anchor=(0.13, 0.4, 0.17, 0.2), frameon=True, fontsize=labelsize)
        legend = ax.get_legend()
        legend.legendHandles[0].set_color('red')
        legend.legendHandles[1].set_color('green')
        legend.legendHandles[2].set_color('purple')

if case == 'fig17':
    fig17, ax = plt.subplots(figsize=(4.5, 4.5))
    database = 'H2UV'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if name in nonthermal_sys:
                    print('remove sys:', name)
                else:
                    if (c.e['H2'].col.val > 19):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                        nH.append(c.e['PDRnH'].col.val)
                        uv.append(c.e['PDRuv'].col.val)
                        me.append(q.e['Me'].col.val)
                        t01.append(t.log().val)
                        NH2.append(c.e['H2'].col.val)
                        xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                      q.e[
                                                                                                          'HI'].col.plus,
                                                                                                      q.e[
                                                                                                          'HI'].col.minus)
                        NH.append(xNH.val)
                        y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')

                        if 1:
                            ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                        yerr=[[y.minus], [y.plus]],
                                        xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                        fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

    c3 = ax.scatter(np.array(nH), np.array(uv) , 50, me, cmap='Reds', edgecolors='black', vmin=-2, vmax=1,
                    label='DLAs')
    # ax.scatter(np.array(nH), np.array(uv), 50, NH2, cmap='Greens', edgecolors='black', vmin=18, vmax=21)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(-0.1, 3.6)
    ax.set_ylim(-1.5, 3.5)
    ax.set_ylabel('$\log I_{\\rm{UV}}$', fontsize=labelsize)
    ax.set_xlabel('$\log n_{\\rm{H}}$, cm$^{-3}$',
                  fontsize=labelsize)
    if 1:
        cax = fig17.add_axes([0.18, 0.69, 0.01, 0.15])
        fig17.colorbar(c3, cax=cax, ticks=[-2,-1,0,1])
        cax.tick_params(labelsize=labelsize)
        cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        cax.tick_params(which='minor', length=3, direction='in')
        cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        ax.text(0.2, 1.9, '$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)

    ax.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=4)

    if 1:
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                print(name)
                if 'PDRnH' in c.e:
                    if name in nonthermal_sys:
                        print('remove sys:', name)
                    else:
                        if (c.e['H2'].col.val > 18):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.log().val)
                            NH2.append(c.e['H2'].col.val)
                            xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(
                                q.e['HI'].col.val,
                                q.e['HI'].col.plus,
                                q.e['HI'].col.minus)
                            NH.append(xNH.val)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')

                            if 1:
                                ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                            yerr=[[y.minus], [y.plus]],
                                            xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                            fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

        c2 = ax.scatter(np.array(nH), np.array(uv), 40, me, cmap='Greens', marker='D', edgecolors='black',
                        vmin=-2, vmax=1, label='MW')

        if 0:
            cax = fig17.add_axes([0.24, 0.69, 0.01, 0.15])
            fig17.colorbar(c2, cax=cax, ticks=[18, 19, 20, 21])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')

    if 1:
        database = 'Magellan'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                print(name)
                if 'PDRnH' in c.e:
                    if name in nonthermal_sys:
                        print('remove sys:', name)
                    else:
                        if (c.e['H2'].col.val > 18):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.log().val)
                            NH2.append(c.e['H2'].col.val)
                            xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(
                                q.e['HI'].col.val,
                                q.e['HI'].col.plus,
                                q.e['HI'].col.minus)
                            NH.append(xNH.val)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')

                            if 1:
                                ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                            yerr=[[y.minus], [y.plus]],
                                            xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                            fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

        c = ax.scatter(np.array(nH), np.array(uv) , 50, me, cmap='Purples', marker='v', edgecolors='black',
                       vmin=-2, vmax=1, label='MC')
        if 0:
            cax = fig17.add_axes([0.3, 0.69, 0.01, 0.15])
            fig17.colorbar(c, cax=cax, ticks=[18, 19, 20, 21])  # ,boundaries=[18,21])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')

        # ax.plot(100, 100, 'o', color='red', label='DLAs',markeredgecolor='black',markersize=6)
        # ax.plot(100, 100, 'o', color='green', label='MW',markeredgecolor='black',markersize=6,marker='D')
        # ax.plot(100, 100, 'o', color='purple', label='MC',markeredgecolor='black',markersize=6,marker='v')

    ax.legend(bbox_to_anchor=(0.8, 0.8, 0.17, 0.2), frameon=True, fontsize=labelsize)
    legend = ax.get_legend()
    legend.legendHandles[0].set_color('red')
    legend.legendHandles[1].set_color('green')
    legend.legendHandles[2].set_color('purple')

if case == 'fig18':
    fig18, axs = plt.subplots(1,2,figsize=(9, 4))
    if 1:
        ax = axs[0]
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                print(name)
                if 'PDRnH' in c.e:
                    if name in nonthermal_sys:
                        print('remove sys:', name)
                    else:
                        if (c.e['H2'].col.val > 18):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.log().val)
                            NH2.append(c.e['H2'].col.val)
                            xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                          q.e[
                                                                                                              'HI'].col.plus,
                                                                                                          q.e[
                                                                                                              'HI'].col.minus)
                            NH.append(xNH.val)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')

                            if 1:
                                ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                            yerr=[[y.minus], [y.plus]],
                                            xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                            fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

        c3 = ax.scatter(np.array(nH), np.array(uv) , 50, me, cmap='Reds', edgecolors='black', vmin=-1.5, vmax=0,
                        label='DLAs')
        # ax.scatter(np.array(nH), np.array(uv), 50, NH2, cmap='Greens', edgecolors='black', vmin=18, vmax=21)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlim(-0.1, 3.6)
        ax.set_ylim(-1.5, 4)
        ax.set_ylabel('$\log I_{\\rm{UV}}$', fontsize=labelsize)
        ax.set_xlabel('$\log n_{\\rm{H}}$, cm$^{-3}$',
                      fontsize=labelsize)
        if 1:
            cax = fig18.add_axes([0.15, 0.69, 0.01, 0.15])
            fig18.colorbar(c3, cax=cax, ticks=[-1.5, -1, -0.5, 0])
            cax.tick_params(labelsize=labelsize-2)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
            ax.text(0.2, 2.3, '$\log\\,{\\rm{Z}}$', fontsize=labelsize)

        ax.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
        ax.tick_params(which='major', length=5)
        ax.tick_params(which='minor', length=4)

        if 1:
            database = 'Galaxy'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            NH = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    print(name)
                    if 'PDRnH' in c.e:
                        if name in nonthermal_sys:
                            print('remove sys:', name)
                        else:
                            if (c.e['H2'].col.val > 18):
                                t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                                p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                me.append(q.e['Me'].col.val)
                                t01.append(t.log().val)
                                NH2.append(c.e['H2'].col.val)
                                xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(
                                    q.e['HI'].col.val,
                                    q.e['HI'].col.plus,
                                    q.e['HI'].col.minus)
                                NH.append(xNH.val)
                                y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')

                                if 1:
                                    ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                                yerr=[[y.minus], [y.plus]],
                                                xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                                fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

            c2 = ax.scatter(np.array(nH), np.array(uv), s=40, c='green', marker='D', edgecolors='black', label='MW')

            if 0:
                cax = fig18.add_axes([0.18, 0.69, 0.01, 0.15])
                fig18.colorbar(c2, cax=cax, ticks=[-1.5, -1, -0.5, 0])
                cax.tick_params(labelsize=labelsize-2)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        if 1:
            database = 'Magellan'
            H2 = H2_exc(H2database=database)
            nH = []
            uv = []
            me = []
            t01 = []
            NH2 = []
            NH = []
            for q in H2.H2.values():
                for i_c, c in enumerate(q.comp):
                    name = "".join([q.name, '_', str(i_c)])
                    print(name)
                    if 'PDRnH' in c.e:
                        if name in nonthermal_sys:
                            print('remove sys:', name)
                        else:
                            if (c.e['H2'].col.val > 18):
                                t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                                p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                                nH.append(c.e['PDRnH'].col.val)
                                uv.append(c.e['PDRuv'].col.val)
                                me.append(q.e['Me'].col.val)
                                t01.append(t.log().val)
                                NH2.append(c.e['H2'].col.val)
                                xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(
                                    q.e['HI'].col.val,
                                    q.e['HI'].col.plus,
                                    q.e['HI'].col.minus)
                                NH.append(xNH.val)
                                y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')

                                if 1:
                                    ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                                yerr=[[y.minus], [y.plus]],
                                                xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                                fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

            c = ax.scatter(np.array(nH), np.array(uv) , s=50, c='purple', marker='v', edgecolors='black',
                           vmin=-1.5, vmax=0, label='MC')
            if 0:
                cax = fig18.add_axes([0.23, 0.69, 0.01, 0.15])
                fig18.colorbar(c, cax=cax, ticks=[-1.5, -1, -0.5, 0])
                cax.tick_params(labelsize=labelsize-2)
                cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                cax.tick_params(which='minor', length=3, direction='in')
                cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')

            # ax.plot(100, 100, 'o', color='red', label='DLAs',markeredgecolor='black',markersize=6)
            # ax.plot(100, 100, 'o', color='green', label='MW',markeredgecolor='black',markersize=6,marker='D')
            # ax.plot(100, 100, 'o', color='purple', label='MC',markeredgecolor='black',markersize=6,marker='v')

        ax.legend(bbox_to_anchor=(0.8, 0.8, 0.17, 0.2), frameon=True, fontsize=labelsize)
        legend = ax.get_legend()
        legend.legendHandles[0].set_color('red')
        legend.legendHandles[1].set_color('green')
        legend.legendHandles[2].set_color('purple')
##################################################################################################
    ##################################################################################################
    ##################################################################################################
    ax = axs[1]
    database = 'H2UV'
    H2 = H2_exc(H2database=database)
    nH = []
    uv = []
    me = []
    t01 = []
    NH2 = []
    NH = []
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            print(name)
            if 'PDRnH' in c.e:
                if name in nonthermal_sys:
                    print('remove sys:', name)
                else:
                    if (c.e['H2'].col.val > 18):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                        nH.append(c.e['PDRnH'].col.val)
                        uv.append(c.e['PDRuv'].col.val)
                        me.append(q.e['Me'].col.val)
                        t01.append(t.log().val)
                        NH2.append(c.e['H2'].col.val)
                        xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                      q.e[
                                                                                                          'HI'].col.plus,
                                                                                                      q.e[
                                                                                                          'HI'].col.minus)
                        NH.append(xNH.val)
                        y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')  / a(
                            q.e['Me'].col.val, q.e['Me'].col.plus, q.e['Me'].col.minus, 'l')
                        if 1:
                            ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                        yerr=[[y.minus], [y.plus]],
                                        xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                        fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)
                        #ax.text(c.e['PDRnH'].col.val, y.val,''.join(['$',name,'$']))

    c3 = ax.scatter(np.array(nH), np.array(uv)-me, 50, NH2, cmap='Reds', edgecolors='black', vmin=18, vmax=21,
                    label='DLAs')
    # ax.scatter(np.array(nH), np.array(uv), 50, NH2, cmap='Greens', edgecolors='black', vmin=18, vmax=21)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(-0.1, 3.6)
    ax.set_ylim(-1.5, 4)
    ax.set_ylabel('$\log I_{\\rm{UV}}/Z$', fontsize=labelsize)
    ax.set_xlabel('$\log n_{\\rm{H}}$, cm$^{-3}$',
                  fontsize=labelsize)
    if 1:
        cax = fig18.add_axes([0.58, 0.69, 0.01, 0.15])
        fig18.colorbar(c3, cax=cax, ticks=[18,19,20,21])
        cax.tick_params(labelsize=labelsize)
        cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        cax.tick_params(which='minor', length=3, direction='in')
        cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        ax.text(0.2, 2.3, '$\log\\,N({\\rm{H_2}})$', fontsize=labelsize)

    ax.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=4)

    if 1:
        database = 'Galaxy'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                print(name)
                if 'PDRnH' in c.e:
                    if name in nonthermal_sys:
                        print('remove sys:', name)
                    else:
                        if (c.e['H2'].col.val > 18):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.log().val)
                            NH2.append(c.e['H2'].col.val)
                            xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(
                                q.e['HI'].col.val,
                                q.e['HI'].col.plus,
                                q.e['HI'].col.minus)
                            NH.append(xNH.val)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')  / a(
                            q.e['Me'].col.val, q.e['Me'].col.plus, q.e['Me'].col.minus, 'l')

                            if 1:
                                ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                            yerr=[[y.minus], [y.plus]],
                                            xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                            fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

        c2 = ax.scatter(np.array(nH), np.array(uv)-me, 40, NH2, cmap='Greens', marker='D', edgecolors='black',
                        vmin=18, vmax=21, label='MW')

        if 1:
            cax = fig18.add_axes([0.58+0.04, 0.69, 0.01, 0.15])
            fig18.colorbar(c2, cax=cax, ticks=[18, 19, 20, 21])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')

    if 1:
        database = 'Magellan'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                print(name)
                if 'PDRnH' in c.e:
                    if name in nonthermal_sys:
                        print('remove sys:', name)
                    else:
                        if (c.e['H2'].col.val > 18):
                            t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                            p = a(c.e['PDRnH'].col.val, c.e['PDRnH'].col.plus, c.e['PDRnH'].col.minus, 'l') * t
                            nH.append(c.e['PDRnH'].col.val)
                            uv.append(c.e['PDRuv'].col.val)
                            me.append(q.e['Me'].col.val)
                            t01.append(t.log().val)
                            NH2.append(c.e['H2'].col.val)
                            xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(
                                q.e['HI'].col.val,
                                q.e['HI'].col.plus,
                                q.e['HI'].col.minus)
                            NH.append(xNH.val)
                            y = a(c.e['PDRuv'].col.val, c.e['PDRuv'].col.plus, c.e['PDRuv'].col.minus, 'l')  / a(
                            q.e['Me'].col.val, q.e['Me'].col.plus, q.e['Me'].col.minus, 'l')

                            if 1:
                                ax.errorbar(x=c.e['PDRnH'].col.val, y=y.val,
                                            yerr=[[y.minus], [y.plus]],
                                            xerr=[[c.e['PDRnH'].col.minus], [c.e['PDRnH'].col.plus]],
                                            fmt='^', color='black', zorder=-10, capsize=1, ecolor='black', markersize=3)

        c = ax.scatter(np.array(nH), np.array(uv)-me, 50, NH2, cmap='Purples', marker='v', edgecolors='black',
                       vmin=18, vmax=21, label='MC')
        if 1:
            cax = fig18.add_axes([0.58+0.08, 0.69, 0.01, 0.15])
            fig18.colorbar(c, cax=cax, ticks=[18, 19, 20, 21])  # ,boundaries=[18,21])
            cax.tick_params(labelsize=labelsize)
            cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
            cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
            cax.tick_params(which='minor', length=3, direction='in')
            cax.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')

        # ax.plot(100, 100, 'o', color='red', label='DLAs',markeredgecolor='black',markersize=6)
        # ax.plot(100, 100, 'o', color='green', label='MW',markeredgecolor='black',markersize=6,marker='D')
        # ax.plot(100, 100, 'o', color='purple', label='MC',markeredgecolor='black',markersize=6,marker='v')
    if 1:
        x = np.linspace(0.5,2.7,10)
        ax.plot(x,x-0.8,'--',color='black')
        x = np.linspace(1,3,10)
        ax.plot(x, x - 1.8, '--', color='black')
        #x = np.linspace(1, 3.2, 10)
        #ax.plot(x, x - 0.8 - 0.8-0.8, '--', color='green')

    if 1:
        ax.text(1.7,3.5,'$J0843+0221_{0,1}$')
        ax.text(2.45, 2.9, '$J2140-0321_0$')
        ax.text(1.4, 1.6, '$J1513+0352_0$',rotation=30)
    if 0:
        ax.legend(bbox_to_anchor=(0.11, 0.4, 0.1, 0.2), frameon=True, fontsize=labelsize)
        legend = ax.get_legend()
        legend.legendHandles[0].set_color('red')
        legend.legendHandles[1].set_color('green')
        legend.legendHandles[2].set_color('purple')

if case == 'fig19':
    fig19,ax = plt.subplots(figsize=(9,4))
    if 0:
        database = 'H2UV'
        H2 = H2_exc(H2database=database)
        nH = []
        uv = []
        me = []
        t01 = []
        NH2 = []
        NH = []
        tcmb = []
        z_dla = []
        press = []
        color= 'green'
        lalbelsize=14
        msize = 7
        label = 'CI DLAs'
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if 'CiTcmb' in c.e:
                    if (c.e['H2'].col.val > 17):
                        t = a(c.e['T01'].col.val, c.e['T01'].col.plus, c.e['T01'].col.minus, 'd').log()
                        p = a(c.e['CinH'].col.val, c.e['CinH'].col.plus, c.e['CinH'].col.minus, 'l') * t
                        nH.append(c.e['CinH'].col.val)
                        uv.append(c.e['Ciuv'].col.val)
                        me.append(q.e['Me'].col.val)
                        t01.append(t.log().val)
                        NH2.append(c.e['H2'].col.val)
                        xNH = a(c.e['H2'].col.val + 0.3, c.e['H2'].col.plus, c.e['H2'].col.minus) + a(q.e['HI'].col.val,
                                                                                                      q.e['HI'].col.plus,
                                                                                                      q.e['HI'].col.minus)
                        z_dla.append(q.z_dla)
                        tcmb.append(c.e['CiTcmb'].col.val)
                        NH.append(xNH.val)
                        press.append(p.val)
                        y = a(c.e['CiTcmb'].col.val, c.e['CiTcmb'].col.plus, c.e['CiTcmb'].col.minus, 'l')
                        print(name, tcmb[-1],nH[-1])
                        if c.e['CiTcmb'].col.type == 'm':
                            ax.errorbar(x=q.z_dla, y=y.val,
                                        yerr=[[y.minus], [y.plus]],
                                        fmt='o',  ecolor='black',
                                        markerfacecolor=color, markeredgecolor='black',
                                        markeredgewidth=2, markersize=msize, capsize=2,alpha=0.9,label=label)
                            label = "_nolegend_"
                        elif c.e['CiTcmb'].col.type == 'u':
                            ax.errorbar(x=q.z_dla, y=y.val,
                                        yerr=1,
                                        fmt='o', markerfacecolor=color,  markeredgecolor='black', markeredgewidth=1,
                                        markersize=msize, capsize=2, ecolor='black',uplims = True)
                        #ax.text(q.z_dla, y.val + 0.3, ''.join(['$',name,'$']))

        #ax.scatter(np.array(z_dla), np.array(tcmb), 50, edgecolors='black', label='DLAs')
        # ax.scatter(np.array(nH), np.array(uv), 50, NH2, cmap='Greens', edgecolors='black', vmin=18, vmax=21)

    #add CO measurements
    if 1:
        database = 'all'
        H2 = H2_exc(H2database=database)
        label = 'CO DLAs'
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if 'T03_co' in c.e:
                    if 0:
                        color='red'
                        ax.errorbar(x=q.z_dla, y=c.e['T03_co'].col.val,
                                yerr=[[c.e['T03_co'].col.minus], [c.e['T03_co'].col.plus]],
                                fmt='o',  ecolor=color,
                                markerfacecolor=color, markeredgecolor='black',
                                markeredgewidth=0.5, markersize=msize, capsize=1,alpha=0.6,label=label)
                    if 1:
                        color = 'blue'
                        ax.errorbar(x=q.z_dla, y=c.e['Tcmbcorr'].col.val,
                                yerr=[[c.e['Tcmbcorr'].col.minus], [c.e['Tcmbcorr'].col.plus]],
                                fmt='v', ecolor=color,
                                markerfacecolor=color, markeredgecolor='black',
                                markeredgewidth=0.5, markersize=msize, capsize=1, alpha=0.6, label=label)
                    label = "_nolegend_"


    # add SZ estimate for clusters
    if 0:
        #Luzzi2009ApJ705
        z = np.array([0.023,0.152,0.183,0.200,0.202,
                     0.216, 0.232, 0.252, 0.282, 0.291,
                     0.451, 0.546, 0.550])
        tcmb = np.array([2.72,2.90,2.95, 2.4, 3.36,
                        3.85, 3.51, 3.39, 3.22, 4.05,
                        3.97, 3.69, 4.59])
        tcmberr = np.array([0.10,0.17,0.27,0.28,0.20,
                           0.64, 0.25, 0.26, 0.26, 0.66,
                           0.19, 0.37, 0.36])
        ax.errorbar(x=z, y = tcmb, yerr=tcmberr,fmt='o', markerfacecolor='blue',
                                        markeredgecolor='black',
                                        zorder=-10, capsize=1, ecolor='black', markersize=msize,alpha=0.9,
                                        label='SZ')
    # add CO
    if 1:
        #from Noterdeeme 2011
        z = np.array([1.7293, 1.7738, 2.6896, 2.4184, 2.0377])
        tcmb = np.array([7.5, 7.8, 10.5, 9.15, 8.6])
        tcmberr = np.array([1.6,0.7,0.8,0.7,1.0])
        ax.errorbar(x=z, y=tcmb, yerr=tcmberr, fmt='o', markerfacecolor='red',
                    markeredgecolor='black',
                    zorder=-10, capsize=1, ecolor='black', markersize=msize, alpha=0.5,
                    label='CO DLAs',markeredgewidth=0.5)



    ax.plot(np.linspace(0, 5, 5), 2.73 * (1 + np.linspace(0, 5, 5)), '--', color='black',label='$T_{\\rm{CMB}}^0(1+z)$')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim(-0.1, 3.7)
    ax.set_ylim(0, 18)
    ax.set_ylabel('$T_{\\rm{CMB}}$, K', fontsize=labelsize)
    ax.set_xlabel('$z$', fontsize=labelsize)
    ax.legend(bbox_to_anchor=(0.12, 0.78, 0.1, 0.2), frameon=False, fontsize=labelsize)
    ax.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True', top='True')
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=4)


save = 0
if save:

    figname='Figs/poster.pdf'
    if case == 'fig5':
        figname = "lnL_z-1.pdf"
    if case == 'fig6':
        figname = "fit_H2_z-1.pdf"
    if case == 'fig7':
        figname = "fit_CI_z-1.pdf"
    fig02.savefig("".join((figname)), bbox_inches='tight')

plt.show()