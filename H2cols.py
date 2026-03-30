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
from spectro.atomic import e

#names = ['temp/H2j0','temp/H2j1','temp/H2j2','temp/H2j3','temp/H2j4']
#for el in names:
if 0:
    e= 'H2j2_col'
    str = ('temp/',e)
    str = "".join(str)
    str_nodes = ('temp/', e,'_nodes')
    str_nodes = "".join(str_nodes)
    with open(str, 'rb') as f:
        XI, YI, ZI = pickle.load(f)
    with open(str_nodes, 'rb') as f:
        x, y, z = pickle.load(f)
    e= 'H2j0_col'
    str = ('temp/',e)
    str = "".join(str)
    str_nodes = ('temp/', e,'_nodes')
    str_nodes = "".join(str_nodes)
    with open(str, 'rb') as f:
        XIa, YIa, ZIa = pickle.load(f)
    with open(str_nodes, 'rb') as f:
        xa, ya, za = pickle.load(f)
    # n = plt.normalize(10., 20.)
    Z1 = ZI #10**(ZI - ZIa)
    z1= z #10**(z - za)

    if 0:
        XI, YI = np.meshgrid(x, y)
        plt.plot(XI, YI,'o',color='grey')
        plt.xlabel('$\log (n)$ [cm$^{-3}]$', fontsize=18)
        plt.ylabel('$\log (I_{UV})$ [Draine field]', fontsize=18)

    if 1:
        XI, YI = np.meshgrid(XI, YI)
        plt.subplot(1, 1, 1)
        #plt.pcolor(XI, YI, Z1, cmap=cm.jet, vmin=12, vmax=20)
        plt.scatter(x, y, 100, z1, cmap=cm.jet, vmin=7, vmax=7,edgecolors='black')
        #plt.title(''.join(('RBF interpolation of',e)))
        #plt.title(e)
        #plt.xlim(-1.2, 3.2)
        #plt.ylim(0.8, 5.2)
        #plt.colorbar()
        plt.xlabel('$\log (n)$ [cm$^{-3}]$', fontsize=18)
        plt.ylabel('$\log (I_{UV})$ [Draine field]', fontsize=18)
        plt.text(-0.9,4.5,('H2 J1/J0'),color='white',fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=13)
        #plt.figure(figsize=(15, 10))
        # plt.savefig('rbf2d.png')
    plt.show()
    #plt.savefig('rbf2d.png')

if 0:
    for k,e in enumerate(['H2j0','H2j1','H2j2','H2j3']):
        str = ('temp/',e)
        str = "".join(str)
        str_nodes = ('temp/', e,'_nodes')
        str_nodes = "".join(str_nodes)
        with open(str, 'rb') as f:
            XI, YI, ZI = pickle.load(f)
        with open(str_nodes, 'rb') as f:
            x, y, z = pickle.load(f)
        # n = plt.normalize(10., 20.)
        if 1:
            XI, YI = np.meshgrid(XI, YI)
            plt.subplot(2, 2, k+1)
            plt.pcolor(XI, YI, ZI, cmap=cm.jet, vmin=12, vmax=20)
            plt.scatter(x, y, 100, z, cmap=cm.jet, vmin=12, vmax=20,edgecolors='black')
            #plt.title(''.join(('RBF interpolation of',e)))
            #plt.title(e)
            plt.xlim(-1.2, 3.2)
            plt.ylim(0.8, 5.2)
            plt.colorbar()
            plt.ylabel('$\log (n)$ [cm$^{-3}]$', fontsize=18)
            plt.xlabel('$\log (I_{UV})$ [Draine field]', fontsize=18)
            plt.text(-0.9,4.5,(''.join(('$\log N$',e))),color='black',fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=13)
            #plt.figure(figsize=(15, 10))
            # plt.savefig('rbf2d.png')
    plt.show()

if 0:
    with open('temp/H2j5', 'rb') as f:
        XI,YI,ZI = pickle.load(f)
    with open('temp/H2j5_nodes', 'rb') as f:
        x,y,z = pickle.load(f)
    #n = plt.normalize(10., 20.)

    if 1:
        XI, YI = np.meshgrid(XI,YI)
        plt.subplot(1, 1, 1)
        plt.pcolor(XI,YI, ZI, cmap=cm.jet, vmin=12, vmax=20)
        plt.scatter(x,y, 100, z, cmap=cm.jet, vmin=12, vmax=20) #,edgecolors='black')
        plt.title('RBF interpolation - testing')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()
        #plt.savefig('rbf2d.png')
        plt.show()
    if 0:
        xi = np.linspace(-1,3,90)
        yi = np.linspace(1,5,90)
        X1, Y1 = np.meshgrid(xi,yi)

        if 1:
            # use rBF
            #rbf = Rbf(x,y,z,function='gaussian',epsilon=2)
            rbf = Rbf(x,y,z,function='multiquadric',smooth=0.1)
            #rbf = Rbf(x, y, z, function='inverse', epsilon=0.14)
            Z1 = rbf(X1,Y1)
            print(rbf(1.0,1.0))
        if 0:
            interp_spline = interp2d(x, y, z, kind='cubic')
            Z1 = interp_spline(xi,yi)

        #plot
        plt.subplot(1, 1, 1)
        plt.pcolor(X1, Y1, Z1, cmap=cm.jet, vmin=13, vmax=20)
        plt.scatter(x, y, 100, z, cmap=cm.jet, vmin=13, vmax=20) #,edgecolors='black')
        plt.title('RBF interpolation - test')
        plt.xlim(-1, 3)
        plt.ylim(1, 5)
        plt.colorbar()
        print('stop')
        #plt.savefig('rbf2d.png')
        plt.show()

if 0:
    name = 'B0405-4418_0'
    str = ('result/all/', name,'.pkl')
    str = "".join(str)
    strnodes = ('result/all/nodes/', name,'.pkl')
    strnodes = "".join(strnodes)
    with open(strnodes, 'rb') as f:
        x1,y1,z1 = pickle.load(f)
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
        #d.plot(color=None)
        ax = plt.subplot(1, 2, 2)
        d.plot_contour(ax=ax, color='red', xlabel='$\log n$ [cm$^{-3}$]', ylabel='$\log I_{UV}$ [Draine field]',cmap=None)

        plt.show()

if 0:
    name = 'J0812+3208_0'
    str = ('result/all/', name,'.pkl')
    str = "".join(str)
    strnodes = ('result/all/lnL_nodes/', name,'.pkl')
    strnodes = "".join(strnodes)
    with open(strnodes, 'rb') as f:
        x1,y1,z1 = pickle.load(f)
    with open(str, 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        d = distr2d(x=x, y=y, z=np.exp(z))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dx.stats(latex=2, name='log UV')
        dy.stats(latex=2, name='log n')
        ax = plt.subplot(1, 1, 1)
        d.plot( color='black', xlabel='$\log (I_{UV})$ [Draine field]', ylabel='$\log (n)$ [cm$^{-3}]$',color_point='white',font=22,cmap='Greys',color_marg='black')
        #d.plot_contour(ax=ax, conf_levels=[0.95], color='magenta', xlabel='$\log (I_{UV})$ [Draine field]', ylabel='$\log (n)$ [cm$^{-3}]$',cmap=None, color_point=None)
        plt.show()


# plot the grid of chi2 distributions and exc_diagrams for systems in H2 list
if 1:

    database = 'H2UV'
    file_path = 'result/cr_fixed/'
    if 0:
        H2 = H2_exc(H2database=database)

        k = 0
        for q in H2.H2.values():
            Me = q.e['Me'].col.val
            if Me < -0.75 and Me > -1.25:
                database = 'z=-1'
            elif Me < -0.25 and Me > -0.75:
                database = 'z=-0.5'
            elif Me < 0.25 and Me > -0.25:
                database = 'z=0'
            elif Me < 0.75 and Me > 0.25:
                database = 'z=0.5'

            file_path_J5 = "".join([file_path, 'J05/', database, '/'])
            file_path_J2 = "".join([file_path, 'J02/', database, '/'])
            file_path_CI = "".join([file_path, 'CI/', database, '/'])

            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                object = "".join([q.name, '_', str(i_c), '_lnL.pkl'])
                bf_name = "".join([q.name, '_', str(i_c), '_H2.pkl'])
                ci_name = "".join([q.name, '_', str(i_c), '_CI_lnL.pkl'])
                k += 1

                with open("".join([file_path_J5, object]), 'rb') as f:
                    x, y, z_H2 = pickle.load(f)


                prior = 0
                if prior == 1 or name in ['B0405-4418_0','J0643-5041_0','B1444+0126_0','B1444+0126_1','0551-3638_0','J1443+2724_0','J1443+2724_1']:
                    z_prior = np.zeros([len(y), len(x)])
                    x_mean = 2.0
                    x_s = 0.5
                    for i_x, xi in enumerate(x):
                        for k_y, yk in enumerate(y):
                            z_prior[k_y, i_x] = -(xi - x_mean) ** 2 / x_s ** 2.
                else:
                    with open("".join([file_path_CI, ci_name]), 'rb') as f:
                        x, y, z_CI = pickle.load(f)
                    z_prior = z_CI


                d = distr2d(x=x, y=y, z=np.exp(z_H2 + z_prior))
                dx, dy = d.marginalize('y'), d.marginalize('x')
                dy.stats(latex=2, name='log UV')
                dx.stats(latex=2, name='log n')

                H2.H2[q.name].comp[i_c].physcond['n0_J5'] = e('n0', dx.point, dx.interval[1] - dx.point,
                                                           dx.point - dx.interval[0])
                H2.H2[q.name].comp[i_c].physcond['uv_J5'] = e('uv', dy.point, dy.interval[1] - dy.point,
                                                           dy.point - dy.interval[0])
                H2.H2[q.name].comp[i_c].exc_CI_path = "".join([file_path_CI, ci_name])
                H2.H2[q.name].comp[i_c].exc_h2_j05_path = "".join([file_path_J5, object])
                H2.H2[q.name].comp[i_c].exc_h2_j05_path_to_logN = "".join([file_path_J5, 'bf/',bf_name])



                with open("".join([file_path_J2, object]), 'rb') as f:
                    x, y, z_H2 = pickle.load(f)

                d = distr2d(x=x, y=y, z=np.exp(z_H2 + z_prior))
                dx, dy = d.marginalize('y'), d.marginalize('x')
                dy.stats(latex=2, name='log UV')
                dx.stats(latex=2, name='log n')

                H2.H2[q.name].comp[i_c].physcond['n0_J2'] = e('n0', dx.point, dx.interval[1] - dx.point,
                                                              dx.point - dx.interval[0])
                H2.H2[q.name].comp[i_c].physcond['uv_J2'] = e('uv', dy.point, dy.interval[1] - dy.point,
                                                              dy.point - dy.interval[0])
                H2.H2[q.name].comp[i_c].exc_h2_j02_path = "".join([file_path_J2, object])
                H2.H2[q.name].comp[i_c].exc_h2_j02_path_to_logN = "".join([file_path_J2, 'bf/', bf_name])


                if name in ['J0000+0048_0','J1439+1118_0','J1513+0352_0','J2100-0641_0','J2123-0050_0','J0816+1446_0']:
                    H2.H2[q.name].comp[i_c].physcond['flag'] = 'J02'
                    H2.H2[q.name].comp[i_c].physcond['n0'] = H2.H2[q.name].comp[i_c].physcond['n0_J2']
                    H2.H2[q.name].comp[i_c].physcond['uv'] = H2.H2[q.name].comp[i_c].physcond['uv_J2']
                else:
                    H2.H2[q.name].comp[i_c].physcond['flag'] = 'J05'
                    H2.H2[q.name].comp[i_c].physcond['n0'] = H2.H2[q.name].comp[i_c].physcond['n0_J5']
                    H2.H2[q.name].comp[i_c].physcond['uv'] = H2.H2[q.name].comp[i_c].physcond['uv_J5']

        with open("".join([file_path, 'H2_CI_exc_all.pkl']), 'wb') as f:
            pickle.dump(H2, f)




    with open("".join([file_path, 'H2_CI_exc_all.pkl']), 'rb') as f:
        H2 = pickle.load(f)

    # print data for the table:
    print('selected systems:')
    i = 0
    for q in H2.H2.values():
        for i_c, c in enumerate(q.comp):
            name = "".join([q.name, '_', str(i_c)])
            i +=1
            #print(i,name, c.z, q.e['HI'].col.val,q.e['HI'].col.plus,q.e['HI'].col.minus,
            #      c.e['H2'].col.val,c.e['H2'].col.plus,c.e['H2'].col.minus)
            print(i,name, q.e['Me'].col.val,q.e['Me'].col.plus, q.e['Me'].col.minus,
                  c.e['T01'].col.val, c.e['T01'].col.plus,c.e['T01'].col.minus,
                  c.physcond['n0'].col.val, c.physcond['n0'].col.plus, c.physcond['n0'].col.minus,
                  c.physcond['uv'].col.val, c.physcond['uv'].col.plus, c.physcond['uv'].col.minus)

    # plot histogram of mean UV
    if 0:
        fig_hist,ax = plt.subplots(figsize=(9, 9))
        uv_hist =[]
        uv_J2_hist =[]

        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                uv_hist.append(c.physcond['uv'].col.val)
                uv_J2_hist.append(c.physcond['uv_J2'].col.val)

        bins = np.linspace(-1, 2, 7)
        ax.hist(uv_hist,bins=bins,color='red',alpha=0.4,label='J=0-5')
        ax.hist(uv_J2_hist,bins=bins,color='blue',alpha=0.4,label='J=0-2')
        ax.legend(fontsize=12)
        ax.set_xlabel('$\log I_{UV}$, Draine unit', fontsize=12)
        ax.set_ylabel('Число систем', fontsize=12)

        #fig_hist.savefig(fname="".join([file_path, 'H2_all_hist.png']), bbox_inches='tight')

    # plot uv-n0 probability contours
    if 0:
        fig_lnL = plt.figure(figsize=(28, 20))
        gs = gridspec.GridSpec(4, 6, wspace=0.3, hspace=0.5, figure=fig_lnL)
        k = 0
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                #if k<2:
                    ax = plt.subplot(gs[k])
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.set_xlabel('$\log n_0$, cm$^{-3}$', fontsize=12)
                    ax.set_ylabel('$\log I_{UV}$, Draine unit', fontsize=12)

                    k += 1
                    name = "".join([q.name, '_', str(i_c)])
                    print(name)
                    left, width = .1, .5
                    bottom, height = .25, .8
                    right = left + width
                    top = bottom + height
                    ax.text(left, top, name, fontsize=12, color='red',transform=ax.transAxes)

                    if H2.H2[q.name].comp[i_c].physcond['flag'] == 'J05':

                        with open(q.comp[i_c].exc_h2_j05_path, 'rb') as f:
                            x, y, z = pickle.load(f)
                        d = distr2d(x=x, y=y, z=np.exp(z))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dy.stats(latex=-1, name='log UV')
                        dx.stats(latex=-1, name='log n')
                        d.plot_contour(ax=ax, color='purple', color_point=None,cmap='Purples',alpha=0, lw=2.0)

                    elif H2.H2[q.name].comp[i_c].physcond['flag'] == 'J02':

                        with open(q.comp[i_c].exc_h2_j02_path, 'rb') as f:
                            x, y, z = pickle.load(f)
                        d = distr2d(x=x, y=y, z=np.exp(z))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dy.stats(latex=2, name='log UV')
                        dx.stats(latex=2, name='log n')
                        d.plot_contour(ax=ax, color='purple', alpha=0.0, cmap='Purples', color_point=None, lw=2.0)


                    prior = 0
                    if prior == 1 or name in ['B0405-4418_0','J0643-5041_0','B1444+0126_0','B1444+0126_1','0551-3638_0','J1443+2724_0','J1443+2724_1']:
                        z_prior = np.zeros([len(y), len(x)])
                        x_mean = 2.0
                        x_s = 0.5
                        for i_x, xi in enumerate(x):
                            for k_y, yk in enumerate(y):
                                z_prior[k_y, i_x] = (-(xi - x_mean) ** 2 / x_s ** 2.)
                        #ax.bar(x=2.0, y=-2., height=5.0, width=1.0, color='green', alpha=0.3)
                    else:
                        with open(q.comp[i_c].exc_CI_path, 'rb') as f:
                            x, y, z_CI = pickle.load(f)
                        z_prior = z_CI
                    d = distr2d(x=x, y=y, z=np.exp(z_prior))
                    dx, dy = d.marginalize('y'), d.marginalize('x')
                    dy.stats(latex=-1, name='log UV')
                    dx.stats(latex=-1, name='log n')
                    d.plot_contour(ax=ax, color='green', alpha=0.0, cmap='Greens', color_point=None, lw=2.0)

                    d = distr2d(x=x, y=y, z=np.exp(z + z_prior))
                    dx, dy = d.marginalize('y'), d.marginalize('x')
                    dy.stats(latex=2, name='log UV')
                    dx.stats(latex=2, name='log n')
                    d.plot_contour(ax=ax, color='red', alpha=0, cmap='Reds', lw=5.0,ylabel='$\log (I_{UV})$', xlabel='$\log (n)$',font=12)






        fig_lnL.savefig(fname="".join([file_path,'H2_all_n0_UV_test.png']),bbox_inches='tight')

    # plot H2 level excitation for the best_fit estimate of (uv,n0)
    if 0:
        fig_bf = plt.figure(figsize=(28, 20))
        gs = gridspec.GridSpec(4, 6, wspace=0.3, hspace=0.5, figure=fig_bf)
        k = 0
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                ax = plt.subplot(gs[k])
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.xaxis.set_major_locator(MultipleLocator(1000))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_major_locator(MultipleLocator(2))
                ax.set_xlabel('Энергия уровня H$_2$(J), K', fontsize=12)
                ax.set_ylabel('$\log N(H_2,J)/g(J)$', fontsize=12)

                k += 1
                name = "".join([q.name, '_', str(i_c)])

                if H2.H2[q.name].comp[i_c].physcond['flag'] == 'J05':
#                    with open(q.comp[i_c].exc_h2_j05_path_to_logN, 'rb') as f:
#                        Ejh2, logNh2 = pickle.load(f)
#                    ax.plot(Ejh2, logNh2,color='blue')
                    color = 'blue'
                elif H2.H2[q.name].comp[i_c].physcond['flag'] == 'J02':
#                    with open(q.comp[i_c].exc_h2_j02_path_to_logN, 'rb') as f:
#                        Ejh2, logNh2 = pickle.load(f)
#                    ax.plot(Ejh2, logNh2,color='red')
                    color = 'red'
                with open('result/cr_fixed/bf_H2/{:s}_H2.pkl'.format(name), 'rb') as f:
                    Ejh2, logNh2 = pickle.load(f)

                ax.plot(Ejh2, logNh2,color=color)

                H2exc = H2_exc(H2database='H2UV')
                H2exc.plot_objects(objects=name, ax=ax,syst=0.2)
                ax.set_xbound(lower=-100, upper=2500)
                ax.set_ybound(lower=10, upper=21.5)
                left, width = .1, .5
                bottom, height = .25, .8
                right = left + width
                top = bottom + height
                ax.text(left, top, name, fontsize=12, color='red', transform=ax.transAxes)
        fig_bf.savefig(fname="".join([file_path,'H2_all_logN_test.png']),bbox_inches='tight')

    if 0:
        fig_lnL = plt.figure(figsize=(28, 15))
        gs = gridspec.GridSpec(3, 5, wspace=0.0, hspace=0.3, figure=fig_lnL)
        k = 0
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                name = "".join([q.name, '_', str(i_c)])
                if name in ['B0528-2505_0','J0812+3208_0','J2140-0321_0','J1513+0352_0','J1237+0647_0']:
                    ax = plt.subplot(gs[k])
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    if k==0:
                        ax.set_ylabel('$\log I_{UV}$, Draine unit', fontsize=12)

                    ax.set_xlabel('$\log n_0$, cm$^{-3}$', fontsize=12)

                    k += 1
                    name = "".join([q.name, '_', str(i_c)])
                    print(name)
                    left, width = .53, .5
                    bottom, height = .25, .6
                    right = left + width
                    top = bottom + height
                    ax.text(left, top, q.name, fontsize=14, color='black', transform=ax.transAxes)

                    if H2.H2[q.name].comp[i_c].physcond['flag'] == 'J05':

                        with open(q.comp[i_c].exc_h2_j05_path, 'rb') as f:
                            x, y, z = pickle.load(f)
                        d = distr2d(x=x, y=y, z=np.exp(z))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dy.stats(latex=-1, name='log UV')
                        dx.stats(latex=-1, name='log n')
                        d.plot_contour(ax=ax, color='purple', color_point=None, cmap='Purples', alpha=0, lw=2.0)

                    elif H2.H2[q.name].comp[i_c].physcond['flag'] == 'J02':

                        with open(q.comp[i_c].exc_h2_j02_path, 'rb') as f:
                            x, y, z = pickle.load(f)
                        d = distr2d(x=x, y=y, z=np.exp(z))
                        dx, dy = d.marginalize('y'), d.marginalize('x')
                        dy.stats(latex=2, name='log UV')
                        dx.stats(latex=2, name='log n')
                        d.plot_contour(ax=ax, color='purple', alpha=0.0, cmap='Purples', color_point=None, lw=2.0)

                    prior = 0
                    if prior == 1 or name in ['B0405-4418_0', 'J0643-5041_0', 'B1444+0126_0', 'B1444+0126_1', '0551-3638_0',
                                              'J1443+2724_0', 'J1443+2724_1']:
                        z_prior = np.zeros([len(y), len(x)])
                        x_mean = 2.0
                        x_s = 0.5
                        for i_x, xi in enumerate(x):
                            for k_y, yk in enumerate(y):
                                z_prior[k_y, i_x] = (-(xi - x_mean) ** 2 / x_s ** 2.)
                        # ax.bar(x=2.0, y=-2., height=5.0, width=1.0, color='green', alpha=0.3)
                    else:
                        with open(q.comp[i_c].exc_CI_path, 'rb') as f:
                            x, y, z_CI = pickle.load(f)
                        z_prior = z_CI
                    d = distr2d(x=x, y=y, z=np.exp(z_prior))
                    dx, dy = d.marginalize('y'), d.marginalize('x')
                    dy.stats(latex=-1, name='log UV')
                    dx.stats(latex=-1, name='log n')
                    d.plot_contour(ax=ax, color='green', alpha=0.0, cmap='Greens', color_point=None, lw=2.0)

                    d = distr2d(x=x, y=y, z=np.exp(z + z_prior))
                    dx, dy = d.marginalize('y'), d.marginalize('x')
                    dy.stats(latex=2, name='log UV')
                    dx.stats(latex=2, name='log n')
                    d.plot_contour(ax=ax, color='red', alpha=0, cmap='Reds', lw=5.0,
                                   xlabel='$\log (n)$', font=12)

                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                    ax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                    ax.tick_params(which='minor', length=3, direction='in')
                    ax.tick_params(axis='both', which='major', labelsize=14, direction='in')
                    if k > 1:
                        ax.set_yticklabels([])

                    ax.set_xlabel('$\log n_H, cm^{-3}$', fontsize=14)
                    if k==1:
                        ax.set_ylabel('$\log I_{UV}, Draine unit$', fontsize=14)

                    ax.set_xbound(lower=0, upper=4.99)
                    ax.set_ybound(lower=-1, upper=3)

                    ax = plt.subplot(gs[k+4])

                    if H2.H2[q.name].comp[i_c].physcond['flag'] == 'J05':
                        #                    with open(q.comp[i_c].exc_h2_j05_path_to_logN, 'rb') as f:
                        #                        Ejh2, logNh2 = pickle.load(f)
                        #                    ax.plot(Ejh2, logNh2,color='blue')
                        color = 'blue'
                    elif H2.H2[q.name].comp[i_c].physcond['flag'] == 'J02':
                        #                    with open(q.comp[i_c].exc_h2_j02_path_to_logN, 'rb') as f:
                        #                        Ejh2, logNh2 = pickle.load(f)
                        #                    ax.plot(Ejh2, logNh2,color='red')
                        color = 'red'
                    with open('result/cr_fixed/bf_H2/{:s}_H2.pkl'.format(name), 'rb') as f:
                        Ejh2, logNh2 = pickle.load(f)

                    ax.plot(Ejh2, logNh2, color=color)

                    H2exc = H2_exc(H2database='H2UV')
                    H2exc.plot_objects(objects=name, ax=ax, syst=0.2)
                    ax.set_xbound(lower=-100, upper=2500)
                    ax.set_ybound(lower=12, upper=21)
                    left, width = .1, .5
                    bottom, height = .25, .8
                    right = left + width
                    top = bottom + height
                    ax.text(0.8, 0.8, 'H$_2$', fontsize=20, color='black', transform=ax.transAxes)

                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.xaxis.set_major_locator(MultipleLocator(1000))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_major_locator(MultipleLocator(2))
                    ax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                    ax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                    ax.tick_params(which='minor', length=3, direction='in')
                    ax.tick_params(axis='both', which='major', labelsize=14, direction='in')
                    if k > 1:
                        ax.set_yticklabels([])

                    ax.set_xlabel('Energy level (H_2) E_J, K', fontsize=14)
                    if k==1:
                        ax.set_ylabel('$\log N(H_2,J)/g(J)$', fontsize=14)


                    ax = plt.subplot(gs[k + 4+5])

                    if H2.H2[q.name].comp[i_c].physcond['flag'] == 'J05':
                        #                    with open(q.comp[i_c].exc_h2_j05_path_to_logN, 'rb') as f:
                        #                        Ejh2, logNh2 = pickle.load(f)
                        #                    ax.plot(Ejh2, logNh2,color='blue')
                        color = 'blue'
                    elif H2.H2[q.name].comp[i_c].physcond['flag'] == 'J02':
                        #                    with open(q.comp[i_c].exc_h2_j02_path_to_logN, 'rb') as f:
                        #                        Ejh2, logNh2 = pickle.load(f)
                        #                    ax.plot(Ejh2, logNh2,color='red')
                        color = 'red'
                    with open('result/cr_fixed/CI/BF/{:s}_CI.pkl'.format(name), 'rb') as f:
                        EjCI, logNCI = pickle.load(f)

                    ax.plot(EjCI, logNCI-logNCI[0], color=color)

                    H2exc = H2_exc(H2database='H2UV')
                    H2exc.plot_objects(objects=name, ax=ax, syst=0.2,species='CI')
                    ax.set_xbound(lower=-5, upper=55)
                    ax.set_ybound(lower=-2, upper=1)
                    left, width = .1, .5
                    bottom, height = .25, .8
                    right = left + width
                    top = bottom + height
                    ax.text(0.8, 0.8, 'CI', fontsize=20, color='black', transform=ax.transAxes)
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.xaxis.set_major_locator(MultipleLocator(10))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    if k>1:
                        #ax.yaxis.set_visible(False)
                        ax.set_yticklabels([])
                    ax.tick_params(which='both', width=1, direction='in', right='True', top='True')
                    ax.tick_params(which='major', length=4, direction='in', right='True', top='True')
                    ax.tick_params(which='minor', length=3, direction='in')
                    ax.tick_params(axis='both', which='major', labelsize=14, direction='in')

                    ax.set_xlabel('Energy level (CI) E_J, K', fontsize=14)
                    if k==1:
                        ax.set_ylabel('$\log N(CI,J)/g(J)/N(CI,J=0)$', fontsize=14)
        str = ".pdf"
        f_name = "".join(('H2ci_exc', str))
        #fig_lnL.savefig(f_name, bbox_inches='tight')

    # plot correlation
    if 1:
        fig_me, ax = plt.subplots(figsize=(9, 6))
        uv = []
        uverrp = []
        uverrm = []
        uverr= []

        n0 = []
        n0errp = []
        n0errm = []
        n0err= []
        me = []
        NH = []
        NH2 = []
        NCI = []
        z=[]
        test= []
        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                uv.append(c.physcond['uv'].col.val)
                n0.append(c.physcond['n0'].col.val)
                me.append(q.e['Me'].col.val)
                NH.append(q.e['H'].col.val-c.e['H2'].col.val)
                NH2.append(c.e['H2'].col.val)

                n0errp.append(c.physcond['n0'].col.plus)
                n0errm.append(c.physcond['n0'].col.minus)
                uverrp.append(c.physcond['uv'].col.plus)
                uverrm.append(c.physcond['uv'].col.minus)

                if q.name in ['B0405-4418','J0643-5041','B1444+0126','B1444+0126','0551-3638','J1443+2724','J1443+2724']:
                    NCI.append(10)
                else:
                    NCI.append(c.e['CIj0'].col.val)
                z.append(c.z)
                test.append(c.physcond['uv'].col.val-c.physcond['n0'].col.val)

        #ax.scatter(x=n0,y=uv,marker='o',markersize=14,markeredgecolor='black',color=me,cmap='PiYG')
        n0err = np.array([n0errm, n0errp])
        uverr = np.array([uverrm, uverrp])

        # ax.scatter(x=n0,y=uv,marker='o',markersize=14,markeredgecolor='black',color=me,cmap='PiYG')

        c = plt.scatter(n0, uv, 300, me, cmap='hot',vmin=-1.5,vmax=0.5,edgecolors='black')
        plt.errorbar(n0, uv, yerr=uverr, fmt='^', color='black', zorder=-10)  # fmt=None, marker=None)
        plt.errorbar(n0, uv, xerr=n0err, fmt='^', color='black', zorder=-10)

        #plt.colorbar()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        ax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        ax.tick_params(which='minor', length=3, direction='in')
        ax.tick_params(axis='both', which='major', labelsize=14, direction='in')

        ax.set_xlim(-0.1,3)
        ax.set_ylim(-1, 3)
        ax.set_xlabel('Hydrogen density, $\log n_H$, cm$^{-3}$', fontsize=16)
        ax.set_ylabel('UV intensity, $\log I_{UV}$, Draine unit', fontsize=16)        #ax.set_xlabel('$\log I_{UV}$, Draine unit', fontsize=12)
        #ax.set_ylabel('Число систем', fontsize=12)
        #plt.hist(n0)
        cax = fig_me.add_axes([0.16, 0.8, 0.4, 0.05])
        fig_me.colorbar(c, cax=cax,orientation='horizontal')
        cax.tick_params(labelsize=20)
        cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        cax.tick_params(which='minor', length=3, direction='in')
        cax.tick_params(axis='both', which='major', labelsize=14, direction='in')
        ax.text(0.2,2.1,'Metallicity',fontsize=18)

        adax = fig_me.add_axes([0.125, 0.11+0.77, 0.775, 0.1])
        adax.hist(n0,color='gold')
        adax.axis('off') #get_xaxis().set_visible(False)
        adax.set_xlim(-0.1, 3)

        aday = fig_me.add_axes([0.125+0.775, 0.11, 0.1, 0.77])
        aday.hist(uv,bins=4,color='darkred',orientation="horizontal")
        aday.set_ylim(-1, 3)
        aday.axis('off') #get_xaxis().set_visible(False)

        str = ".pdf"
        f_name = "".join(('Me', str))
        #fig_me.savefig(f_name, bbox_inches='tight')
    if 0:
        fig_me, ax = plt.subplots(figsize=(9, 6))
        uv = []
        n0 = []
        me = []
        NH = []
        NH2 = []
        NCI = []
        z=[]
        test= []
        T01 = []
        T01errp = []
        T01errm = []
        T01err = []
        NH2errp = []
        NH2errm = []
        NH2err = []

        for q in H2.H2.values():
            for i_c, c in enumerate(q.comp):
                uv.append(c.physcond['uv'].col.val)
                n0.append(c.physcond['n0'].col.val)
                me.append(q.e['Me'].col.val)
                NH.append(q.e['H'].col.val-c.e['H2'].col.val)
                NH2.append(c.e['H2'].col.val)
                NH2errp.append(c.e['H2'].col.plus)
                NH2errm.append(c.e['H2'].col.minus)
                print(q.name)
                #print(c.e['T01'].col.val,c.e['T01'].col.plus,c.e['T01'].col.minus)
                if q.name in ['J1439+1118']:
                    T01.append(110.0)
                    T01errp.append(np.log10((46+110)/110))
                    T01errm.append(-np.log10((110-25)/110))
                else:
                    T01.append(c.e['T01'].col.val)
                    T01errp.append(np.log10((c.e['T01'].col.plus+c.e['T01'].col.val)/c.e['T01'].col.val))
                    T01errm.append(-np.log10((-c.e['T01'].col.minus+c.e['T01'].col.val)/c.e['T01'].col.val))

                    #T01err.append([np.log10((c.e['T01'].col.plus+c.e['T01'].col.val)/c.e['T01'].col.val),
                    #-np.log10((-c.e['T01'].col.minus+c.e['T01'].col.val)/c.e['T01'].col.val)])
                if q.name in ['B0405-4418','J0643-5041','B1444+0126','B1444+0126','0551-3638','J1443+2724','J1443+2724']:
                    NCI.append(10)
                else:
                    NCI.append(c.e['CIj0'].col.val)
                z.append(c.z)
                test.append(c.physcond['uv'].col.val-c.physcond['n0'].col.val)
        T01 = np.log10(T01)

        print('test', np.mean(n0), np.std(n0))

        T01err = np.array([T01errm,T01errp])
        NH2err = np.array([NH2errm,NH2errp])
        #ax.scatter(x=n0,y=uv,marker='o',markersize=14,markeredgecolor='black',color=me,cmap='PiYG')
        c = plt.scatter(NH2, T01, 300, uv, cmap='hot',vmin=-0.1,vmax=1.8,edgecolors='black')
        plt.errorbar(NH2, T01, yerr=T01err,  fmt='^',color='black',zorder=-10) #fmt=None, marker=None)
        plt.errorbar(NH2, T01, xerr=NH2err, fmt='^', color='black', zorder=-10)
        #plt.colorbar()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        ax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        ax.tick_params(which='minor', length=3, direction='in')
        ax.tick_params(axis='both', which='major', labelsize=14, direction='in')


        #ax.set_xlim(-0.1,3)
        ax.set_ylim(1.5, 3.2)
        ax.set_xlabel('$\log N(H_2)$', fontsize=16)
        ax.set_ylabel('$\log T_{01}$', fontsize=16)        #ax.set_xlabel('$\log I_{UV}$, Draine unit', fontsize=12)
        #ax.set_ylabel('Число систем', fontsize=12)
        #plt.hist(n0)
        cax = fig_me.add_axes([0.52, 0.8, 0.35, 0.05])
        fig_me.colorbar(c, cax=cax,orientation='horizontal')
        cax.tick_params(labelsize=20)
        cax.tick_params(which='both', width=1, direction='in', right='True', top='True')
        cax.tick_params(which='major', length=4, direction='in', right='True', top='True')
        cax.tick_params(which='minor', length=3, direction='in')
        cax.tick_params(axis='both', which='major', labelsize=14, direction='in')

        ax.text(20,2.8,'$\log I_{UV}$',fontsize=18)
        str = ".pdf"
        f_name = "".join(('T01', str))
        #fig_me.savefig(f_name, bbox_inches='tight')

        print('test', np.mean(uv), np.disp(uv))

    plt.show()






if 0:
    fig_bf = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(2, 6, wspace=0.3, hspace=0.5, figure=fig_bf)

    onlyfiles = [f for f in listdir('result/z=-1/') if isfile(join('result/z=-1/', f))]
    qlist = sorted(onlyfiles, key=lambda s: s[1:5])
    #qlist = ['J0812+3208_2_H2.pkl']
    for i, object in enumerate(qlist):
        if i < 22:
            k = i
            name = object.split(".")[0][0:12]
            print(k, name)
            ax = plt.subplot(gs[k])

            with open("".join(['result/z=-1/bf/', '_'.join([name,'H2.pkl'])]), 'rb') as f:
                Ejh2, logNh2 = pickle.load(f)

            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_major_locator(MultipleLocator(1000))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.set_xlabel('Энергия уровня H$_2$(J), K', fontsize=12)
            ax.set_ylabel('$\log N(H_2,J)/g(J)$', fontsize=12)
            H2exc = H2_exc(H2database='all')
            H2exc.plot_objects(objects=name, ax=ax)
            ax.plot(Ejh2, logNh2)
            ax.set_xbound(lower=-100, upper=2500)
            ax.set_ybound(lower=10, upper=21.5)
            left, width = .1, .5
            bottom, height = .25, .8
            right = left + width
            top = bottom + height
            ax.text(left, top, name, fontsize=12, color='red', transform=ax.transAxes)


    plt.show()
    #plt.savefig('out.pdf')

# plot all contours and histogram
if 0:
    fig, ax = plt.subplots(figsize=(9, 9))  # ,sharex=True)
    fig2,ax2 = plt.subplots(figsize=(9, 9))
    #fig3,ax3 = plt.subplots(figsize=(9, 9))

    onlyfiles = [f for f in listdir('result/reduced/') if isfile(join('result/reduced/', f))]
    mod =  sorted(onlyfiles, key=lambda s: s[1:5])
    point_estimate = []
    for i, object in enumerate(mod):
        with open("".join(['result/all/', object]), 'rb') as f:
            x, y, z = pickle.load(f)
        d = distr2d(x=x, y=y, z=np.exp(z))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dx.stats(latex=2, name='log UV')
        point_estimate.append(float(dx.point))
        #dy.stats(latex=2, name='log n')
        if i == 0:
            Iuv_prob=dx.y
        else:
            Iuv_prob+=dx.y
        d.plot_contour(ax=ax, color='red', xlabel='$\log I_{UV}$', ylabel='$\log n$', font=12,cmap=None)
        ax2.plot(dx.x,dx.y)
    ax2.plot(dx.x,Iuv_prob,'--',color='red',label='Sum of likelihoods')
    print(np.mean(point_estimate), moment(point_estimate,moment=1),moment(point_estimate,moment=2),moment(point_estimate,moment=3))
    interp_uv = interpolate.interp1d(dx.x, Iuv_prob)
    #ax3.plot(dx.x,Iuv_prob)
    #f = []
    #for e in dx.x:
    #    f.append(interp_uv(e))
    #ax3.plot(dx.x,f,'--')
    num = 6
    bins = np.linspace(dx.x[0], dx.x[-1], num)
    bin_size = (dx.x[-1] - dx.x[0])/ num
    #dx_hist = [integrate.quad(interp_uv, bins[0],bins[0]+bin_size/2)[0]]
    dx.interpolate()
    dx_hist = [0]
    for i in range(1, num):
        y = integrate.quad(interp_uv, bins[i-1],bins[i])
        dx_hist.append(y[0])
        print(i,bins[i-1],bins[i],y[0])

#    dx_hist.append(integrate.quad(interp_uv, bins[len(bins)-1]- bin_size / 2, bins[len(bins)-1])[0])

    ax2.step(bins,dx_hist,where='pre',linewidth=2.0, color='black',label='histogram=#sys per bin')

    interp_uv_x = interpolate.interp1d(dx.x, Iuv_prob*dx.x)
    uv_mean = integrate.quad(interp_uv_x, dx.x[0],dx.x[-1])[0]/integrate.quad(interp_uv, dx.x[0],dx.x[-1])[0]

    interp_uv_xx = interpolate.interp1d(dx.x, Iuv_prob * (dx.x-uv_mean)*(dx.x-uv_mean))
    uv_disp =integrate.quad(interp_uv_xx, dx.x[0],dx.x[-1])[0]/integrate.quad(interp_uv, dx.x[0],dx.x[-1])[0]

    print(uv_mean,uv_disp)
    ax2.errorbar(uv_mean,10,xerr=uv_disp**0.5,fmt='o',markersize=12, linewidth=2.0, color='black',capsize=3)
    ax2.text(1.0,10.5,'$<I_{UV}>=1.23\pm0.77$',fontsize=14)
    ax2.legend(fontsize='medium')
    ax2.hist(point_estimate) #,bins=bins)
    plt.show()

    #print(np.average(dx.x, weights=interp_uv), nmoment(dx.x, interp_uv, np.average(dx.x, weights=interp_uv), 2))

if 0:
    with open('temp/lnL_nodes.pkl', 'rb') as f:
        x1, y1, z1 = pickle.load(f)
    with open('temp/lnL.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        plt.subplot(1, 3, 1)
        plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
        plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
        plt.title('RBF interpolation - likelihood')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()
        # plt.savefig('rbf2d.png')
    with open('temp/lnL_nodes_0.2.pkl', 'rb') as f:
        x1, y1, z1 = pickle.load(f)
    with open('temp/lnL_0.2.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        plt.subplot(1,3, 2)
        plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
        plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
        plt.title('RBF interpolation - likelihood')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()
        # plt.savefig('rbf2d.png')
    with open('temp/lnL_nodes_0.3.pkl', 'rb') as f:
        x1, y1, z1 = pickle.load(f)
    with open('temp/lnL_0.3.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        plt.subplot(1, 3, 3)
        plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
        plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
        plt.title('RBF interpolation - likelihood')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()
        # plt.savefig('rbf2d.png')
        plt.show()

print('exit')

def nmoment(x, counts, c, n):
    return np.sum(counts*(x-c)**n) / np.sum(counts)