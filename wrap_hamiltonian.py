#%matplotlib inline
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
from optparse import OptionParser
import os
from mpl_toolkits.mplot3d import Axes3D
import random
from mpl_toolkits import mplot3d
from matplotlib import rc
#plt.rcParams['mathtext.fontset'] = 'dejavuserif'
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})

#rc('text', usetex=True) #open this for latex
######## Constant defined here ########
pi        =     3.1415926535897932384626
q0        =     1.602176565e-19 # C
m0        =     9.10938291e-31  # kg
v0        =     2.99792458e8    # m/s^2
kb        =     1.3806488e-23   # J/K
mu0       =     4.0e-7*pi       # N/A^2
epsilon0  =     8.8541878176203899e-12 # F/m
h_planck  =     6.62606957e-34  # J s
wavelength=     1.0e-6
frequency =     v0*2*pi/wavelength

exunit    =     m0*v0*frequency/q0
bxunit    =     m0*frequency/q0
denunit    =     frequency**2*epsilon0*m0/q0**2
print('electric field unit: '+str(exunit))
print('magnetic field unit: '+str(bxunit))
print('density unit nc: '+str(denunit))
font = {'family' : 'monospace',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 25,
        }

font2 = {'family' : 'monospace',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 20,
        }

font_size =25
font_size2=20

c_red  = matplotlib.colors.colorConverter.to_rgba('red')
c_blue = matplotlib.colors.colorConverter.to_rgba('blue')
c_black = matplotlib.colors.colorConverter.to_rgba('black')
c_green= matplotlib.colors.colorConverter.to_rgba('lime')
c_white_trans = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0.0)
cmap_mycolor1 = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_red,c_green],128)
cmap_my_rw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_red,c_white_trans],128)
cmap_my_bw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_blue,c_white_trans],128)
cmap_my_kw = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_black,c_white_trans],128)
cmap_my_wr = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_red],128)
cmap_my_wb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_blue],128)
cmap_my_wk = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white_trans,c_black],128)

upper = matplotlib.cm.jet(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
    lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_jet = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

mp  = 1836. # proton mass
V_s = 0.145  # in unit [c]
K   = 100.0 # electric field ratio
E_0 = 30
px_1d  = np.linspace(-0.2,1.2,201)
xi_1d  = np.linspace(-15,15,201)
xi_0= 0*2*np.pi
xi_1= 0.45*2*np.pi
xi_2= 0.90*2*np.pi
xi_3=-0.45*2*np.pi
xi_4=-0.90*2*np.pi
xi, px = np.meshgrid(xi_1d, px_1d)
phi = np.zeros_like(xi)
E_xi= np.zeros_like(xi)

condition = (xi<=xi_4)
phi[condition] = 0
E_xi[condition]= 0
condition = (xi_4<xi)&(xi<=xi_3)
phi[condition] = E_0*(xi[condition]-xi_4)**2*0.5/(xi_3-xi_4)
E_xi[condition]=-E_0*(xi[condition]-xi_4)/(xi_3-xi_4)
condition = (xi_3<xi)&(xi<=xi_0)
phi[condition] = E_0*(xi_2-xi_0)*0.5-E_0/(xi_0-xi_3)*0.5*(xi_0-xi[condition])**2
E_xi[condition]=-E_0*(xi_0-xi[condition])/(xi_0-xi_3)
condition = (xi_0<xi)&(xi<=xi_1)
phi[condition] = E_0*(xi_2-xi_0)*0.5-E_0/(xi_1-xi_0)*0.5*(xi[condition]-xi_0)**2
E_xi[condition] = E_0*(xi[condition]-xi_0)/(xi_1-xi_0)
condition = (xi_1<xi)&(xi<=xi_2)
phi[condition] = E_0*(xi_2-xi[condition])**2*0.5/(xi_2-xi_1)
E_xi[condition] = E_0*(xi_2-xi[condition])/(xi_2-xi_1)
condition = (xi>xi_2)
phi[condition] = 0
E_xi[condition]= 0

Hami= ((px**2+1)**0.5-1)*mp-V_s*px*mp+phi
Hami_1=Hami/mp

mp  = 1836. # proton mass
V_s = 0.507  # in unit [c]
K   = 100.0 # electric field ratio
E_0 = 15
px_1d  = np.linspace(-0.2,1.2,201)
xi_1d  = np.linspace(-15,15,201)
xi_0= 0*2*np.pi
xi_1= 0.90*2*np.pi
xi_2= 1.80*2*np.pi
xi_3=-0.90*2*np.pi
xi_4=-1.80*2*np.pi
xi, px = np.meshgrid(xi_1d, px_1d)
phi = np.zeros_like(xi)
E_xi= np.zeros_like(xi)

condition = (xi<=xi_4)
phi[condition] = 0
E_xi[condition]= 0
condition = (xi_4<xi)&(xi<=xi_3)
phi[condition] = E_0*(xi[condition]-xi_4)**2*0.5/(xi_3-xi_4)
E_xi[condition]=-E_0*(xi[condition]-xi_4)/(xi_3-xi_4)
condition = (xi_3<xi)&(xi<=xi_0)
phi[condition] = E_0*(xi_2-xi_0)*0.5-E_0/(xi_0-xi_3)*0.5*(xi_0-xi[condition])**2
E_xi[condition]=-E_0*(xi_0-xi[condition])/(xi_0-xi_3)
condition = (xi_0<xi)&(xi<=xi_1)
phi[condition] = E_0*(xi_2-xi_0)*0.5-E_0/(xi_1-xi_0)*0.5*(xi[condition]-xi_0)**2
E_xi[condition] = E_0*(xi[condition]-xi_0)/(xi_1-xi_0)
condition = (xi_1<xi)&(xi<=xi_2)
phi[condition] = E_0*(xi_2-xi[condition])**2*0.5/(xi_2-xi_1)
E_xi[condition] = E_0*(xi_2-xi[condition])/(xi_2-xi_1)
condition = (xi>xi_2)
phi[condition] = 0
E_xi[condition]= 0

Hami= ((px**2+1)**0.5-1)*mp-V_s*px*mp+phi
Hami_2=Hami/mp

plt.subplot(1,2,1)
levels = np.linspace(-0.15,-0.05, 21) 
plt.contourf(xi/(2*np.pi), px, Hami_2, cmap=cmap_my_bw,levels=levels)
levels = np.linspace(-0.02,0.05, 21) 
plt.contourf(xi/(2*np.pi), px, Hami_1, cmap=cmap_my_kw,levels=levels)
#plt.contour(xi/(2*np.pi), px, Hami_1, levels=[0,0.03],colors='k', linewidths=1, linestyles='--',zorder=4)
#plt.contour(xi/(2*np.pi), px, Hami_2, levels=[-0.125,-0.092],colors='k',linewidths=1,linestyles=':')
#plt.plot((np.zeros_like(px_1d)+xi_0)/(2*np.pi),px_1d,'--',color='k')
#plt.plot((np.zeros_like(px_1d)+xi_2)/(2*np.pi),px_1d,'--',color='k')
#### manifesting colorbar, changing label and axis properties ####
part_number=100
nsteps=4000
v_s   =0.2
insert1='./test_particle/run_hami_3d_001/'
insert_n='_0000'
t1=np.loadtxt(insert1+'t'+insert_n+'.txt')
x1=np.loadtxt(insert1+'x'+insert_n+'.txt')
px1=np.loadtxt(insert1+'px'+insert_n+'.txt')
t1=np.reshape(t1,(part_number,nsteps))/(2.0*np.pi)
x1=np.reshape(x1,(part_number,nsteps))/(2.0*np.pi)
px1=np.reshape(px1,(part_number,nsteps))
part_id1 = np.linspace(0,part_number-1,part_number).reshape(part_number,1)+np.zeros_like(x1)
xi1=np.zeros_like(x1)
condition1 = (t1<=10)
print(x1.shape,t1.shape,xi1.shape)
xi1[condition1]=x1[condition1]-0.145*t1[condition1]
condition1 = (t1>10)
xi1[condition1]=x1[condition1]-0.507*(t1[condition1]-10.0)-0.145*10
for n in range(part_number):
    if n%2!=0:
        continue
    if xi1[n,0]<0.35:
        continue
    plt.scatter(xi1[n,:], px1[n,:], c=part_id1[n,:]/100.0*0.9, s=0.9 ,cmap='autumn', norm=colors.Normalize(vmin=0.35,vmax=0.9), edgecolors='None')
plt.xlabel(r'$\xi\ [\mu m]$',fontdict=font)
plt.ylabel(r'$p_x\ [m_pc]$',fontdict=font)
plt.xlim(-0.2,2.0)
plt.ylim(-0.02,1.1)
plt.xticks([0,0.5,1.0,1.5,2.0],fontsize=font_size); 
plt.yticks(fontsize=font_size);


plt.subplot(1,2,2)
plt.contour(xi/(2*np.pi), px, Hami_1, levels=[0,0.015,0.03],colors='k', linewidths=4, linestyles='--',zorder=4)
plt.contour(xi/(2*np.pi), px, Hami_2, levels=[-0.125,-0.092,-0.05,0.0],colors='b',linewidths=4,linestyles='--')
#plt.plot((np.zeros_like(px_1d)+xi_0)/(2*np.pi),px_1d,'--',color='k')
#plt.plot((np.zeros_like(px_1d)+xi_2)/(2*np.pi),px_1d,'--',color='k')
#### manifesting colorbar, changing label and axis properties ####

#plt.plot((np.zeros_like(px_1d)+xi_0)/(2*np.pi),px_1d,'--',color='k')
#plt.plot((np.zeros_like(px_1d)+xi_2)/(2*np.pi),px_1d,'--',color='k') 
v_s1=0.145; v_s2=0.507
for dir_list in ['n5','n15','n15_n5']:
    from_path = './spin_a70_'+dir_list+'_fine/'
    part_name = 'ion_s'
    part_mass = 1836
    ion_px=np.loadtxt(from_path+part_name+'_px.txt')
    ion_py=np.loadtxt(from_path+part_name+'_py.txt')
    ion_pz=np.loadtxt(from_path+part_name+'_pz.txt')
    ion_xx=np.loadtxt(from_path+part_name+'_xx.txt')
    ion_yy=np.loadtxt(from_path+part_name+'_yy.txt')
    ion_zz=np.loadtxt(from_path+part_name+'_zz.txt')
    ion_sx=np.loadtxt(from_path+part_name+'_sx.txt')*(part_mass*m0*v0)
    ion_sy=np.loadtxt(from_path+part_name+'_sy.txt')*(part_mass*m0*v0)
    ion_sz=np.loadtxt(from_path+part_name+'_sz.txt')*(part_mass*m0*v0)
    ion_ww=np.loadtxt(from_path+part_name+'_ww.txt')
    ion_tt=np.linspace(0.333333,75,225)  
    ion_pp=(ion_px**2+ion_py**2+ion_pz**2)**0.5
    ion_ek=((ion_px**2+ion_py**2+ion_pz**2+1)**0.5-1)*918.0
    ion_ss=(ion_sx**2+ion_sy**2+ion_sz**2)**0.5
    for n in range(np.size(ion_ww[:,0])):
      if dir_list == 'n15_n5':
          xi = np.zeros_like(ion_tt)
          condition = (ion_tt<=55.0)
          xi[condition] = ion_xx[n,condition]-v_s1*(ion_tt[condition]-30.0)/3.333333333333
          condition = (ion_tt>55.0)
          xi[condition] = ion_xx[n,condition]-v_s2*(ion_tt[condition]-55.0)/3.333333333333-v_s1*25.0/3.333333333333
          color_choice='red'
          lwidth=0.3
      if dir_list == 'n15':
          xi = np.zeros_like(ion_tt)
          xi = ion_xx[n,:]-v_s1*(ion_tt-30.0)/3.333333333333
          color_choice='limegreen'
          lwidth=0.8
      if dir_list == 'n5':
          xi = np.zeros_like(ion_tt)
          xi = ion_xx[n,:]-v_s2*(ion_tt-30.0)/3.333333333333
          color_choice='gold'
          lwidth=0.3
      #plt.scatter(xi, ion_px[n,:], c=ion_tt, norm=colors.Normalize(vmin=0,vmax=70), s=3, cmap='viridis', edgecolors='None', alpha=1,zorder=10)
      #plt.scatter(xi, ion_px[n,:], c=ion_tt, norm=colors.Normalize(vmin=0,vmax=70), s=3, cmap='viridis', edgecolors='None', alpha=1,zorder=10)
      print(dir_list,n)
      plt.plot(xi, ion_px[n,:], color=color_choice, linewidth=lwidth, linestyle='-',zorder=10)
#      if (dir_list=='n15_n5') and (n ==10):
#        plt.scatter(xi, ion_px[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=20)
#      if (dir_list=='n15_n5') and (n ==47):
#        plt.scatter(xi, ion_px[n,:], c=ion_ek[n,:], norm=colors.Normalize(vmin=0,vmax=150), s=25, cmap='magma', edgecolors='None', alpha=1,zorder=20)
    #cbar=plt.colorbar(pad=0.01,ticks=[0,10,20,30,40,50,60,70])
    #cbar.set_label('$t\ [\mathrm{fs}]$',fontdict=font)
    #cbar.ax.tick_params(labelsize=font_size2)
#### manifesting colorbar, changing label and axis properties ####
plt.xlabel(r'$\xi\ [\mu m]$',fontdict=font)
plt.ylabel(r'$p_x\ [m_pc]$',fontdict=font)
plt.xlim(-0.2,2.0)
plt.ylim(-0.02,1.1)
plt.xticks([0,0.5,1.0,1.5,2.0],fontsize=font_size); 
plt.yticks(fontsize=font_size);

plt.subplots_adjust(left=0.08, bottom=0.11, right=0.98, top=0.98, wspace=0.2, hspace=0.03)
fig = plt.gcf()
fig.set_size_inches(18, 7.5)
#plt.show()
fig.savefig('./wrap_hamiltonian.png',format='png',dpi=160, transparent=None)
plt.close("all")
