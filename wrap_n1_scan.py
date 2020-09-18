#import sdf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal



def reflec_en(a0):
#a0=np.linspace(1,140,140)
    beta_s=(0.006*a0)**0.5/(1+(0.006*a0)**0.5)
    print('beta_s:',beta_s)
#    beta_s=0.38
#    beta_s=0.05*a0**0.5
    gg_s=1/(1-beta_s**2)**0.5
    print('gg_s:',gg_s)
    phi=3.14*3./8.*a0/1836
    B=phi+1/gg_s
    print(B)
    p=(beta_s*B+(B**2+beta_s**2-1)**0.5)/(1-beta_s**2)
#    p=(beta_s*B)/(1-beta_s**2)
    ek=1836*0.51*((p**2.0+1.)**0.5-1)
    beta_i=2.*beta_s/(1+beta_s**2)
    gg_i=1/(1-beta_i**2)**0.5
    ek_hb = 2*(0.006*a0)/(1+2*(0.006*a0)**0.5)*1836*0.51 # 1836.*0.51*(gg_i-1) 
    ek_2 = 1836*0.51*(0.5*0.006*a0+phi/(1-0.006*a0)+(2*phi)**0.5/(1-0.006*a0)/(0.006*a0)**0.5)
    return ek, ek_hb, ek_2

if __name__ == "__main__":
  print ('This is main of module "test2d.py"')
  ######## Constant defined here ########
  pi        =     3.1415926535897932384626
  q0        =     1.602176565e-19 # C
  m0        =     9.10938291e-31  # kg
  v0        =     2.99792458e8    # m/s^2
  kb        =     1.3806488e-23   # J/K
  mu0       =     4.0e-7*np.pi       # N/A^2
  epsilon0  =     8.8541878176203899e-12 # F/m
  h_planck  =     6.62606957e-34  # J s
  wavelength=     0.8e-6
  frequency =     v0*2*pi/wavelength
  
  exunit    =     m0*v0*frequency/q0
  bxunit    =     m0*frequency/q0
  denunit    =     frequency**2*epsilon0*m0/q0**2
  jalf      =     4*np.pi*epsilon0*m0*v0**3/q0/wavelength**2
  print('electric field unit: '+str(exunit))
  print('magnetic field unit: '+str(bxunit))
  print('density unit nc: '+str(denunit))
  
  font = {'family' : 'monospace',  
          'color'  : 'black',  
          'weight' : 'normal',  
          'size'   : 25,  
          }  
  font_size = 25
##below is for norm colorbar
  class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y)) 
##end for norm colorbar####


  to_path='./'
  set_relativistic =0 

  sim_vs         = np.array([0.632, 0.617, 0.570, 0.266, 0.2046, 0.1441, 0.1411, 0.1268, 0.1095])
  sim_n1         = np.array([1,  3,   5,   7,  10,  15,  20,  30,  40])  # n1 electron density in n_c
  sim_en         = np.array([65, 80, 90,  377, 653, 432, 241, 80,  75])  # in MeV
  sim_charge     = np.array([0.0214, 0.1, 0.128, 0.216, 0.152, 0.109, 0.1, 0.0416, 0.0291]) # total proton charge in nC
  sim_sx         = np.array([0.938, 0.923, 0.938, 0.944, 0.966, 0.946, 0.954, 0.989,  0.983]) # averaged proton s_x

  plt.subplot(1,3,1)
  plt.plot(sim_n1,sim_vs,c='green', label=r'$\beta_1$', linewidth=3,alpha=1,zorder=0)
  plt.scatter(sim_n1,sim_vs,c='green',marker='o',s=300, label=r'$\beta_1$', edgecolors='black', linewidth='2.5',alpha=1,zorder=2)
  ## for hole boring velocity line 
  a0=0.65*70 
  ne=np.linspace(4,50,200)
  mi=12*1836.0/6.0
  pp=1.0/ne*1.0/mi*a0**2
  beta_ho = pp**0.5/(1+pp**0.5)
  plt.plot(ne,beta_ho,':',color='green',linewidth=4)
  ## for relativistic transparency line
  a0=0.65*70
  ne=np.linspace(0.5,8,50)
  mi=1836.0
  K1=(8/np.pi**2/(ne/a0)*((1+8/np.pi**2/(ne/a0)/27.0)**0.5+1))**0.33333
  K2=(8/np.pi**2/(ne/a0)*((1+8/np.pi**2/(ne/a0)/27.0)**0.5-1))**0.33333
  beta_re = 1.0/(1.0+2*np.pi**2*ne/a0)**0.5
  beta_real = K1-K2-1
  plt.plot(ne,beta_re,'--',color='green',linewidth=4)
#  plt.plot(ne,beta_real,'-',color='green',linewidth=4)

  plt.xlabel('$n_{e1}\ [n_c]$',fontdict=font)
  plt.ylabel(r'$\beta_1$',fontdict=font)
  plt.xticks([0,10,20,30,40],fontsize=font_size); 
  plt.yticks(fontsize=font_size);
#  plt.grid(which='major',color='k', linestyle=':', linewidth=0.15)
#  plt.grid(which='minor',color='k', linestyle=':', linewidth=0.15)
  plt.xlim(-4,45) 
#  plt.xscale('log')
  plt.ylim(0,1.02) 


  plt.subplot(1,3,2)
#  plt.plot(a0,ek_reflec,'--',color='red',linewidth=4, label='Theoretic equation',zorder=0)
#  Te_2 = 2.5*a0**0.667*0.51
#  theory_st_1 = Te_1*(np.exp(7.5)*6.5+1)/(np.exp(7.5)-1)
#  theory_st_2 = Te_2*(np.exp(7.5)*6.5+1)/(np.exp(7.5)-1)
#  plt.fill_between(a0,theory_st_1,theory_st_2,color='blue',alpha=.25,label='Thermal model',zorder=0)
  plt.plot(sim_n1,sim_en,c='red',linewidth=3,alpha=1,zorder=0)
  plt.scatter(sim_n1,sim_en,c='red',marker='o',s=300, label=r'$\varepsilon_p^\mathrm{cut}$', edgecolors='black', linewidth='2.5',alpha=1,zorder=2)
  y_line = np.linspace(0,800,100)
  x_line = np.zeros_like(y_line)+6.48
  x_line1= np.zeros_like(y_line)+24.02
  plt.plot(x_line,y_line,':',color='red',linewidth=4)
  plt.plot(x_line1,y_line,':',color='black',linewidth=4)
  plt.xlabel('$n_{e1}\ [n_c]$',fontdict=font)
  plt.ylabel(r'$\varepsilon_p^\mathrm{cut}$'+' [MeV]',fontdict=font)
  plt.xticks([0,10,20,30,40],fontsize=font_size); 
  plt.yticks([0,150,300,450,600,750],fontsize=font_size);
#  plt.grid(which='major',color='k', linestyle=':', linewidth=0.15)
#  plt.grid(which='minor',color='k', linestyle=':', linewidth=0.15)
  plt.xlim(-4,45) 
#  plt.xscale('log')
  plt.ylim(0,760) 
#  plt.legend(loc='best',fontsize=18,framealpha=1)

  plt.subplot(1,3,3)
  plt.plot(sim_n1,sim_sx,c='gray',linewidth=3,alpha=1,zorder=0)
  plt.scatter(sim_n1,sim_sx,c='gray',marker='^',s=340, label=r'$\left<s_x\right>$', edgecolors='black', linewidth='2.5',alpha=1,zorder=2)
  ne     = np.linspace(-2,43,200)
  spin_x = np.zeros_like(ne) + 0.946
  plt.plot(ne,spin_x,':',color='black',linewidth=4)
  plt.xlabel('$n_{e1}\ [n_c]$',fontdict=font)
  plt.ylabel(r'$\left<s_x\right>$',fontdict=font)
  plt.xticks([0,10,20,30,40],fontsize=font_size); 
  plt.yticks(fontsize=font_size);
#  plt.grid(which='major',color='k', linestyle=':', linewidth=0.15)
#  plt.grid(which='minor',color='k', linestyle=':', linewidth=0.15)
  plt.xlim(-4,45) 
#  plt.xscale('log')
  plt.ylim(0,1.12)
#  plt.legend(loc='best',fontsize=18,framealpha=1)

  par1 = plt.twinx()
  plt.plot(sim_n1,sim_charge*1e3,c='dodgerblue',linewidth=3,alpha=1,zorder=0)
  par1.scatter(sim_n1,sim_charge*1e3,c='dodgerblue',marker='o',s=300, label=r'$\mathcal{Q}$', edgecolors='black', linewidth='2.5',alpha=1,zorder=2)
#  par1.legend(loc='best',fontsize=18,framealpha=1.0)
  par1.set_ylim(0,336)
  par1.set_ylabel('$\mathcal{Q}$ [pC]',fontdict=font,color='dodgerblue')
  par1.tick_params(axis='y',labelsize=25,colors='dodgerblue')
  par1.set_yticks([0,60,120,180,240,300]) 


  plt.subplots_adjust(left=0.08, bottom=0.15, right=0.93, top=0.98, wspace=0.3, hspace=None)
    #        plt.text(250,6e9,'t='+str(round(time/1.0e-15,0))+' fs',fontdict=font)
  fig = plt.gcf()
  fig.set_size_inches(21., 6.0)
  fig.savefig('./wrap_n1_scan.png',format='png',dpi=160)
  plt.close("all")
