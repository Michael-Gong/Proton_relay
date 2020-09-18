import sdf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
 

upper = matplotlib.cm.viridis(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
    lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_viridis = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
 
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
 
  font_size=25 
  font_size2=20
  marker_size = 0.1 
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

  def pxpy_to_energy(gamma, weight):
      binsize = 200
      en_grid = np.linspace(0,1000,200)
      en_bin  = np.linspace(0,1000,201)
      en_value = np.zeros_like(en_grid) 
      for i in range(binsize):
        en_value[i] = sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])/(en_bin[-1]-en_bin[-2])
      return (en_grid, en_value)

  def spin_to_bin(spin, weight):
      binsize = 100
      en_grid = np.linspace(0,1,100)
      en_bin  = np.linspace(0,1,101)
      en_value = np.zeros_like(en_grid) 
      for i in range(binsize):
        en_value[i] = sum(weight[ (en_bin[i]<=spin) & (spin<en_bin[i+1]) ])/(en_bin[-1]-en_bin[-2])
      return (en_grid, en_value)


  from_path='./spin_a70_n15_n5_fine_2/'
  to_path  ='./'

  ######### Parameter you should set ###########
  plt.subplot(2,2,2)
  data_bin1   = np.load(from_path+'dNds_bin.npy') 
  data_time1  = np.load(from_path+'dNds_t.npy') 
  data_value = np.load(from_path+'dNds_data.npy')
#  data_value2 = np.zeros([113,101])
#  data_value2 =  
  print(data_value[:,-3:])
  print(data_time1)
  data_bin1 = np.linspace(0,1.003,100)
  print(data_bin1)
  eee = np.max(data_value)
  print(eee/4e10)
  data_value[data_value>4e10] =4e10
  data_value[:,:-1] = data_value[:,1:]
  data_time1, data_bin1 = np.meshgrid(data_time1,data_bin1)
#  plt.pcolormesh(data_time1, data_bin1, data_value.T, norm=colors.Normalize(vmin=0, vmax=eee), cmap='magma')
  plt.pcolormesh(data_time1, data_bin1, data_value.T, norm=colors.LogNorm(vmin=1e7, vmax=4e10), cmap=mycolor_viridis)
#  cbar=plt.colorbar(pad=0.01,ticks=[1e7,1e8,1e9,1e10])
#  cbar.set_label('$dN/ds_x$',fontdict=font)
#  cbar.ax.tick_params(labelsize=font_size2)
#  plt.pcolormesh(x, y*10./3., data_no.T, norm=colors.LogNorm(vmin=10, vmax=1e4), cmap=mycolor_viridis)
  plt.xlabel(r'$t\ [\mathrm{fs}]$',fontdict=font)
  plt.ylabel(r'$s_x$',fontdict=font)
  plt.xticks(fontsize=font_size); 
  plt.yticks(fontsize=font_size);
#  plt.yscale('log')
#  plt.ylim(0.0,1.003)
  plt.grid(linestyle=':', linewidth=0.4, color='grey')
  plt.xlim(11,74)
#  plt.legend(loc='best',fontsize=font_size2-15,framealpha=1.0)

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
  ion_ex=np.loadtxt(from_path+part_name+'_ex.txt')
  ion_ey=np.loadtxt(from_path+part_name+'_ey.txt')
  ion_ez=np.loadtxt(from_path+part_name+'_ez.txt')
  ion_bx=np.loadtxt(from_path+part_name+'_bx.txt')
  ion_by=np.loadtxt(from_path+part_name+'_by.txt')
  ion_bz=np.loadtxt(from_path+part_name+'_bz.txt')
  
  ion_tt=np.linspace(0.333333,75,225)  
  ion_pp=(ion_px**2+ion_py**2+ion_pz**2)**0.5
  ion_gg=(ion_pp**2+1)**0.5
  ion_ek=((ion_px**2+ion_py**2+ion_pz**2+1)**0.5-1)*918.0
  ion_ss=(ion_sx**2+ion_sy**2+ion_sz**2)**0.5
  ion_vx=ion_px/ion_gg
  ion_vy=ion_py/ion_gg
  ion_vz=ion_pz/ion_gg

  lwdth = 3

  n=0
  plt.subplot(5,2,1)
  plt.plot(ion_tt, ion_sx[n,:], linestyle='-',color='crimson',linewidth=lwdth)
  plt.plot(ion_tt, ion_ss[n,:], linestyle='--',color='k',linewidth=lwdth)
  plt.ylabel('$s_x$',fontdict=font)
#  plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
  plt.xticks(fontsize=0.001); 
  plt.yticks(fontsize=font_size);
  plt.xlim(11,74)
  plt.ylim(0.89,1.003)
  plt.grid(linestyle=':', linewidth=0.4, color='grey')

  plt.subplot(5,2,3)
  plt.plot(ion_tt, ion_sy[n,:], linestyle='-',color='mediumseagreen',linewidth=lwdth)
  plt.plot(ion_tt, ion_sz[n,:], linestyle='-',color='dodgerblue',linewidth=lwdth)
  plt.ylabel('$s_{y,z}$',fontdict=font)
#  plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
  plt.xticks(fontsize=0.001); 
  plt.yticks(fontsize=font_size);
  plt.xlim(11,74)
  plt.ylim(-0.37,0.37)
  plt.grid(linestyle=':', linewidth=0.4, color='grey')

  af=1.79
  O_x = -1.0/1836.0*((af+1/ion_gg)*ion_bx-af*ion_gg/(1+ion_gg)*(ion_vx*ion_bx+ion_vy*ion_by+ion_vz*ion_bz)*ion_vx-(af+1/(1+ion_gg))*(ion_vy*ion_ez-ion_vz*ion_ey))
  O_y = -1.0/1836.0*((af+1/ion_gg)*ion_by-af*ion_gg/(1+ion_gg)*(ion_vx*ion_bx+ion_vy*ion_by+ion_vz*ion_bz)*ion_vy-(af+1/(1+ion_gg))*(ion_vz*ion_ex-ion_vx*ion_ez))
  O_z = -1.0/1836.0*((af+1/ion_gg)*ion_bz-af*ion_gg/(1+ion_gg)*(ion_vx*ion_bx+ion_vy*ion_by+ion_vz*ion_bz)*ion_vz-(af+1/(1+ion_gg))*(ion_vx*ion_ey-ion_vy*ion_ex))
  d_sx = O_y*ion_sz-O_z*ion_sy
  d_sy = O_z*ion_sx-O_x*ion_sz
  d_sz = O_x*ion_sy-O_y*ion_sx

  O_x_non = -1.0/1836.0*(af+1.0)*ion_bx
  O_y_non = -1.0/1836.0*(af+1.0)*ion_by
  O_z_non = -1.0/1836.0*(af+1.0)*ion_bz
  d_sx_non = O_y_non*ion_sz-O_z_non*ion_sy
  d_sy_non = O_z_non*ion_sx-O_x_non*ion_sz
  d_sz_non = O_x_non*ion_sy-O_y_non*ion_sx

  dt_fac = 2*np.pi/3.333333
  
  plt.subplot(5,2,5)
  y1 = np.zeros_like(ion_tt)
  y2 = d_sy[n,:]*dt_fac
  plt.fill_between(ion_tt,y1,y2,where=y1<=y2,color='red',alpha=.5,zorder=0)
  plt.fill_between(ion_tt,y1,y2,where=y1>y2,color='dodgerblue',alpha=.6,zorder=0)
  plt.plot(ion_tt, d_sy[n,:]*dt_fac, linestyle='-',color='mediumseagreen',linewidth=lwdth)
  plt.plot(ion_tt, d_sy_non[n,:]*dt_fac, linestyle=':',color='k',linewidth=lwdth)
#  plt.plot(ion_tt, d_sz[n,:], linestyle='-',color='dodgerblue',linewidth=lwdth)
  plt.ylabel('$ds_y/dt\ [\mathrm{fs}^{-1}]$',fontdict=font)
#  plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
  plt.xticks(fontsize=0.001); 
  plt.yticks(fontsize=font_size);
  plt.xlim(11,74)
  plt.ylim(-0.13,0.13)
  plt.grid(linestyle=':', linewidth=0.4, color='grey')

  plt.subplot(5,2,7)
  y1 = np.zeros_like(ion_tt)
  y2 = d_sz[n,:]*dt_fac
  plt.fill_between(ion_tt,y1,y2,where=y1<=y2,color='red',alpha=.5,zorder=0)
  plt.fill_between(ion_tt,y1,y2,where=y1>y2,color='dodgerblue',alpha=.6,zorder=0)
  plt.plot(ion_tt, d_sz[n,:]*dt_fac, linestyle='-',color='dodgerblue',linewidth=lwdth)
  plt.plot(ion_tt, d_sz_non[n,:]*dt_fac, linestyle=':',color='k',linewidth=lwdth)
  plt.ylabel('$ds_z/dt\ [\mathrm{fs}^{-1}]$',fontdict=font)
#  plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
  plt.xticks(fontsize=0.001); 
  plt.yticks(fontsize=font_size);
  plt.xlim(11,74)
  plt.ylim(-0.13,0.13)
  plt.grid(linestyle=':', linewidth=0.4, color='grey')

  plt.subplot(5,2,9)
#  plt.plot(ion_tt, d_sx[n,:], linestyle='-',color='crimson',linewidth=lwdth)
#  plt.plot(ion_tt, d_sy[n,:], linestyle='-',color='mediumseagreen',linewidth=lwdth)
  y1 = np.zeros_like(ion_tt)
  y2 = d_sx[n,:]*dt_fac
  plt.fill_between(ion_tt,y1,y2,where=y1<=y2,color='red',alpha=.5,zorder=0)
  plt.fill_between(ion_tt,y1,y2,where=y1>y2,color='dodgerblue',alpha=.6,zorder=0)
  plt.plot(ion_tt, d_sx[n,:]*dt_fac, linestyle='-',color='crimson',linewidth=lwdth)
  plt.plot(ion_tt, d_sx_non[n,:]*dt_fac, linestyle=':',color='k',linewidth=lwdth)
  plt.ylabel('$ds_x/dt\ [\mathrm{fs}^{-1}]$',fontdict=font)
  plt.xlabel('$t\ [\mathrm{fs}]$',fontdict=font)
  plt.xticks(fontsize=font_size); 
  plt.yticks(fontsize=font_size);
  plt.xlim(11,74)
  plt.ylim(-0.022,0.022)
  plt.grid(linestyle=':', linewidth=0.4, color='grey')


  plt.subplots_adjust(left=0.12, bottom=0.08, right=0.99, top=0.99, wspace=0.18, hspace=0.05)

  fig = plt.gcf()
  fig.set_size_inches(15, 14)
  fig.savefig(to_path+'wrap_spin_evolution.png',format='png',dpi=160)
  plt.close("all")
  print(to_path+'wrap_spin_evolution.png')
