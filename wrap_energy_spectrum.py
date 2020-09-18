import sdf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
  
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

  def frac_to_spin(gamma, weight, spin):
      weight_spin = weight*spin
      binsize = 30
      en_grid = np.linspace(0,500,30)
      en_bin  = np.linspace(0,500,31)
      en_value = np.zeros_like(en_grid) 
      for i in range(binsize):
        print(en_bin[i])
        if sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ]) ==0:
            continue
        en_value[i] = sum(weight_spin[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])/sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])
      return (en_grid, en_value)

  def pxpy_to_energy(gamma, weight):
      binsize = 200
      en_grid = np.linspace(0,1000,200)
      en_bin  = np.linspace(0,1000,201)
      en_value = np.zeros_like(en_grid) 
      for i in range(binsize):
        en_value[i] = sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])/(en_bin[-1]-en_bin[-2])
      return (en_grid, en_value)

  from_path='./spin_a70_n15_n5_revised/'
  to_path  ='./'

  ######### Parameter you should set ###########
  n=5
  data = sdf.read(from_path+'track'+str(n).zfill(4)+".sdf",dict=True)
  header=data['Header']
  print(header['time']/1e-15)
  ion_x = data['Grid/Particles/subset_high_p/ion_s'].data[0]/1e-6
  ion_y = data['Grid/Particles/subset_high_p/ion_s'].data[1]/1e-6
  ion_z = data['Grid/Particles/subset_high_p/ion_s'].data[2]/1e-6
  ion_px = data['Particles/Px/subset_high_p/ion_s'].data/(1836*m0*v0)
  ion_py = data['Particles/Py/subset_high_p/ion_s'].data/(1836*m0*v0)
  ion_pz = data['Particles/Pz/subset_high_p/ion_s'].data/(1836*m0*v0)
  ion_ww = data['Particles/Weight/subset_high_p/ion_s'].data
  ion_theta = np.arctan2((ion_py**2+ion_pz**2)**0.5,ion_px)/np.pi*180
  condition = ((ion_y**2+ion_z**2)**0.5<2.0)&(ion_theta<10.0)
  ion_x = ion_x[condition] 
  ion_y = ion_y[condition] 
  ion_z = ion_z[condition] 
  ion_px = ion_px[condition] 
  ion_py = ion_py[condition] 
  ion_pz = ion_pz[condition]
  ion_ww = ion_ww[condition]  
  gg = (ion_px**2+ion_py**2+ion_pz**2+1)**0.5
  ek = (gg-1)*1836*0.51 
  grid_en, value_en = pxpy_to_energy(ek, ion_ww) 
 
  plt.subplot(2,1,2)
  plt.plot(grid_en,value_en,color='dodgerblue',linewidth=3,label='t = 50 fs')


  n=9
  data = sdf.read(from_path+'track'+str(n).zfill(4)+".sdf",dict=True)
  header=data['Header']
  print(header['time']/1e-15)
  ion_x = data['Grid/Particles/subset_high_p/ion_s'].data[0]/1e-6
  ion_y = data['Grid/Particles/subset_high_p/ion_s'].data[1]/1e-6
  ion_z = data['Grid/Particles/subset_high_p/ion_s'].data[2]/1e-6
  ion_px = data['Particles/Px/subset_high_p/ion_s'].data/(1836*m0*v0)
  ion_py = data['Particles/Py/subset_high_p/ion_s'].data/(1836*m0*v0)
  ion_pz = data['Particles/Pz/subset_high_p/ion_s'].data/(1836*m0*v0)
  ion_ww = data['Particles/Weight/subset_high_p/ion_s'].data
  ion_sx = data['Particles/Sx/subset_high_p/ion_s'].data
  ion_sy = data['Particles/Sy/subset_high_p/ion_s'].data
  ion_sz = data['Particles/Sz/subset_high_p/ion_s'].data
  ion_theta = np.arctan2((ion_py**2+ion_pz**2)**0.5,ion_px)/np.pi*180

  condition = ((ion_y**2+ion_z**2)**0.5<2.0)&(ion_theta<10.0)
  ion_x = ion_x[condition] 
  ion_y = ion_y[condition] 
  ion_z = ion_z[condition] 
  ion_px = ion_px[condition] 
  ion_py = ion_py[condition] 
  ion_pz = ion_pz[condition]
  ion_ww = ion_ww[condition]  
  ion_sx = ion_sx[condition] 
  ion_sy = ion_sy[condition] 
  ion_sz = ion_sz[condition]
  gg = (ion_px**2+ion_py**2+ion_pz**2+1)**0.5
  ek = (gg-1)*1836*0.51 
  grid_en, value_en = pxpy_to_energy(ek, ion_ww) 
 
  plt.plot(grid_en,value_en,color='crimson',linewidth=3,label='t = 83 fs')

  plt.xlabel(r'$\varepsilon_p$ [MeV]',fontdict=font)
  plt.ylabel(r'$dN/d\varepsilon_p$ [MeV$^{-1}$]',fontdict=font)
  plt.xticks(fontsize=font_size); 
  plt.yticks(fontsize=font_size);
  plt.yscale('log')
  plt.xlim(0,620)
  plt.grid(linestyle=':', linewidth=0.4, color='grey')
#    plt.ylim(1e1,1e5)
  plt.legend(loc='best',fontsize=font_size2,framealpha=1.0)

  plt.subplot(2,1,1)
  width = 15
  grid_en, value_en = frac_to_spin(ek, ion_ww, ion_sx)
  pl=plt.bar(grid_en, value_en, width, color='crimson',edgecolor='black',linewidth=2,alpha=0.7,label=r'$dp_x/dt$')
  value_complement = 1- value_en
  pl=plt.bar(grid_en, value_complement, width, bottom=value_en, color='white',edgecolor='black',linewidth=2,alpha=0.7,label=r'$dp_x/dt$')
  #plt.xlabel(r'$\varepsilon_p$ [MeV]',fontdict=font)
#  plt.ylabel(r'$\mathcal{S}_x$',fontdict=font)
  plt.ylabel(r'$\left<s_x\right>$',fontdict=font)
  plt.xticks(fontsize=0.001); 
  plt.yticks(fontsize=font_size);
  plt.xlim(0,620)

  plt.subplots_adjust(left=0.18, bottom=0.15, right=0.98, top=0.97, wspace=0.1, hspace=0.05)



  fig = plt.gcf()
  fig.set_size_inches(7, 6.5)
  fig.savefig(to_path+'wrap_energy_spectrum.png',format='png',dpi=160)
  plt.close("all")
  print(to_path+'wrap_energy_spectrum.png')
