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
  plt.subplot(2,2,1)
  data_value = np.zeros([113,100])
  data_time1 = np.zeros(113)
  data_bin1  = np.zeros(100) 
  if -1 > 0: 
    part_name='subset_high_p/ion_s'
    part_mass=1836.0*1
    data = sdf.read(from_path+"track0200.sdf",dict=True)
    grid_x = data['Grid/Particles/'+part_name].data[0]/wavelength
    grid_y = data['Grid/Particles/'+part_name].data[1]/wavelength
    grid_z = data['Grid/Particles/'+part_name].data[2]/wavelength
    px = data['Particles/Px/'+part_name].data/(part_mass*m0*v0)
    py = data['Particles/Py/'+part_name].data/(part_mass*m0*v0)
    pz = data['Particles/Pz/'+part_name].data/(part_mass*m0*v0)
    theta = np.arctan2((py**2+pz**2)**0.5,px)*180.0/np.pi
    grid_r = (grid_y**2+grid_z**2)**0.5
    gg = (px**2+py**2+pz**2+1)**0.5
    Ek = (gg-1)*part_mass*0.51
    part13_id = data['Particles/ID/'+part_name].data
    part13_id = part13_id[ (Ek>100.0) & (theta<10.0) ]
    #part13_id = part13_id[abs(grid_y)<0.5]
    #choice = np.random.choice(range(part13_id.size), 10000, replace=False)
    #part13_id = part13_id[choice]
    print('part13_id size is ',part13_id.size,' max ',np.max(part13_id),' min ',np.min(part13_id))

    for n in range(0,226,2):
      data = sdf.read(from_path+'track'+str(n).zfill(4)+".sdf",dict=True)
      header=data['Header']
      print(header['time']/1e-15)
      data_time1[n//2] = header['time']/1e-15
      ion_x = data['Grid/Particles/subset_high_p/ion_s'].data[0]/1e-6
      ion_y = data['Grid/Particles/subset_high_p/ion_s'].data[1]/1e-6
      ion_z = data['Grid/Particles/subset_high_p/ion_s'].data[2]/1e-6
      ion_px = data['Particles/Px/subset_high_p/ion_s'].data/(1836*m0*v0)
      ion_py = data['Particles/Py/subset_high_p/ion_s'].data/(1836*m0*v0)
      ion_pz = data['Particles/Pz/subset_high_p/ion_s'].data/(1836*m0*v0)
      ion_sx = data['Particles/Sx/subset_high_p/ion_s'].data
      ion_sy = data['Particles/Sy/subset_high_p/ion_s'].data
      ion_sz = data['Particles/Sz/subset_high_p/ion_s'].data
      ion_ww = data['Particles/Weight/subset_high_p/ion_s'].data
      temp_id = data['Particles/ID/subset_high_p/ion_s'].data
      gg = (ion_px**2+ion_py**2+ion_pz**2+1)**0.5
      ek = (gg-1)*1836*0.51 
      ion_theta = np.arctan2((ion_py**2+ion_pz**2)**0.5,ion_px)/np.pi*180
      #condition = ((ion_y**2+ion_z**2)**0.5<2.0)&(ion_theta<10)
      condition = np.in1d(temp_id,part13_id) #& (ek>30.0)
      ion_x = ion_x[condition] 
      ion_y = ion_y[condition] 
      ion_z = ion_z[condition] 
      ion_px = ion_px[condition] 
      ion_py = ion_py[condition] 
      ion_pz = ion_pz[condition]
      ion_sx = ion_sx[condition] 
      ion_sy = ion_sy[condition] 
      ion_sz = ion_sz[condition]
      ion_ww = ion_ww[condition]  
      gg = (ion_px**2+ion_py**2+ion_pz**2+1)**0.5
      ek = (gg-1)*1836*0.51 
      if np.size(ek) == 0:
          continue 
      value_x, value_y = spin_to_bin(ion_sx, ion_ww)
      data_bin1   = value_x
      data_value[n//2,:] = value_y

      np.save(from_path+'dNds_bin',data_bin1) 
      np.save(from_path+'dNds_t',data_time1) 
      np.save(from_path+'dNds_data',data_value) 
    #  plt.plot(grid_en,value_en,color='dodgerblue',linewidth=3,label='time '+str(header['time']/1e-15))
      #plt.plot(value_x,value_y,linewidth=3,label='time '+str(int(header['time']/1e-15)))

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
  cbar=plt.colorbar(pad=0.01,ticks=[1e7,1e8,1e9,1e10])
  cbar.set_label('$dN/ds_x$',fontdict=font)
  cbar.ax.tick_params(labelsize=font_size2)
#  plt.pcolormesh(x, y*10./3., data_no.T, norm=colors.LogNorm(vmin=10, vmax=1e4), cmap=mycolor_viridis)

  plt.xlabel(r'$t\ [\mathrm{fs}]$',fontdict=font)
  plt.ylabel(r'$s_x$',fontdict=font)
  plt.xticks(fontsize=font_size); 
  plt.yticks(fontsize=font_size);
#  plt.yscale('log')
#  plt.ylim(0.0,1.01)
  plt.grid(linestyle=':', linewidth=0.4, color='grey')
#    plt.ylim(1e1,1e5)
#  plt.legend(loc='best',fontsize=font_size2-15,framealpha=1.0)


  plt.subplots_adjust(left=0.16, bottom=0.15, right=0.98, top=0.97, wspace=0.1, hspace=0.1)

  fig = plt.gcf()
  fig.set_size_inches(16, 13)
  fig.savefig(to_path+'wrap_spin_spectrum.png',format='png',dpi=160)
  plt.close("all")
  print(to_path+'wrap_spin_spectrum.png')
