import sdf
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
import numpy as np
#from numpy import ma
#from matplotlib import colors, ticker, cm
#from matplotlib.mlab import bivariate_normal
#from optparse import OptionParser
#import os
#from colour import Color

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
#print 'electric field unit: '+str(exunit)
#print 'magnetic field unit: '+str(bxunit)
#print 'density unit nc: '+str(denunit)

font = {'family' : 'monospace',  
        'color'  : 'black',  
    	'weight' : 'normal',  
        'size'   : 20,  
       }  

from_path = './spin_a70_n5_fine/'
to_path   = './spin_a70_n5_fine/'
part_name='subset_high_p/ion_s'
part_mass=1836.0*1
data = sdf.read(from_path+"track0200.sdf",dict=True)
print(data.keys())
grid_x = data['Grid/Particles/'+part_name].data[0]/wavelength
grid_y = data['Grid/Particles/'+part_name].data[1]/wavelength
grid_z = data['Grid/Particles/'+part_name].data[2]/wavelength
px = data['Particles/Px/'+part_name].data/(part_mass*m0*v0)
py = data['Particles/Py/'+part_name].data/(part_mass*m0*v0)
pz = data['Particles/Pz/'+part_name].data/(part_mass*m0*v0)
sx = data['Particles/Sx/'+part_name].data
sy = data['Particles/Sy/'+part_name].data
sz = data['Particles/Sz/'+part_name].data
#theta = np.arctan2(py,px)*180.0/np.pi
grid_r = (grid_y**2+grid_z**2)**0.5
gg = (px**2+py**2+pz**2+1)**0.5
Ek = (gg-1)*part_mass*0.51

part13_id = data['Particles/ID/'+part_name].data
part13_id = part13_id[ (px>0.1) & (px<0.18) & (grid_r<0.2) & (grid_x>2) & (grid_x<2.9) ]
#part13_id = part13_id[abs(grid_y)<0.5]
#choice = np.random.choice(range(part13_id.size), 10000, replace=False)
#part13_id = part13_id[choice]
print('part13_id size is ',part13_id.size,' max ',np.max(part13_id),' min ',np.min(part13_id))

######### Parameter you should set ###########
start   =  0  # start time
stop    =  224  # end time
step    =  1  # the interval or step

#  if (os.path.isdir('jpg') == False):
#    os.mkdir('jpg')
######### Script code drawing figure ################
part_id = part13_id
for n in range(start,stop+step,step):
    #### header data ####
    data = sdf.read(from_path+'track'+str(n).zfill(4)+".sdf",dict=True)
    header=data['Header']
    time=header['time']
    tt_id = data['Particles/ID/'+part_name].data
    grid_y = data['Grid/Particles/'+part_name].data[1]/wavelength
    grid_z = data['Grid/Particles/'+part_name].data[2]/wavelength
    grid_r = (grid_y**2+grid_z**2)**0.5
    tt_id = tt_id[grid_r<0.2] 
    part_id = np.intersect1d(tt_id, part_id)
    
    print('Particle_ID size is ',part_id.size,' max ',np.max(part_id),' min ',np.min(part_id))

print('After intersecting with all.sdf')
print('Particle_ID size is ',part_id.size,' max ',np.max(part_id),' min ',np.min(part_id))


px_d = np.zeros([part_id.size,stop-start+1])
py_d = np.zeros([part_id.size,stop-start+1])
pz_d = np.zeros([part_id.size,stop-start+1])
xx_d = np.zeros([part_id.size,stop-start+1])
yy_d = np.zeros([part_id.size,stop-start+1])
zz_d = np.zeros([part_id.size,stop-start+1])
sx_d = np.zeros([part_id.size,stop-start+1])
sy_d = np.zeros([part_id.size,stop-start+1])
sz_d = np.zeros([part_id.size,stop-start+1])
ww_d = np.zeros([part_id.size,stop-start+1])
for n in range(start,stop+step,step):
    #### header data ####
    data = sdf.read(from_path+'track'+str(n).zfill(4)+".sdf",dict=True)
    px = data['Particles/Px/'+part_name].data/(part_mass*m0*v0)
    py = data['Particles/Py/'+part_name].data/(part_mass*m0*v0)
    pz = data['Particles/Pz/'+part_name].data/(part_mass*m0*v0)
    sx = data['Particles/Sx/'+part_name].data/(part_mass*m0*v0)
    sy = data['Particles/Sy/'+part_name].data/(part_mass*m0*v0)
    sz = data['Particles/Sz/'+part_name].data/(part_mass*m0*v0)
    grid_x = data['Grid/Particles/'+part_name].data[0]/wavelength
    grid_y = data['Grid/Particles/'+part_name].data[1]/wavelength
    grid_z = data['Grid/Particles/'+part_name].data[2]/wavelength
    ww = data['Particles/Weight/'+part_name].data
    temp_id =  data['Particles/ID/'+part_name].data

    px = px[np.in1d(temp_id,part_id)]
    py = py[np.in1d(temp_id,part_id)]
    pz = pz[np.in1d(temp_id,part_id)]
    sx = sx[np.in1d(temp_id,part_id)]
    sy = sy[np.in1d(temp_id,part_id)]
    sz = sz[np.in1d(temp_id,part_id)]
    grid_x = grid_x[np.in1d(temp_id,part_id)]
    grid_y = grid_y[np.in1d(temp_id,part_id)]
    grid_z = grid_z[np.in1d(temp_id,part_id)]
    ww = ww[np.in1d(temp_id,part_id)]
    temp_id = temp_id[np.in1d(temp_id,part_id)]

    for ie in range(part_id.size):
        px_d[ie,n-start] = px[temp_id==part_id[ie]]
        py_d[ie,n-start] = py[temp_id==part_id[ie]]
        pz_d[ie,n-start] = pz[temp_id==part_id[ie]]
        xx_d[ie,n-start] = grid_x[temp_id==part_id[ie]]
        yy_d[ie,n-start] = grid_y[temp_id==part_id[ie]]
        zz_d[ie,n-start] = grid_z[temp_id==part_id[ie]]
        sx_d[ie,n-start] = sx[temp_id==part_id[ie]]
        sy_d[ie,n-start] = sy[temp_id==part_id[ie]]
        sz_d[ie,n-start] = sz[temp_id==part_id[ie]]
        ww_d[ie,n-start] = ww[temp_id==part_id[ie]]
    print('finised '+part_name+' '+str(round(100.0*(n-start+step)/(stop-start+step),4))+'%')

np.savetxt(to_path+part_name[-5:]+'_px.txt',px_d)
np.savetxt(to_path+part_name[-5:]+'_py.txt',py_d)
np.savetxt(to_path+part_name[-5:]+'_pz.txt',pz_d)
np.savetxt(to_path+part_name[-5:]+'_xx.txt',xx_d)
np.savetxt(to_path+part_name[-5:]+'_yy.txt',yy_d)
np.savetxt(to_path+part_name[-5:]+'_zz.txt',zz_d)
np.savetxt(to_path+part_name[-5:]+'_sx.txt',sx_d)
np.savetxt(to_path+part_name[-5:]+'_sy.txt',sy_d)
np.savetxt(to_path+part_name[-5:]+'_sz.txt',sz_d)
np.savetxt(to_path+part_name[-5:]+'_ww.txt',ww_d)
