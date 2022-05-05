import numpy as np

graph_file = 'hcpl_graph.npz'
b0_file = 'hcpl_b0.nii.gz'

output_file = 'hcpl_odf.nii.gz'

import os

import scipy.ndimage as ndi

''' ----------------------------------------
    Create graph or read from file if exists 
    ----------------------------------------'''
if (os.path.isfile(graph_file)):
    print('The graph file found. Loading the graph.')
    loaded = np.load(graph_file)
    nbh = np.float32(loaded['nbh'])
    nbh_pdf = np.float32(loaded['nbh_pdf'])
else:
    print('The graph file does not exist.')

print('Graph is ready.')

normnbh = np.sqrt(np.sum(nbh*nbh, axis=1))
nbh[:,0] = nbh[:,0] / normnbh
nbh[:,1] = nbh[:,1] / normnbh
nbh[:,2] = nbh[:,2] / normnbh

from dipy.core.sphere import Sphere
pmfsphere = Sphere(xyz=nbh)  

rep362 = np.loadtxt('repulsion362_hemisphere.txt')
newsphere = Sphere(xyz=rep362)

'''
# One way is to calculate spherical harmonics first, then resample the spherical function
import dipy.reconst.shm
sh = dipy.reconst.shm.sf_to_sh(nbh_pdf,pmfsphere)
newodf = dipy.reconst.shm.sh_to_sf(sh,newsphere)
'''

'''
#Second way is to interpolate spherical function using scipy
from scipy.interpolate import LSQSphereBivariateSpline,SmoothSphereBivariateSpline
theta = pmfsphere.theta 
phi = pmfsphere.phi
phi[phi<0] += 2*np.pi
theta[theta>=np.pi] = np.pi-0.00001
phi[phi>=2*np.pi] = 2*np.pi-0.00001

newtheta = newsphere.theta 
newphi = newsphere.phi
newphi[newphi<0] += 2*np.pi
newtheta[newtheta>=np.pi] = np.pi-0.00001
newphi[newphi>=2*np.pi] = 2*np.pi-0.00001

from scipy.interpolate import SmoothSphereBivariateSpline
newodf = np.zeros((nbh_pdf.shape[0],nbh_pdf.shape[1],nbh_pdf.shape[2],len(newtheta)),dtype=np.float32)
for i in range(nbh_pdf.shape[0]):
    for j in range(nbh_pdf.shape[1]):
        for k in range(nbh_pdf.shape[2]):
            data = nbh_pdf[i,j,k,:]
            lut = SmoothSphereBivariateSpline(theta, phi, data, s=0.5)
            new_data = lut(newtheta, newphi, grid=False)
            newodf[i,j,k,:] = new_data
'''

#Third way is to interpolate spherical function using scipy LSQSphereBivariateSpline
from scipy.interpolate import LSQSphereBivariateSpline,SmoothSphereBivariateSpline
theta = pmfsphere.theta 
phi = pmfsphere.phi
phi[phi<0] += 2*np.pi
theta[theta>=np.pi] = np.pi-0.00001
phi[phi>=2*np.pi] = 2*np.pi-0.00001

newtheta = newsphere.theta 
newphi = newsphere.phi
newphi[newphi<0] += 2*np.pi
newtheta[newtheta>=np.pi] = np.pi-0.00001
newphi[newphi>=2*np.pi] = 2*np.pi-0.00001

knotst = np.linspace(0, np.pi, num=5)
knotsp = np.linspace(0, 2*np.pi, num=9)
knotst[0] += .0001
knotst[-1] -= .0001
knotsp[0] += .0001
knotsp[-1] -= .0001


newodf = np.zeros((nbh_pdf.shape[0],nbh_pdf.shape[1],nbh_pdf.shape[2],len(newtheta)),dtype=np.float32)
for i in range(nbh_pdf.shape[0]):
    for j in range(nbh_pdf.shape[1]):
        for k in range(nbh_pdf.shape[2]):
            data = nbh_pdf[i,j,k,:]
            lut = LSQSphereBivariateSpline(theta, phi, data, knotst, knotsp)
            #lut = SmoothSphereBivariateSpline(theta, phi, data, s=0.8)
            new_data = lut(newtheta, newphi, grid=False)
            newodf[i,j,k,:] = new_data

import nibabel as nib
b0nii = nib.load(b0_file)
odfnii = nib.Nifti1Image(newodf, b0nii.affine)
nib.save(odfnii, output_file)

#Visualize

from dipy.viz import window,actor
scene = window.Scene()

#fs_actor = actor.odf_slicer(nbh_pdf[60:70,60:70,:],sphere=pmfsphere,colormap='plasma',scale=0.4)
#fs_actor = actor.odf_slicer(newodf[60:70,60:70,:],sphere=newsphere,colormap='plasma',scale=0.4)
fs_actor = actor.odf_slicer(newodf[40:50,60:70,:],sphere=newsphere,colormap='plasma',scale=0.4)
actor.
fs_actor.display(z=50)
scene.add(fs_actor)

window.show(scene)

