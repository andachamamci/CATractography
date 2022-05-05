
dwi_file = 'prep.dwi.hcpl.nii.gz'
bval_file = 'prep.bvalues.hcpl.txt'
bvec_file = 'prep.gradients.hcpl.txt'
graph_file = 'hcpl_graph.npz'
b0_file = 'hcpl_b0.nii.gz'
seed_file = 'prep.inject.nii.gz'

ConvertTo2Shells = True
#FA mask biraz dilate edilerek denenmeli !!!
FA_THRESHOLD = 0.22 #The graph_file should be deleted if you change this!!!

import nibabel as nib

img = nib.load(dwi_file)
b0imgdata = img.dataobj[:,:,:,0]
b0img = nib.Nifti1Image(b0imgdata, img.affine)
nib.save(b0img,b0_file)

import numpy as np

if (ConvertTo2Shells):
    # Convert bvalues to 3 shells:
    print('Converting data to 2 shells')    
    bvaldata = np.genfromtxt(bval_file, delimiter=' ')
    #The delimiter is 2 spaces for old HCP bvals
    if (np.isnan(bvaldata).any()):
        bvaldata = np.genfromtxt(bval_file, delimiter='  ')
 
    bvaldata[bvaldata<500] = 0
    bvaldata[(bvaldata>5500) & (bvaldata<6500)] = 6000
    bvaldata[(bvaldata>11500) & (bvaldata<12500)] = 12000
    #bvaldata[bvaldata>2500] = 3000
    
    np.savetxt('new_' + bval_file, bvaldata, delimiter=' ',fmt='%d')
    bval_file = 'new_' + bval_file
#####################

bvf = np.loadtxt(bvec_file)

for i in range(bvf.shape[0]):
    if (np.linalg.norm(bvf[i,:]) > 0.1):
        bvf[i,:] = bvf[i,:] / np.linalg.norm(bvf[i,:]) 
    else:
        bvf[i,:] = 0

np.savetxt('new_' + bvec_file, bvf, delimiter=' ',fmt='%4.3f')
bvec_file = 'new_' + bvec_file
##########################################
# TISSUE PRIORS using FA map

import dipy.reconst.dti as dti

from dipy.segment.mask import median_otsu

data = img.get_fdata()

#maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=False, dilate=2)
maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=False)
print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)
nib.save(mask_img, 'median_otsu_mask.nii.gz')

#maskdata = data

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

bvals,bvecs = read_bvals_bvecs(bval_file,bvec_file)
gtab = gradient_table(bvals, bvecs)
tenmodel = dti.TensorModel(gtab)

tenfit = tenmodel.fit(maskdata)

from dipy.reconst.dti import fractional_anisotropy

FA = fractional_anisotropy(tenfit.evals)

FA[np.isnan(FA)] = 0

fa_img = nib.Nifti1Image(FA.astype(np.float32), img.affine)
nib.save(fa_img, 'tensor_fa.nii.gz')

##########################################

import catractography as CAT
import os

import scipy.ndimage as ndi

''' ----------------------------------------
    Create graph or read from file if exists 
    ----------------------------------------'''
if (os.path.isfile(graph_file)):
    print('The graph file found. Loading the graph.')
    loaded = np.load(graph_file)
    nbh = loaded['nbh']
    nbh_pdf = np.float32(loaded['nbh_pdf'])
else:
    print('The graph file does not exist. Creating the graph.')
    csd_odf, sphere = CAT.fit_csd_model(dwi_file,bval_file,bvec_file,UseParallel=False,Print_Response=True, roi_radius=30, fa_thr=0.6)
    tissueprior = np.ones(b0imgdata.shape,dtype=np.float32)
    tissueprior[FA<FA_THRESHOLD] = 0
    
    #closed_tissueprior = ndi.binary_closing(tissueprior).astype(tissueprior.dtype)
    #nbh_pdf, nbh = CAT.create_graph(csd_odf, sphere, closed_tissueprior)
    
    #dilated_tissueprior = ndi.binary_dilation(tissueprior).astype(tissueprior.dtype)
    #nbh_pdf, nbh = CAT.create_graph(csd_odf, sphere, dilated_tissueprior)
    
    nbh_pdf, nbh = CAT.create_graph(csd_odf, sphere, tissueprior)
    np.savez_compressed(graph_file, nbh_pdf = nbh_pdf, nbh=nbh)      

print('Graph is ready.')
''' -------------------
    Arrange Seed Labels
    -------------------'''
seedimg = nib.load(seed_file)
seed_data = seedimg.get_fdata()

seeds = np.zeros(seed_data.shape,dtype=np.int32)
lbl=1
for j in range(seed_data.shape[1]):
    for i in range(seed_data.shape[0]):
        for k in range(seed_data.shape[2]):
            if (seed_data[i,j,k]>0):
                seeds[i,j,k]=lbl
                #lbl=lbl+1

print('Seed labels ready.')

''' --------------------------------------------
    Perform Cellular Automata based tractography
    --------------------------------------------'''
import time
CA = CAT.CATractography()
CA.set_gpu_graph(nbh_pdf,nbh)
CA.set_gpu_variables(seeds)
ti=time.time()
CA.gpu_N_iterations(130)
tf=time.time()
CA.get_gpu_variables()

''' --------------------------------------------
    Results
    --------------------------------------------'''
from nilearn import plotting
connimg = nib.Nifti1Image(CA.conn, seedimg.affine)
plotting.plot_glass_brain(connimg,
                          #cmap='cold_white_hot',
                          threshold=0.25,
                          colorbar=True,
                          #vmin=0.45,
                          #vmax=1,
                          #symmetric_cbar=False,
                          #output_file='conn_glass.pdf',
                          black_bg=True,
                          title= 'ATTENTION: This is just an approximate visualization. For a correct \n visualization the result should be normalized to ICBM152 template.')

plotting.plot_img(connimg,threshold=0.15)


#OLD SAMPLING
#for trs in np.arange(0.25,0.5,0.002):
#    connimg = nib.Nifti1Image(np.float32(CA.conn>trs), seedimg.affine)
#    nib.save(connimg,'hcplresult/hcpl_result_trs'+str(int(trs*1000)))

#NONUNIFORM SAMPLING
'''
for trs in np.arange(0.25,0.45,0.01):
    connimg = nib.Nifti1Image(np.float32(CA.conn>trs), seedimg.affine)
    nib.save(connimg,'hcplresult/hcpl_result_trs'+str(int(trs*1000)))

for trs in np.arange(0.45,0.5,0.001):
    connimg = nib.Nifti1Image(np.float32(CA.conn>trs), seedimg.affine)
    nib.save(connimg,'hcplresult/hcpl_result_trs'+str(int(trs*1000)))
'''
# WITHOUT SAMPLING
nib.save(connimg, 'hcpl_connectivities')
