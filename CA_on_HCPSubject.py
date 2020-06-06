# -*- coding: utf-8 -*-
""" Sample Application of Cellular Automata Tractography on a Human Connectome Project Subject.
    Copyright (C) 2019  Andac Hamamci

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Andac Hamamci
	andachamamci@gmail.com

"""
import numpy as np
import time
import matplotlib.pyplot as plt
import os


import nibabel as nib

import catractography as CAT

''' -----------------------------------------
    Pre-processed DWI data for a subject in
    1200 Subject Release of the Young Adult
    dataset should be downloaded to process.
    https://www.humanconnectome.org/
    
    Note that the whole process is in subject
    coordinates.
    ----------------------------------------'''

DWIFILE_NAME = './HCPData/data.nii.gz'
BVALFILE_NAME = './HCPData/bvals'
BVECFILE_NAME = './HCPData/bvecs'
SEEDFILE_NAME = './HCPData/cc_seed.nii.gz'
GRAPHFILE_NAME = './HCPData/graph.npz'
TISSUEPRIORFILE_NAME = './HCPData/tissueprior.nii.gz'

''' -----------------------------------------
    Converting HCP diffusion data to 3 shells
    -----------------------------------------'''
# Convert bvalues to 3 shells:
bvaldata = np.genfromtxt(BVALFILE_NAME, delimiter=' ')
#The delimiter is 2 spaces for old HCP bvals
if (np.isnan(bvaldata).any()):
    bvaldata = np.genfromtxt(BVALFILE_NAME, delimiter='  ')
 
bvaldata[bvaldata<500] = 0
bvaldata[(bvaldata>500) & (bvaldata<1500)] = 1000
bvaldata[(bvaldata>1500) & (bvaldata<2500)] = 2000
bvaldata[bvaldata>2500] = 3000

BVALFILE_NAME = BVALFILE_NAME + 'new'
np.savetxt(BVALFILE_NAME, bvaldata, delimiter='  ',fmt='%d')

''' -------------------
    Arrange Seed Labels
    -------------------'''
seedimg = nib.load(SEEDFILE_NAME)
seed_data = seedimg.get_data()

seeds = np.zeros(seed_data.shape,dtype=np.int32)
lbl=1
for j in range(seed_data.shape[1]):
    for i in range(seed_data.shape[0]):
        for k in range(seed_data.shape[2]):
            if (seed_data[i,j,k]>0):
                seeds[i,j,k]=lbl
                lbl=lbl+1

''' ----------------------------------------
    Create graph or read from file if exists 
    ----------------------------------------'''

if (os.path.isfile(GRAPHFILE_NAME)):
    print('The graph file found. Loading the graph.')
    loaded = np.load(GRAPHFILE_NAME)
    nbh = loaded['nbh']
    nbh_pdf = np.float32(loaded['nbh_pdf'])
else:
    print('The graph file does not exist. Creating the graph.')
    tissuepriorimg = nib.load(TISSUEPRIORFILE_NAME)
    tissueprior = tissuepriorimg.get_data()
    csd_odf, sphere = CAT.fit_csd_model(DWIFILE_NAME, BVALFILE_NAME, BVECFILE_NAME, Print_Response=False, UseMemoryEfficiently=True, UseParallel=False, roi_radius=10, fa_thr=0.7)
    nbh_pdf, nbh = CAT.create_graph(csd_odf, sphere,tissueprior)
    np.savez_compressed(GRAPHFILE_NAME, nbh_pdf = nbh_pdf, nbh=nbh)      
    
''' --------------------------------------------
    Perform Cellular Automata based tractography
    --------------------------------------------'''
CA = CAT.CATractography()
CA.set_gpu_graph(nbh_pdf,nbh)
CA.set_gpu_variables(seeds)
ti=time.time()
CA.gpu_N_iterations(70)
tf=time.time()
CA.get_gpu_variables()

''' --------------------------------------------
    Results
    --------------------------------------------'''
from nilearn import plotting

data = (CA.conn - 0.99)/0.01
data[data<0] = 0
connimg = nib.Nifti1Image(data, seedimg.affine)
plotting.plot_glass_brain(connimg,
                          #cmap='cold_white_hot',
                          threshold=0,
                          colorbar=True,
                          vmin=0,
                          vmax=1,
                          #symmetric_cbar=False,
                          output_file='conn_glass.pdf',
                          black_bg=True,
                          title= 'ATTENTION: This is just an approximate visualization. For a correct \n visualization the result should be normalized to ICBM152 template.')