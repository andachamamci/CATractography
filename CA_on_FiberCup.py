# -*- coding: utf-8 -*-
""" Sample Application of Cellular Automata Tractography on FiberCup Phantom.
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

import nibabel as nib

import catractography as CAT

DWIFILE_NAME = './FiberCupData/acq-1_b-2000.nii.gz'
BVALFILE_NAME = './FiberCupData/grad64-2000.bvals'
BVECFILE_NAME = './FiberCupData/grad64.bvecs'
SEEDFILE_NAME = './FiberCupData/groundtruthseeds.nii'

csd_odf, sphere = CAT.fit_csd_model(DWIFILE_NAME, BVALFILE_NAME, BVECFILE_NAME, Print_Response=False, UseMemoryEfficiently=False, UseParallel=False, roi_radius=30, fa_thr=0.6)

nbh_pdf, nbh = CAT.create_graph(csd_odf, sphere)

seedimg = nib.load(SEEDFILE_NAME)
seed_data = seedimg.get_data()

for seedid in range(0,7):
    seed = np.int32(seed_data[:,:,1:4,seedid]) # The groundtruth seeds were defined as 5 slices!
       
    CA = CAT.CATractography()
    CA.set_gpu_graph(nbh_pdf,nbh)
    CA.set_gpu_variables(seed)
    ti=time.time()
    CA.gpu_N_iterations(70)
    tf=time.time()
    CA.get_gpu_variables()
    
    x=CA.conn[:,:,1]
    plt.imshow(CA.conn[:,:,1] > 0.45,cmap='Oranges',interpolation='gaussian')
    plt.title('Seed #'+str(seedid+1))
    plt.show()
    print('CA Computation Time : ' + str(tf-ti))



