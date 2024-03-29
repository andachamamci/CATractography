from dipy.io.streamline import load_trk

import os
import sys  

import numpy as np
import nibabel as nib
import scipy.ndimage.morphology

from dipy.tracking import utils
from dipy.tracking.streamlinespeed import length

###########################################################################################################################
# The volumetric version of EPFL post processing.
# ARGUMENTS: connectivity volume filename ; output folder                                                    #
# e.g.: python ../compute_density_volumes_EPFL_vol.py tractogram_connectivity.nii.gz test_EPFL #
###########################################################################################################################

density_filename = sys.argv[1]
output_folder= sys.argv[2]

if os.path.isdir(output_folder):
    print("Warning! output folder already exist '%s'." % output_folder)
else:
    os.mkdir(output_folder)

density_img = nib.load(density_filename)

#ref_img = nib.load(reference_filename)
#tractogram = nib.streamlines.load(streamline_filename)

#only keep streamlines with length > 1mm
#lengths = length(tractogram.streamlines)
#streamlines = tractogram.streamlines[lengths > 1]

# Compute the visitation count image and apply a small gaussian smoothing.
# The gaussian smoothing is especially usefull to increase voxel coverage of deterministic algorithms.
#density = utils.density_map(streamlines, ref_img.affine, ref_img.shape)
density = np.array(density_img.get_data())
density = scipy.ndimage.gaussian_filter(density.astype("float32"), 0.5)

# Iteratively threshold the log of density map, 200 volumes/operation points
log_density = np.log10(density+1)
vol=-1
for i, t in enumerate(np.arange(0, np.max(log_density), np.max(log_density) / 200)):
    vol += 1
    nbr = str(vol)
    nbr = nbr.zfill(3)
    mask = log_density >= t
    vol_filename = os.path.join(output_folder, "vol" + nbr + "_t"+str(t) + ".nii.gz")
    nib.Nifti1Image(mask.astype("int32"), density_img.affine, density_img.header).to_filename(vol_filename)


