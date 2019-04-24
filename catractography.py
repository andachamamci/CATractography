# -*- coding: utf-8 -*-
""" Cellular Automata Tractography: Fast Geodesic Diffusion MR Tractography and Connectivity Based Segmentation on the GPU
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
import os

import nibabel as nib

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model

import pyopencl as cl

class CATractography:
    def __init__(self):
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        
        self.ctx = cl.create_some_context(interactive=True)
        self.queue = cl.CommandQueue(self.ctx)
        
        self.ngpuprocs = 0
        
        # Using UINT32 type for the label variable resulted 32bits/64bits confusion. 
        # I couldn't arrange the pointers in a correct way so the labels are also float. Andac

        self.prg = cl.Program(self.ctx, """
          __kernel void gpuca(__global float *label, __global float *nlabel, __global float *strn, __global float *nstrn, __global float *dist, __global float *ndist, __global const int *nbr, __global const float *nbrdist, __global const int *shape)
          {
            int pid = get_global_id(0);
            int idx = pid / (shape[1]*shape[2]);
            int idy = (pid - idx*shape[1]*shape[2])/shape[2];
            int idz = pid - idx*shape[1]*shape[2] - idy*shape[2];
            
            int nbrx,nbry,nbrz,nbrid,nbrdistid,nb;
            if ( (idx>0) && (idx<shape[0]-1) && (idy>0) && (idy<shape[1]-1) && (idz>0) && (idz<shape[2]-1) )
            {
                for(nb=0; nb<shape[3]; nb++)
                {
                    nbrx = idx + nbr[3*nb];
                    nbry = idy + nbr[3*nb+1];
                    nbrz = idz + nbr[3*nb+2];
        
                    nbrid = nbrx * shape[1] * shape[2] + nbry * shape[2] + nbrz;
                    nbrdistid = pid*shape[3]+nb;
                    
                    if (nstrn[pid] < strn[nbrid]*nbrdist[nbrdistid])
                    {
                        nlabel[pid] = label[nbrid];
                        nstrn[pid] = strn[nbrid]*nbrdist[nbrdistid]; 
                        ndist[pid] = dist[nbrid]+1;
                    }
                }
            }
          }
          """).build()
    
    def set_gpu_variables(self,seed):
        self.label = np.zeros(seed.shape, dtype=np.float32)
        for i in range(seed.shape[0]):
            for j in range(seed.shape[1]):
                for k in range(seed.shape[2]):
                    self.label[i,j,k] = np.float32(seed[i,j,k])    
        
        self.ngpuprocs = (np.prod(self.label.shape),1)
            
        mf = cl.mem_flags            
        self.gpu_label = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.label)
        self.gpu_nlabel = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.label)            
    
        self.strn = np.zeros(seed.shape, dtype=np.float32)
        self.strn[seed>0] = 1.0
        self.gpu_strn = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.strn)
        self.gpu_nstrn = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.strn)

        self.dist = np.zeros(seed.shape, dtype=np.float32)
        self.dist[:,:,:] = 1.0      
        self.gpu_dist = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dist)
        self.gpu_ndist = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dist)

    def set_gpu_graph(self,nbh_pdf,nbh):
        mf = cl.mem_flags
        self.gpu_nbh = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nbh)
        self.gpu_nbh_w = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(nbh_pdf))
        
        nbh_w_shape = np.int32(nbh_pdf.shape)
        self.gpu_nbh_w_shape = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nbh_w_shape)
        
    def gpu_one_iteration(self):
        self.prg.gpuca(self.queue, self.ngpuprocs, None, self.gpu_label, self.gpu_nlabel, self.gpu_strn, self.gpu_nstrn, self.gpu_dist, self.gpu_ndist, self.gpu_nbh, self.gpu_nbh_w, self.gpu_nbh_w_shape)
        tmp = self.gpu_label
        self.gpu_label = self.gpu_nlabel
        self.gpu_nlabel = tmp
        tmp = self.gpu_strn
        self.gpu_strn = self.gpu_nstrn
        self.gpu_nstrn = tmp
        tmp = self.gpu_dist
        self.gpu_dist = self.gpu_ndist
        self.gpu_ndist = tmp        
    
    def get_gpu_variables(self):    
        cl.enqueue_copy(self.queue, self.label, self.gpu_label)
        cl.enqueue_copy(self.queue, self.strn, self.gpu_strn)
        cl.enqueue_copy(self.queue, self.dist, self.gpu_dist)
        self.conn = self.strn**(1/self.dist)
        
    def gpu_N_iterations(self,NIterations=70):
        for i in range(NIterations):
            self.gpu_one_iteration()


def create_graph(csd_odf, sphere, tissueprior):

    # All 26 directions are defined.
    nbh=np.array([[1,0,0],[0,1,0],[1,1,0],[1,-1,0],[-1,1,0],[-1,0,0],[0,-1,0],[-1,-1,0],[-1,0,-1],[0,0,1],[0,-1,-1],[0,0,-1],[-1,-1,-1],[-1,0,1],[0,1,-1],[0,-1,1],[1,1,-1],[-1,1,1],[1,0,1],[0,1,1],[1,1,1],[1,-1,1],[1,0,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])
    
    # Transform 724 directions to 26 directions for every voxels.
    # Find the minimum angle value by calculating the angle between 
    # the vector in 724 direcitons and the vector in 26 directions.
    # This transformation data is saved in transform[] array.
    transform=np.zeros(sphere.theta.shape[0]) 
    for i in range(0,sphere.theta.shape[0]): 
        min_ang=360
        for j in range(0,nbh.shape[0]): #26 neighborhoods
            dot=(nbh[j,0]*sphere.x[i]+nbh[j,1]*sphere.y[i]+nbh[j,2]*sphere.z[i])/(np.linalg.norm(nbh[j,:]))
            ang=np.arccos(dot)
    
            if min_ang>ang:
                min_ang=ang
                transform[i]=j
    transform = np.int32(transform)
    
    nbh_pdf=np.zeros((csd_odf.shape[0],csd_odf.shape[1],csd_odf.shape[2],nbh.shape[0]))
    
    # In this part, we trasnform odf's from 724 directions 
    # to distrubuted 26 directions. Then we normalize the ODF's for all voxels.
    for k in range(0,csd_odf.shape[0]):  
        for m in range(0,csd_odf.shape[1]):
            for sl in range(0,csd_odf.shape[2]):
                for i in range(0,sphere.theta.shape[0]): 
                    nbh_pdf[k,m,sl,transform[i]]+=csd_odf[k,m,sl,i] 
                    
                if (np.sum(nbh_pdf[k,m,sl,:])<0.0001):
                    nbh_pdf[k,m,sl,:]=0
                else:
                    nbh_pdf[k,m,sl,:]=0.5*(nbh_pdf[k,m,sl,:]/np.max(nbh_pdf[k,m,sl,:]))
                    #22/3/2018: Scale maximum to 0.5 (Instead of normalizing to 1) see: medina2007
    
    #Obtain a symmetric edge weighting
    negnbh = np.zeros(nbh.shape[0],dtype=np.int32)
    for i in range(nbh.shape[0]):
        for j in range(nbh.shape[0]):
            if np.all(-nbh[i] == nbh[j]):
                negnbh[i]=j
    
    for d in range(0,nbh_pdf.shape[3]):
        #we will take only one of the  directions to prevent double count
        if (negnbh[d]>d):
            for i in range(1,nbh_pdf.shape[0]-1):
                for j in range(1,nbh_pdf.shape[1]-1):
                    for k in range(1,nbh_pdf.shape[2]-1):
                            cur = (i,j,k)
                            other = tuple(cur+nbh[d])
                            otherd = negnbh[d]
                            # No need to use tissue priors. Andac
                            newval = tissueprior[cur] * tissueprior[other] * 0.5 * (nbh_pdf[cur][d] + nbh_pdf[other][otherd]) #averaging is problematic
                            #newval = 0.5 * (nbh_pdf[cur][d] + nbh_pdf[other][otherd])
                            nbh_pdf[cur][d] = newval
                            nbh_pdf[other][otherd] = newval
                        
    nbh_pdf[nbh_pdf<0]=0        
    return nbh_pdf, nbh

def fit_csd_model(data_file, bval_file, bvec_file, UseMemoryEfficiently=True, UseParallel=True, Print_Response=False, roi_radius=10, fa_thr=0.7):
    """ Calculate the fiber orientation distribution function (fODF)
	Uses DIPY library

    Parameters
    ----------
    data_file : File name of the diffusion weighted data in NII format.
    bval_file : File name of the bval file.
        Text file listing b-values of each volume in the dataset.
    bvec_file : File name of the bvec file.
        Text file listing diffusion gradient directions 
        of each volume in the dataset.
    Print_Response : Print detailed information on the estimated fiber response function

    Returns
    -------
    csd_odf : numpy array
    sphere : dipy.core.sphere.Sphere object
    
    Notes
    -----
   
    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
           the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical
           deconvolution
    """

    bvals,bvecs = read_bvals_bvecs(bval_file,bvec_file)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(data_file)

    sphere = get_sphere('symmetric724')

    # ESTIMATION OF THE FIBRE RESPONSE FUNCTION ###################################
    # The auto_response function will calculate FA for an ROI of radius equal to roi_radius 
    # in the center of the volume and return the response function estimated in that region 
    # for the voxels with FA higher than 0.7.
   
    response, ratio = auto_response(gtab, img.dataobj, roi_radius=roi_radius, fa_thr=fa_thr)

    if (Print_Response):
        print('------------------------------------------------------------------------------------------------')
        print('Response Function = ' + str(response))
        print('Ratio = ' + str(ratio) + '[value should be about 0.2]')
        print('------------------------------------------------------------------------------------------------')
        print('The tensor generated from the response must be prolate (two smaller eigenvalues should be equal)')
        print('and look anisotropic with a ratio of second to first eigenvalue of about 0.2.')
        print('Or in other words, the axial diffusivity of this tensor should be around 5 times larger than the')
        print('radial diffusivity.')
        print('------------------------------------------------------------------------------------------------')
    
    csd_model = ConstrainedSphericalDeconvModel(gtab,response)
    
    if UseParallel:
        if (UseMemoryEfficiently):
            csd_odf = np.zeros((img.shape[0],img.shape[1],img.shape[2],sphere.phi.shape[0]),dtype=np.float64)
            slimg = np.zeros((img.shape[0],img.shape[1],1,img.shape[3]))
            for sl in range(0,img.shape[2]):
                print('processing slice ' + str(sl+1) + ' of ' + str(img.shape[2]) )
                slimg[:,:,0,:] = img.dataobj[:,:,sl,:]
                tmp_csd_fit =  peaks_from_model(model=csd_model,
                                          data=slimg,
                                          sphere=sphere,
                                          relative_peak_threshold=.5,
                                          min_separation_angle=25,
                                          mask=None,
                                          return_sh=False,
                                          return_odf=True,
                                          normalize_peaks=False,
                                          npeaks=5,
                                          parallel=True,
                                          nbr_processes=None)
                
                csd_odf[:,:,sl,:] = np.squeeze(tmp_csd_fit.odf)
        else:
            tmp_csd_fit =  peaks_from_model(model=csd_model,
                                      data=img.get_data(),
                                      sphere=sphere,
                                      relative_peak_threshold=.5,
                                      min_separation_angle=25,
                                      mask=None,
                                      return_sh=False,
                                      return_odf=True,
                                      normalize_peaks=False,
                                      npeaks=5,
                                      parallel=True,
                                      nbr_processes=None)
            
            csd_odf = np.squeeze(tmp_csd_fit.odf)
    else:
        if (UseMemoryEfficiently):
            # Do CSD fit slice by slice to use memory efficiently in processing big data (such as HCP)
            csd_odf = np.zeros((img.shape[0],img.shape[1],img.shape[2],sphere.phi.shape[0]),dtype=np.float64)
            slimg = np.zeros((img.shape[0],img.shape[1],1,img.shape[3]))
            for sl in range(0,img.shape[2]):
                print('processing slice ' + str(sl+1) + ' of ' + str(img.shape[2]) )
                slimg[:,:,0,:] = img.dataobj[:,:,sl,:]
                tmp_csd_fit = csd_model.fit(slimg)
                csd_odf[:,:,sl,:] = np.squeeze(tmp_csd_fit.odf(sphere))
        else:
            csd_fit = csd_model.fit(img.get_data())
            csd_odf = csd_fit.odf(sphere)
            
    #csd_odf = csd_odf.clip(min=0)    #consumes too much memory
    return csd_odf, sphere
    

