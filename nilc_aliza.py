import sys
import healpy as hp

import numpy as np

import mtneedlet as nd
import os
from astropy.io import fits

# generate needlet (mexican or standard) bands stored in array b
def get_needlet_bands(needlets,B,lmax,b_max,n_bands):
    # needlets: 1: mexican needlets; 2: standard needlets
    # B: parameter which sets widht of needlet bands
    # lmax: maximum multipole
    # b_max: maximum band to merge in the first needlet band
    # n_bands: total number of bands
    
    j_min = 0
    j_max = b_max + n_bands

    if needlets==1:
        band=f'mexB{str(B)}_b0b{str(b_max)}_wl01'
        b_in=nd.mexicanneedlet(B,list(range(j_min,j_max+1)),lmax)
    if needlets==2:
        band=f'standB{str(B)}_b0b{str(b_max)}_wl01'
        b_in=nd.standardneedlet(B,list(range(j_min,j_max+1)),lmax)
        
    if needlets != 3:
        b=np.zeros((n_bands,lmax+1))
        b[0,:]=np.sqrt(np.sum((b_in**2)[:b_max+1,:lmax+1],axis=0))
        for i in range(1,n_bands):
            b[i]=b_in[b_max+i]
    
    b[0,:2]=1.
    
    return b, band

# function that computes the nside of needlet maps corresponding to the input needlet windows b
def get_nside_nl(b,nside,lmax,resol_nl):
    if resol_nl:
        try:
            if (int(np.max(np.nonzero(b))/2) <= 16) & (int(np.max(np.nonzero(b))/2) > 8):
                nside_nl=16
            if (int(np.max(np.nonzero(b))/2) <= 32) & (int(np.max(np.nonzero(b))/2) > 16):
                nside_nl=32
            if (int(np.max(np.nonzero(b))/2) <= 64) & (int(np.max(np.nonzero(b))/2) > 32):
                nside_nl=64
            if (int(np.max(np.nonzero(b))/2) <= 128) & (int(np.max(np.nonzero(b))/2) > 64):
                nside_nl=128
            if (int(np.max(np.nonzero(b))/2) <= 256) & (int(np.max(np.nonzero(b))/2) > 128):
                nside_nl=256
            if (int(np.max(np.nonzero(b))/2) <= 512) & (int(np.max(np.nonzero(b))/2) > 256):
                nside_nl=512
            if (int(np.max(np.nonzero(b))/2) <= 1024) & (int(np.max(np.nonzero(b))/2) > 512):
                nside_nl=1024
            lmax_nl=2*nside_nl
        except ValueError:
            nside_nl=nside
            lmax_nl=lmax
    else:
        nside_nl=nside
        lmax_nl=lmax

    return nside_nl, lmax_nl


def nilc(alm_map,alm_fore,alm_noi,b,lmax,nside,bias,resol_nl=True):
    # alm_map: alms of input maps, dimension should be (n_freqs,lm)
    # alm_fore: alms of input foregrounds, dimension should be (n_freqs,lm)
    # alm_map: alms of input noise, dimension should be (n_freqs,lm)
    # b: set of needlet windows, with shape (n_bands,lmax+1)
    # bias: bias parameter. The covariance matrix elements in each pixel can be estimated as the average of products of needlet maps within a Gaussian domain centered in that pixel.
    #       bias sets the size of these domains. bias should be smaller than 0.01. Good choice is 0.005.
    #       if bias=0, covariance matrix elements are estimated with averages over the whole sphere
    # resol_nl: if True, needlet maps are generated at different Nside depending on the maximum multipole accessed by the needlet windows. 
    #           if False, all needlet maps are generated at input Nside.

    n_freq = alm_map.shape[0]    
    n_bands=b.shape[0]
    lm=hp.Alm.getsize(lmax)
    npix=12*nside**2

    alm_out = np.zeros(lm,dtype=complex)
    alm_fres = np.zeros(lm,dtype=complex)
    alm_nres = np.zeros(lm,dtype=complex)
    
    # needlet component separation
    for j in (range(n_bands)):
        nside_nl, lmax_nl = get_nside_nl(b[j],nside,lmax,resol_nl)
             
        npix_nl=12*nside_nl**2
        lmax_nl=np.min([lmax,lmax_nl])
        
        map_nl = np.zeros((n_freq,npix_nl))
        noi_nl = np.zeros((n_freq,npix_nl))
        fore_nl = np.zeros((n_freq,npix_nl))
        
        lm_nl = hp.Alm.getsize(lmax_nl)
        
        # needlet decomposition
        for i in range(n_freq):
            alm_map_nl = hp.almxfl(alm_map[i],(b)[j])
            alm_fore_nl = hp.almxfl(alm_fore[i],(b)[j])
            alm_noi_nl = hp.almxfl(alm_noi[i],(b)[j])

            alm_map_nl_=np.zeros(lm_nl,dtype=complex)
            alm_noi_nl_=np.zeros(lm_nl,dtype=complex)
            alm_fore_nl_=np.zeros(lm_nl,dtype=complex)
            
            for l in range(2,lmax_nl+1):
                for m in range(0,l+1):
                    ind=hp.Alm.getidx(lmax, l, m)
                    ind_nl=hp.Alm.getidx(lmax_nl, l, m)
                    
                    alm_map_nl_[ind_nl]=alm_map_nl[ind]
                    alm_noi_nl_[ind_nl]=alm_noi_nl[ind]
                    alm_fore_nl_[ind_nl]=alm_fore_nl[ind]
                      
            map_nl[i]=hp.alm2map(alm_map_nl_,nside=nside_nl,lmax=lmax_nl)
            noi_nl[i]=hp.alm2map(alm_noi_nl_,nside=nside_nl,lmax=lmax_nl)
            fore_nl[i]=hp.alm2map(alm_fore_nl_,nside=nside_nl,lmax=lmax_nl)
        
        # computation of covariance, NILC weights
        if bias == 0.:
            cov=np.mean(np.einsum('ik,jk->ijk', map_nl, map_nl),axis=2)
            inv_cov=np.linalg.inv(cov)

            w=np.sum(inv_cov,axis=1)/np.sum(inv_cov)

            cmb_nl = np.einsum('i,ik->k', w, map_nl)
            fres_nl = np.einsum('i,ik->k', w, fore_nl)
            nres_nl = np.einsum('i,ik->k', w, noi_nl)
            
        else:
            nmodes_band  = np.sum((2.*np.arange(0,lmax+1)+1.) * (b[j])**2 )
            pps = np.sqrt(float(npix_nl) * float(n_freq-1) / (bias * nmodes_band) )

            cov = np.zeros((n_freq,n_freq,int(np.min([npix_nl,12*128**2]))))

            for i in (range(n_freq)):
                for k in range(i,n_freq):
                    cov[i,k]=localcovar((map_nl[i]), (map_nl[k]), pps, b[j], int(np.min([nside_nl,128])))

            for i in range(n_freq):
                for k in range(i):
                    cov[i,k]=cov[k,i]

            inv_cov=np.linalg.inv(cov.T).T

            del cov

            w=np.zeros((n_freq,npix_nl))
            for i in range(n_freq):
                w[i]=hp.ud_grade((np.sum(inv_cov,axis=1)[i])/np.sum(inv_cov,axis=(0,1)),nside_nl)

            del inv_cov

            cmb_nl = np.einsum('ik,ik->k', w, map_nl)
            fres_nl = np.einsum('ik,ik->k', w, fore_nl)
            nres_nl = np.einsum('ik,ik->k', w, noi_nl)
        
        # inverse needlet transform
        alm_cmb_nl = hp.almxfl(hp.map2alm(cmb_nl,lmax=lmax_nl,pol=False),b[j,:lmax_nl+1])
        alm_fres_nl = hp.almxfl(hp.map2alm(fres_nl,lmax=lmax_nl,pol=False),b[j,:lmax_nl+1])
        alm_nres_nl = hp.almxfl(hp.map2alm(nres_nl,lmax=lmax_nl,pol=False),b[j,:lmax_nl+1])
        
        for l in range(2,lmax_nl+1):
            for m in range(0,l+1):
                ind=hp.Alm.getidx(lmax, l, m)
                ind_nl=hp.Alm.getidx(lmax_nl, l, m)

                alm_out[ind] += alm_cmb_nl[ind_nl]
                alm_nres[ind] += alm_nres_nl[ind_nl]
                alm_fres[ind] += alm_fres_nl[ind_nl]
        
    cmb_cleaned = hp.alm2map(alm_cmb,nside,lmax=lmax, pixwin=True ,pol=False)
    foreres = hp.alm2map(alm_fres,nside,lmax=lmax, pixwin=True ,pol=False)
    noires = hp.alm2map(alm_nres,nside,lmax=lmax, pixwin=True ,pol=False)
    
    return cmb_cleaned, foreres, noires

def localcovar(map1, map2, pixperscale, b, *nsidecovar):
    """Local covariance of two maps in the pixel space."""

    map_ = map1 * map2
    npix_ = map_.size
    nside = hp.pixelfunc.npix2nside(npix_)
    if not nsidecovar:
        nsidecovar = nside
    else:
        nsidecovar = nsidecovar[0]

    # First degrade a bit to speed-up smoothing

    if (nside / 4) > 1:
        nside_out = int(nside / 4)
    else:
        nside_out = 1

    stat = hp.pixelfunc.ud_grade(map_, nside_out = nside_out, order_in = 'RING', order_out = 'RING')
    
    # Compute alm

    lmax = lmax = 2 * nside_out #3 * nside_out - 1 # 
    nlm_tot = hp.sphtfunc.Alm.getsize(lmax)
    alm = hp.sphtfunc.map2alm(stat, lmax=lmax, iter=1, use_weights=True)# iter=0?

    # Find smoothing size

    pixsize = np.sqrt(4.0 * np.pi / npix_)
    fwhm = pixperscale * pixsize
#     print(np.rad2deg(fwhm))
    bl = hp.sphtfunc.gauss_beam(fwhm, lmax)
#     bl = b[:lmax]

    # Smooth the alm

    alm_s = hp.sphtfunc.almxfl(alm, bl)
    
    # Back to pixel space

    stat_out = hp.sphtfunc.alm2map(alm_s, nsidecovar, lmax=lmax, verbose=False)

    return stat_out   

