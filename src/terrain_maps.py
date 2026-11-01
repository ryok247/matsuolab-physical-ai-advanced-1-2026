"""Deterministic metric height fields; no simulator dependency."""
import hashlib
import numpy as np


def generate(seed, amplitude, size=24., spacing=.1, coarse=.4):
    if amplitude < 0 or spacing <= 0 or size < 4:
        raise ValueError('invalid terrain dimensions')
    rng=np.random.default_rng(seed)
    coarse_axis=np.linspace(-size/2,size/2,round(size/coarse)+1)
    axis=np.linspace(-size/2,size/2,round(size/spacing)+1)
    values=rng.uniform(-1,1,(len(coarse_axis),len(coarse_axis)))
    intermediate=np.stack([np.interp(axis,coarse_axis,row) for row in values])
    height=np.stack([np.interp(axis,coarse_axis,intermediate[:,j]) for j in range(len(axis))],axis=1)
    height-=height.mean()
    height/=max(float(np.abs(height).max()),1e-12)
    x,y=np.meshgrid(axis,axis,indexing='ij')
    blend=np.clip((np.maximum(np.abs(x),np.abs(y))-1.)/.5,0,1)
    height=np.rint(height*amplitude*blend/.001).astype(np.int16)
    return height,dict(seed=int(seed),amplitude_m=amplitude,horizontal_scale=spacing,vertical_scale=.001,
        pos=[-size/2,-size/2,0],shape=list(height.shape),sha256=hashlib.sha256(height.tobytes()).hexdigest(),
        rms_m=float(np.sqrt(np.mean((height.astype(float)*.001)**2))),
        max_abs_m=float(np.abs(height).max()*.001))


def sample(height,metadata,xy):
    """Bilinear reference sampler. Physical interpolation must be calibrated separately."""
    coords=(np.asarray(xy)-np.asarray(metadata['pos'][:2]))/metadata['horizontal_scale']
    if np.any(coords<0) or np.any(coords>np.asarray(height.shape)-1):
        raise ValueError('outside terrain')
    lower=np.minimum(np.floor(coords).astype(int),np.asarray(height.shape)-2)
    f=coords-lower;i,j=lower[...,0],lower[...,1];u,v=f[...,0],f[...,1]
    return ((1-u)*(1-v)*height[i,j]+u*(1-v)*height[i+1,j]+(1-u)*v*height[i,j+1]+u*v*height[i+1,j+1])*metadata['vertical_scale']
