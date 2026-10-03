import numpy as np
p=np.load('/mnt/data/cache/claude-work/models/photo640.npy'); print(p.shape,p.dtype)
a=p if p.shape[-1]==3 else p.transpose(1,2,0)
def rs(a,h,w):
    yi=np.linspace(0,a.shape[0]-1,h); xi=np.linspace(0,a.shape[1]-1,w)
    y0=np.floor(yi).astype(int); y1=np.minimum(y0+1,a.shape[0]-1); fy=(yi-y0)[:,None,None]
    x0=np.floor(xi).astype(int); x1=np.minimum(x0+1,a.shape[1]-1); fx=(xi-x0)[None,:,None]
    a=a.astype(np.float32)
    top=a[y0][:,x0]*(1-fx)+a[y0][:,x1]*fx; bot=a[y1][:,x0]*(1-fx)+a[y1][:,x1]*fx
    return top*(1-fy)+bot*fy
s=np.clip(np.rint(rs(a,512,512)),0,255).astype(np.uint8)[None]; s.tofile('sam_photo.bin'); np.save('sam_l0_enc_photo_in.npy',s)
r=(a.astype(np.float32)/255)[None]; r.astype(np.float32).tofile('rtdetr_photo.bin'); np.save('rtdetr_pre_photo_in.npy',r.astype(np.float32))
print(s.shape,r.shape,s.mean(),r.mean())
