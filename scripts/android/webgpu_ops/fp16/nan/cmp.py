import numpy as np, glob, sys, os
name=sys.argv[1]
for tag in sys.argv[2:]:
    out=[]
    for f in sorted(glob.glob(f'dump/{name}_cpu/*.bin')):
        a=np.fromfile(f,np.float32); g=f.replace(f'{name}_cpu',f'{name}_{tag}')
        if not os.path.exists(g): out.append('missing'); continue
        b=np.fromfile(g,np.float32)
        if a.size!=b.size: out.append('size'); continue
        nn=int(np.isnan(b).sum())+int(np.isinf(b).sum())
        if nn: out.append('NaN/Inf:%d/%d'%(nn,b.size)); continue
        d=a-b
        out.append('max%.1e/rel-l2 %.1e'%(np.abs(d).max()/(np.abs(a).max()+1e-12), np.sqrt((d**2).sum()/((a**2).sum()+1e-20))))
    print(name,tag,out)
