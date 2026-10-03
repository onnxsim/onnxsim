import numpy as np, sys
name,tag,ref,dumpd=sys.argv[1:5]
rows=[l.split('\t') for l in open(f'{name}_{tag}_bis.txt').read().split('\n')]
prev=0
print('idx op maxref  rel-l2   max-rel  nan  name')
for k,(i,op,nm,t) in enumerate(rows):
    a=np.fromfile(f'{ref}/{k}.bin',np.float32); b=np.fromfile(f'{dumpd}/{k}.bin',np.float32)
    d=a-b; nn=int(np.isnan(b).sum()+np.isinf(b).sum())
    l2=np.sqrt(np.nansum(d**2)/((a**2).sum()+1e-20)); mx=np.nanmax(np.abs(d))/(np.abs(a).max()+1e-12) if nn<b.size else float('nan')
    if k%8==0 or l2>2*prev+0.005 or nn:
        print(f'{k:4d} {op:10s} {np.abs(a).max():9.1f} {l2:8.1e} {mx:8.1e} {nn:7d} {nm.split("/enc/")[-1][-60:]}')
    prev=max(prev*0.9,l2)
