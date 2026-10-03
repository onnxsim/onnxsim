import numpy as np, sys
for m in sys.argv[1:]:
    ref=np.fromfile(f'iso/{m}_ref.bin',np.float32)
    out=[]
    for acc in (0,1):
        try: b=np.fromfile(f'diso/{m}_{acc}/0.bin',np.float32)
        except Exception: out.append('missing'); continue
        d=ref-b; out.append('acc%s rel-l2 %.1e max-rel %.1e nonfinite %d'%('f32' if acc else 'def', np.sqrt((d**2).sum()/(ref**2).sum()), np.abs(d).max()/np.abs(ref).max(), int((~np.isfinite(b)).sum())))
    print(f'{m:16s}', ' | '.join(out))
