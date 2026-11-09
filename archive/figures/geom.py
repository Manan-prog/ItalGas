import struct, glob, re, sys, numpy as np, json
from pyproj import Transformer
ROOT="/Users/manansarupria/Library/CloudStorage/GoogleDrive-manansarupria599@gmail.com/My Drive/Smartauger/ItalGas/Savona"
def read_path(fn):
    b=open(fn,'rb').read(); n=(len(b)-112)//36
    a=np.frombuffer(b[112:112+36*n],dtype=np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('seq','<i4'),('tms','<i4'),('flag','<i4')]))
    return a
# ---- all swaths
swaths=[]
for fn in sorted(glob.glob(ROOT+"/*/Survey_*/Swath*.path")):
    a=read_path(fn); sess=fn.split('/')[-3]; day=re.search(r'_(2025\.10\.\d\d)_',sess).group(1)
    swaths.append(dict(file=fn,day=day,x=a['x'],y=a['y']))
print("swaths",len(swaths)); 
for d in sorted(set(s['day'] for s in swaths)): print(" ",d, sum(1 for s in swaths if s['day']==d),"swaths")
np.savez(sys.argv[1]+"/swaths.npz", **{f"s{i}":np.c_[s['x'],s['y']] for i,s in enumerate(swaths)}, days=np.array([s['day'] for s in swaths]))
# ---- test swath + KML validation
fn=ROOT+"/savona 14.10.2025 matt_2025.10.14_11.31.18/Survey_2025-10-14_11-37-03/Swath001.path"
a=read_path(fn); x,y=a['x'],a['y']
kml=open(glob.glob(ROOT+"/savona 14.10.2025 matt_2025.10.14_11.31.18/*.kml")[0]).read()
T=Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)
K=[]
for m in re.finditer(r'<coordinates>(.*?)</coordinates>',kml):
    pts=np.array([[float(v) for v in p.split(',')[:2]] for p in m.group(1).split()])
    X,Y=T.transform(pts[:,0],pts[:,1]); K.append(np.c_[X,Y])
Kall=np.vstack(K); print("KML segments",len(K),"points",len(Kall))
# antenna geometry from Survey file (x along, y across) and GPS offset
ant600=[(0.347,yy) for yy in (0.671,0.335,0,-0.335,-0.671)]
ant200=[(0.0,yy) for yy in (0.671,0.335,0,-0.335,-0.671)]
gps=(0.694758,-0.175)
# heading from path
dx=np.gradient(x); dy=np.gradient(y); hd=np.arctan2(dy,dx)
def lines(sign_x, sign_y):
    out=[]
    for ax,ay in ant600+ant200:
        fx=sign_x*(ax-gps[0]); fy=sign_y*(ay-gps[1])   # offsets in vehicle frame (forward, left)
        X=x+fx*np.cos(hd)-fy*np.sin(hd); Y=y+fx*np.sin(hd)+fy*np.cos(hd); out.append(np.c_[X,Y])
    return out
from scipy.spatial import cKDTree
tree=cKDTree(Kall)
# evaluate on the stretch covered by the KML (KML bbox)
bb=(Kall[:,0].min(),Kall[:,0].max(),Kall[:,1].min(),Kall[:,1].max())
sel=(x>=bb[0]-2)&(x<=bb[1]+2)&(y>=bb[2]-2)&(y<=bb[3]+2); print("path pts in KML bbox",sel.sum())
best=None
for sx in (1,-1):
  for sy in (1,-1):
    L=lines(sx,sy); d=np.concatenate([tree.query(l[sel])[0] for l in L]); print(f" sign_x={sx:+d} sign_y={sy:+d}: mean dist to KML lines {d.mean():.3f} m, 90th pct {np.percentile(d,90):.3f} m")
    if best is None or d.mean()<best[0]: best=(d.mean(),sx,sy)
print("BEST",best)
# also how far is raw GPS track from nearest KML line (sanity: should be ~0.17-0.7 m)
print("raw GPS track to KML mean", tree.query(np.c_[x,y][sel])[0].mean())
json.dump(dict(sign_x=best[1],sign_y=best[2]), open(sys.argv[1]+"/convention.json","w"))
