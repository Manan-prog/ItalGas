import re, glob, sys, numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
S=sys.argv[1]
ROOT="/Users/manansarupria/Library/CloudStorage/GoogleDrive-manansarupria599@gmail.com/My Drive/Smartauger/ItalGas/Savona"
z=np.load(S+"/swaths.npz"); days=z['days']; sw=[z[f"s{i}"] for i in range(len(days))]
T=Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)
# KML of 17 Oct only (covers swath 154)
kf=glob.glob(ROOT+"/savona 1410 matt_2025.10.17_09.03.29/*.kml")[0]; txt=open(kf).read()
segs=[]
for m in re.finditer(r'<coordinates>(.*?)</coordinates>',txt):
    pts=np.array([[float(v) for v in p.split(',')[:2]] for p in m.group(1).split()]); X,Y=T.transform(pts[:,0],pts[:,1]); segs.append(np.c_[X,Y])
print("segments",len(segs),"pts per seg (min/median/max)",min(map(len,segs)),int(np.median(list(map(len,segs)))),max(map(len,segs)))
p=sw[154]; x,y=p[:,0],p[:,1]; hd=np.arctan2(np.gradient(y),np.gradient(x)); ptree=cKDTree(p)
# Web Mercator scale factor at this latitude (metres on ground = mercator metres * cos(lat))
lat=44.31; k=np.cos(np.radians(lat))
# signed cross-track offset of KML points relative to GPS track (positive = left of travel)
Kall=np.vstack(segs); d,idx=ptree.query(Kall); ok=d<3
dxk=Kall[ok,0]-x[idx[ok]]; dyk=Kall[ok,1]-y[idx[ok]]; h=hd[idx[ok]]
cross=(-dxk*np.sin(h)+dyk*np.cos(h))*k; along=(dxk*np.cos(h)+dyk*np.sin(h))*k
hist,edges=np.histogram(cross,bins=np.arange(-1.5,1.5,0.04))
print("cross-track offset histogram of KML channel lines relative to GPS track (m, +left):")
for c,e in zip(hist,edges):
    if c>0.01*hist.max(): print(f" {e:+.2f} {'#'*int(60*c/hist.max())}")
print("along-track offset: mean %.3f  median %.3f"%(along.mean(),np.median(along)))
# per-segment: mean cross offset -> cluster into lines
segoff=[]
for s in segs:
    d,i=ptree.query(s); 
    if (d<3).mean()<0.5: continue
    dx_=s[:,0]-x[i]; dy_=s[:,1]-y[i]; hh=hd[i]; segoff.append(np.median((-dx_*np.sin(hh)+dy_*np.cos(hh))*k))
segoff=np.array(segoff); print("distinct per-segment lateral offsets (rounded 5 cm):", sorted(set(np.round(segoff/0.05)*0.05)))
