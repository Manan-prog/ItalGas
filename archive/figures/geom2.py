import re, glob, sys, json, numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
S=sys.argv[1]
ROOT="/Users/manansarupria/Library/CloudStorage/GoogleDrive-manansarupria599@gmail.com/My Drive/Smartauger/ItalGas/Savona"
z=np.load(S+"/swaths.npz"); days=z['days']; sw=[z[f"s{i}"] for i in range(len(days))]
T=Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)
K=[]
for kf in sorted(glob.glob(ROOT+"/*/*.kml")):
    txt=open(kf).read(); n=0
    for m in re.finditer(r'<coordinates>(.*?)</coordinates>',txt):
        pts=np.array([[float(v) for v in p.split(',')[:2]] for p in m.group(1).split()])
        X,Y=T.transform(pts[:,0],pts[:,1]); K.append(np.c_[X,Y]); n+=len(pts)
    print(kf.split('/')[-2][:35], "kml pts", n)
Kall=np.vstack(K); tree=cKDTree(Kall); print("total KML points",len(Kall))
# pick swath with best KML coverage (fraction of path points within 1.5 m of a KML point)
cov=[]
for i,p in enumerate(sw):
    d=tree.query(p)[0]; cov.append((np.mean(d<1.5), i, len(p)))
cov.sort(reverse=True); print("best covered swaths:",cov[:5])
frac,i,n=cov[0]; p=sw[i]; x,y=p[:,0],p[:,1]; print("using swath",i,days[i],"n",n,"coverage",frac)
ant600=[(0.347,yy) for yy in (0.671,0.335,0,-0.335,-0.671)]
ant200=[(0.0,yy) for yy in (0.671,0.335,0,-0.335,-0.671)]
gps=(0.694758,-0.175)
dx=np.gradient(x); dy=np.gradient(y); hd=np.arctan2(dy,dx); sel=tree.query(p)[0]<1.5
def lines(sx,sy,x=x,y=y,hd=hd):
    out=[]
    for ax,ay in ant600+ant200:
        fx=sx*(ax-gps[0]); fy=sy*(ay-gps[1])
        out.append(np.c_[x+fx*np.cos(hd)-fy*np.sin(hd), y+fx*np.sin(hd)+fy*np.cos(hd)])
    return out
res={}
for sx in (1,-1):
  for sy in (1,-1):
    d=np.concatenate([tree.query(l[sel])[0] for l in lines(sx,sy)]); res[(sx,sy)]=d.mean()
    print(f" sign_x={sx:+d} sign_y={sy:+d}: mean dist {d.mean():.3f} m  median {np.median(d):.3f}  90th {np.percentile(d,90):.3f}")
print(" raw GPS track mean dist", tree.query(p[sel])[0].mean())
# also test: no GPS offset at all (lines centred on GPS track) 
d=np.concatenate([tree.query((p[sel]+ (np.c_[-fy*np.sin(hd[sel]), fy*np.cos(hd[sel])])))[0] for fy in (0.671,0.335,0,-0.335,-0.671)]); print(" centred-on-GPS (no lever arm) mean dist",d.mean())
best=min(res,key=res.get); print("BEST",best,res[best])
json.dump(dict(sign_x=best[0],sign_y=best[1],swath=int(i),day=str(days[i])),open(S+"/convention.json","w"))
