import re, glob, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import contextily as cx
import matplotlib.patheffects as pe
from pyproj import Transformer
from scipy.spatial import cKDTree
S=sys.argv[1]
ROOT="/Users/manansarupria/Library/CloudStorage/GoogleDrive-manansarupria599@gmail.com/My Drive/Smartauger/ItalGas/Savona"
C=dict(surface="#fcfcfb",text="#0b0b0b",text2="#52514e",muted="#8a8984",s1="#2a78d6",s2="#eb6834",grid="#e6e5e1")
z=np.load(S+"/swaths.npz"); days=z['days']; sw=[z[f"s{i}"] for i in range(len(days))]
T=Transformer.from_crs("EPSG:3857","EPSG:4326",always_xy=True)
lat0=44.31; k=np.cos(np.radians(lat0))   # web-mercator scale: ground m = mercator m * k

# ---------- zoom segment: straight 24 m stretch of swath 154 (17 Oct)
SW=int(sys.argv[2]) if len(sys.argv)>2 else 154
p=sw[SW]; x,y=p[:,0],p[:,1]; hd=np.arctan2(np.gradient(y),np.gradient(x))
best=None
for i in range(50,len(p)-50):
    j=i
    while j<len(p)-1 and np.hypot(x[j]-x[i],y[j]-y[i])*k<24: j+=1
    if j-i<20: continue
    v=np.std(np.unwrap(hd[i:j]))
    if best is None or v<best[0]: best=(v,i,j)
_,i0,i1=best; xs,ys,hs=x[i0:i1],y[i0:i1],hd[i0:i1]
cxm,cym=xs.mean(),ys.mean()
print("zoom segment: records",i0,i1,"heading std deg",np.degrees(best[0]))

# antenna geometry (vehicle frame: x forward, y left), GPS antenna position in that frame
ant600=[(0.347,yy) for yy in (0.671,0.335,0,-0.335,-0.671)]
ant200=[(0.0,yy) for yy in (0.671,0.335,0,-0.335,-0.671)]
gps=(0.694758,-0.175)
def to_map(fx,fy,xg,yg,h):   # vehicle-frame offset (m, ground) -> mercator coords
    fx,fy=fx/k,fy/k
    return xg+fx*np.cos(h)-fy*np.sin(h), yg+fx*np.sin(h)+fy*np.cos(h)
# KML channel lines near the segment (IDS's own output) for comparison
kf=glob.glob(ROOT+"/savona 1410 matt_2025.10.17_09.03.29/*.kml")[0]; txt=open(kf).read()
T2=Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True); ksegs=[]
for m in re.finditer(r'<coordinates>(.*?)</coordinates>',txt):
    pts=np.array([[float(v) for v in q.split(',')[:2]] for q in m.group(1).split()]); X,Y=T2.transform(pts[:,0],pts[:,1]); ksegs.append(np.c_[X,Y])

# ---------- figure
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":C["grid"],"axes.labelcolor":C["text2"],
                     "xtick.color":C["text2"],"ytick.color":C["text2"],"text.color":C["text"],"figure.facecolor":C["surface"],"axes.facecolor":C["surface"]})
fig=plt.figure(figsize=(14,7.6),dpi=150)
gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=0.08,left=0.03,right=0.985,top=0.86,bottom=0.09)
axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1])

# --- A: overview, all 155 swaths
allxy=np.vstack(sw); pad=600
axA.set_xlim(allxy[:,0].min()-pad,allxy[:,0].max()+pad); axA.set_ylim(allxy[:,1].min()-pad,allxy[:,1].max()+pad)
for q in sw: axA.plot(q[:,0],q[:,1],color=C["s1"],lw=1.1,alpha=0.9,solid_capstyle="round")
GSAT="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
cx.add_basemap(axA,source=GSAT,crs="EPSG:3857",zoom=15,attribution=False,alpha=0.75)
axA.plot(cxm,cym,"o",ms=9,mfc="none",mec=C["s2"],mew=2); axA.annotate("zoom panel",(cxm,cym),xytext=(-70,14),textcoords="offset points",color=C["s2"],fontsize=9,path_effects=[pe.withStroke(linewidth=2.5,foreground=C["surface"])])
# scale bar 2 km
x0=allxy[:,0].min()-pad+700; y0=allxy[:,1].min()-pad+450; L=2000/k
axA.plot([x0,x0+L],[y0,y0],color=C["text"],lw=2.5,solid_capstyle="butt",path_effects=[pe.withStroke(linewidth=5,foreground=C["surface"])]); axA.text(x0+L/2,y0+90,"2 km",ha="center",fontsize=8,color=C["text"],path_effects=[pe.withStroke(linewidth=2.5,foreground=C["surface"])])
axA.set_xticks([]); axA.set_yticks([]); 
axA.set_title("A.  All 155 swaths (GPS antenna track), Savona, 13 to 17 Oct 2025",loc="left",fontsize=10.5,color=C["text"],pad=6)
# lon/lat corner labels
for (xx,yy,ha,va) in [(axA.get_xlim()[0],axA.get_ylim()[0],"left","bottom"),(axA.get_xlim()[1],axA.get_ylim()[1],"right","top")]:
    lo,la=T.transform(xx,yy); axA.text(xx,yy,f"  {la:.3f}°N, {lo:.3f}°E  ",ha=ha,va=va,fontsize=7,color=C["text2"],
        bbox=dict(boxstyle="square,pad=0.15",fc=C["surface"],ec="none",alpha=0.85))

# --- B: zoom, antenna lines
half=15/k
axB.set_xlim(cxm-half,cxm+half); axB.set_ylim(cym-half,cym+half); axB.set_aspect("equal")
try: cx.add_basemap(axB,source=GSAT,crs="EPSG:3857",zoom=21,attribution=False)
except Exception as e: print("imagery failed",e)
# IDS channel lines (29 channel midpoints) in neutral gray
for s_ in ksegs:
    m=(np.abs(s_[:,0]-cxm)<half*1.2)&(np.abs(s_[:,1]-cym)<half*1.2)
    if m.sum()>1: axB.plot(s_[m,0],s_[m,1],color="#f3f2ee",lw=0.6,alpha=0.9,zorder=2)
# our 5 antenna-column centre lines (600 and 200 MHz rows share the same lateral positions)
for ax_,ay_ in ant200:
    X,Y=to_map(ax_-gps[0],ay_-gps[1],xs,ys,hs); axB.plot(X,Y,color=C["s1"],lw=2,zorder=4)
# GPS track
axB.plot(xs,ys,color=C["text"],lw=1.2,ls=(0,(3,2)),zorder=5)
# array footprint at one instant (mid segment)
im=len(xs)//2; xg,yg,h=xs[im],ys[im],hs[im]
def rect(cxv,cyv,w,hgt,col):   # w along-track, hgt across-track, in m ground
    corners=[(-w/2,-hgt/2),(w/2,-hgt/2),(w/2,hgt/2),(-w/2,hgt/2)]
    P=np.array([to_map(cxv-gps[0]+a,cyv-gps[1]+b,xg,yg,h) for a,b in corners])
    axB.fill(P[:,0],P[:,1],fc=col,ec=C["surface"],lw=1,alpha=0.95,zorder=6)
for ax_,ay_ in ant600: rect(ax_,ay_,0.26,0.32,C["s1"])
for ax_,ay_ in ant200: rect(ax_,ay_,0.44,0.32,C["s2"])
axB.plot(xg,yg,"o",ms=8,mfc=C["surface"],mec=C["text"],mew=1.8,zorder=7)
# direction arrow
X1,Y1=to_map(2.5,0,xg,yg,h); axB.annotate("",xy=(X1,Y1),xytext=(xg,yg),arrowprops=dict(arrowstyle="-|>",color=C["surface"],lw=1.5),zorder=7)
# labels
X,Y=to_map(4.0,-1.6,xg,yg,h); axB.text(X,Y,"5 antenna columns\n0.335 m apart",color=C["surface"],fontsize=8,ha="right",va="top",zorder=8)
Xl,Yl=to_map(-0.9,-0.9,xg,yg,h); axB.text(Xl,Yl,"GPS antenna",color=C["surface"],fontsize=8,ha="right",va="top",zorder=8)
# scale bar 2 m
sx0=cxm-half+1.0/k; sy0=cym-half+0.8/k; axB.plot([sx0,sx0+2/k],[sy0,sy0],color=C["surface"],lw=2.5); axB.text(sx0+1/k,sy0+0.25/k,"2 m",ha="center",color=C["surface"],fontsize=8)
axB.set_xticks([]); axB.set_yticks([])
lo,la=T.transform(cxm,cym)
axB.set_title("B.  Antenna geometry on the ground, one swath of 17 Oct 2025",loc="left",fontsize=10.5,color=C["text"],pad=6)
axB.text(axB.get_xlim()[1],axB.get_ylim()[1],f"  {la:.5f}°N, {lo:.5f}°E  ",ha="right",va="top",fontsize=7,color=C["text2"],bbox=dict(boxstyle="square,pad=0.15",fc=C["surface"],ec="none",alpha=0.85),zorder=9)
# legend (manual)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
hand=[Line2D([],[],color=C["s1"],lw=2,label="Antenna column centre line (computed from GPS track + lever arm)"),
      Patch(fc=C["s1"],label="600 MHz antenna footprint (front row)"),
      Patch(fc=C["s2"],label="200 MHz antenna footprint (rear row)"),
      Line2D([],[],color=C["text"],lw=1.2,ls=(0,(3,2)),label="GPS antenna track (10 Hz RTK)"),
      Line2D([],[],color="#bdbcb6",lw=0.8,label="IDS software channel lines from the delivered KML (29 channels)")]
fig.legend(handles=hand,loc="lower center",ncol=3,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,0.0),handlelength=2.2)
fig.suptitle("Stream UP array swaths over Savona: where each antenna column actually sampled",x=0.03,ha="left",fontsize=13,fontweight="bold",y=0.955)
fig.text(0.03,0.905,"Panel A: GPS antenna track of every swath. Panel B: a 30 m stretch, with the five antenna columns offset 0.17 m left of the GPS track and 0.335 m apart, on aerial imagery.",fontsize=9,color=C["text2"])
fig.text(0.985,0.015,"Basemap: Google satellite imagery, 75% opacity in A",ha="right",fontsize=7,color=C["muted"])
out=S+f"/savona_swaths_antennas_{SW}.png"; fig.savefig(out,facecolor=C["surface"]); print("saved",out)
