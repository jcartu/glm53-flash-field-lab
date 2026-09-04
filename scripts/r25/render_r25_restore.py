#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

BG='#090D13'; PANEL='#111821'; GRID='#263242'; TEXT='#EDF4FA'; MUTED='#92A2B5'
BLUE='#4FA8FF'; GREEN='#46D785'; AMBER='#F2B84B'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':PANEL,'axes.edgecolor':GRID,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'text.color':TEXT,'font.family':'DejaVu Sans','font.size':11})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,5.2))
fig.subplots_adjust(top=.78,bottom=.18,wspace=.35)

def panel(ax,vals,title,ylabel,labels):
    bars=ax.bar(['R24','R25'],vals,color=[BLUE,GREEN],width=.58,zorder=3)
    for bar,val,label in zip(bars,vals,labels):
        ax.annotate(label,(bar.get_x()+bar.get_width()/2,val),xytext=(0,6),textcoords='offset points',ha='center',color=TEXT,fontweight='bold',fontsize=12,path_effects=[pe.withStroke(linewidth=2,foreground=BG)])
    ax.set_title(title,fontweight='bold',color=TEXT); ax.set_ylabel(ylabel)
    ax.grid(axis='y',color=GRID,alpha=.5,zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

panel(ax1,[2.418,.761],'LMCache internal restore','seconds, lower is better',['2.418s','0.761s'])
ax1.text(.5,2.08,'3.2× faster',ha='center',color=GREEN,fontweight='bold',fontsize=14)
panel(ax2,[12.267,11.879],'Whole request wall time','seconds, lower is better',['12.27s','11.88s'])
ax2.text(.5,10.3,'3.2% faster',ha='center',color=AMBER,fontweight='bold',fontsize=14)
fig.suptitle('R25 does exactly what it says',fontsize=20,fontweight='bold',color=GREEN,y=.96)
fig.text(.5,.885,'same cached 950,272-token prompt, same DCP4 DFlash K7 packed-NVFP4 config',ha='center',color=TEXT,fontsize=10)
fig.text(.5,.84,'the cache transfer is much faster, total request time is mostly generation and setup now',ha='center',color=MUTED,fontsize=9)
fig.text(.5,.045,'4× RTX PRO 6000 Blackwell · PCIe 5 single root · pinned R24/R25 digests · exact output hit on both',ha='center',color=MUTED,fontsize=8)
fig.savefig('/home/josh/omp-workspace/drock-lmcache/r24-battery/charts/05-r25-restore.png',dpi=160,bbox_inches='tight',facecolor=BG)
