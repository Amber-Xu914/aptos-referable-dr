import sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
P=sys.argv[1]
INK,INK2,BLUE,LBLUE,GREY,LINE='#0b0b0b','#52514e','#2a78d6','#eaf2fc','#f5f4f1','#d6d4ce'
fig,ax=plt.subplots(figsize=(13.33,5.6)); ax.set_xlim(0,13.33); ax.set_ylim(0,5.6); ax.axis('off'); fig.patch.set_facecolor('white')
# --- pathway
ax.text(0.3,5.3,'PATHWAY TO CLINICAL USE',fontsize=10.5,weight='bold',color=INK2)
steps=[('Prototype','APTOS, retrospective'),('External validation','Australian clinics'),('Prospective trial','real primary care'),('TGA approval','Class IIa, ARTG'),('MSAC funding','new MBS item'),('Routine use','monitor & update')]
n=len(steps); x0,x1=0.9,12.43; xs=[x0+i*(x1-x0)/(n-1) for i in range(n)]; y=4.5
ax.plot([x0,x1],[y,y],color=LINE,lw=3,zorder=1); ax.plot([x0,xs[1]-0.5],[y,y],color=BLUE,lw=3,zorder=1)
for i,(x,(h,b)) in enumerate(zip(xs,steps)):
    here=i==0
    ax.add_patch(Circle((x,y),0.3,fc=BLUE if here else 'white',ec=BLUE if here else LINE,lw=2.5,zorder=2))
    ax.text(x,y,str(i+1),ha='center',va='center',fontsize=12,weight='bold',color='white' if here else INK2,zorder=3)
    ax.text(x,y-0.5,h,ha='center',va='top',fontsize=10.8,weight='bold',color=BLUE if here else INK,linespacing=1.1)
    ax.text(x,y-0.88,b,ha='center',va='top',fontsize=9.3,color=INK2)
ax.text(xs[0],y+0.48,'WE ARE HERE',ha='center',fontsize=9,weight='bold',color=BLUE)
# --- cards
cards=[('Regulation','Class IIa medical device\n(decision support, TGA rule 4.5)','ARTG listing + ISO 13485'),
       ('Reimbursement','Photos already funded\n(MBS 12325 / 12326)','AI grading not funded → MSAC'),
       ('Evidence needed','Australian external +\nprospective validation','Benchmark: IDx-DR, 900-patient trial')]
w,g,x=4.1,0.22,0.3; yb,hb=1.2,1.55
for h,a,b in cards:
    ax.add_patch(FancyBboxPatch((x,yb),w,hb,boxstyle='round,pad=0,rounding_size=0.12',fc=GREY,ec='none'))
    ax.add_patch(FancyBboxPatch((x,yb+hb-0.08),w,0.08,boxstyle='square,pad=0',fc=BLUE,ec='none'))
    ax.text(x+0.25,yb+hb-0.35,h,fontsize=12.5,weight='bold',color=INK,va='center')
    ax.text(x+0.25,yb+hb-0.62,a,fontsize=10.2,color=INK,va='top',linespacing=1.3)
    ax.text(x+0.25,yb+0.2,b,fontsize=9.4,color=INK2,va='center')
    x+=w+g
# --- answer
ax.add_patch(FancyBboxPatch((0.3,0.05),12.74,0.9,boxstyle='round,pad=0,rounding_size=0.12',fc=BLUE,ec='none'))
ax.text(0.6,0.5,'Answer',fontsize=12.5,weight='bold',color='white',va='center')
ax.text(1.85,0.5,'A promising triage tool (AUROC 0.98, specificity 97%), but not clinically validated yet:\nit needs Australian validation, TGA approval and funding before use with patients.',fontsize=11,color='white',va='center',linespacing=1.4)
fig.savefig(P+'/SLIDE14_translation_v2.png',dpi=200,bbox_inches='tight',pad_inches=0.08)
