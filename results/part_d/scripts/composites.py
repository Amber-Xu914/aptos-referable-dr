import pandas as pd, numpy as np, json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve
S=sys.argv[1]; O=S+'/out'; P=S+'/slides'
INK,INK2,GRID,SURF='#0b0b0b','#52514e','#e4e3df','#ffffff'
C={'baseline':'#eb6834','A':'#1baf7a','cnn':'#2a78d6'}
plt.rcParams.update({'font.size':11,'axes.edgecolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False})
summ=lambda m: json.load(open(f'{O}/{m}/summary.json'))
d=pd.read_csv(S+'/all_preds.csv'); d['y']=(d.true_grade>=2).astype(int)
show=[('baseline_colour_logreg','baseline','Colour baseline'),('A_rf_all','A','Classical: 38 features + RF'),('B_efficientnet_b0_finetune_s1','cnn','CNN: EfficientNet-B0 fine-tuned')]
# ================= SLIDE 9
fig=plt.figure(figsize=(13.33,6.4)); gs=fig.add_gridspec(2,3,height_ratios=[0.75,1.75],width_ratios=[1.1,1,1.05],hspace=0.22,wspace=0.32,left=0.05,right=0.98,top=0.92,bottom=0.13)
fig.text(0.05,0.955,'CNN and classical pipeline perform similarly on the test set; both clearly beat the colour baseline',fontsize=14,weight='bold',color=INK)
ax=fig.add_subplot(gs[0,:]); ax.axis('off')
T=[['Model','AUROC (95% CI)','Grade 1 vs 2–4\nAUROC','Sensitivity','Specificity','Precision','F1','Accuracy'],
['Colour baseline','0.898 (0.871–0.923)','0.637','0.90','0.80','0.75','0.82','0.84'],
['Classical: 38 features + RF','0.976 (0.965–0.986)','0.892','0.84','0.96','0.94','0.89','0.91'],
['CNN: EfficientNet-B0 fine-tuned\n(mean ± SD, 3 seeds)','0.984 ± 0.002','0.917 ± 0.012','0.83 ± 0.01','0.97 ± 0.01','0.96 ± 0.01','0.89 ± 0.01','0.91 ± 0.01']]
t=ax.table(cellText=T[1:],colLabels=T[0],loc='center',cellLoc='center',colWidths=[.22,.15,.11,.1,.1,.1,.1,.1])
t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1,1.75)
for (i,j),c in t.get_celld().items():
    c.set_edgecolor(GRID); c.set_facecolor(SURF)
    if i==0: c.set_text_props(weight='bold'); c.set_facecolor('#eef4fc')
    if j==0 and i>0: c.get_text().set_color({1:C['baseline'],2:'#0f7a52',3:C['cnn']}[i]); c.set_text_props(weight='bold')
# ROC
ax=fig.add_subplot(gs[1,0]); ax.plot([0,1],[0,1],color=GRID,ls='--',lw=1.2)
for m,k,lab in show:
    tt=d[(d.model_type==m)&(d.split=='test')]; fpr,tpr,_=roc_curve(tt.y,tt.model_score); s=summ(m)['test_metrics_at_main_threshold']
    ax.plot(fpr,tpr,color=C[k],lw=2,label=lab); ax.scatter([1-s['specificity']],[s['sensitivity']],s=70,color=C[k],edgecolor='white',lw=2,zorder=3)
ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity'); ax.set_xlim(-0.01,1); ax.set_ylim(0,1.02); ax.grid(color=GRID,lw=0.7); ax.set_axisbelow(True)
ax.set_title('ROC, test set (dot = operating point)',loc='left',fontsize=11.5,weight='bold')
ax.legend(frameon=False,fontsize=8.8,loc='lower right')
# calibration
ax=fig.add_subplot(gs[1,1]); ax.plot([0,1],[0,1],color=GRID,ls='--',lw=1.2)
for m,k,lab in show[1:]:
    tt=d[(d.model_type==m)&(d.split=='test')]; b=pd.cut(tt.model_score,np.linspace(0,1,11),include_lowest=True)
    g=tt.groupby(b,observed=True).agg(p=('model_score','mean'),o=('y','mean')); e=summ(m)['calibration']['ECE']
    ax.plot(g.p,g.o,color=C[k],lw=2,marker='o',ms=6,mec='white',mew=1.5,label=f"{lab.split(':')[0]} (ECE {e:.3f})")
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed fraction referable'); ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.grid(color=GRID,lw=0.7); ax.set_axisbelow(True)
ax.set_title('Calibration, test set',loc='left',fontsize=11.5,weight='bold'); ax.legend(frameon=False,fontsize=9,loc='lower right')
# threshold box
ax=fig.add_subplot(gs[1,2]); ax.axis('off')
ax.add_patch(FancyBboxPatch((0,0.02),1,0.96,boxstyle='round,pad=0,rounding_size=0.03',fc='#f5f9fe',ec='#cfe0f6',transform=ax.transAxes))
txt=[('How the threshold was set',True),('Set on validation only: the highest\nthreshold that still gives ≥ 85%\nsensitivity. Applied once, unchanged.',False),
('Why sensitivity first',True),('A missed referable eye can progress to\nvision loss before the next screen;\na false alarm costs one extra visit.',False),
('What happened on test',True),('CNN: sensitivity 83% (CI 77–87%),\nspecificity 97%; slightly under target.\nCNN vs classical: AUROC +0.005\n(CI −0.003 to +0.014), not significant.',False)]
y=0.92
for s,b in txt:
    ax.text(0.05,y,s,transform=ax.transAxes,fontsize=10.5 if b else 9.6,weight='bold' if b else 'normal',color=INK if b else INK2,va='top')
    y-=0.075 if b else 0.064*(s.count('\n')+1)+0.04
fig.text(0.05,0.012,'Test set n = 550 (225 referable). Test set used once. ROC and calibration curves: CNN seed 1 (pre-specified). Significance: paired bootstrap, 2,000 resamples.',fontsize=8.5,color=INK2)
fig.savefig(P+'/SLIDE09_test_results.png',dpi=200); plt.close()
# ================= SLIDE 10
from PIL import Image
fig=plt.figure(figsize=(13.33,6.4))
fig.text(0.04,0.955,'Errors sit at the mild / moderate boundary; some severe eyes are still missed',fontsize=14,weight='bold',color=INK)
ex=[('TP_352e4a939242','True positive','Grade 3 · score 1.00 · referred','Many haemorrhages and exudates','#0f7a52'),
    ('TN_0125fbd2e791','True negative','Grade 0 · score 0.00 · not referred','Healthy retina','#0f7a52'),
    ('FP_f481f76a6b75','False positive','Grade 1 · score 0.99 · referred','Yellow exudate-like spots: borderline case','#c0392b'),
    ('FN_bfdee9be1f1d','False negative','Grade 4 · score 0.0003 · not referred','"Quiet" eye: few visible lesions; missed by all 3 seeds','#c0392b')]
pos=[(0.035,0.5),(0.255,0.5),(0.035,0.06),(0.255,0.06)]
for (f,h,sub,note,col),(x,y) in zip(ex,pos):
    ax=fig.add_axes([x,y+0.075,0.2,0.3]); ax.imshow(Image.open(f'{S}/fundus/{f}.png')); ax.axis('off')
    fig.text(x,y+0.39,h,fontsize=11.5,weight='bold',color=col); fig.text(x,y+0.05,sub,fontsize=9.3,color=INK)
    fig.text(x,y+0.015,note,fontsize=8.6,color=INK2,wrap=True)
ax=fig.add_axes([0.53,0.33,0.45,0.52]); w=0.26
for i,(m,k,lab) in enumerate(show):
    g=pd.read_csv(f'{O}/{m}/referral_by_grade.csv'); ax.bar(g.grade+(i-1)*w,g.fraction_referred,width=w-0.03,color=C[k],label=lab)
ax.set_xticks(range(5)); ax.set_xticklabels(['0\nNo DR','1\nMild','2\nModerate','3\nSevere','4\nProliferative'],fontsize=10)
ax.axvspan(1.5,4.5,color=C['cnn'],alpha=0.05,zorder=0); ax.text(3,1.05,'should be referred',ha='center',fontsize=9.5,color=INK2); ax.text(0.5,1.05,'should not be referred',ha='center',fontsize=9.5,color=INK2)
ax.set_ylim(0,1.12); ax.set_ylabel('Fraction referred'); ax.grid(axis='y',color=GRID,lw=0.7); ax.set_axisbelow(True)
ax.set_title('Referral rate by original grade (test set)',loc='left',fontsize=11.5,weight='bold')
ax.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(0.5,-0.16),ncol=3)
fig.text(0.53,0.035,'CNN seed 1: all 10 false positives are grade 1 (none grade 0).\n31 of 40 misses are grade 2; 5 grade-4 eyes missed by all 3 seeds.\nAll 18 label-conflict test images were classified correctly.',fontsize=9.6,color=INK,linespacing=1.4)
fig.text(0.035,0.012,'Images shown after CLAHE preprocessing (model input). Labels = APTOS grade; scores = CNN seed 1.',fontsize=8.3,color=INK2)
fig.savefig(P+'/SLIDE10_errors.png',dpi=200); plt.close()
# ================= SLIDE 14 roadmap
fig,ax=plt.subplots(figsize=(13.33,3.4)); ax.set_xlim(0,13.33); ax.set_ylim(0,3.4); ax.axis('off')
steps=[('Research prototype','Retrospective test on\none public dataset\n(APTOS, India)',True),
('External validation','Australian images &\ncameras, incl. Aboriginal\nMedical Services',False),
('Prospective study','Real primary-care\nworkflow; multi-grader\nor OCT reference',False),
('TGA approval','ARTG inclusion, likely\nClass IIa (rule 4.5);\nISO 13485 QMS',False),
('Funding pathway','MSAC application;\nphotos already funded\n(MBS 12325 / 12326)',False),
('Routine use','Post-market monitoring,\nchange control\nfor retraining',False)]
x0,wd,gap=0.25,1.95,0.22
for i,(h,b,here) in enumerate(steps):
    x=x0+i*(wd+gap)
    ax.add_patch(FancyBboxPatch((x,0.5),wd,2.05,boxstyle='round,pad=0,rounding_size=0.12',fc='#2a78d6' if here else '#f5f4f1',ec='#2a78d6' if here else '#d6d4ce',lw=1.4))
    ax.text(x+wd/2,2.25,h,ha='center',va='center',fontsize=10.3,weight='bold',color='white' if here else INK)
    ax.text(x+wd/2,1.35,b,ha='center',va='center',fontsize=8.7,color='white' if here else INK2,linespacing=1.4)
    if i<len(steps)-1: ax.annotate('',xy=(x+wd+gap-0.02,1.52),xytext=(x+wd+0.02,1.52),arrowprops=dict(arrowstyle='->',color=INK2,lw=1.4))
ax.text(x0+wd/2,0.25,'▲ we are here',ha='center',fontsize=10.5,weight='bold',color='#2a78d6')
ax.text(x0+wd+gap,0.25,'Benchmark: IDx-DR needed a 900-patient prospective trial before FDA authorisation (2018)',fontsize=9.5,color=INK2,va='center')
ax.text(0.25,3.12,'Pathway to clinical use',fontsize=14,weight='bold',color=INK)
fig.savefig(P+'/SLIDE14_pathway.png',dpi=200,bbox_inches='tight'); plt.close()
