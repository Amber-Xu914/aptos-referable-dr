import pandas as pd, numpy as np, json, sys, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
S=sys.argv[1]; O=S+'/out'; P=S+'/slides'
C={'surf':'#fcfcfb','ink':'#0b0b0b','ink2':'#52514e','grid':'#e4e3df',
   'baseline':'#eb6834','A_rf_all':'#1baf7a','finetune':'#2a78d6'}
plt.rcParams.update({'font.size':13,'axes.edgecolor':C['ink2'],'axes.labelcolor':C['ink'],
 'xtick.color':C['ink2'],'ytick.color':C['ink2'],'figure.facecolor':C['surf'],'axes.facecolor':C['surf'],
 'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
mc=pd.read_csv(O+'/model_comparison.csv')
def summ(m): return json.load(open(f'{O}/{m}/summary.json'))
rows=[]
for m in mc.model:
    s=summ(m); t=s['test_metrics_at_main_threshold']; nc=s['test_without_label_conflicts']
    rows.append(dict(model=m,val_auroc=mc.set_index('model').loc[m,'val_AUROC'],auroc=s['test_AUROC'],
      auroc_g1=s['test_AUROC_grade1_vs_2to4'],sens=t['sensitivity'],spec=t['specificity'],ppv=t['PPV_precision'],
      npv=t['NPV'],f1=t['F1'],fn=t['FN'],fp=t['FP'],ece=s['calibration']['ECE'],brier=s['calibration']['Brier'],
      auroc_nc=nc['AUROC'],sens_nc=nc.get('sensitivity',np.nan),spec_nc=nc.get('specificity',np.nan),
      lo=s['test_95CI_bootstrap']['AUROC'][0],hi=s['test_95CI_bootstrap']['AUROC'][1]))
per=pd.DataFrame(rows); per.to_csv(P+'/per_run_test_metrics.csv',index=False)
per['base']=per.model.str.replace(r'_s[123]$','',regex=True)
num=[c for c in per.columns if c not in('model','base')]
agg=per.groupby('base')[num].agg(['mean','std','count'])
agg.columns=['_'.join(c) for c in agg.columns]; agg=agg.sort_values('val_auroc_mean',ascending=False)
agg.to_csv(P+'/seed_aggregated_test_metrics.csv')
names={'B_efficientnet_b0_finetune':'CNN: EfficientNet-B0 fine-tuned (3 seeds)','B_hybrid_efficientnet_b0_imagenet':'Hybrid CNN + features (1 run)',
 'B_efficientnet_b0_frozen':'CNN: EfficientNet-B0 frozen (3 seeds)','A_rf_all':'Part A: 38 features + random forest',
 'B_smallcnn':'SmallCNN from scratch (3 seeds)','A_rf_bright_dark':'Part A: lesion features only + RF','baseline_colour_logreg':'Baseline: colour + logistic regression'}
def f(r,k,d=3):
    m=r[k+'_mean']; s=r[k+'_std']
    return f"{m:.{d}f}" if np.isnan(s) else f"{m:.{d}f} ± {s:.{d}f}"
tab=[]
for b,r in agg.iterrows():
    tab.append({'Model':names[b],'Val AUROC':f(r,'val_auroc'),'Test AUROC':f(r,'auroc'),'Grade 1 vs 2–4 AUROC':f(r,'auroc_g1'),
     'Sensitivity':f(r,'sens'),'Specificity':f(r,'spec'),'PPV':f(r,'ppv'),'NPV':f(r,'npv'),
     'Missed (FN /225)':f(r,'fn',1),'False alarms (FP /325)':f(r,'fp',1),'ECE':f(r,'ece'),
     'Test AUROC w/o label conflicts':f(r,'auroc_nc')})
T=pd.DataFrame(tab); T.to_csv(P+'/slide7_results_table.csv',index=False); print(T.to_string())
# ---- per 1000 at 10% prevalence
pr=[]
for b,r in agg.iterrows():
    se,sp=r.sens_mean,r.spec_mean; tp=100*se; fp=900*(1-sp)
    pr.append({'Model':names[b],'Sensitivity':round(se,3),'Specificity':round(sp,3),'Caught /100 referable':round(tp,1),'Missed /100':round(100-tp,1),
     'Unnecessary referrals /1000':round(fp,1),'PPV at 10%':round(tp/(tp+fp),3),'Referrals /1000':round(tp+fp,1),'Cost of unnecessary referrals /1000 (AUD, $79.80 each)':round(fp*79.80)})
PR=pd.DataFrame(pr); PR.to_csv(P+'/per1000_at_10pct_prevalence.csv',index=False); print(PR.to_string())
# ---- ROC figure: 3 models, finetune_s1 as pre-specified representative seed
d=pd.read_csv(S+'/all_preds.csv'); d['y']=(d.true_grade>=2).astype(int)
show=[('baseline_colour_logreg','baseline','Baseline colour LR'),('A_rf_all','A_rf_all','Part A features + RF'),('B_efficientnet_b0_finetune_s1','finetune','EfficientNet-B0 fine-tuned')]
fig,ax=plt.subplots(figsize=(7,6.2))
ax.plot([0,1],[0,1],color=C['grid'],lw=1.5,ls='--',zorder=1)
for m,k,lab in show:
    t=d[(d.model_type==m)&(d.split=='test')]; fpr,tpr,_=roc_curve(t.y,t.model_score)
    s=summ(m); tm=s['test_metrics_at_main_threshold']
    ax.plot(fpr,tpr,color=C[k],lw=2,label=f"{lab}  (AUROC {s['test_AUROC']:.3f})",zorder=2)
    ax.scatter([1-tm['specificity']],[tm['sensitivity']],s=90,color=C[k],edgecolor=C['surf'],linewidth=2,zorder=3)
ax.set_xlabel('1 − specificity (false-alarm rate)'); ax.set_ylabel('Sensitivity')
ax.set_xlim(-0.01,1); ax.set_ylim(0,1.01); ax.grid(color=C['grid'],lw=0.8); ax.set_axisbelow(True)
ax.legend(loc='lower right',frameon=False,fontsize=11)
ax.set_title('Test set ROC (n = 550); dots = threshold fixed on validation',fontsize=12.5,color=C['ink'],loc='left')
fig.tight_layout(); fig.savefig(P+'/slide7_roc_test.png',dpi=200); plt.close()
# ---- Bar chart sens/spec by model (mean ± sd)
order=['baseline_colour_logreg','A_rf_all','B_efficientnet_b0_finetune']; lbl=['Baseline\ncolour LR','Part A\nfeatures + RF','EfficientNet-B0\nfine-tuned (3 seeds)']
fig,axs=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
for ax,(k,tt) in zip(axs,[('sens','Sensitivity (catch referable DR)'),('spec','Specificity (avoid false alarms)')]):
    v=[agg.loc[o,k+'_mean'] for o in order]; e=[0 if np.isnan(agg.loc[o,k+'_std']) else agg.loc[o,k+'_std'] for o in order]
    cols=[C['baseline'],C['A_rf_all'],C['finetune']]
    ax.bar(range(3),v,yerr=e,color=cols,width=0.6,capsize=4,error_kw=dict(ecolor=C['ink2'],lw=1.2))
    for i,x in enumerate(v): ax.text(i,x+0.02,f'{x:.2f}',ha='center',color=C['ink'],fontsize=12)
    ax.set_xticks(range(3)); ax.set_xticklabels(lbl,fontsize=11); ax.set_title(tt,loc='left',fontsize=12.5)
    ax.set_ylim(0,1.08); ax.grid(axis='y',color=C['grid'],lw=0.8); ax.set_axisbelow(True)
axs[0].axhline(0.85,color=C['ink2'],ls=':',lw=1.2); axs[0].text(-0.45,0.995,'dotted line = 85% target set on validation',fontsize=9.5,color=C['ink2'],ha='left',va='bottom')
fig.suptitle('Test set, threshold fixed on validation (≥85% sensitivity)',x=0.01,ha='left',fontsize=13)
fig.tight_layout(); fig.savefig(P+'/slide7_sens_spec_bars.png',dpi=200); plt.close()
# ---- referral by grade, 3 models
fig,ax=plt.subplots(figsize=(8,5.2)); w=0.26
for i,(m,k,lab) in enumerate(show):
    g=pd.read_csv(f'{O}/{m}/referral_by_grade.csv')
    ax.bar(g.grade+(i-1)*w,g.fraction_referred,width=w-0.03,color=C[k],label=lab)
ax.set_xticks(range(5)); ax.set_xticklabels(['0\nNo DR','1\nMild','2\nModerate','3\nSevere','4\nProliferative'])
ax.axvspan(1.5,4.5,color='#2a78d6',alpha=0.05,zorder=0); ax.text(3,1.06,'should be referred',ha='center',fontsize=10,color=C['ink2'])
ax.text(0.5,1.06,'should not be referred',ha='center',fontsize=10,color=C['ink2'])
ax.set_ylabel('Fraction referred'); ax.set_ylim(0,1.12); ax.grid(axis='y',color=C['grid'],lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=3)
ax.set_title('Who gets referred, by true grade (test set)',loc='left',fontsize=12.5)
fig.tight_layout(); fig.savefig(P+'/slide8_referral_by_grade.png',dpi=200); plt.close()
# ---- confusion matrix finetune_s1
s=summ('B_efficientnet_b0_finetune_s1')['test_metrics_at_main_threshold']
M=np.array([[s['TN'],s['FP']],[s['FN'],s['TP']]])
fig,ax=plt.subplots(figsize=(5.8,4.8))
ax.imshow(M,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b',['#eef4fc','#2a78d6']))
lab=[['True negative','False alarm'],['Missed','True referral']]
for i in range(2):
  for j in range(2): ax.text(j,i,f'{M[i,j]}\n{lab[i][j]}',ha='center',va='center',fontsize=13,color='white' if M[i,j]>150 else C['ink'])
ax.set_xticks([0,1]); ax.set_xticklabels(['Not referred','Referred']); ax.set_yticks([0,1]); ax.set_yticklabels(['Non-referable\n(grade 0–1)','Referable\n(grade 2–4)'])
ax.set_xlabel('Model decision'); ax.set_ylabel('Reference grade'); ax.spines[:].set_visible(False)
ax.set_title(f"EfficientNet-B0 fine-tuned, seed 1\ntest n = 550, threshold {s['threshold']:.2f} from validation",fontsize=11.5,loc='left')
fig.tight_layout(); fig.savefig(P+'/slide8_confusion_finetune_s1.png',dpi=200); plt.close()
# ---- calibration finetune_s1 vs A_rf_all
fig,ax=plt.subplots(figsize=(6,5.4)); ax.plot([0,1],[0,1],color=C['grid'],ls='--',lw=1.5)
for m,k,lab in show[1:]:
    t=d[(d.model_type==m)&(d.split=='test')]; b=pd.cut(t.model_score,np.linspace(0,1,11),include_lowest=True)
    g=t.groupby(b,observed=True).agg(p=('model_score','mean'),o=('y','mean'),n=('y','size'))
    e=summ(m)['calibration']['ECE']
    ax.plot(g.p,g.o,color=C[k],lw=2,marker='o',ms=8,mec=C['surf'],mew=2,label=f'{lab} (ECE {e:.3f})')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed fraction referable'); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.grid(color=C['grid'],lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False,fontsize=10.5,loc='upper left')
ax.set_title('Calibration on test set (10 bins)',loc='left',fontsize=12.5)
fig.tight_layout(); fig.savefig(P+'/slide8_calibration.png',dpi=200); plt.close()
