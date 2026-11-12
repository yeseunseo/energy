"""① 재현 결과를 논문 표 8 · 9와 비교."""
import pandas as pd, numpy as np
d=pd.read_csv('results_dunke/replication_grid.csv')
d['cost']=d['cost'].astype(float)
def stats(x):
    x=np.array(x); return dict(n=len(x),최소=round(x.min(),2),평균=round(x.mean(),2),최대=round(x.max(),2),중앙=round(np.median(x),2),
                               within1=round(np.mean(np.abs(x)<=1)*100,2))
def imp(a,b): return (a-b)/np.abs(a)*100   # a 대비 b의 개선율
print('[표 8] 정확한 방식(ex)의 근사 방식(heu) 대비 비용 개선율 (%)')
paper8={3:dict(최소=0.0,평균=6.17,최대=59.65,중앙=0.66,within1=55.21),5:dict(최소=0.0,평균=4.98,최대=46.09,중앙=1.12,within1=47.92)}
for nj in [3,5]:
    x=d[d.jobs==nj].pivot_table(index=['day','start','fixed','ppa'],columns='alg',values='cost').dropna()
    if len(x)==0 or len(x.columns)<2: continue
    print(f'  {nj}개 우리:', stats(imp(x['heu'],x['ex'])), '\n      논문:', paper8[nj])
print('[표 9] 00:00 시작의 08:00 시작 대비 비용 개선율 (%)')
paper9={3:dict(최소=0.0,평균=4.08,최대=38.92,중앙=0.0,within1=65.62),5:dict(최소=0.0,평균=10.18,최대=66.83,중앙=0.0,within1=65.62),10:dict(최소=0.0,평균=14.39,최대=89.62,중앙=1.22,within1=50.0)}
for nj in [3,5,10]:
    x=d[d.jobs==nj].pivot_table(index=['day','alg','fixed','ppa'],columns='start',values='cost').dropna()
    if len(x)==0 or len(x.columns)<2: continue
    print(f'  {nj}개 우리:', stats(imp(x[32],x[0])), '\n      논문:', paper9[nj])
print('최적해 비율:', round((d.status=='OPTIMAL').mean()*100,1),'% / 행', len(d))
