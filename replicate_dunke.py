"""① 재현: Dunke & Nickel (2025) 실험 480개를 우리 계획 모형으로 다시 풂. 중간에 멈춰도 다시 실행하면 이어서 계산.
실행: py replicate_dunke.py  → results_dunke/replication_grid.csv, 끝나면 py replicate_tables.py
"""
import sys, time, os, csv
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from peakshift import dunke as D, planner
base='dunke_code/'
prices=D.load_prices(base+[f for f in os.listdir(base) if f.endswith('.xlsx')][0])
out='results_dunke/replication_grid.csv'
os.makedirs('results_dunke', exist_ok=True)
done=set()
if os.path.exists(out):
    for r in csv.DictReader(open(out)): done.add((r['jobs'],r['day'],r['alg'],r['start'],r['fixed'],r['ppa']))
f=open(out,'a',newline=''); w=csv.writer(f)
if not done: w.writerow(['jobs','day','alg','start','fixed','ppa','status','cost','bound','sec']); f.flush()
for nj in [3,5,10]:
    for day in D.CALENDAR_DAYS:
        machines,jobs,ops,precs,seqs,seed=D.generate_instance_data(nj,day)
        for start in [0,32]:
            for fixed in ['dyn','fix']:
                for ppa in ['dyn','ppa']:
                    for alg in (['heu','ex'] if nj<=5 else ['heu']):
                        key=(str(nj),str(day),alg,str(start),fixed,ppa)
                        if key in done: continue
                        inst=D.build_instance(machines,jobs,ops,seqs,D.effective_price(prices[day],fixed,ppa),start,alg)
                        t=time.time(); pl=planner.solve(inst,np.zeros(96),time_limit=60)
                        w.writerow([nj,day,alg,start,fixed,ppa,pl.status,round(D.paper_cost(pl.objective),4),round(D.paper_cost(pl.bound),4),round(time.time()-t,1)]); f.flush()
print('DONE')
