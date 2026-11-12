"""설명하는 수읽기 시연: 하루를 롤아웃으로 돌리고, 판단이 크게 갈린 결정 5개를 문장으로 출력.
실행: py explain_demo.py [날짜번호] [작업수]
"""
import sys, os
import numpy as np, pandas as pd
from peakshift import dunke as D, sim as S, planner
from peakshift.policies import RolloutPolicy, explain_decisions
from peakshift.pvreal import RealPV
from run_dunke import KOR_DATE
prices=D.load_prices('dunke_code/'+[f for f in os.listdir('dunke_code') if f.endswith('.xlsx')][0])
day=int(sys.argv[1]) if len(sys.argv)>1 else 258; nj=int(sys.argv[2]) if len(sys.argv)>2 else 10
machines,jobs,ops,precs,seqs,seed=D.generate_instance_data(nj,day)
inst=D.build_instance(machines,jobs,ops,seqs,D.effective_price(prices[day],'dyn','dyn'),32,'ex' if nj<=5 else 'heu')
rpv=RealPV('data/pv_weather_hourly.csv','data/kosep/*.csv',D.PV_CAP_RATIO*10,seed=1)
fc,act=rpv.day(pd.Timestamp(KOR_DATE[day])); q50=fc.predict()[1]
plan=planner.solve(inst,q50,time_limit=10)
maker=D.urgent_maker(machines,nj); rng=np.random.default_rng(day*1000+7)
events=[(h*4,'urgent',maker(rng,h*4)) for h in range(9,17) if rng.random()<0.1]; pact=D.sample_actual_times(inst.ops,rng,0.2)
pol=RolloutPolicy(plan,fc,None,maker,n_scen=8,k=2,seed=day,urgent_rate=0.1,p_spread=0.2,slack_guard=True,terminal='realistic'); pol.explain=True
st=S.State.initial(inst,act,events,p_act=pact); pol.install(st); S.run(st,inst,pol)
print('결정 기록', len(pol.log), '개, 계획과 다른 선택', sum(1 for r in pol.log if r['chosen']!=r['base']), '개')
for l in explain_decisions(pol,st,inst,top=5,unit=' (비용 단위)',scale=1/25): print(l)
