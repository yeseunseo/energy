set -e
C="--plan-time 20 --mpc-time 2 --rollout-n 10 --rollout-k 2 --days 20"
for sd in 1 2 3; do python3 run_experiment.py $C --seed $sd --out val/base_s$sd > val/log_base_s$sd.txt 2>&1; done
python3 run_experiment.py $C --seed 1 --season springfall --policies A B D --out val/spring > val/log_spring.txt 2>&1
python3 run_experiment.py $C --seed 1 --pbar loose --policies A B D --out val/loose > val/log_loose.txt 2>&1
python3 run_experiment.py $C --seed 1 --urgent-rate 0.3 --policies A B D --out val/urgent > val/log_urgent.txt 2>&1
python3 run_experiment.py $C --seed 1 --cloud 0.55 --policies A B D --out val/cloudy > val/log_cloudy.txt 2>&1
echo ALLDONE > val/done.txt
