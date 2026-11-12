import mostRemainingWork as mrw
import docplex.mp.model
import numpy
import copy
import time
import json

def determineParetoSolutions(machines, jobs, ops, precs, energy, startingSlot, fixedTariff, ppaPrice, algorithmType):
    energyNew = copy.deepcopy(energy)
    if ppaPrice == "ppa":
        for slot in range(0, 96):
            energyNew["costs"]["renewable"][slot] = 95.18
    if fixedTariff == "fix":
        avgCosts = numpy.average(list(energyNew["costs"]["conventional"].values()))
        for slot in range(0, 96):
            energyNew["costs"]["conventional"][slot] = avgCosts
    solution = mrw.sequenceByMRW(machines, jobs, ops, precs)
    solution = solveJobShop(machines, jobs, ops, precs, energyNew, solution, "makespan", 999999, 999999, startingSlot, algorithmType)
    minMakespan = round(solution["makespan"])
    solution = mrw.sequenceByMRW(machines, jobs, ops, precs)
    solution = solveJobShop(machines, jobs, ops, precs, energyNew, solution, "energy costs", 999999, 999999, startingSlot, algorithmType)
    maxMakespan = round(solution["makespan"])
    paretoSolutions = []
    for ms in range(int(minMakespan), min(96, int(maxMakespan)) + 1):
        solution = mrw.sequenceByMRW(machines, jobs, ops, precs)
        start = time.time()
        solution = solveJobShop(machines, jobs, ops, precs, energyNew, solution, "energy costs", 999999, ms, startingSlot, algorithmType)
        stop = time.time()
        if len(paretoSolutions) == 0 or solution["makespan"] != paretoSolutions[len(paretoSolutions) - 1]["makespan"]:
            paretoSol = {"makespan": solution["makespan"], "energy costs": solution["energy costs"], "start time": solution["start time"], "finish time": solution["finish time"], "energy consumption": solution["energy consumption"], "emissions": solution["emissions"], "renewable share": solution["renewable share"], "CPU time": stop - start}
            paretoSolutions.append(paretoSol)
        print(str(int(ms)) + "/" + str(int(minMakespan)) + "-" + str(int(maxMakespan)))
        with open("paretoSolutions.json", 'w') as file:
          json.dump(paretoSolutions, file, indent=4)
    return paretoSolutions

def solveJobShop(machines, jobs, ops, precs, energy, solution, objectiveString, epsilonEnergyCosts, epsilonMakespan, startingSlot, algorithmType):
    M, T, R = [machine for machine in machines], range(0, 96), ["conventional", "renewable"]
    mdl = docplex.mp.model.Model()
    C_max = mdl.continuous_var()
    xz_idx = [(op["machine"], op["job"], t) for op in ops for t in T]
    x = mdl.binary_var_dict(xz_idx)
    z = mdl.binary_var_dict(xz_idx)
    z_proj_idx = [(m, t) for m in M for t in T]
    z_on, z_idle, z_off1, z_off2 = mdl.binary_var_dict(z_proj_idx), mdl.binary_var_dict(z_proj_idx), mdl.binary_var_dict(z_proj_idx), mdl.binary_var_dict(z_proj_idx)
    alpha_idx = [(r, t) for r in R for t in T]
    alpha = mdl.binary_var_dict(alpha_idx)
    ec_idx = [(m, r, t) for m in M for r in R for t in T]
    em_idx = [(m, t) for m in M for t in T]
    ec_on, em_on, ec_idle, em_idle = mdl.continuous_var_dict(ec_idx), mdl.continuous_var_dict(em_idx), mdl.continuous_var_dict(ec_idx), mdl.continuous_var_dict(em_idx)
    ec_em_proj_costs_idx = [(m, t) for m in M for t in T]
    ec_proj, em_proj = mdl.continuous_var_dict(ec_em_proj_costs_idx), mdl.continuous_var_dict(ec_em_proj_costs_idx)
    ec_projproj = mdl.continuous_var_dict([(t) for t in T])
    costs_on, costs_idle = mdl.continuous_var_dict(ec_em_proj_costs_idx, lb = -9999), mdl.continuous_var_dict(ec_em_proj_costs_idx, lb = -9999)
    t_start = mdl.continuous_var()
    
    makespan = C_max - t_start
    energyConsumption = mdl.sum(ec_proj[m, t] for m in M for t in T)
    emissions = mdl.sum(em_proj[m, t] for m in M for t in T)
    energyCosts = mdl.sum((costs_on[m, t] + costs_idle[m, t]) for m in M for t in T)
    
    if objectiveString == "makespan":
        mdl.objective = makespan
    if objectiveString == "energy costs":
        mdl.objective = energyCosts + 0.0001 * makespan
    if objectiveString == "none":
        mdl.objective = 0
    mdl.minimize(mdl.objective)
    
    if objectiveString != "none":
        mdl.add_constraint(energyCosts <= epsilonEnergyCosts)
        mdl.add_constraint(makespan <= epsilonMakespan)
    mdl.add_constraint(t_start >= startingSlot)
    
    mdl.add_constraints(t_start <= mdl.sum(t * x[jobs[j]["sequence"][0], j, t] for t in T) for j in jobs)
    mdl.add_constraints(mdl.sum(z[m, op["job"], t] for op in ops if op["machine"] == m) == z_on[m, t] for m in M for t in T)
    mdl.add_constraints(z_on[m, t] + z_idle[m, t] + z_off1[m, t] + z_off2[m, t] == 1 for m in M for t in T)    
    mdl.add_constraints(z_off1[m, t] + z_idle[m, t+1] <= 1 for m in M for t in T if t < max(T))    
    mdl.add_constraints(z_on[m, t] + z_off1[m, t+1] <= 1 for m in M for t in T if t < max(T))    
    mdl.add_constraints(z_idle[m, t] + z_off1[m, t+1] <= 1 for m in M for t in T if t < max(T))    
    mdl.add_constraints(z_idle[m, t] + z_off2[m, t+1] <= 1 for m in M for t in T if t < max(T))     
    mdl.add_constraints(z_off2[m, t] + z_off1[m, t+1] <= 1 for m in M for t in T if t < max(T))    
    mdl.add_constraints(z_off2[m, t] + z_on[m, t+1] <= 1 for m in M for t in T if t < max(T))
    mdl.add_constraints(z_off2[m, t] + z_idle[m, t+1] <= 1 for m in M for t in T if t < max(T)) 
    mdl.add_constraints(z_idle[m, 0] == 0 for m in M)
    mdl.add_constraints(z_off2[m, max(T)] == 1 for m in M)    
    mdl.add_constraints(mdl.sum(z[op["machine"], op["job"], t2] for t2 in T if t2 >= t and t2 <= t + op["processing time"] - 1) >= op["processing time"] * x[op["machine"], op["job"], t] for op in ops for t in T)
    mdl.add_constraints(z[op["machine"], op["job"], t] <= mdl.sum(x[op["machine"], op["job"], t2] for t2 in T if t2 >= t - op["processing time"] + 1 and t2 <= t) for op in ops for t in T)

    mdl.add_constraints(mdl.sum(x[op["machine"], op["job"], t] for t in T) == 1 for op in ops)
    mdl.add_constraints(mdl.sum((t + op["processing time"])* x[op["machine"], op["job"], t] for t in T) <= C_max for op in ops)
    mdl.add_constraints(mdl.sum(x[m, op["job"], t2] for op in ops if op["machine"] == m for t2 in T if t2 >= t - op["processing time"] + 1 and t2 <= t) <= 1 for m in M for t in T)
    mdl.add_constraints(mdl.sum((t + prec["from"]["processing time"]) * x[prec["from"]["machine"], prec["from"]["job"], t] for t in T) <= mdl.sum(t * x[prec["to"]["machine"], prec["to"]["job"], t] for t in T) for prec in precs)

    if objectiveString != "none":
        mdl.add_constraints(mdl.sum(alpha[r, t] for r in R) == 1 for t in T)     
        mdl.add_constraints(ec_on[m, r, t] <= machines[m]["consumption"]["on"] * z_on[m, t] for m in M for r in R for t in T)
        mdl.add_constraints(ec_on[m, r, t] <= machines[m]["consumption"]["on"] * alpha[r, t] for m in M for r in R for t in T)
        mdl.add_constraints(ec_on[m, r, t] >= machines[m]["consumption"]["on"] * (z_on[m, t] + alpha[r, t] - 1) for m in M for r in R for t in T)
        mdl.add_constraints(ec_idle[m, r, t] <= machines[m]["consumption"]["idle"] * z_idle[m, t] for m in M for r in R for t in T)
        mdl.add_constraints(ec_idle[m, r, t] <= machines[m]["consumption"]["idle"] * alpha[r, t] for m in M for r in R for t in T)
        mdl.add_constraints(ec_idle[m, r, t] >= machines[m]["consumption"]["idle"] * (z_idle[m, t] + alpha[r, t] - 1) for m in M for r in R for t in T)
        mdl.add_constraints(ec_proj[m, t] == mdl.sum(ec_on[m, r, t] + ec_idle[m, r, t] for r in R) for m in M for t in T)
        mdl.add_constraints(ec_projproj[t] == mdl.sum(ec_proj[m, t] for m in M) for t in T)
        mdl.add_constraints(em_on[m, t] == machines[m]["emissions"]["on"] * z_on[m, t] for m in M for t in T)
        mdl.add_constraints(em_idle[m, t] == machines[m]["emissions"]["idle"] * z_idle[m, t] for m in M for t in T)
        mdl.add_constraints(em_proj[m, t] == em_on[m, t] + em_idle[m, t] for m in M for t in T)    
        mdl.add_constraints(costs_on[m, t] == mdl.sum(ec_on[m, r, t] * energy["costs"][r][t] for r in R) for m in M for t in T)
        mdl.add_constraints(costs_idle[m, t] == mdl.sum(ec_idle[m, r, t] * energy["costs"][r][t] for r in R) for m in M for t in T)
    
    if algorithmType != "ex":
        b_idx = [(op1["machine"], op1["job"], op2["machine"], op2["job"]) for op1 in ops for op2 in ops if op1["machine"] == op2["machine"] and op1["job"] != op2["job"]]
        b = mrw.transformJobSequencesInto_b(solution["job sequences"])
        for i in b_idx:
            if i in b:
                mdl.add_constraint(mdl.sum(t * x[i[0], i[1], t] for t in T) <= mdl.sum(t * x[i[2], i[3], t] for t in T))
    
    mdl.solve(log_output = False)
    
    if mdl.solution == None:
        print("War nix")
        return None    
    
    if objectiveString != "none":
        solution["energy costs"], solution["emissions"], solution["makespan"], solution["energy consumption"], solution["start time"], solution["finish time"], solution["renewable share"] = mdl.solution.get_value(energyCosts), mdl.solution.get_value(emissions), round(mdl.solution.get_value(makespan)), mdl.solution.get_value(energyConsumption), round(mdl.solution.get_value(t_start)), round(mdl.solution.get_value(C_max)), sum(mdl.solution.get_value(ec_idle[m, "renewable", t]) + mdl.solution.get_value(ec_on[m, "renewable", t]) for m in M for t in T) / mdl.solution.get_value(energyConsumption)
    return solution