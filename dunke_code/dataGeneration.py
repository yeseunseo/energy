import mostRemainingWork as mrw
import pareto as pareto
import random
import copy
import pandas
import math

def generateMachineData(numMachines):
    random.seed(0)
    machines = {}
    for i in range(0, numMachines):
        machine = {}
        kilowatts = random.randint(1, 5)
        machine["consumption"] = {"on": kilowatts, "idle": 0.25 * kilowatts}
        emissions = random.randint(1, 5)
        machine["emissions"] = {"on": emissions, "idle": 0.25 * emissions}
        machines["M" + str(i + 1)] = machine
    return machines

def generateJobData(numJobs, machines, seed):    
    random.seed(seed)
    jobs = {}
    if numJobs == 3:
        maxProcessingTime = 20
    if numJobs == 5:
        maxProcessingTime = 15
    if numJobs == 10:
        maxProcessingTime = 10
    for j in range(1, numJobs + 1):
        sequence = random.sample(machines.keys(), k=random.randint(2, len(machines.keys())))
        machineTimes = {}
        for s in range(0, len(sequence)):
            machineTimes[sequence[s]] = random.randint(2, maxProcessingTime)
        jobs["J" + str(j)] = {"sequence": sequence, "machine times": machineTimes}
    for machine in machines:
         machineNotUsed = True
         for job in jobs:
             if machine in jobs[job]["sequence"]:
                 machineNotUsed = False
         if machineNotUsed:
             jobs[(list(jobs))[0]]["sequence"].append(machine)
             jobs[(list(jobs))[0]]["machine times"][machine] = random.randint(2, maxProcessingTime)
    return jobs

def generateJobsOpsPrecsData(numJobs, machines, seed, energy):
    solvable = False
    while not solvable:
        jobs = generateJobData(numJobs, machines, seed)
        ops, precs = getOperationsAndPrecedences(jobs)
        solution = mrw.sequenceByMRW(machines, jobs, ops, precs)
        solution = pareto.solveJobShop(machines, jobs, ops, precs, energy, solution, "none", 999999, 999999, 32, "heu")
        if solution != None:
            solvable = True
        else:
            seed += 1
    return jobs, ops, precs

def getHistoricalEnergyData(numSamples, filenameHistoricalEnergyData):
    dataframe = pandas.read_excel(filenameHistoricalEnergyData)
    historicalEnergyCosts = {}
    for sampleNo in range(1, numSamples + 1):
        historicalEnergyCosts[sampleNo] = {}
        for slot in range(0, 96):
            historicalEnergyCosts[sampleNo][slot] = -9999
    counter = -4
    for index, row in dataframe.iterrows():
        if not math.isnan(row["Tag"]):
            historicalEnergyCosts[row["Tag"]][counter] = row["Day Ahead Auktion (DE-LU)"]
            historicalEnergyCosts[row["Tag"]][counter+1] = row["Day Ahead Auktion (DE-LU)"]
            historicalEnergyCosts[row["Tag"]][counter+2] = row["Day Ahead Auktion (DE-LU)"]
            historicalEnergyCosts[row["Tag"]][counter+3] = row["Day Ahead Auktion (DE-LU)"]
        counter += 4
        if counter == 96:
            counter = 0
    return historicalEnergyCosts

def generateEnergyData(historicalEnergyData, dayNo, machines, seed):
    random.seed(seed)
    energy = {}
    energy["costs"] = {}
    energy["costs"]["conventional"] = copy.deepcopy(historicalEnergyData[dayNo])
    energy["costs"]["renewable"] = copy.deepcopy(historicalEnergyData[dayNo])
    return energy

def getOperationsAndPrecedences(jobs):
    ops, precs = [], []
    for job in jobs:
        for machine in jobs[job]["sequence"]:
            ops.append({"machine": machine, "job": job, "processing time": jobs[job]["machine times"][machine]})
            machineIndex = jobs[job]["sequence"].index(machine)
            if machineIndex > 0:
                precs.append({"from": ops[len(ops) - 2], "to": ops[len(ops) - 1]})
    return ops, precs