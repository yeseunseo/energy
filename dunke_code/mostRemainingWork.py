def sequenceByMRW(machines, jobs, ops, precs):
    knoten, kanten = generateInitialGraph(ops, precs)
    unscheduledNodes = knoten
    unscheduledNodes.remove(("source"))
    unscheduledNodes.remove(("sink"))
    noNodes = len(knoten)
    scheduledNodes = []

    nextNodeCandidates = getSuccessors(("source"), kanten) 
    while len(scheduledNodes) < noNodes:
        remainingWorks = getRemainingWorks(machines, unscheduledNodes, kanten)
        nextMachine = getMachineWithMostRemainingWork(nextNodeCandidates, remainingWorks)
        nextNode = getNextNode(nextMachine, nextNodeCandidates)
        nextNodeCandidates.remove(nextNode)
        unscheduledNodes.remove(nextNode)

        if not isinstance(getSuccessor(nextNode, kanten), str):
            nextNodeCandidates.append(getSuccessor(nextNode, kanten))
        scheduledNodes.append(nextNode)
    mrw_jobSequences = createJobSequencesOfMachines(machines, scheduledNodes)
    
    solution = {"name": "", "job sequences": mrw_jobSequences}
    return solution
    
def createJobSequencesOfMachines(machines, scheduledNodes):
    jobSequences = {}
    for machine in machines:
        sequence = []
        for i in range(0, len(scheduledNodes)):
            if scheduledNodes[i][0] == machine:
                successorFound = False
                for j in range(i+1, len(scheduledNodes)):
                    if not successorFound:
                        idx1 = scheduledNodes[i][0] #first machine
                        idx2 = scheduledNodes[i][1] #first job
                        idx3 = scheduledNodes[j][0] #second machine
                        idx4 = scheduledNodes[j][1] #second job
                        if idx1 == idx3:
                            if len(sequence) == 0:
                                sequence.append(idx2)
                            sequence.append(idx4)
                            successorFound = True
        jobSequences[machine] = sequence
    return jobSequences

def getNextNode(nextMachine, nextNodeCandidates):
    for node in nextNodeCandidates:
        if node[0] == nextMachine:
            return node

def getRemainingWorks(machines, unscheduledNodes, kanten):
    remainingWorks = {}
    for machine in machines:
        remainingWorks[machine] = 0
        for kante in kanten:
            if kante["start"] in unscheduledNodes and kante["start"][0] == machine :
                remainingWorks[machine] += kante["length"]
    machine = getMachineWithMostRemainingWork
    return remainingWorks

def getMachineWithMostRemainingWork(nextNodeCandidates, remainingWorks):
    machine = ""
    mostRemainingWork = 0
    for node in nextNodeCandidates:
        if remainingWorks[node[0]] > mostRemainingWork:
            mostRemainingWork = remainingWorks[node[0]]
            machine = node[0]
    return machine            
    
def getSuccessors(node, kanten):
    successors = []
    for kante in kanten:
        if kante["start"] == node:
            successors.append(kante["end"])
    return successors

def getSuccessor(node, kanten):
    for kante in kanten:
        if kante["start"] == node:
            return kante["end"]
    
def generateInitialGraph(ops, precs):
    knoten = []
    for op in ops:
        knoten.append((op["machine"], op["job"]))
    kanten = []
    for prec in precs:
        tuple1 = (prec["from"]["machine"], prec["from"]["job"])
        tuple2 = (prec["to"]["machine"], prec["to"]["job"])
        length = prec["from"]["processing time"]
        kanten.append({"start": tuple1, "end": tuple2, "length": length})
    
    jobStartknoten = []
    jobEndknoten = []
    for knot in knoten:
        isStartknoten = True
        isEndknoten = True
        for kante in kanten:
            if kante["end"] == knot:
                isStartknoten = False
            if kante["start"] == knot:
                isEndknoten = False
        if isStartknoten:
            jobStartknoten.append(knot)
        if isEndknoten:
            jobEndknoten.append(knot)
    
    knoten.append(("source"))
    knoten.append(("sink"))
    for knot in jobStartknoten:
        kanten.append({"start": ("source"), "end": knot, "length": 0})
    for knot in jobEndknoten:
        for op in ops:
            if (op["machine"], op["job"]) == knot:
                kanten.append({"start": knot, "end": ("sink"), "length": op["processing time"]})
    return knoten, kanten

def transformJobSequencesInto_b(jobSequences):
    b = {}
    for machine in jobSequences:
        jobSequence = jobSequences[machine]
        for i in range(0, len(jobSequence) - 1):
            for j in range(i + 1, len(jobSequence)):
                b[machine, jobSequence[i], machine, jobSequence[j]] = 1
    return b