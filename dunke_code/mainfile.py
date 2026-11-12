import dataGeneration as datGen
import pareto as pareto
import json

numSamples, filenameEnergyData  = 365, "energy-charts_Stromproduktion_und_Börsenstrompreise_in_Deutschland_2023.xlsx"
historicalEnergyData = datGen.getHistoricalEnergyData(numSamples, filenameEnergyData)
machines = datGen.generateMachineData(3)
numbersJobs = [3, 5, 10]
calendarDays = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
startingSlots = [0, 32]
fixedTariffs = ["dyn", "fix"]
ppaPrices = ["dyn", "ppa"]
algorithmTypes = ["heu", "ex"]
for numberJobs in numbersJobs:
      for calendarDay in calendarDays:
          energy = datGen.generateEnergyData(historicalEnergyData, calendarDay, machines, calendarDay*numberJobs + 1)
          jobs, ops, precs = datGen.generateJobsOpsPrecsData(numberJobs, machines, calendarDay*numberJobs, energy)
          for startingSlot in startingSlots:
              for fixedTariff in fixedTariffs:
                  for ppaPrice in ppaPrices:
                      for algorithmType in algorithmTypes: 
                          filename = "" + str(numberJobs) + "-" + str(calendarDay) + "-" + algorithmType + "-" + str(startingSlot) + "-" + fixedTariff + "-" + ppaPrice + ".json"
                          print(filename)
                          paretoSolutions = pareto.determineParetoSolutions(machines, jobs, ops, precs, energy, startingSlot, fixedTariff, ppaPrice, algorithmType)
                          paretoSolutionsDict = {"algorithm type": algorithmType, "calendar day": calendarDay, "# jobs": numberJobs, "starting slot": startingSlot, "fixed tariff": fixedTariff, "ppa price": ppaPrice, "Pareto solutions": paretoSolutions}                        
                          with open(filename, 'w') as file:
                              json.dump(paretoSolutionsDict, file, indent=4)