import Reporter
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

# Modify the class name to match your student number.
class r0826058:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.populationSize1 = 300
		self.offspringSize1 = 600
		self.populationSize2 = 160
		self.offspringSize2 = 160
		self.alpha = 0.9
		self.stepa = 0.1
		self.startk = 2
		self.stepk = 1
		self.exchange = 2
		self.lap = 1

		random.seed()

	def pathToAdjacency(self, individual):
		new = np.zeros(len(individual), dtype=int)
		for i in range(len(individual)):
			new[int(individual[i])] = individual[int((i + 1) % len(individual))]
		return new

	def fitnessFunction(self, individual):
		distance = 0
		for i in range(len(individual) - 1):
			distance += self.distanceMatrix[individual[i]][individual[i + 1]]
		distance += self.distanceMatrix[individual[len(individual) - 1]][individual[0]]
		return distance

	def fitnesses(self, individuals):
		fit = []
		for i in individuals:
			fit.append(self.fitnessFunction(i))
		return fit

	def initialisation(self, numberOfNodes, size):
		candidateList = []
		for i in range(size):
			candidate = np.random.permutation(numberOfNodes)
			candidateList.append(candidate)
		return candidateList

	def hInit(self, numberOfNodes):
		toVisit = list(range(numberOfNodes))
		start = random.randint(0, numberOfNodes - 1)
		individual = np.zeros(numberOfNodes, dtype=int)
		individual[0] = start
		toVisit.pop(toVisit.index(start))
		for i in range(numberOfNodes - 1):
			minDist = self.distanceMatrix[individual[i]][toVisit[0]]
			best = toVisit[0]
			for candidate in toVisit:
				if self.distanceMatrix[individual[i]][candidate] < minDist:
					minDist = self.distanceMatrix[individual[i]][candidate]
					best = candidate
			individual[i + 1] = best
			toVisit.pop(toVisit.index(best))

		return individual

	def heuristicInitialisation(self, numberOfNodes, size):
		candidateList = []
		for i in range(size):
			candidate = self.hInit(numberOfNodes)
			candidateList.append(candidate)
		return candidateList

	def kselection(self, candidateList, fitnesses, k, size):
		parents = []
		for i in range(size):
			candIndexes = np.random.choice(len(candidateList), size=k)
			indexMin = 0
			for m in range(1, k):
				if fitnesses[candIndexes[m]] < fitnesses[candIndexes[indexMin]]:
					indexMin = m
			parents.append(candidateList[candIndexes[indexMin]])
		return parents

	def inversionMutation(self, individuals, alpha):
		for k in range(len(individuals)):
			if random.uniform(0, 1) < alpha:

				individual = individuals[k]
				breakPoints = []
				for i in range(len(individual) // 100 + 1):
					breakPoints.append(random.randint(0, len(individual) - 1))
					breakPoints.append(random.randint(0, len(individual) - 1))
				breakPoints.sort()

				for i in range(0, 2 * (len(individual) // 100 + 1), 2):
					a = breakPoints[i]
					b = breakPoints[i + 1]
					if a != b:
						subList = individual[a:b]
						subList[:] = subList[::-1]
						individual = np.concatenate((individual[:a], subList, individual[b:]))
				individuals[k] = individual

		return individuals

	def DeplacementMutation(self, individuals, alpha):

		for i in range(len(individuals)):
			if random.uniform(0, 1) < alpha:

				individual = individuals[i]
				breakPoint1 = random.randint(0, len(individual) - 1)
				breakPoint2 = random.randint(0, len(individual) - 1)
				a = min(breakPoint1, breakPoint2)
				b = max(breakPoint1, breakPoint2)

				if a != b:
					subList = individual[a:b]
					individual = np.concatenate((individual[:a], individual[b:]))
					newSpot = random.randint(0, len(individual) - 1)
					individual = np.concatenate((individual[:newSpot], subList, individual[newSpot:]))
					individuals[i] = individual

		return individuals

	def orderCrossover(self, parent1, parent2):
		index1 = np.random.randint(0, len(parent1) - 1)  # take a first index, not the last one
		index2 = np.random.randint(index1 + 1, len(parent1))  # take a second index, bigger than index1

		# the first child
		substring1 = parent1[index1:index2 + 1]  # first is inclusive, second is exclusive, so +1
		child1 = parent1.copy()
		childIndex = index2 + 1
		if childIndex == len(parent1):
			childIndex = 0
		for i in range(index2 + 1, len(parent1)):
			if not parent2[i] in substring1:
				child1[childIndex] = parent2[i]
				childIndex = childIndex + 1
				if childIndex == len(parent1):
					childIndex = 0
		for i in range(0, index2 + 1):
			if not parent2[i] in substring1:
				child1[childIndex] = parent2[i]
				childIndex = childIndex + 1
				if childIndex == len(parent1):
					childIndex = 0

		# the second child
		substring2 = parent2[index1:index2 + 1]  # first is inclusive, ssecond is exclusive, so +1
		child2 = parent2.copy()
		childIndex = index2 + 1
		if childIndex == len(parent1):
			childIndex = 0
		for i in range(index2 + 1, len(parent2)):
			if not parent1[i] in substring2:
				child2[childIndex] = parent1[i]
				childIndex = childIndex + 1
				if childIndex == len(parent2):
					childIndex = 0
		for i in range(0, index2 + 1):
			if not parent1[i] in substring2:
				child2[childIndex] = parent1[i]
				childIndex = childIndex + 1
				if childIndex == len(parent1):
					childIndex = 0

		return [child1, child2]

	def heuristicRecombination(self, parent1, parent2):
		parent1 = self.pathToAdjacency(parent1)
		parent2 = self.pathToAdjacency(parent2)

		child = np.zeros(len(parent1), dtype=int)
		visited = np.zeros(len(parent1), dtype=bool)
		visited[0] = 1
		n = 1
		for i in range(len(parent1) - 1):
			lastVisited = child[i]
			# we want to check, where we should go from the city "lastVisited"
			edgeP1 = self.distanceMatrix[lastVisited][parent1[lastVisited]]
			edgeP2 = self.distanceMatrix[lastVisited][parent2[lastVisited]]
			# we check if there wont be a cycle
			if edgeP1 <= edgeP2 and visited[parent1[lastVisited]] == 0:
				child[i + 1] = parent1[lastVisited]
				visited[parent1[lastVisited]] = 1
			elif visited[parent2[lastVisited]] == 0:
				child[i + 1] = parent2[lastVisited]
				visited[parent2[lastVisited]] = 1
			elif visited[parent1[lastVisited]] == 0:
				child[i + 1] = parent1[lastVisited]
				visited[parent1[lastVisited]] = 1
			else:  # looking for a city I havent been to yet
				while visited[n] == 1:
					n += 1
				child[i + 1] = n
				visited[n] = 1

		return child

	def recombinationH(self, parents):
		offspring = []
		for i in range(0, len(parents)-1, 2):
			child = self.heuristicRecombination(parents[i], parents[i+1])
			offspring.append(child)

		return offspring

	def recombinationO(self, parents):
		offspring = []
		for i in range(0, len(parents),2):
			[child1, child2] = self.orderCrossover(parents[i], parents[i+1])
			offspring.append(child1)
			offspring.append(child2)

		return offspring

	def eliminate(self, combined, fitnesses, size):
		# lambda+mu selection
		over = len(combined) - size
		for i in range(over):
			indexMax = 0
			for m in range(1, len(combined)):
				if fitnesses[m] > fitnesses[indexMax]:
					indexMax = m
			combined.pop(indexMax)
			fitnesses.pop(indexMax)

		return combined

	def distanceIndividuals (self, individual1, individual2):
		individual1a = self.pathToAdjacency(individual1)
		individual2a = self.pathToAdjacency(individual2)
		d = len(individual1)
		for i in range(len(individual1)):
			if individual1a[i] == individual2a[i]:
				d -= 1
		return d

	def diverseEliminate(self, combined, fitnesses, size):

		survivers = []
		for i in range(size):
			indexMin = 0
			for m in range(1, len(combined)):
				if fitnesses[m] < fitnesses[indexMin]:
					indexMin = m

			chosen = combined[indexMin]
			survivers.append(chosen)
			combined.pop(indexMin)
			fitnesses.pop(indexMin)
			competitor1 = np.random.choice(len(combined))
			competitor2 = np.random.choice(len(combined))
			if self.distanceIndividuals(chosen, combined[competitor1]) < self.distanceIndividuals(chosen, combined[competitor2]):
				combined.pop(competitor1)
				fitnesses.pop(competitor1)
			else:
				combined.pop(competitor2)
				fitnesses.pop(competitor2)

		return survivers

	def neighborhood(self, individual):
		l = len(individual)
		best = 0
		profit = 0
		for i in range(1, l):
			dist1 = self.distanceMatrix[l - 1][0] + self.distanceMatrix[0][1] + self.distanceMatrix[i][(i + 1) % l] + self.distanceMatrix[i - 1][i]
			dist2 = self.distanceMatrix[0][(i + 1) % l] + self.distanceMatrix[i - 1][0] + self.distanceMatrix[i][1] + self.distanceMatrix[l - 1][i]
			if dist1 - dist2 > profit:
				profit = dist1 - dist2
				best = i
		individual[0], individual[best] = individual[best], individual[0]
		return individual

	def Neighborhood(self, individual):
		l = len(individual)
		best = 0
		profit = 0
		for i in range(1, l):
			dist1 = self.distanceMatrix[i - 1][i] + self.distanceMatrix[i][(i + 1) % l] + self.distanceMatrix[(i + 1) % l][(i + 2) % l]
			dist2 = self.distanceMatrix[i - 1][(i + 1) % l] + self.distanceMatrix[(i + 1) % l][i] + self.distanceMatrix[i][(i + 2) % l]
			if dist1 - dist2 > profit:
				profit = dist1 - dist2
				best = i
		individual[best], individual[(best + 1) % l] = individual[(best + 1) % l], individual[best]
		return individual

	def island1(self, candidateList, fitnesses, k, a, bestScore, bestIndividual, q):

		flag = True

		while True:

			parents1 = self.kselection(candidateList, fitnesses, k, self.populationSize1)
			parents2 = self.kselection(candidateList, fitnesses, k, self.populationSize1)
			parents = np.concatenate((parents2, parents1))
			offspring = self.recombinationH(parents)

			if flag:
				offspring = self.DeplacementMutation(offspring, a)
				offFitnesses = self.fitnesses(offspring)
				candidateList = self.DeplacementMutation(candidateList, a)
				fitnesses = self.fitnesses(candidateList)
				flag = False
			else:
				offspring = self.inversionMutation(offspring, a)
				offFitnesses = self.fitnesses(offspring)
				candidateList = self.inversionMutation(candidateList, a)
				fitnesses = self.fitnesses(candidateList)
				flag = True

			candidateList = self.diverseEliminate(candidateList + offspring, fitnesses + offFitnesses, self.populationSize1)
			fitnesses = self.fitnesses(candidateList)

			# updating best solution
			indexMin = 0
			for m in range(1, len(fitnesses)):
				if fitnesses[m] < fitnesses[indexMin]:
					indexMin = m

			newBestScore = fitnesses[indexMin]
			newBestIndividual = candidateList[indexMin]
			if newBestScore < bestScore:  # no improvement
				bestIndividual = newBestIndividual
				bestScore = newBestScore

			end = time.time()
			if end - self.start >= self.lap * 60 - 1:
				break

		mean = sum(fitnesses) / len(candidateList)
		q.put((1, candidateList, fitnesses, bestScore, bestIndividual, mean))


	def island2(self, candidateList, fitnesses, k, a, bestScore, bestIndividual, q):

		flag = True

		while True:

			parents = self.kselection(candidateList, fitnesses, k, self.populationSize2)
			offspring = self.recombinationO(parents)

			if flag:
				offspring = self.DeplacementMutation(offspring, a)
				offFitnesses = self.fitnesses(offspring)
				flag = False
			else:
				offspring = self.inversionMutation(offspring, a)
				offFitnesses = self.fitnesses(offspring)
				flag = True

			candidateList = self.eliminate(candidateList + offspring, fitnesses + offFitnesses, self.populationSize2)
			for i in range(len(candidateList)):
				candidateList[i] = self.neighborhood(candidateList[i])
			fitnesses = self.fitnesses(candidateList)

			# updating best solution
			indexMin = 0
			for m in range(1, len(fitnesses)):
				if fitnesses[m] < fitnesses[indexMin]:
					indexMin = m

			newBestScore = fitnesses[indexMin]
			newBestIndividual = candidateList[indexMin]
			if newBestScore < bestScore:  # no improvement
				bestIndividual = newBestIndividual
				bestScore = newBestScore

			end = time.time()
			if end-self.start >= self.lap * 60 - 1:
				break

		mean = sum(fitnesses) / len(candidateList)
		q.put((2, candidateList, fitnesses, bestScore, bestIndividual, mean))


	# The evolutionary algorithm's main loop
	def optimize(self, filename):

		# Read distance matrix from file.
		self.start = time.time()
		file = open(filename)
		self.distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.

		numberOfNodes = self.distanceMatrix.shape[0]
		candidateList1 = self.initialisation(numberOfNodes, self.populationSize1)
		candidateList2 = self.heuristicInitialisation(numberOfNodes, self.populationSize2)

		fitnesses1 = self.fitnesses(candidateList1)
		fitnesses2 = self.fitnesses(candidateList2)

		mean = (sum(fitnesses1) + sum(fitnesses2)) / (len(candidateList1) + len(candidateList2))

		# finding minimal element - best value
		indexMin1 = 0
		for m in range(1, len(fitnesses1)):
			if fitnesses1[m] < fitnesses1[indexMin1]:
				indexMin1 = m
		bestScore1 = fitnesses1[indexMin1]
		bestIndividual1 = candidateList1[indexMin1]


		indexMin2 = 0
		for m in range(1, len(fitnesses2)):
			if fitnesses2[m] < fitnesses2[indexMin2]:
				indexMin2 = m
		bestScore2 = fitnesses2[indexMin2]
		bestIndividual2 = candidateList2[indexMin2]

		if bestScore1 < bestScore2:
			timeLeft = self.reporter.report(mean, bestScore1, bestIndividual1)
			#print(mean, bestScore1, bestIndividual1)
		else:
			timeLeft = self.reporter.report(mean, bestScore2, bestIndividual2)
			#print(mean, bestScore2, bestIndividual2)

		k = self.startk
		a = self.alpha
		if min(bestScore1,bestScore2) - 27154.48839924464 <= 0.00000000001:
			print(min(bestScore1, bestScore2), mean)
			return (min(bestScore1, bestScore2), mean)

		while True:

			q = Queue()
			p1 = Process(target=self.island1, args=(candidateList1, fitnesses1, k, a, bestScore1, bestIndividual1, q))
			p1.start()
			p2 = Process(target=self.island2, args=(candidateList2, fitnesses2, k, a, bestScore2, bestIndividual2, q))
			p2.start()

			result1 = q.get(block=True)
			result2 = q.get(block=True)

			if result1[0] == 1:
				candidateList1 = result1[1]
				fitnesses1 = result1[2]
				bestScore1 = result1[3]
				bestIndividual1 = result1[4]
				mean1 = result1[5]

				candidateList2 = result2[1]
				fitnesses2 = result2[2]
				bestScore2 = result2[3]
				bestIndividual2 = result2[4]
				mean2 = result2[5]
			else:
				candidateList1 = result2[1]
				fitnesses1 = result2[2]
				bestScore1 = result2[3]
				bestIndividual1 = result2[4]
				mean1 = result2[5]

				candidateList2 = result1[1]
				fitnesses2 = result1[2]
				bestScore2 = result1[3]
				bestIndividual2 = result1[4]
				mean2 = result1[5]

			# exchange between islands
			candIndexes = np.random.choice(min(len(candidateList2), len(candidateList1)), size=self.exchange)
			for s in candIndexes:
				candidateList1[s], candidateList2[s] = candidateList2[s], candidateList1[s]
				fitnesses1[s], fitnesses2[s] = fitnesses2[s], fitnesses1[s]

			mean = (mean1 * self.populationSize1 + mean2 * self.populationSize2) / (self.populationSize1 + self.populationSize2)

			p1.join()
			p2.join()

			k += self.stepk
			a -= self.stepa
			self.lap += 1

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0

			if bestScore1 < bestScore2:
				timeLeft = self.reporter.report(mean, bestScore1, bestIndividual1)
				print(bestScore1, bestIndividual1)
			else:
				timeLeft = self.reporter.report(mean, bestScore2, bestIndividual2)
				print(bestScore2, bestIndividual2)

			if min(bestScore1, bestScore2) - 27154.48839924464 <= 0.00000000001:
				break

			if timeLeft < 0:
				break
			#end = time.time()
			#if end-self.start >= 300:
			#	break

		print(min(bestScore1,bestScore2),mean)
		return (min(bestScore1,bestScore2),mean)


bests = []
means = []

for i in range(50):
	myTSPproblem = r0826058()
	b, m = myTSPproblem.optimize("tour100.csv")
	bests.append(b)
	means.append(m)






