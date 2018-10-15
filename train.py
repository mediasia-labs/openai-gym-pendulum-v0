import gym
import numpy as np
import math
import copy

'''

Training Class
--------------
Solve Pendulum-v0 Gym Challenge with genetic evolutionary strategy

Inspired by:
------------
- https://arxiv.org/abs/1703.03864
- http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
- https://github.com/MorvanZhou/Evolutionary-Algorithm/blob/master/tutorial-contents/Using%20Neural%20Nets/Evolution%20Strategy%20with%20Neural%20Nets.py
------------

'''


class PendulumSolver:

	def __init__(self, generations=1000):
		# Load MountainCar
		self.env = gym.make('Pendulum-v0')

		# Learning rate
		self.learningRate = 0.05

		# Exploration rate (mutations)
		self.explorationRate = 0.75

		self.iterations = 5

		# Generations
		# ---
		# Number of generations we will train on
		# Similar concept to `number of epochs`
		# At each generation a subset of the population mutates and
		# another subset dies and is replaced by children from other individuals
		self.generations = generations

		# Population size
		# ---
		# Number of individual in the population,
		# each generation has the same population size
		self.populationSize = 20

		# Number of kids per generation
		# ---
		# Kids replace individuals which did not survive the previous generation
		# half of the kids are copies of the best performing individuals with few mutations,
		# the other half are new random individuals
		# This allows to quickly explore the feature space, while 
		# encouraging best performing individual to last, and less performing to evolve
		self.kidsPerGeneration = 5

		# Network shape
		# - number of inputs
		# - number of neurons in hidden layer
		# - number of neurons in hidden layer
		# - number of outputs 
		self.networkShape = (
			len(self.env.observation_space.high), 
			20,
			10,
			5,
			len(self.env.action_space.high)
		)

		# Create initial population
		# ---
		# Each individual is a randomly generated feed forward neural net
		# An individual role is to select the best `action` given an `observation`
		self.population = [Individual(self.networkShape) for i in range(self.populationSize)]

		# Save best individual and best reward
		self.bestIndividual = None
		self.bestReward = -200.0

		# Start training
		self.run()


	# Run game
	def run(self):

		# Loop through generations
		for i in range(self.generations):
			finished = 0.0
			best = -10000

			# Loop through individuals
			for individual in self.population:
				episode_reward = 0.

				for x in range(self.iterations):
					# Get default state
					current_state = self.env.reset()

					# An episode ends after 200 steps
					for step in range(500):
						# self.env.render()

						action = individual.forward(current_state)

						# Run step with chosen action
						# retrieve current action reward and new state
						new_state, reward, done, info = self.env.step(action)

						# Episode reward
						episode_reward += reward

						# Remember state
						current_state = new_state

						# If game is solved before 200 steps
						if done and step < 199:
							finished += 1
							episode_reward += 2 * self.iterations
							break


				# Set individual's reward
				episode_reward /= self.iterations 
				episode_reward -= 2 * self.iterations
				individual.reward(episode_reward)

				# Remember best score
				if episode_reward > best:
					best = episode_reward

			# If average best score is above 90 for 100 iterations
			# network is considered trained, and challenge passed
			finished /= self.iterations
			if best > -500:
				self.rank()
				print 'Generation', i + 1, 'Best score', self.population[0]._reward, 'Finished', finished
				break

			self.next_generation(i, finished)


		# Training is finished
		# Run the 10 best performing individuals
		for i in range(10):
			current_state = self.env.reset()
			for step in range(500):
				self.env.render()
				action = self.population[0].forward(current_state)

				# Run step with chosen action
				new_state, reward, done, info = self.env.step(action)

				current_state = new_state

				if done:
					break


	def next_generation(self, generation, finished):
		# Rank individuals based on rewards
		self.rank()

		# Save best performing network
		if self.population[0]._reward > self.bestReward:
			self.bestIndividual = copy.deepcopy(self.population[0].layers)
			self.bestReward = self.population[0]._reward

		for i in range(self.populationSize):
			# Slightly update first half of population 
			if i < (self.populationSize - self.kidsPerGeneration) / 2:
				self.population[i].evolve(((i + 1) / (self.populationSize - self.kidsPerGeneration) * self.learningRate))
			
			# Explore space with second half of population
			elif i < self.populationSize - self.kidsPerGeneration:
				self.population[i].evolve(self.explorationRate)

			# Add best performing ever to population
			elif i == self.populationSize - self.kidsPerGeneration and self.bestIndividual:
				self.population[i] = Individual(self.networkShape, self.bestIndividual)
			
			# Replace less performing individuals
			else:
				# With kids from best performing
				if i % 2 == 0:
					# Make children
					self.population[i] = self.population[(self.populationSize - 1) - i].child(self.learningRate * self.explorationRate)
				
				else:
					# Or with randomly generated new individuals (to keep exploring new possibilities)
					self.population[i] = Individual(self.networkShape)

		print 'Generation', generation + 1, 'Best score', self.population[0]._reward, 'Finished', finished


	def rank(self):
		def sort(e):
			return e._reward

		# sort population per reward
		self.population.sort(key=sort, reverse=True)


'''

Individual class
----------------
Simple `fast forward` neural net implementation

'''

class Individual:
	def __init__(self, shapes, layers=None):
		self.shapes = shapes
		self.layers = []
		self._reward = 0

		# Create layers
		for i, shape in enumerate(self.shapes):
			if i < len(self.shapes) - 1:
				self.layers.append(self.layer(self.shapes[i], self.shapes[i + 1]))

		if layers:
			self.layers = layers


	# Network layer
	def layer(self, inputs, outputs):
		# Create layer inputs
		w = np.random.randn(inputs * outputs).astype(np.float32)
		w = w.reshape((inputs, outputs))

		# Create outputs
		b = np.random.randn(outputs).astype(np.float32) * .1
		b = b.reshape((1, outputs))

		return [w, b]


	# Forward function with continuous output
	def forward(self, x):

		# Forward X to layers
		for i, layer in enumerate(self.layers):
			# loop over each layer, except last
			if i < len(self.layers) - 1:
				x = np.tanh(x.dot(layer[0]) + layer[1])

			# compute last layer
			else:
				x = x.dot(layer[0]) + layer[1]

		# Return computed Y
		return x[0]
		# return np.argmax(x, axis=1)[0]

	# Evolve
	def evolve(self, learningRate):
		for i, shape in enumerate(self.shapes):
			if i < len(self.shapes) - 1:
				gradient = self.layer(self.shapes[i], self.shapes[i + 1])
				self.layers[i][0] += gradient[0] * learningRate
				self.layers[i][1] += gradient[1] * learningRate


	# Save reward
	def reward(self, reward=None):
		self._reward = reward

	# Make child
	def child(self, learningRate):
		child = Individual(self.shapes, copy.deepcopy(self.layers))
		child.evolve(learningRate)
		return child



PendulumSolver()

