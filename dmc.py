import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

class DiffusionMonteCarlo:
  def __init__(self, d, n, steps=1000, N0=500, Nmax=2000, num_buckets=200, x_min=-20.0, x_max=20.0, dt=0.1, alpha=10.0):
    """
    __init__ initializes base class

    d: Spatial dimension
    n: Number of particles
    steps: Number of time steps to run
    N0: Initial number of replicas
    Nmax: Maximum number of replicas
    num_buckets: Number of bins
    x_min: Lower bound of grid
    x_max: Upper bound of grid
    dt: Time step
    alpha: Emperically chosen positive parameter to update energy, based on 1/dt
    """
    self.d = d
    self.n = n
    self.steps = steps
    self.dn = self.d * self.n # Effective dimension
    self.N0 = N0
    self.N = N0 # Variable to allow number of replicas to change
    self.Nmax = Nmax
    self.num_buckets = num_buckets
    self.x_min = x_min
    self.x_max = x_max
    self.dt = dt
    self.alpha = alpha

    self.flags = np.zeros(self.Nmax, dtype=int) # Three integer types: 0 = dead, 1 = alive, 2 = newly made replica
    for i in range(self.N):
      self.flags[i] = 1 # Set the flags of initial replicas to "alive"
    self.points = np.zeros((self.Nmax, self.dn)) # (Initial) positions of replicas, shape: (Nmax, dn)
    self.E1 = self.averagePotentialEnergy() # Reference energy (1)
    self.E2 = self.E1 # Reference energy (2)
    self.energy_storage = [] # Storage list to calculate average reference energies at the end
    self.hist_storage = np.zeros(self.num_buckets, dtype=int) # Histogram storage array

  def U(self, x):
    """
    U: potential energy (for a single replica)

    x: array shape (d), position of replica

    returns: float, energy of the replica
    """
    pot = 0.5 * np.linalg.norm(x)**2
    return pot

  def averagePotentialEnergy(self):
    """
    potential energy (for the entire system/all the replicas)

    returns: float, energy of all replicas
    """
    pot = 0.0
    for i in range(self.Nmax):
      if self.flags[i]:
        pot += self.U(self.points[i])
    avg_pot = pot / self.N

    return avg_pot

  def walk(self):
    """
    walk: moves points around steps once

    returns nothing
    """
    self.points += np.sqrt(self.dt) * np.random.normal(0, 1, size=(self.Nmax, self.dn))

  def calculateM(self, E_ref, index):
    """
    calculateM: calculates m :)

    E_ref: float, reference energy
    index: (nd), position of replica

    returns: int, what to do with the replica in branch()
    """
    return int(1 - (self.U(self.points[index]) - E_ref)*self.dt + np.random.uniform(0, 1) )

  def replicateSingleReplica(self, i):
    """
    replicate point i into next available point, update flags by 2 to not loop over new replica
    i: index of points[i]

    returns nothing
    """
    for j in range(self.Nmax):
      if self.flags[j] == 0:
        self.flags[j] = 2
        self.points[j] = self.points[i]
        self.N += 1
        break

  def branch(self):
    """
    branch: replicates and removes points from the distribution as needed

    returns nothing
    """
    for i in range(self.Nmax): # Loop through all replicas
      if self.flags[i] == 1: # If replica is alive, calculate m
        m = self.calculateM(self.E2, i)

        if m == 0: # KILL
          self.flags[i] = 0
          self.N -= 1
        elif m == 1: # Do nothing
          pass
        elif m == 2: # Replicate once
          self.replicateSingleReplica(i)
        else: # Replicate twice if m >= 3
          self.replicateSingleReplica(i)
          self.replicateSingleReplica(i)
    
    for i in range(self.Nmax): # Set previously created replica to "alive"
      if self.flags[i] == 2:
        self.flags[i] = 1

    self.E1 = self.E2
    self.E2 = self.E1 + self.alpha*(1.0 - self.N / self.N0)

  def bucketNumber(self, x):
    """
    bucketNumber: returns the index of the bucket that x falls in the interval [x_min, x_max] for num_buckets buckets

    x: float, position of replica

    return: int, bucket number
    """
    if x < self.x_min:
      return 0
    elif x > self.x_max:
      return int(self.num_buckets - 1)
    else:
      return int((x - self.x_min)*self.num_buckets / (self.x_max - self.x_min))

  def count(self):
    """
    count: counts the number of replicas in each bucket

    returns nothing
    """
    self.energy_storage.append(self.averagePotentialEnergy())
    
    for i in range(self.Nmax):
      if self.flags[i]:
        for j in range(self.dn):
          self.hist_storage[self.bucketNumber(self.points[i])] += 1

  def output(self):
    """
    output: outputs stuff to the screen

    returns nothing
    """
    avg_energy = np.average(np.array(self.energy_storage))
    print("Average Reference Energy: {}".format(avg_energy))
    print("Analytic Energy:          {}".format(0.5)) # Exact energy for the 1D ground state harmonic oscillator in dimensionless units
    print("Percent Difference:       {:.2f}%".format(np.abs(0.5 - avg_energy)/0.5*100))

    ax = plt.gca()

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', axis="x", direction="in")
    ax.tick_params(which='both', axis="y", direction="in")
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)

    plt.rcParams["figure.figsize"] = [10.00, 7.00]
    plt.rcParams.update({'font.size': 14})

    plt.title('Reference Energy Per Time Step')
    plt.xlabel(r'$\tau$')
    plt.ylabel('<$E_{R}$>')
    x = [i for i in range(len(self.energy_storage))]
    plt.plot(x, self.energy_storage)
    plt.show()

    count = []
    bins = []
    for i, value in enumerate(self.hist_storage):
      count.append(self.x_min + (self.x_max - self.x_min) * (i+0.5) / self.num_buckets)
      bins.append(value / np.max(self.hist_storage))
    plt.title('Ground State Wavefunction')
    plt.xlabel('x')
    plt.ylabel('$\Phi_{0}$(x)')
    plt.bar(count, bins)
    plt.show()

  def simulate(self):
    """
    Do the simulation
    """
    current_step = 0
    progress = int(self.steps / 10)
    while current_step < self.steps:

      if(current_step == progress):
        print("[ 10 |", end="", flush=True)
      elif(current_step == 2*progress):
        print(" 20 |", end="", flush=True)
      elif(current_step == 3*progress):
        print(" 30 |", end="", flush=True)
      elif(current_step == 4*progress):
        print(" 40 |", end="", flush=True)
      elif(current_step == 5*progress):
        print(" 50 |", end="", flush=True)
      elif(current_step == 6*progress):
        print(" 60 |", end="", flush=True)
      elif(current_step == 7*progress):
        print(" 70 |", end="", flush=True)
      elif(current_step == 8*progress):
        print(" 80 |", end="", flush=True)
      elif(current_step == 9*progress):
        print(" 90 |", end="", flush=True)
      elif(current_step == 10*progress - 1):
        print(" 100 ]\n", flush=True)

      self.walk()
      self.branch()
      self.count()
      current_step += 1

    print("****************************************************")
    self.output()

# The inital parameters indicated in __init__ are ones the reference suggests but the 
# below parameters produce a more "stable" energy curve. Takes ~12 min
# Curve continues to flatten with increasing replicas (N0, Nmax)
DMC = DiffusionMonteCarlo(1, 1, steps=2000, dt=0.01, x_min=-5.0, x_max=5.0, N0=10000, Nmax=50000)
DMC.simulate()
