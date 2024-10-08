import numpy as np
from tqdm import tqdm
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
        Calculate the potential energy (for the entire system/all the replicas)
        using vectorized NumPy operations.

        Returns:
            float: Average energy of all replicas
        """
        alive_indices = self.flags == 1
        pot = 0.5 * np.sum(np.linalg.norm(self.points[alive_indices], axis=1) ** 2)
        avg_pot = pot / self.N
        return avg_pot


    def walk(self):
        """
        Vectorized walk: moves points around once.
        """
        self.points += np.sqrt(self.dt) * rng.normal(0, 1, size=(self.Nmax, self.dn))

    def calculateM(self, E_ref, index):
        """
        calculateM: calculates m :)

        E_ref: float, reference energy
        index: (nd), position of replica

        returns: int, what to do with the replica in branch()
        """
        energy = self.U(self.points[index])
        m = int(1 - (energy - E_ref) * self.dt + np.random.uniform(0, 1))
        return m
        
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
        branch: replicates and removes points from the distribution as needed.
        """
        alive_indices = np.where(self.flags == 1)[0]  # Get indices of alive replicas

        # Calculate m for all alive replicas at once
        energies = np.array([self.U(self.points[i]) for i in alive_indices])
        m_values = np.floor(1 - (energies - self.E2) * self.dt + np.random.uniform(0, 1, len(alive_indices)))

        # Process each replica based on its m value
        for i, m in zip(alive_indices, m_values):
            if m <= 0:  # KILL
                self.flags[i] = 0
                self.N -= 1
            elif m == 2:  # Replicate once
                self.replicateSingleReplica(i)
            elif m >= 3:  # Replicate twice if m >= 3
                self.replicateSingleReplica(i)
                self.replicateSingleReplica(i)

        # Update energy and set new replicas to alive
        self.flags[self.flags == 2] = 1
        self.E1 = self.E2
        self.E2 = self.E1 + self.alpha * (1.0 - self.N / self.N0)

    def bucketNumber(self, x):
        """
        bucketNumber: returns the index of the bucket that x falls in the interval [x_min, x_max] for num_buckets buckets

        x: float or array-like, position of replica

        return: int, bucket number
        """
        # Convert to the scalar range if x is an array
        x_val = np.linalg.norm(x) if isinstance(x, np.ndarray) else x
        
        if x_val < self.x_min:
            return 0
        elif x_val > self.x_max:
            return int(self.num_buckets - 1)
        else:
            return int((x_val - self.x_min) * self.num_buckets / (self.x_max - self.x_min))

    def count(self):
        """
        count: counts the number of replicas in each bucket

        returns nothing
        """
        self.energy_storage.append(self.averagePotentialEnergy())
        
        for i in range(self.Nmax):
            if self.flags[i]:
                for j in range(self.dn):
                    # Use self.points[i, j] for the j-th dimension
                    self.hist_storage[self.bucketNumber(self.points[i, j])] += 1

    def output(self):
        """
        output: outputs stuff to the screen

        returns nothing
        """
        avg_energy = np.average(np.array(self.energy_storage))
        
        # Calculate the analytic energy for the ground state harmonic oscillator
        analytic_energy = 0.5 * self.n * self.d
        
        print("Average Reference Energy: {:.4f}".format(avg_energy))
        print("Analytic Energy:          {:.4f}".format(analytic_energy))
        print("Percent Difference:       {:.2f}%".format(np.abs(analytic_energy - avg_energy) / analytic_energy * 100))

        ax = plt.gca()

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', axis="x", direction="in")
        ax.tick_params(which='both', axis="y", direction="in")
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        
        # Plot the reference energy per time step
        plt.rcParams["figure.figsize"] = [10.00, 7.00]
        plt.rcParams.update({'font.size': 14})

        plt.title('Reference Energy Per Time Step')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\langle E_{R} \rangle$')
        x = [i for i in range(len(self.energy_storage))]
        plt.plot(x, self.energy_storage)
        plt.show()

        count = []
        bins = []
        for i, value in enumerate(self.hist_storage):
            count.append(self.x_min + (self.x_max - self.x_min) * (i + 0.5) / self.num_buckets)
            bins.append(value / np.max(self.hist_storage))
            
        plt.title('Ground State Wavefunction')
        plt.xlabel('x')
        plt.ylabel(r'$\Phi_{0}(x)$')
        plt.bar(count, bins)
        plt.show()

    def simulate(self):
        """
        Do the simulation
        """
        progress = tqdm(total=self.steps, desc="Simulating", unit="step")
        
        current_step = 0
        while current_step < self.steps:
            self.walk()
            self.branch()
            self.count()
            current_step += 1
            
            progress.update(1)

        progress.close()
        print("****************************************************")
        self.output()

if __name__ == '__main__':
    rng = np.random.default_rng()
    DMC = DiffusionMonteCarlo(1, 2, steps=2000, dt=0.01, x_min=-5.0, x_max=5.0, N0=10000, Nmax=50000)
    DMC.simulate()
