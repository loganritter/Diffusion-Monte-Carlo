#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>  // For std::accumulate
#include <iterator> // For std::back_inserter

class DiffusionMonteCarlo {
public:
    DiffusionMonteCarlo(int d, int n, int steps = 1000, int N0 = 500, int Nmax = 2000,
                        int num_buckets = 200, double x_min = -20.0, double x_max = 20.0,
                        double dt = 0.1, double alpha = 10.0)
        : d(d), n(n), steps(steps), dn(d * n), N0(N0), N(N0), Nmax(Nmax), num_buckets(num_buckets),
          x_min(x_min), x_max(x_max), dt(dt), alpha(alpha), flags(Nmax, 0), points(Nmax, std::vector<double>(d * n, 0.0)),
          energy_storage(), hist_storage(num_buckets, 0) {
        
        // Initialize flags for initial replicas to "alive"
        std::fill(flags.begin(), flags.begin() + N, 1);

        // Calculate initial energy
        E1 = averagePotentialEnergy();
        E2 = E1;
    }

    void simulate() {
        int progress = steps / 10;
        for (int current_step = 0; current_step < steps; ++current_step) {
            if (current_step % progress == 0) {
                std::cout << "[ " << (current_step / steps) * 100 << " |";
            }

            walk();
            branch();
            count();
        }
        output();
    }

private:
    int d, n, dn, steps, N0, N, Nmax, num_buckets;
    double x_min, x_max, dt, alpha, E1, E2;
    std::vector<int> flags;
    std::vector<std::vector<double>> points;
    std::vector<double> energy_storage;
    std::vector<int> hist_storage;
    std::default_random_engine generator;
    std::normal_distribution<double> normal_dist{0.0, 1.0};
    std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};

    double U(const std::vector<double>& x) {
        // Potential energy for a single replica
        double pot = 0.5 * std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
        return pot;
    }

    double averagePotentialEnergy() {
        // Average potential energy for all replicas
        double pot = 0.0;
        for (int i = 0; i < Nmax; ++i) {
            if (flags[i]) {
                pot += U(points[i]);
            }
        }
        return pot / N;
    }

    void walk() {
        // Moves points around
        for (auto& point : points) {
            for (auto& p : point) {
                p += std::sqrt(dt) * normal_dist(generator);
            }
        }
    }

    int calculateM(double E_ref, int index) {
        // Calculate the branching factor
        double energy_diff = U(points[index]) - E_ref;
        int m = static_cast<int>(1 - energy_diff * dt + uniform_dist(generator));
        return m;
    }

    void replicateSingleReplica(int i) {
        // Replicate point i into the next available point
        for (int j = 0; j < Nmax; ++j) {
            if (flags[j] == 0) {
                flags[j] = 2;
                points[j] = points[i];
                ++N;
                break;
            }
        }
    }

    void branch() {
        // Replicate or remove points as needed
        for (int i = 0; i < Nmax; ++i) {
            if (flags[i] == 1) {
                int m = calculateM(E2, i);

                if (m == 0) {
                    flags[i] = 0; // KILL
                    --N;
                } else if (m == 2) {
                    replicateSingleReplica(i);
                } else if (m >= 3) {
                    replicateSingleReplica(i);
                    replicateSingleReplica(i);
                }
            }
        }

        // Set previously created replicas to "alive"
        for (int i = 0; i < Nmax; ++i) {
            if (flags[i] == 2) {
                flags[i] = 1;
            }
        }

        E1 = E2;
        E2 = E1 + alpha * (1.0 - static_cast<double>(N) / N0);
    }

    int bucketNumber(double x) {
        // Get the bucket number for x
        if (x < x_min) {
            return 0;
        } else if (x > x_max) {
            return num_buckets - 1;
        } else {
            return static_cast<int>((x - x_min) * num_buckets / (x_max - x_min));
        }
    }

    void count() {
        // Count replicas in each bucket
        energy_storage.push_back(averagePotentialEnergy());

        for (int i = 0; i < Nmax; ++i) {
            if (flags[i]) {
                for (double val : points[i]) {
                    hist_storage[bucketNumber(val)]++;
                }
            }
        }
    }

    void output() {
        // Output results
        double avg_energy = std::accumulate(energy_storage.begin(), energy_storage.end(), 0.0) / energy_storage.size();
        double analytic_energy = 0.5 * n * d;
        double percent_difference = std::abs((analytic_energy - avg_energy) / analytic_energy * 100.0);

        std::cout << "Average Reference Energy (Simulation): " << avg_energy << std::endl;
        std::cout << "Analytic Energy (Ground State Harmonic Oscillator): " << analytic_energy << std::endl;
        std::cout << "Percent Difference: " << percent_difference << "%" << std::endl;
    }
};

int main() {
    DiffusionMonteCarlo DMC(1, 2, 2000, 10000, 50000, 200, -5.0, 5.0, 0.01, 10.0);
    DMC.simulate();
    return 0;
}
