"""
Change log: The only difference between this version and the old PSOManager code is that
it assumes that pso_params is given as a dictionary (might be easier for scalability)
and else case allows for class
"""

import numpy as np
import time

class Particle:
    def __init__(self):
        self.dim = 3
        self.w = 1
        self.c1 = 2
        self.c2 = 2
        self.bounds = np.array([(0.1, 1.1)] * self.dim)
        self.voltage = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
        self.velocity = np.random.uniform(-1, 1, self.dim)
        self.best_voltage = self.voltage.copy()
        self.best_cost = float('inf')
        self.cost = float('inf')

    def update_velocity(self, global_best_voltage, relTol=0.01):
        r1, r2 = np.random.rand(), np.random.rand()
        delta_V_cognitive = (self.best_voltage - self.voltage)
        delta_V_social = (global_best_voltage - self.voltage)
        cognitive = self.c1 * r1 * delta_V_cognitive
        social = self.c2 * r2 * delta_V_social
        apply_velocity = np.int32((delta_V_cognitive > relTol) | (delta_V_social > relTol))
        self.velocity = self.w * self.velocity * apply_velocity + cognitive + social

    def update_voltage(self):
        temp_voltage = self.voltage + self.velocity
        outOfBounds = (temp_voltage > self.bounds[:, 1]) | (temp_voltage < self.bounds[:, 0])
        self.velocity = np.where(outOfBounds, np.random.uniform(-1, 1, self.dim), self.velocity)
        self.voltage = np.where(outOfBounds,
                                np.random.uniform(self.bounds[0, 0], self.bounds[0, 1], self.dim),
                                self.voltage + self.velocity)
        self.voltage = np.clip(self.voltage, self.bounds[:, 0], self.bounds[:, 1])

class PSOManager:
    def __init__(self, pso_params, threshold_cost):
        if isinstance(pso_params, dict):
            self.num_particles = pso_params["num_particles"]
            self.max_iter = pso_params["max_iter"]
            self.Measure_CostFunction = pso_params["Measure_CostFunction"]
            self.log_event = pso_params["log_event"]
            self.log_file = pso_params["log_file"]
            self.meas_device = pso_params["meas_device"]
        else: # Assuming pso_params is PSOParams
            self.num_particles = pso_params.num_particles
            self.max_iter = pso_params.max_iter
            self.Measure_CostFunction = pso_params.Measure_CostFunction
            self.log_event = pso_params.log_event
            self.log_file = pso_params.log_file
            self.meas_device = pso_params.meas_device

        self.threshold_cost = threshold_cost

        self.init_particles()
        # self.particles = [Particle() for _ in range(self.num_particles)]
        # self.global_best_voltage = np.random.uniform(self.particle.bounds[:, 0], self.particle.bounds[:, 1], self.particle.dim)
        # self.global_best_cost = float('inf')

        # Use parameters from pso_params
    
    def init_particles(self):
        self.particle = Particle()
        self.w = self.particle.w
        self.particles = [Particle() for _ in range(self.num_particles)]
        self.global_best_voltage = np.random.uniform(self.particle.bounds[:, 0], self.particle.bounds[:, 1], self.particle.dim)
        self.global_best_cost = float('inf')

    def evaluate_cost(self, user, user_ctrl, channels, voltage, reference_cost):
        max_tries = 5
        temp_cost = np.zeros(max_tries)
        user_ctrl.Vset(voltage)
        for i in range(max_tries):
            temp_measurement, temp_cost[i] = self.Measure_CostFunction(user, self.meas_device, channels)
        mean_cost = np.mean(temp_cost)
        return mean_cost, mean_cost <= reference_cost

    def optimize(self, user, user_ctrl, channels):
        self.log_event(self.log_file, "\nOptimization Summary")
        self.log_event(self.log_file, "=" * 120)
        self.log_event(self.log_file, f"{'Iteration':<12}{'Particle No.':<15}{'Voltage':<25}{'Measurement':<20}{'Cost':<15}{'Best Cost':<15}{'Global Best Cost':<20}")
        self.log_event(self.log_file, "-" * 120)

        for iteration in range(self.max_iter):
            tic = time.perf_counter()
            for particle_no, particle in enumerate(self.particles):
                user_ctrl.Vset(particle.voltage)
                particle.measurement, particle.cost = self.Measure_CostFunction(user, self.meas_device, channels)

                if particle.cost < particle.best_cost:
                    particle.cost, success = self.evaluate_cost(user, user_ctrl, channels, particle.voltage, particle.best_cost)
                    if success:
                        particle.best_cost = particle.cost
                        particle.best_voltage = particle.voltage.copy()

                if particle.cost < self.global_best_cost:
                    particle.cost, success = self.evaluate_cost(user, user_ctrl, channels, particle.voltage, self.global_best_cost)
                    if success:
                        self.global_best_cost=particle.cost
                        self.global_best_voltage = particle.voltage.copy()

                self.log_event(self.log_file, f"{iteration + 1:<12}{particle_no:<15}{str(particle.voltage):<25}{str(particle.measurement):<20}{particle.cost:<15.5f}{particle.best_cost:<15.5f}{self.global_best_cost:<20.5f}")

            if np.round(self.global_best_cost, 4) <= self.threshold_cost:
                self.global_best_cost, success = self.evaluate_cost(user, user_ctrl, channels, self.global_best_voltage, self.threshold_cost)
                if success:
                    self.log_event(self.log_file, f"Threshold cost achieved at iteration {iteration + 1}. Optimization stopped.")
                    break

            toc = time.perf_counter()
            
            self.log_event(self.log_file, f"Iteration {iteration + 1}/{self.max_iter} | Best voltage: {self.global_best_voltage} | Best Cost: {self.global_best_cost}, time elapsed {toc - tic} seconds\n")

            for particle in self.particles:
                particle.update_velocity(self.global_best_voltage)
                particle.update_voltage()
            self.w *= 0.99
            


        return self.global_best_voltage, self.global_best_cost
    
    def optimize_polarization(self, user, pol, channels, user_ctrl):
        success = False
        for _ in range(2):
            #self.init_particles()
            user.PSG.polSET(pol)
            best_voltage, best_cost = self.optimize(user_ctrl=user_ctrl, channels=channels, user=user)
            measurement, user_pol_visibility = self.Measure_CostFunction(user, meas_device=self.meas_device, channels=channels)
            print(f"user pol visibility: {user_pol_visibility}")
            if np.round(user_pol_visibility, 3) <= self.threshold_cost:
                success = True
                break
        self.log_event(self.log_file, f"Optimal Voltage for polarization control: {best_voltage}")
        self.log_event(self.log_file, f"Minimum Visibility: {best_cost}")
        return best_voltage, best_cost, success
