import numpy as np
import matplotlib.pyplot as plt

class UAV:
    def __init__(self, position, field_of_view):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position
        self.best_score = float('inf')
        self.field_of_view = field_of_view

    def update_position_pso(self, target_position):
        # Update velocity and position using PSO algorithm
        # Placeholder for demonstration, random movement
        self.velocity = np.random.uniform(-1, 1, self.position.shape)
        self.position += self.velocity

        # Placeholder for demonstration, update best position based on distance to target
        score = np.linalg.norm(self.position - target_position)
        if score < self.best_score:
            self.best_position = self.position
            self.best_score = score

    def evaluate_fitness(self, target_position):
        # Placeholder for demonstration, fitness function can be based on distance to target
        score = np.linalg.norm(self.position - target_position)
        return score

    def is_target_within_field_of_view(self, target_position):
        # Check if the target is within the field of view of the UAV
        dist_to_target = np.linalg.norm(target_position - self.position)
        return dist_to_target < self.field_of_view

    def is_uav_within_field_of_view(self, other_uav):
        # Check if another UAV is within the field of view of this UAV
        dist_between_uavs = np.linalg.norm(self.position - other_uav.position)
        return dist_between_uavs < self.field_of_view * 2  # Adjust 2 as needed to prevent overlap

class Swarm:
    def __init__(self, num_uavs, target_position, field_of_view):
        self.uavs = self.initialize_uavs(num_uavs, field_of_view)
        self.target_position = target_position
        self.use_pso = True

    def initialize_uavs(self, num_uavs, field_of_view):
        uavs = []
        for _ in range(num_uavs):
            # Generate random position for UAV
            position = np.random.uniform(-10, 10, 2)
            # Ensure no overlap with existing UAVs
            while any(uav.is_uav_within_field_of_view(UAV(position, field_of_view)) for uav in uavs):
                position = np.random.uniform(-10, 10, 2)
            uavs.append(UAV(position, field_of_view))
        return uavs

    def simulate_step_pso(self):
        for uav in self.uavs:
            uav.update_position_pso(self.target_position)

    def evaluate_fitness(self):
        for uav in self.uavs:
            uav_best_score = uav.evaluate_fitness(self.target_position)
            if uav_best_score < uav.best_score:
                uav.best_score = uav_best_score
                uav.best_position = uav.position

    def get_global_best_solution(self):
        best_score = float('inf')
        best_position = None
        for uav in self.uavs:
            if uav.best_score < best_score:
                best_score = uav.best_score
                best_position = uav.best_position
        return best_position, best_score

    def switch_to_aco(self):
        self.use_pso = False

# Main simulation
num_iterations_pso = 50
num_iterations_aco = 50
num_uavs = 10
target_position = np.array([5, 5])
field_of_view = 3.0  # Example field of view radius

swarm = Swarm(num_uavs, target_position, field_of_view)

# Visualization
plt.figure()
for i in range(num_iterations_pso):
    swarm.simulate_step_pso()
    swarm.evaluate_fitness()
    best_position, best_score = swarm.get_global_best_solution()

    # Check if any UAV sees the target within its field of view
    for uav in swarm.uavs:
        if uav.is_target_within_field_of_view(target_position):
            swarm.switch_to_aco()
            break

    if not swarm.use_pso:
        break

    # Clear previous plot
    plt.clf()

    # Plot UAVs for PSO
    plt.scatter(target_position[0], target_position[1], color='red', label='Target')
    for uav in swarm.uavs:
        plt.scatter(uav.position[0], uav.position[1], color='blue', marker='o')
    plt.title(f"Iteration {i+1}: PSO - Best position = {best_position}, Best score = {best_score}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.grid(True)
    plt.pause(0.1)

if not swarm.use_pso:
    # Switched to ACO
    print("Switching to ACO...")

    # Placeholder for ACO simulation
    num_ants = 10
    pheromone_map = np.zeros((100, 100))  # Placeholder for pheromone map

    for i in range(num_iterations_aco):
    # Move ants and update UAV positions
        for ant in range(num_ants):
            for uav in swarm.uavs:
            # Calculate direction towards the target position
                direction_to_target = target_position - uav.position
            # Normalize direction vector
                direction_to_target /= np.linalg.norm(direction_to_target)
            # Move the UAV towards the target position with a random step size
                step_size = np.random.uniform(0, 1)
                uav.position += step_size * direction_to_target

    # Clear previous plot
        plt.clf()

    # Plot UAVs for ACO
        plt.scatter(target_position[0], target_position[1], color='red', label='Target')
    # Plot UAVs for ACO (green)
        for uav in swarm.uavs:
            plt.scatter(uav.position[0], uav.position[1], color='green', marker='o')
        plt.title(f"Iteration {i+1}: ACO Simulation")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.grid(True)
        plt.pause(0.1)

    plt.show()

# mother fucker