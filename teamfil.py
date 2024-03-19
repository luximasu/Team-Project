import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Obstacle:
    def __init__(self, position, radius):
        self.position = position  # Position of the obstacle (center)
        self.radius = radius  # Radius of the obstacle

    def is_colliding(self, point):
        distance = np.linalg.norm(point - self.position)
        return distance <= self.radius


class UAV:
    def __init__(self, position, fov, targ_pos, obstacles=[], alpha=1, beta=1):
        self.position = position
        self.velocity = np.zeros(2)
        self.color = 'blue'
        self.no_hit = True
        self.best_pos = self.position.copy()
        self.best_score = float('inf')
        self.fov = fov
        self.targ_pos = targ_pos
        self.at_targ = False
        self.use_aco = False
        self.obstacles = obstacles  # List of obstacles
        self.alpha = alpha  # ACO parameter alpha
        self.beta = beta    # ACO parameter beta
        self.pheromone_map = np.zeros(8)

    def update_position(self, homebase, plot_area, step_size):
        distance = np.linalg.norm(self.position - homebase)
        self.is_targ_within_fov()

        if not self.at_targ:
            if (distance < plot_area) and (self.no_hit == True):
                direction = (self.position - homebase) / distance
                self.velocity = direction * step_size
                self.position += self.velocity


            elif (distance == plot_area) and (self.no_hit == True):
                self.color = 'red'
                self.no_hit = False
            elif not self.at_targ:  
                self.pso()
                self.color = 'red'
                self.no_hit = False
            else:
                self.color = 'red'
                self.no_hit = False
                self.use_aco = True 
                self.aco(np.zeros(8))  # Placeholder for pheromone map
        elif self.at_targ:
                    self.color = 'green'
                    self.no_hit = False
                    self.use_aco = True 
                    self.aco(np.zeros(8))  
            # Check for obstacle collisions and avoid them
        for obstacle in self.obstacles:
                if obstacle.is_colliding(self.position):
                    self.avoid_obstacle(obstacle)

    def pso(self):
        inertia_weight = 0.5  # Inertia weight
        c_weight = 0.5  # Cognitive weight
        s_weight = 0.5  # Social weight
        max_velocity = 0.50  # Maximum velocity

        personal_best_position = self.best_pos
        global_best_position = self.targ_pos

        # Update velocity
        c_comp = c_weight * np.random.rand() * (personal_best_position - self.position)
        s_comp = s_weight * np.random.rand() * (global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + c_comp + s_comp

        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

        self.position += self.velocity

        score = np.linalg.norm(self.position - self.targ_pos)
        if score < self.best_score:
            self.best_pos = self.position.copy()
            self.best_score = score

    def aco(self, pheromone_map):
        probabilities = np.zeros(8)  # Placeholder for probabilities
        total_probability = 0
        self.color = 'green'

        for direction in range(8):  
            probabilities[direction] = (pheromone_map[direction] ** self.alpha) * ((1 / self.distance_to_target(direction)) ** self.beta)
            total_probability += probabilities[direction]

        total_probability = np.sum(probabilities)
        if total_probability > 0:
            probabilities /= total_probability

        chosen_direction = np.random.multinomial(1, probabilities)
        self.position += self.move_in_direction(chosen_direction)

        self.update_pheromone_map(chosen_direction)
    
    def distance_to_target(self, direction):
        next_position = self.position + self.move_in_direction(direction)
        distance = np.linalg.norm(next_position - self.targ_pos)
        return distance
    
    def move_in_direction(self, direction):
        directions = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])
        movement_vectors = directions * np.linalg.norm(self.velocity)
        return movement_vectors
    
    def update_pheromone_map(self, direction, decay_rate=0.1, increment=0.1):
        self.pheromone_map *= (1 - decay_rate)
        self.pheromone_map[direction] += increment

    def check_fit(self):
        score = np.linalg.norm(self.position - self.targ_pos)
        return score

    def is_targ_within_fov(self):
        dist_2_targ = np.linalg.norm(self.targ_pos - self.position)
        if dist_2_targ < self.fov:
            self.at_targ = True

        return dist_2_targ < self.fov

    def is_uav_within_fov(self, other_uav):
        dist_between_uavs = np.linalg.norm(self.position - other_uav.position)
        return dist_between_uavs < self.fov * 2
    
    def avoid_obstacle(self, obstacle):
        avoidance_vector = self.position - obstacle.position
        avoidance_vector /= np.linalg.norm(avoidance_vector)

        avoidance_distance = 2 * obstacle.radius  
        avoidance_offset = avoidance_distance * avoidance_vector
        self.position += avoidance_offset

class Swarm:
    def __init__(self, num_uavs, homebase, plot_area, fov, targ_pos, num_obstacles=9):
        self.num_uavs = num_uavs
        self.homebase = homebase
        self.plot_area = plot_area
        self.fov = fov
        self.targ_pos = targ_pos
        # Generate random obstacle positions
        self.obstacles = [Obstacle(np.random.uniform(low=-10, high=10, size=2), 0.3) for _ in range(num_obstacles)]
        self.uavs = [UAV(np.random.uniform(low=-1, high=1, size=2), self.fov, self.targ_pos, self.obstacles) for _ in range(num_uavs)]

    def update_positions(self, step_size):
        for uav in self.uavs:
            uav.update_position(self.homebase, self.plot_area, step_size)

    def check_fit(self):
        for uav in self.uavs:
            uav_best_score = uav.check_fit()
            if uav_best_score < uav.best_score:
                uav.best_score = uav_best_score
                uav.best_pos = uav.position

    def get_global_best_solution(self):
        best_score = float('inf')
        best_pos = None
        for uav in self.uavs:
            if uav.best_score < best_score:
                best_score = uav.best_score
                best_pos = uav.best_pos
        return best_pos, best_score

# Parameters
num_uavs = 10  # Number of UAVs
num_steps = 100  # Number of simulation steps
step_size = 0.1  # Step size for movement
homebase = np.array([0, 0])  # Homebase location
plot_area = 10  # Size of the plotted area
fov = 3.0  # Field of view
# targ_pos = np.array(np.random.uniform(-5, 5, 2))
targ_pos = (-7, 2)

swarm = Swarm(num_uavs, homebase, plot_area, fov, targ_pos)

# position
def update_positions(i):
    global swarm

    swarm.update_positions(step_size)
    swarm.check_fit()
    best_pos, best_score = swarm.get_global_best_solution()

    # Plot
    plt.clf()
    plt.scatter(homebase[0], homebase[1], color='green', label='Homebase')
    plt.scatter(targ_pos[0], targ_pos[1], color='purple', label='Target')

    # Plot UAVs
    for uav in swarm.uavs:
        plt.scatter(uav.position[0], uav.position[1], color=uav.color)

    # Plot outer edge
    circle = plt.Circle((0, 0), plot_area, color='green', fill=False, linestyle='--', label='Outer Edge')
    plt.gca().add_artist(circle)

    for obstacle in swarm.obstacles:
        circle = plt.Circle(obstacle.position, obstacle.radius, color='red', alpha=0.5)
        plt.gca().add_artist(circle)

    plt.title('Swarm of UAVs Movement')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-plot_area, plot_area)
    plt.ylim(-plot_area, plot_area)
    plt.legend()

    if not uav.no_hit:
        # Plot best position found
        plt.scatter(best_pos[0], best_pos[1], color='red')

# Create animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_positions, frames=num_steps, interval=100)

plt.show()
