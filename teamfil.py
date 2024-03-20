import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Obstacle:
    def __init__(self, pos, radius):
        self.pos = pos  # pos of the obstacle (center)
        self.radius = radius  # Radius of the obstacle

    def is_colliding(self, point):
        dist = np.linalg.norm(point - self.pos)
        return dist <= self.radius


class UAV:
    def __init__(self, pos, fov, targ_pos, obstacles=[], alpha=1, beta=1):
        self.pos = pos
        self.vel = np.zeros(2)
        self.color = 'blue'
        self.no_hit = True
        self.best_pos = self.pos.copy()
        self.best_score = float('inf')
        self.fov = fov
        self.targ_pos = targ_pos
        self.at_targ = False
        self.use_pso = False
        self.obstacles = obstacles  # List of obstacles
        self.alpha = alpha  # ACO parameter alpha
        self.beta = beta    # ACO parameter beta
        self.pheromone_map = np.zeros(8)

    def update_pos(self, homebase, plot_area, step_size):
        dist = np.linalg.norm(self.pos - homebase)
        self.is_targ_within_fov()

        if not self.at_targ:
            if (dist < plot_area) and (self.no_hit == True):
                direction = (self.pos - homebase) / dist
                self.vel = direction * step_size
                self.pos += self.vel

            elif (dist == plot_area) and (self.no_hit == True):
                self.color = 'black'
                self.no_hit = False

            elif not self.at_targ:  
                self.pso()
                self.color = 'red'
                self.no_hit = False
            
        elif self.at_targ:
                    self.color = 'green'
                    self.no_hit = False
                    self.use_aco = True 
                    self.pso()  
            # Check for obstacle collisions and avoid them
        for obstacle in self.obstacles:
                if obstacle.is_colliding(self.pos):
                    self.avoid_obstacle(obstacle)

    def pso(self):
        inertia_weight = 0.5  # Inertia weight
        c_weight = 0.5  # Cognitive weight
        s_weight = 0.5  # Social weight
        max_vel = 0.50  # Maximum vel

        personal_best_pos = self.best_pos
        global_best_pos = self.targ_pos

        # Update vel
        c_comp = c_weight * np.random.rand() * (personal_best_pos - self.pos)
        s_comp = s_weight * np.random.rand() * (global_best_pos - self.pos)
        self.vel = inertia_weight * self.vel + c_comp + s_comp

        self.vel = np.clip(self.vel, -max_vel, max_vel)

        self.pos += self.vel

        score = np.linalg.norm(self.pos - self.targ_pos)
        if score < self.best_score:
            self.best_pos = self.pos.copy()
            self.best_score = score

    
    def dist2target(self, direction):
        next_pos = self.pos + self.move_in_direction(direction)
        dist = np.linalg.norm(next_pos - self.targ_pos)
        return dist
    
    def move_in_direction(self, direction):
        directions = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])
        movement_vectors = directions * np.linalg.norm(self.vel)
        return movement_vectors
    

    def check_fit(self):
        score = np.linalg.norm(self.pos - self.targ_pos)
        return score

    def is_targ_within_fov(self):
        dist2targ = np.linalg.norm(self.targ_pos - self.pos)
        if dist2targ < self.fov:
            self.at_targ = True
        else: 
            self.at_targ = False
        return dist2targ < self.fov

    def is_uav_within_fov(self, other_uav):
        dist_between_uavs = np.linalg.norm(self.pos - other_uav.pos)

        avoid_vec = self.pos - other_uav.pos
        avoid_vec /= np.linalg.norm(avoid_vec)

        avoid_dist = 2 * (3 + self.fov)  
        avoid_offset = avoid_dist * avoid_vec
        self.pos += avoid_offset

        return dist_between_uavs < self.fov * 2
    
    def avoid_obstacle(self, obstacle):
        avoid_vec = self.pos - obstacle.pos
        avoid_vec /= np.linalg.norm(avoid_vec)

        avoid_dist = 2 * (obstacle.radius + self.fov) 
        avoid_offset = avoid_dist * avoid_vec
        self.pos += avoid_offset

class Swarm:
    def __init__(self, n_uavs, homebase, plot_area, fov, targ_pos, n_obstacles=9):
        self.n_uavs = n_uavs
        self.homebase = homebase
        self.plot_area = plot_area
        self.fov = fov
        self.targ_pos = targ_pos
        # Generate random obstacle poss
        self.obstacles = [Obstacle(np.random.uniform(low=-10, high=10, size=2), 0.3) for _ in range(n_obstacles)]
        self.uavs = [UAV(np.random.uniform(low=-1, high=1, size=2), self.fov, self.targ_pos, self.obstacles) for _ in range(n_uavs)]

    def update_poss(self, step_size):
        target_detected = False  # Flag to track if the target is detected by any UAV
        for uav in self.uavs:
            if not uav.at_targ:
                uav.update_pos(self.homebase, self.plot_area, step_size)
            else:
                target_detected = True

        if target_detected:
            for uav in self.uavs:
                uav.use_pso = True
                uav.pso()


    def check_fit(self):
        for uav in self.uavs:
            uav_best_score = uav.check_fit()
            if uav_best_score < uav.best_score:
                uav.best_score = uav_best_score
                uav.best_pos = uav.pos

    def get_global_best_solution(self):
        best_score = float('inf')
        best_pos = None
        for uav in self.uavs:
            if uav.best_score < best_score:
                best_score = uav.best_score
                best_pos = uav.best_pos
        return best_pos, best_score

# Parameters
n_uavs = 10  # Number of UAVs
n_steps = 100  # Number of simulation steps
step_size = 0.1  # Step size for movement
homebase = np.array([0, 0])  # Homebase location
plot_area = 10  # Size of the plotted area
fov = 3.0  # Field of view
targ_pos = np.array(np.random.uniform(-5, 5, 2))

swarm = Swarm(n_uavs, homebase, plot_area, fov, targ_pos)

def update_poss(i):
    global swarm

    swarm.update_poss(step_size)
    swarm.check_fit()
    best_pos, best_score = swarm.get_global_best_solution()

    
    plt.clf()
    plt.scatter(homebase[0], homebase[1], color='green', label='Homebase')
    plt.scatter(targ_pos[0], targ_pos[1], color='purple', label='Target')

    #Plot UAVs
    for uav in swarm.uavs:
        plt.scatter(uav.pos[0], uav.pos[1], color=uav.color)

    
    circle = plt.Circle((0, 0), plot_area, color='green', fill=False, linestyle='--', label='Outer Edge')
    plt.gca().add_artist(circle)

    for obstacle in swarm.obstacles:
        circle = plt.Circle(obstacle.pos, obstacle.radius, color='red', alpha=0.5)
        plt.gca().add_artist(circle)

    plt.title('Swarm of UAVs Movement')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-plot_area, plot_area)
    plt.ylim(-plot_area, plot_area)
    plt.legend()

    if not uav.no_hit:
        
        plt.scatter(best_pos[0], best_pos[1], color='red')

# Create animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_poss, frames=n_steps, interval=100)

plt.show()
