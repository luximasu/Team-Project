import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



class Obstacle:
    def __init__(self, pos, radius):
        self.pos = pos 
        self.radius = radius  

    def is_colliding(self, point):
        dist = np.linalg.norm(point - self.pos)
        return dist <= self.radius


class UAV:
    def __init__(self, pos, fov, targ_pos, obstacles=[]):
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
       

    def update_pos(self, homebase, plot_area, step_size):
        dist = np.linalg.norm(self.pos - homebase)
        self.is_targ_within_fov()

        direction = (self.pos - homebase) / dist
        self.vel = direction * step_size
        self.pos += self.vel

            # Check for obstacle collisions and avoid them
        for obstacle in self.obstacles:
                if obstacle.is_colliding(self.pos):
                    self.avoid_obstacle(obstacle)

    def avoid_obstacle(self, obstacle):
            avoid_vec = self.pos - obstacle.pos
            avoid_vec /= np.linalg.norm(avoid_vec)

            avoid_dist = 2 * (obstacle.radius + self.fov) 
            avoid_offset = avoid_dist * avoid_vec
            self.pos += avoid_offset



class Swarm:
    def __init__(self, n_uavs, homebase, plot_area, fov, targ_pos, n_obstacles):
        self.n_uavs = n_uavs
        self.homebase = homebase
        self.plot_area = plot_area
        self.fov = fov
        self.targ_pos = targ_pos

        #Generate random obstacles and UAVs
        self.obstacles = [Obstacle(np.random.uniform(low=-10, high=10, size=2), 0.3) for _ in range(n_obstacles)]
        self.uavs = [UAV(np.random.uniform(low=-1, high=1, size=2), self.fov, self.targ_pos, self.obstacles) for _ in range(n_uavs)]



    def update_poss(self, step_size):
        
        for uav in self.uavs:
            
                uav.update_pos(self.homebase, self.plot_area, step_size)
        


# Parameters
n_uavs = 10  # Number of UAVs
n_steps = 100  # Number of simulation steps
step_size = 0.1  # Step size for movement
homebase = np.array([0, 0])  # Homebase location
plot_area = 10  # Size of the plotted area
fov = 3.0  # Field of view
targ_pos = np.array(np.random.uniform(-5, 5, 2))
n_obstacles = 12

swarm = Swarm(n_uavs, homebase, plot_area, fov, targ_pos)

def update_poss(i):
    global swarm

    swarm.update_poss(step_size)



    
    plt.clf()
    plt.scatter(homebase[0], homebase[1], color='green', label='Homebase')


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



# Create animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_poss, frames=n_steps, interval=100)

plt.show()
