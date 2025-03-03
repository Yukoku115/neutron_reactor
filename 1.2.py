import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

def generate_cylinder(radius, height, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)  
    z = np.linspace(0, height, num_points)
    Theta, Z = np.meshgrid(theta, z)

    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)

    # Generate top and bottom disk caps
    theta_cap = np.linspace(0, 2 * np.pi, num_points)
    r_cap = np.linspace(0, radius, num_points)
    Theta_cap, R_cap = np.meshgrid(theta_cap, r_cap)

    X_cap = R_cap * np.cos(Theta_cap)
    Y_cap = R_cap * np.sin(Theta_cap)
    Z_top = np.full_like(X_cap, height)  # Top disk at z = height
    Z_bottom = np.full_like(X_cap, 0)    # Bottom disk at z = 0

    return X, Y, Z, X_cap, Y_cap, Z_top, Z_bottom

def generate_random_positions(radius, height, num_neutrons):
   
    r = radius * np.sqrt(np.random.uniform(0, 1, num_neutrons))  
    theta = np.random.uniform(0, 2 * np.pi, num_neutrons)  
    z = np.random.uniform(0, height, num_neutrons)  

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y, z))  # Return a single NumPy array for efficiency

def generate_random_directions(num_neutrons):
    """Generates isotropic random unit direction vectors."""
    theta_dir = np.random.uniform(0, 2 * np.pi, num_neutrons)  
    cos_phi = np.random.uniform(-1, 1, num_neutrons)  
    sin_phi = np.sqrt(1 - cos_phi**2)

    vx = np.cos(theta_dir) * sin_phi
    vy = np.sin(theta_dir) * sin_phi
    vz = cos_phi

    return np.column_stack((vx, vy, vz))  # Store as a NumPy array

def neutron_simulation(radius, height, num_neutrons, l):
    """Simulates neutron trajectories inside a cylindrical volume."""
    positions = generate_random_positions(radius, height, num_neutrons)
    trajectories = [ [pos.copy()] for pos in positions]  # Store initial positions

    for i in range(num_neutrons):
        while True:
            d = -l * np.log(np.random.uniform())  # Sample step length
            
            # Generate a new random direction at each step
            theta_dir = np.random.uniform(0, 2 * np.pi)
            cos_phi = np.random.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)

            direction = np.array([
                np.cos(theta_dir) * sin_phi,
                np.sin(theta_dir) * sin_phi,
                cos_phi
            ])

            # Update position
            positions[i] += d * direction
            trajectories[i].append(positions[i].copy())  # Store new step

            # Check exit condition
            x, y, z = positions[i]
            if x**2 + y**2 >= radius**2 or z <= 0 or z >= height:
                break

    return [np.array(traj) for traj in trajectories]  # Convert to NumPy arrays for fast processing

def plot_survival_probability(trajectories, radius, height):
    
    max_steps = max(len(traj) for traj in trajectories)
    survival_count = np.zeros(max_steps)

    for traj in trajectories:
        for i, (x, y, z) in enumerate(traj):
            if x**2 + y**2 <= radius**2 and 0 <= z <= height:
                survival_count[i] += 1

    survival_fraction = survival_count / len(trajectories)
    time_steps = np.arange(len(survival_fraction))

    # Fit an exponential function to the survival data
    def exponential_fit(t, N0, tau):
        return N0 * np.exp(-t / tau)

    popt, _ = curve_fit(exponential_fit, time_steps, survival_fraction, p0=[1, 10])

    # Plot the survival probability
    plt.figure(figsize=(8,6))
    plt.plot(time_steps, survival_fraction, 'o-', label="Simulated Data")
    plt.plot(time_steps, exponential_fit(time_steps, *popt), 'r--', 
             label=f"Fit: N(t) = N₀ exp(-t/{popt[1]:.2f})")
    plt.xlabel("Number of Steps", fontsize =24)
    plt.ylabel("Fraction of Neutrons Remaining,", fontsize =24)
    plt.title("Neutron Survival Probability Over Time", fontsize =24)
    plt.legend()
    plt.grid()
    plt.show()

# Cylinder and neutron parameters
radius = 8
height = 8
num_neutrons = 10000
mean_free_path = 0.1

start_time = time.time()
trajectories = neutron_simulation(radius, height, num_neutrons, mean_free_path)
end_time = time.time()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
"""
# Plot cylinder boundary
X, Y, Z = generate_cylinder(radius, height)
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.3, edgecolor='k')
"""
# Plot part of the trajectories
num_to_plot = 50  # Only plot 500 neutrons for clarity
sampled_trajectories = np.random.choice(len(trajectories), num_to_plot, replace=False)

X, Y, Z, X_cap, Y_cap, Z_top, Z_bottom = generate_cylinder(radius, height)


ax.plot_surface(X, Y, Z, color='cyan', alpha=0.3, edgecolor='none')


ax.plot_surface(X_cap, Y_cap, Z_top, color='cyan', alpha=0.3, edgecolor='none')
ax.plot_surface(X_cap, Y_cap, Z_bottom, color='cyan', alpha=0.3, edgecolor='none')


for i in sampled_trajectories:
    ax.plot(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], linewidth=1)
execution_time = end_time - start_time
print(f"Simulation completed in {execution_time:.4f} seconds.")

"""
for traj in trajectories:
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1)
"""
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title(f"Neutron Trajectories in a Cylinder (l = {mean_free_path})",fontsize =24)

plt.show()


#plot_survival_probability(trajectories, radius, height)


def verify_initial_distribution(radius, height, num_neutrons):
    
    positions = generate_random_positions(radius, height, num_neutrons)
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    r = np.sqrt(x**2 + y**2)  # Compute radial distances

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(r, bins=20, color='blue', alpha=0.7, edgecolor='black', density=True)
    axes[0].set_xlabel("Radial Distance (r)", fontsize=16)
    axes[0].set_ylabel("Probability Density", fontsize=16)
    axes[0].set_title("Radial Distribution", fontsize=18)

    theta = np.arctan2(y, x)
    axes[1].hist(theta, bins=20, color='green', alpha=0.7, edgecolor='black', density=True)
    axes[1].set_xlabel("Azimuthal Angle (θ)", fontsize=16)
    axes[1].set_ylabel("Probability Density", fontsize=16)
    axes[1].set_title("Azimuthal Angle Distribution", fontsize=18)

    axes[2].hist(z, bins=20, color='red', alpha=0.7, edgecolor='black', density=True)
    axes[2].set_xlabel("Height (z)", fontsize=16)
    axes[2].set_ylabel("Probability Density", fontsize=16)
    axes[2].set_title("Height Distribution", fontsize=18)

    plt.show()

#verify_initial_distribution(radius, height, num_neutrons)
