import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import time

def generate_sphere(radius, num_points=100):
    """Generates a 3D mesh grid for a sphere surface."""
    phi = np.linspace(0, np.pi, num_points)  # Elevation angle
    theta = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle
    Phi, Theta = np.meshgrid(phi, theta)

    X = radius * np.sin(Phi) * np.cos(Theta)
    Y = radius * np.sin(Phi) * np.sin(Theta)
    Z = radius * np.cos(Phi)
    
    return X, Y, Z

def generate_random_positions_sphere(radius, num_neutrons):
    """Generates uniform neutron positions inside a sphere."""
    u = np.random.uniform(0, 1, num_neutrons)  # To ensure volume-uniform distribution
    r = radius * (u ** (1/3))
    
    theta = np.random.uniform(0, 2 * np.pi, num_neutrons)
    cos_phi = np.random.uniform(-1, 1, num_neutrons)
    sin_phi = np.sqrt(1 - cos_phi**2)

    x = r * np.cos(theta) * sin_phi
    y = r * np.sin(theta) * sin_phi
    z = r * cos_phi

    return np.column_stack((x, y, z))

def generate_random_directions(num_neutrons):
    """Generates isotropic random unit direction vectors."""
    theta_dir = np.random.uniform(0, 2 * np.pi, num_neutrons)  
    cos_phi = np.random.uniform(-1, 1, num_neutrons)  
    sin_phi = np.sqrt(1 - cos_phi**2)

    vx = np.cos(theta_dir) * sin_phi
    vy = np.sin(theta_dir) * sin_phi
    vz = cos_phi

    return np.column_stack((vx, vy, vz))

def neutron_simulation_sphere(radius, num_neutrons, l):
    """Simulates neutron trajectories inside a spherical volume."""
    positions = generate_random_positions_sphere(radius, num_neutrons)
    trajectories = [ [pos.copy()] for pos in positions]

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

            # Check exit condition: if outside sphere
            x, y, z = positions[i]
            if x**2 + y**2 + z**2 > radius**2:
              break 

    return [np.array(traj) for traj in trajectories]

def plot_survival_probability(trajectories, radius):
    """Plots the fraction of neutrons still inside the sphere over time."""
    max_steps = max(len(traj) for traj in trajectories)
    survival_count = np.zeros(max_steps)

    for traj in trajectories:
        for i, (x, y, z) in enumerate(traj):
            if x**2 + y**2 + z**2 <= radius**2:
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
    plt.xlabel("Number of Steps", fontsize=16)
    plt.ylabel("Fraction of Neutrons Remaining", fontsize=16)
    plt.title("Neutron Survival Probability in Sphere", fontsize=18)
    plt.legend()
    plt.grid()
    plt.show()

# Sphere and neutron parameters
radius = 16
num_neutrons = 1000
mean_free_path = 0.5  


trajectories = neutron_simulation_sphere(radius, num_neutrons, mean_free_path)

# Plot trajectories
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot sphere boundary
X, Y, Z = generate_sphere(radius)
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.2, edgecolor='k')

# Plot neutron paths
for traj in trajectories:
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title(f"Neutron Trajectories in a Sphere (l = {mean_free_path})", fontsize=16)

plt.show()

def verify_initial_distribution(radius, height, num_neutrons):
    
    positions = generate_random_positions_sphere(radius, height, num_neutrons)
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

#plot_survival_probability(trajectories, radius)
