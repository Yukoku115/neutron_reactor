import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# 1) Geometry Helpers
###############################################################################
def generate_cylinder(radius, height, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(0, height, num_points)
    Theta, Z = np.meshgrid(theta, z)

    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)
    return X, Y, Z

def generate_random_positions(radius, height, num_neutrons):
    """Uniform random positions inside a cylinder of radius, height."""
    r = radius * np.sqrt(np.random.uniform(0, 1, num_neutrons))
    phi = np.random.uniform(0, 2 * np.pi, num_neutrons)
    z = np.random.uniform(0, height, num_neutrons)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x, y, z))

def generate_random_directions(num_neutrons):
    """Generates isotropic random unit direction vectors."""
    cos_theta = np.random.uniform(-1, 1, num_neutrons)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, num_neutrons)

    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = cos_theta
    return np.column_stack((vx, vy, vz))

###############################################################################
# 2) Cross-Section / Material Data
###############################################################################

def create_material_data(material_name, density, nuclide_list, energy_regime='thermal'):
    """
    Sums up partial cross sections for each nuclide to get total Sigma_tot (cm^-1).
    Returns a dict with 'Sigma_tot', 'mean_free_path', and a list of per-nuclide data.
    """
    N_A = 6.02214076e23  # atoms/mol
    Sigma_tot = 0.0
    output_nuclides = []

    for nuc in nuclide_list:
        name = nuc["name"]
        mass_fraction = nuc["mass_fraction"]
        molar_mass = nuc["molar_mass"]
        nuc_density = nuc["density"]

        # Get cross sections for the chosen regime
        sigma_s_barns = nuc["cross_sections"][energy_regime]["sigma_s"]
        sigma_a_barns = nuc["cross_sections"][energy_regime]["sigma_a"]

        # barns -> cm^2
        sigma_s_cm2 = sigma_s_barns * 1e-24
        sigma_a_cm2 = sigma_a_barns * 1e-24

        # Number density (naive approach)
        N_i = (nuc_density * mass_fraction / molar_mass) * N_A

        # Macroscopic cross sections
        Sigma_s = sigma_s_cm2 * N_i
        Sigma_a = sigma_a_cm2 * N_i
        Sigma_i = Sigma_s + Sigma_a

        Sigma_tot += Sigma_i
        output_nuclides.append({
            "name": name,
            "mass_fraction": mass_fraction,
            "Sigma_s": Sigma_s,
            "Sigma_a": Sigma_a,
            "Sigma_i": Sigma_i
        })

    # Mean free path
    if Sigma_tot > 0:
        mfp = 1.0 / Sigma_tot
    else:
        mfp = math.inf

    return {
        "Sigma_tot": Sigma_tot,
        "mean_free_path": mfp,
        "nuclides": output_nuclides
    }

###############################################################################
# 3) Random Walk Simulation with Scattering/Absorption
###############################################################################
def simulate_neutron_transport(
    radius:int,
    height,
    num_neutrons,
    P_scat,
    P_abs,
    mean_free_path
):
    absorbed_tick = []
    positions = generate_random_positions(radius, height, num_neutrons)
    directions = generate_random_directions(num_neutrons)
    results = []  # will store (trajectory, status)

    for i in range(num_neutrons):
        traj = [positions[i].copy()]  # store initial position
        tick = 0
        while True:
            # 1) Sample a free-flight distance
            d = -mean_free_path * math.log(np.random.rand())

            # 2) Move the neutron
            direction = directions[i]
            positions[i] += d * direction
            traj.append(positions[i].copy())

            x, y, z = positions[i]

            # 3) Check if neutron leaves geometry
            if x**2 + y**2 > radius**2 or z < 0 or z > height:
                # Neutron escapes
                results.append((traj, "escaped"))
                break

            # 4) Otherwise, collision occurs:
            u = np.random.rand()
            if u < P_scat:
                # scattering
                theta_dir = np.random.uniform(0, 2 * np.pi)
                cos_phi = np.random.uniform(-1, 1)
                sin_phi = math.sqrt(1 - cos_phi**2)
                new_dir = np.array([
                    math.cos(theta_dir) * sin_phi,
                    math.sin(theta_dir) * sin_phi,
                    cos_phi
                ])
                directions[i] = new_dir
                tick += 1
                # continue loop
            else:
                # absorbed
                results.append((traj, "absorbed"))
                absorbed_tick.append(tick)
                break

    return results, absorbed_tick

def plot_trajectories_3d(results, radius, height, num_to_plot=100):
    """
    results: list of (trajectory, status) pairs.
    radius, height: cylinder geometry
    num_to_plot: how many trajectories to plot for clarity
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1) Plot the cylinder surface
    X, Y, Z = generate_cylinder(radius, height)
    #ax.plot_surface(X, Y, Z, color='cyan', alpha=0.3, edgecolor='k')

    # 2) Subset of trajectories
    if len(results) <= num_to_plot:
        subset_indices = range(len(results))
    else:
        subset_indices = np.random.choice(len(results), size=num_to_plot, replace=False)

    # 3) Plot each trajectory with color coding
    for i in subset_indices:
        trajectory, status = results[i]
        traj = np.array(trajectory)
        if status == "escaped":
            color = "tomato"
        else:  # "absorbed"
            color = "olivedrab"

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1, color=color)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("Neutron Trajectories (Green=Escaped, Red=Absorbed)")
    plt.tight_layout()
    plt.show()