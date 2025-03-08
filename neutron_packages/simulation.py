import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

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

                if x**2 + y**2 > radius**2:
                    traj[-1][0] = x * radius / math.sqrt(x**2 + y**2)
                    traj[-1][1] = y * radius / math.sqrt(x**2 + y**2)
                
                if z < 0 :
                    traj[-1][2] = 0
                elif z > height:
                    traj[-1][2] = height

                results.append((traj, "escaped"))
                #print(f'Position: {(traj, "escaped")}, radius: {np.sqrt(radius_bigin[0]**2 + radius_bigin[1]**2 + radius_bigin[2]**2)}')
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

def count_absorbed_tick(absorbed_tick):
    """
    Plots a histogram of the absorbed_tick array.
    """
    
    if not absorbed_tick:
        print("No absorbed ticks to plot.")
        return
    
    # Build the bins from min to max
    min_tick = min(absorbed_tick)
    max_tick = max(absorbed_tick)
    bins = range(min_tick, max_tick + 2)  # +2 so the last bin is inclusive
    
    plt.figure()
    plt.hist(absorbed_tick, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Tick')
    plt.ylabel('Count')
    plt.title('Distribution of Absorbed Ticks')
    plt.show()

def volume_shell(a, b, R, H):
    """
    Calculate the volume of the region inside a vertical cylinder of radius R and height H
    and within a spherical shell defined by
      a <= sqrt(r^2 + (z-H/2)**2) <= b,
    where the sphere is centered at (0,0,H/2).
    
    Parameters:
        a (float): inner spherical radius (with a < b)
        b (float): outer spherical radius
        R (float): radius of the cylinder
        H (float): height of the cylinder
        
    Returns:
        float: the approximate volume of the region.
    """
    def integrand(z):
        # Shift z-coordinate to center the sphere at H/2
        z_shift = z - H/2
        
        # For a fixed z, the spherical upper limit for r is given by:
        r_up_sphere = np.sqrt(b**2 - z_shift**2)
        # But the region is also limited by the cylinder (r < R)
        r_max = min(R, r_up_sphere)
        
        # For the inner sphere, if (z-H/2)^2 < a^2 then r must be at least sqrt(a^2 - (z-H/2)^2).
        # Otherwise, if (z-H/2)^2 >= a^2, the condition a <= sqrt(r^2+(z-H/2)^2) is automatically satisfied by r>=0.
        r_min = np.sqrt(a**2 - z_shift**2) if z_shift**2 < a**2 else 0.0
        
        # If the spherical shell gives no region at this z (i.e. r_max < r_min), return 0.
        if r_max < r_min:
            return 0.0
        
        # The r-integral can be computed analytically:
        # ∫[r=r_min to r_max] r dr = 1/2*(r_max^2 - r_min^2)
        # The theta-integral (0 to 2π) gives a factor of 2π.
        # Therefore, the z-integrand is:
        return np.pi * (r_max**2 - r_min**2)
    
    # z must be within both the cylinder (0 <= z <= H) and the spherical shell (z in [H/2-b, H/2+b])
    z_lower = max(0, H/2 - b)
    z_upper = min(H, H/2 + b)
    
    vol, err = quad(integrand, z_lower, z_upper)
    return vol

def verify_end_distribution(results, radius, height, particle_num, num_bins=50):

    # Compute particle density in the cylinder:
    # density = particle_num / (Volume of cylinder)
    density = particle_num / (np.pi * radius**2 * height)

    # Lists to store the final radial positions for escaped vs. absorbed
    escaped_r = []
    absorbed_r = []

    for (trajectory, status) in results:
        # final position
        final_pos = trajectory[0]
        x, y, z = final_pos
        r = math.sqrt(x**2 + y**2 + (z-height/2)**2)

        if status == "escaped":
            # It's typical that the last position is outside the geometry,
            # but we'll just store the radial distance anyway.
            escaped_r.append(r)
        else:
            # "absorbed"
            absorbed_r.append(r)

    # Create a figure with two subplots side by side
    # Create a figure with two subplots side by side (without sharing y axis)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # --- Escaped neutrons ---
    escaped_r = np.array(escaped_r)  # your data array
    escaped_counts, bin_edges = np.histogram(escaped_r, bins=num_bins, range=(0, radius*1.2))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Re-weight the escaped radial distribution
    eff_area_escaped = np.array([volume_shell(bin_edges[i], bin_edges[i+1], radius, height) for i in range(len(bin_edges)-1)])
    escaped_bin_heights = escaped_counts / (eff_area_escaped)
    escaped_errors = np.sqrt(escaped_counts) / (eff_area_escaped)

    axes[0].bar(bin_centers, escaped_bin_heights, width=(bin_edges[1] - bin_edges[0]),
                color='tomato', alpha=0.7, edgecolor='black', label='Escaped')
    axes[0].errorbar(bin_centers, escaped_bin_heights, yerr=escaped_errors, fmt='none',
                    ecolor='black', capsize=5)
    
    # Plot density line on escaped subplot
    axes[0].axhline(density, color='blue', linestyle='--', label='Density')

    # Compute the average histogram height (excluding empty bins) for escaped neutrons
    nonempty = escaped_counts > 0
    if np.any(nonempty):
        avg_escaped_height = np.mean(escaped_bin_heights[nonempty])
        axes[0].axhline(avg_escaped_height, color='red', linestyle=':', label='Avg Escaped Height')

    axes[0].set_title("Final Radial Distribution (Escaped)", fontsize=24)
    axes[0].set_xlabel("Radius (cm)", fontsize=20)
    axes[0].set_ylabel("density", fontsize=20)
    axes[0].legend()

    # --- Absorbed neutrons ---
    absorbed_r = np.array(absorbed_r)  # your data array
    absorbed_counts, absorbed_bin_edges = np.histogram(absorbed_r, bins=num_bins, range=(0, radius*1.2))
    absorbed_bin_centers = 0.5 * (absorbed_bin_edges[:-1] + absorbed_bin_edges[1:])
    bin_width_absorbed = absorbed_bin_edges[1] - absorbed_bin_edges[0]

    # Calculate the effective area for each bin center
    eff_area = np.array([volume_shell(absorbed_bin_edges[i], absorbed_bin_edges[i+1], radius, height) for i in range(len(absorbed_bin_edges)-1)])
    # Re-weight using the effective spherical area within the cylinder
    absorbed_bin_heights = absorbed_counts / (eff_area)
    # Calculate error bars with the same scaling
    absorbed_errors = np.sqrt(absorbed_counts) / (eff_area)

    axes[1].bar(absorbed_bin_centers, absorbed_bin_heights, width=(absorbed_bin_edges[1] - absorbed_bin_edges[0]),
                color='olivedrab', alpha=0.7, edgecolor='black', label='Absorbed')
    axes[1].errorbar(absorbed_bin_centers, absorbed_bin_heights, yerr=absorbed_errors, fmt='none',
                    ecolor='black', capsize=5)
    
    # Plot density line on absorbed subplot
    axes[1].axhline(density, color='blue', linestyle='--', label='Density')

    # Compute the average histogram height (excluding empty bins) for absorbed neutrons
    nonempty_abs = absorbed_counts > 0
    if np.any(nonempty_abs):
        avg_absorbed_height = np.mean(absorbed_bin_heights[nonempty_abs])
        axes[1].axhline(avg_absorbed_height, color='red', linestyle=':', label='Avg Absorbed Height')

    axes[1].set_title("Final Radial Distribution (Absorbed)", fontsize=24)
    axes[1].set_xlabel("Radius (cm)", fontsize=20)
    axes[1].set_ylabel("density", fontsize=20)
    axes[1].legend()

    plt.tight_layout()
    plt.show()