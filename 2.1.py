import numpy as np
import math
import sys
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
nuclide_list = [
    {
        'name': 'U235',
        'mass_fraction': 0.007,
        'molar_mass': 235.0,
        'density': 19.1,
        'cross_sections': {
            'thermal': {'sigma_s': 10.0, 'sigma_a': 99.0},  # barns
            'fast':    {'sigma_s': 4.0,  'sigma_a': 0.09},  # barns
        }
    },
    {
        'name': 'U238',
        'mass_fraction': 0.993,
        'molar_mass': 238.0,
        'density': 19.1,
        'cross_sections': {
            'thermal': {'sigma_s': 9.0, 'sigma_a': 2.0},    # barns
            'fast':    {'sigma_s': 5.0, 'sigma_a': 0.07},   # barns
        }
    },
    {
        'name': 'graphite',
        'mass_fraction': 0.03,
        'molar_mass': 12.0,
        'density': 2.267,
        'cross_sections': {
            'thermal': {'sigma_s': 5.0,   'sigma_a': 0.02},     # barns
            'fast':    {'sigma_s': 2.0,   'sigma_a': 0.00001},  # barns
        }
    }
]

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
    positions = generate_random_positions(radius, height, num_neutrons)
    directions = generate_random_directions(num_neutrons)
    results = []  # will store (trajectory, status)

    for i in range(num_neutrons):
        traj = [positions[i].copy()]  # store initial position
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
                # continue loop
            else:
                # absorbed
                results.append((traj, "absorbed"))
                break

    return results

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

###############################################################################
# 4) Main Program
###############################################################################
def main():
    print("=== Mean Free Path Calculator ===")

    # 1) Ask how many nuclides
    num_str = input("How many nuclides in your mixture? ")
    try:
        num_nuclides = int(num_str)
    except ValueError:
        num_nuclides = 1
        print("Invalid input. Defaulting to 1.")
    if num_nuclides <= 0:
        print("Number of nuclides must be > 0. Defaulting to 1.")
        num_nuclides = 1

    # Build a lookup dict for known nuclides
    known_dict = {nuc['name']: nuc for nuc in nuclide_list}
    chosen_nuclides = []

    # 2) Gather user inputs for each nuclide
    for i in range(num_nuclides):
        print(f"\n--- Nuclide {i+1} of {num_nuclides} ---")
        nuc_name = input("  Which nuclide? (e.g. 'U235', 'U238', 'graphite'): ")
        if nuc_name not in known_dict:
            print(f"  '{nuc_name}' not found. Defaulting to 'U235'.")
            nuc_name = 'U235'

        mf_str = input("  Mass fraction (0 < mf < 1): ")
        try:
            mf = float(mf_str)
        except ValueError:
            mf = 0.0
            print("  Invalid fraction. Defaulting to 0.0")

        base_data = known_dict[nuc_name]
        nuc_dict = {
            'name': base_data['name'],
            'molar_mass': base_data['molar_mass'],
            'density': base_data['density'],
            'cross_sections': base_data['cross_sections'],
            'mass_fraction': mf
        }
        chosen_nuclides.append(nuc_dict)

    # Check total fraction
    total_fraction = sum(nuc['mass_fraction'] for nuc in chosen_nuclides)
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-5):
        print(f"Warning: total mass fraction = {total_fraction:.3f}, not 1.0!")
        sys.exit("Exiting program due to invalid mass fraction sum.")

    # 3) Ask for energy regime
    regime = input("\nEnergy regime? ('thermal' or 'fast'): ")
    if regime not in ["thermal", "fast"]:
        print("Invalid choice. Defaulting to 'thermal'.")
        regime = "thermal"

    # 4) Create the material data (Sigma_tot, etc.)
    material_name = "User-Defined Mixture"
    result = create_material_data(
        material_name,
        density=1.0,  # dummy
        nuclide_list=chosen_nuclides,
        energy_regime=regime
    )

    Sigma_tot = result["Sigma_tot"]
    mfp = result["mean_free_path"]

    # Overall scattering & absorption
    Sigma_s_total = sum(nuc["Sigma_s"] for nuc in result["nuclides"])
    Sigma_a_total = sum(nuc["Sigma_a"] for nuc in result["nuclides"])
    if Sigma_tot > 0:
        P_scat = Sigma_s_total / Sigma_tot
        P_abs = Sigma_a_total / Sigma_tot
    else:
        P_scat = 0.0
        P_abs = 0.0

    # Print summary
    print("\n=== RESULTS ===")
    print(f"Energy Regime: {regime}")
    print(f"Mean Free Path: {mfp:.4e} cm")
    print(f"Probability of scattering: {P_scat:.4f}")
    print(f"Probability of absorption: {P_abs:.4f}")
    print("Nuclide Breakdown:")
    for nuc in result["nuclides"]:
        frac_percent = nuc['mass_fraction'] * 100
        print(f"  - {nuc['name']}: fraction={frac_percent:.1f}%, Sigma_i={nuc['Sigma_i']:.3e} cm^-1")

    # 5) Prompt user for random-walk simulation
    ans = input("\nDo you want to run a neutron random walk simulation? (y/n): ")
    if ans.lower().startswith('y'):
        # Ask how many neutrons
        num_neutrons_str = input("How many neutrons to simulate? ")
        try:
            num_neutrons = int(num_neutrons_str)
        except ValueError:
            num_neutrons = 10002 

        # Cylinder geometry
        radius = 20.0
        height = 20.0

        # Run the random walk. 
        # NOTE: your function should return a list of (trajectory, status) pairs
        results = simulate_neutron_transport(
            radius=radius,
            height=height,
            num_neutrons=num_neutrons,
            P_scat=P_scat,
            P_abs=P_abs,
            mean_free_path=mfp
        )
        print("Simulation complete.")

        # Count how many escaped vs. absorbed using the status field
        escaped_count = sum(1 for (_, status) in results if status == "escaped")
        absorbed_count = sum(1 for (_, status) in results if status == "absorbed")
        print(f"Neutrons escaped:   {escaped_count}")
        print(f"Neutrons absorbed: {absorbed_count}")
        print("Done.")

        # Plot color-coded trajectories
        plot_trajectories_3d(results, radius, height, num_to_plot=100)

if __name__ == "__main__":
    main()

