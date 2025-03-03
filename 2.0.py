import numpy as np
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import sys

def generate_cylinder(radius, height, num_points=100):
 
    theta = np.linspace(0, 2 * np.pi, num_points)  
    z = np.linspace(0, height, num_points)
    Theta, Z = np.meshgrid(theta, z)

    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)
    
    return X, Y, Z

def generate_random_positions(radius, height, num_neutrons):
   
    r = radius * np.sqrt(np.random.uniform(0, 1, num_neutrons))  
    phi = np.random.uniform(0, 2 * np.pi, num_neutrons)  
    z = np.random.uniform(0, height, num_neutrons)  

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    return np.column_stack((x, y, z)) 

def generate_random_directions(num_neutrons):
    """Generates isotropic random unit direction vectors."""
    
    cos_theta = np.random.uniform(-1, 1, num_neutrons)  # Sample cos(θ) uniformly
    theta = np.arccos(cos_theta)  # Transform to get correct θ distribution
    phi = np.random.uniform(0, 2 * np.pi, num_neutrons)  # Uniform azimuthal angle
    
    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = cos_theta  # Already correctly sampled

    return np.column_stack((vx, vy, vz))  # Store as a NumPy array

def neutron_simulation(radius, height, num_neutrons, l):
    positions = generate_random_positions(radius, height, num_neutrons)
    trajectories = [ [pos.copy()] for pos in positions]  # Store initial positions

    for i in range(num_neutrons):
        while True:
            d = -l * np.log(np.random.uniform())  
            
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
            if x**2 + y**2 > radius**2 or z < 0 or z > height:
                break

    return [np.array(traj) for traj in trajectories]  # Convert to NumPy arrays for fast processing


# Cylinder and neutron parameters
radius = 12
height = 12
num_neutrons = 400000
mean_free_path = 0.2
"""
start_time = time.time()
trajectories = neutron_simulation(radius, height, num_neutrons, mean_free_path)
end_time = time.time()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot cylinder boundary
X, Y, Z = generate_cylinder(radius, height)
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.3, edgecolor='k')

# Plot neutron paths
num_to_plot = 50  # Only plot 500 neutrons for clarity
sampled_trajectories = np.random.choice(len(trajectories), num_to_plot, replace=False)
execution_time = end_time - start_time
#print(f"Simulation completed in {execution_time:.4f} seconds.")


for i in sampled_trajectories:
    ax.plot(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], linewidth=1)
    

for traj in trajectories:
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title(f"Neutron Trajectories in a Cylinder (l = {mean_free_path})",fontsize =24)

#plt.show()
"""
# nuclide_data.py (for example)
nuclide_list = [
    {
        'name': 'U235',
        'mass_fraction': 0.007,
        'molar_mass': 235.0,
        'density':19.1,
        'cross_sections': {
            'thermal': {'sigma_s': 10.0,   'sigma_a': 99.0},  # barns
            'fast':    {'sigma_s': 4.0,   'sigma_a': 0.09},    # barns
        }
    },
    {
        'name': 'U238',
        'mass_fraction': 0.993,
        'molar_mass': 238.0,
        'density':19.1,
        'cross_sections': {
            'thermal': {'sigma_s': 9.0,   'sigma_a': 2.0},    # barns
            'fast':    {'sigma_s': 5.0,   'sigma_a': 0.07},    # barns
        }
    },
    {
        'name': 'graphite',
        'mass_fraction': 0.03,
        'molar_mass': 12.0,
        'density': 2.267,
        'cross_sections': {
            'thermal': {'sigma_s': 5.0,   'sigma_a': 0.02},    # barns
            'fast':    {'sigma_s': 2,   'sigma_a': 0.00001},    # barns
        }
    }
    
]

def create_material_data(material_name, density, nuclide_list, energy_regime='thermal'):

    N_A = 6.02214076e23  # atoms/mol

    Sigma_tot = 0.0
    output_nuclides = []

    for nuc in nuclide_list:
        name = nuc["name"]
        mass_fraction = nuc["mass_fraction"]
        molar_mass = nuc["molar_mass"]
        density = nuc["density"]  # from the hard-coded data

        # Get cross sections for the chosen regime
        sigma_s_barns = nuc["cross_sections"][energy_regime]["sigma_s"]
        sigma_a_barns = nuc["cross_sections"][energy_regime]["sigma_a"]

        # Convert barns -> cm^2
        sigma_s_cm2 = sigma_s_barns * 1e-24
        sigma_a_cm2 = sigma_a_barns * 1e-24

        # Number density for this nuclide
        # (Naive approach: use "density * mass_fraction" for partial mass, 
        # though physically you'd usually have one overall mixture density.)
        N_i = (density * mass_fraction / molar_mass) * N_A

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
    

    # We'll create a dictionary that maps each 'name' to the existing data
    # so we can look them up easily.
    known_dict = { nuc['name']: nuc for nuc in nuclide_list }

    chosen_nuclides = []

    # 2) Gather user inputs for each nuclide
    for i in range(num_nuclides):
        print(f"\n--- Nuclide {i+1} of {num_nuclides} ---")

        # Prompt for nuclide name
        nuc_name = input("  Which nuclide? (e.g. 'U235', 'U238', 'graphite'): ")
        if nuc_name not in known_dict:
            print(f"  '{nuc_name}' not found in nuclide_list. Defaulting to 'U235'.")
            nuc_name = 'U235'

        # Prompt for mass fraction
        mf_str = input("  Mass fraction (0 < mf < 1): ")
        try:
            mf = float(mf_str)
        except ValueError:
            mf = 0.0
            print("  Invalid fraction. Defaulting to 0.0")

        # Retrieve the base data for this nuclide from your existing list
        base_data = known_dict[nuc_name]

        # Build a new dictionary that overrides mass_fraction with user input
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
       print(f"Warning: total mass fraction = {total_fraction:.3f}, which is not 1.0, redo the math!")
       sys.exit("Exiting program due to invalid mass fraction sum.")

    # 3) Ask for energy regime
    regime = input("\nEnergy regime? ('thermal' or 'fast'): ")
    if regime not in ["thermal", "fast"]:
        print("Invalid choice. Defaulting to 'thermal'.")
        regime = "thermal"

    # 4) Compute the total macroscopic cross section & mean free path
    #    Note: We pass a dummy 'density' to create_material_data because
    #    your function ultimately uses each nuclide's own 'density'.
    material_name = "User-Defined Mixture"
    result = create_material_data(
        material_name,
        density=1.0,  # dummy value
        nuclide_list=chosen_nuclides,
        energy_regime=regime
    )
    # Extract total sigma and mean free path
    Sigma_tot = result["Sigma_tot"]
    mfp = result["mean_free_path"]

    print("\n=== RESULTS ===")
    Sigma_s_total=  sum(nuc["Sigma_s"] for nuc in result ["nuclides"])
    Sigma_a_total = sum(nuc["Sigma_a"] for nuc in result ["nuclides"])
    if Sigma_tot > 0:
        P_scat = Sigma_s_total / Sigma_tot
        P_abs = Sigma_a_total / Sigma_tot
    else:
        P_scat = 0.0
        P_abs = 0.0
    print(f"Energy Regime: {regime}")
    print(f"Mean Free Path: {mfp:.4e} cm")
    print(f"Probability of scattering:{P_scat:.4f}")
    print(f"Probability of absorbing:{P_abs:.4f}")
    print("Nuclide Breakdown:")
    for nuc in result["nuclides"]:
        print(f"  - {nuc['name']}: percentage mass ={nuc['mass_fraction']*100:.1f}%", f"Sigma_i={nuc['Sigma_i']:.3e} cm^-1")

if __name__ == "__main__":
    main()