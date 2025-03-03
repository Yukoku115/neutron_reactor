import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import neutron_packages.simulation as sim

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
    result = sim.create_material_data(
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
            num_neutrons = 1000 

        # Cylinder geometry
        radius = 20.0
        height = 20.0

        # Run the random walk. 
        # NOTE: your function should return a list of (trajectory, status) pairs
        results, absorbed_tick = sim.simulate_neutron_transport(
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
        sim.plot_trajectories_3d(results, radius, height, num_to_plot=(num_neutrons if num_neutrons < 1000 else 1000))

        # Plot a bar plot of the absorbed ticks
        plt.figure()
        plt.hist(absorbed_tick, bins=range(min(absorbed_tick), max(absorbed_tick) + 1), edgecolor='black')
        plt.xlabel('Tick')
        plt.ylabel('Count')
        plt.title('Distribution of Absorbed Ticks')
        plt.show()
        


if __name__ == "__main__":
    main()

