import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import neutron_packages.simulation as sim

nuclide_list = [
    {
        'name': 'U235',
        'mass_fraction': 0.007,
        'molar_mass': 235.0,
        'density': 19.1,
        'cross_sections': {
            'thermal': {'sigma_s': 10.0, 'sigma_c': 99.0, 'sigma_f': 583},  # barns
            'fast':    {'sigma_s': 4.0,  'sigma_c': 0.09, 'sigma_f': 1.0},  # barns
        }
    },
    {
        'name': 'U238',
        'mass_fraction': 0.993,
        'molar_mass': 238.0,
        'density': 19.1,
        'cross_sections': {
            'thermal': {'sigma_s': 9.0, 'sigma_c': 2.0, 'sigma_f': 0.00002},    # barns
            'fast':    {'sigma_s': 5.0, 'sigma_c': 0.07, 'sigma_f': 0.3},       # barns
        }
    },
    {
        'name': 'graphite',
        'mass_fraction': 0.03,
        'molar_mass': 12.0,
        'density': 2.267,
        'cross_sections': {
            'thermal': {'sigma_s': 5.0,   'sigma_c': 0.02},     # barns
            'fast':    {'sigma_s': 2.0,   'sigma_c': 0.00001},  # barns
        }
    }
]

def main():
    print("=== Mean Free Path Calculator ===")

    # 0) ask for manual input
    manual = input("Do you want to manually input nuclides? (y/n): ")
    if manual.lower().startswith('y'):
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

    else:
        # Hard-coded fallback mixture
        chosen_nuclides = [
            {
                'name': 'U235', 
                'molar_mass': 235.0, 
                'density': 19.1, 
                'cross_sections': {
                    'thermal': {'sigma_s': 10.0, 'sigma_a': 99.0, 'sigma_f': 583},
                    'fast':    {'sigma_s': 4.0,  'sigma_a': 0.09, 'sigma_f': 1.0}
                },
                'mass_fraction': 0.007
            }, 
            {
                'name': 'U238',
                'molar_mass': 238.0,
                'density': 19.1,
                'cross_sections': {
                    'thermal': {'sigma_s': 9.0, 'sigma_a': 2.0, 'sigma_f': 0.00002},    # barns
                    'fast':    {'sigma_s': 5.0, 'sigma_a': 0.07, 'sigma_f': 0.3},    
                    },
                    'mass_fraction': 0.993
            }
        ]
        regime = "thermal"

    # 4) Create the material data (Sigma_tot, etc.)
    material_name = "User-Defined Mixture"
    result = sim.create_material_data(
        material_name,
        density=1.0,  # dummy
        nuclide_list=chosen_nuclides,
        energy_regime=regime
    )

    # Retrieve totals from the result
    Sigma_tot      = result["Sigma_tot"]
    mfp            = result["mean_free_path"]
    Sigma_a_total  = result["Sigma_a_total"]  # from the updated create_material_data
    Sigma_f_total  = result["Sigma_f_total"]  # from the updated create_material_data

    # Probability of fission given absorption
    if Sigma_a_total > 0.0:
        p_fiss_in_abs = Sigma_f_total / Sigma_a_total

    else:
        print(p_fiss_in_abs)
        #p_fiss_in_abs = 0.0

    # Probability of scattering and absorption (as fraction of total interactions)
    Sigma_s_total = result["Sigma_s_total"]
    if Sigma_tot > 0:
        P_scat = Sigma_s_total / Sigma_tot
        P_abs  = Sigma_a_total / Sigma_tot
    else:
        P_scat = 0.0
        P_abs  = 0.0

    # Print summary
    print("\n=== RESULTS ===")
    print(f"Energy Regime: {regime}")
    print(f"Mean Free Path: {mfp:.4e} cm")
    print(f"Probability of scattering: {P_scat:.4f}")
    print(f"Probability of absorption: {P_abs:.4f}")
    print(f"Fission Probability: {p_fiss_in_abs:.4f}")

    print("Nuclide Breakdown:")
    for nuc in result["nuclides"]:
        frac_percent = nuc['mass_fraction'] * 100
        print(f"  - {nuc['name']}: fraction={frac_percent:.1f}%")

    # 5) Prompt user to run the first (single) generation
    ans = input("\nDo you want to run the first generation? (y/n): ")
    if not ans.lower().startswith('y'):
        print("No simulation requested. Exiting.")
        return

    # Cylinder geometry
    radius = 10.0
    height = 10.0

    # Ask how many neutrons for Gen 1
    num_neutrons_str = input("How many neutrons to simulate in Gen 1? ")
    try:
        initial_num_neutrons = int(num_neutrons_str)
    except ValueError:
        initial_num_neutrons = 1000

    # We'll keep average_nu hardcoded at 2.3 for demonstration
    average_nu = 2.3
    sim.run_multiple_generations(
    radius=radius,
    height=height,
    initial_num_neutrons=initial_num_neutrons,
    P_scat=P_scat,
    P_abs=P_abs,
    p_fiss_in_abs=p_fiss_in_abs,
    average_nu=average_nu,
    mean_free_path=mfp
)

    
if __name__ == "__main__":
    main()
