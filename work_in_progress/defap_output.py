import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
plt.close("all")

def write_output_file(input_data, seedname):

    defects_data = input_data.defects_data

    with open(f"{seedname}.output", "w") as f:

        f.write("Defect Analysis Pacakge")
        f.write(", v: 4.0\n")

        f.write(f"\nDefAP executed on: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

        f.write(f">>> Tasks:\n\n")

        f.write(f"    Number of tasks : {len(input_data.tasks)}\n")
        for task_num, task in enumerate(input_data.tasks, start=1):
            f.write(f"    Task {task_num} : {task}\n")

        f.write("\n\n")

        f.write(">>> Host Material:\n\n")

        f.write(f"    {'{:22s}'.format('Host')} :  {input_data.host['formula']}\n")
        f.write(f"    {'{:22s}'.format('Host energy pfu')} : {input_data.host_energy_pfu}\n")
        f.write(f"    {'{:22s}'.format('Host energy supercell')} : {input_data.host_energy_supercell}\n")

        f.write("\n\n")

        f.write(">>> Electronic Properties:\n\n")

        f.write(f"    {'{:15s}'.format('Bandgap')} : {input_data.bandgap}\n")
        f.write(f"    {'{:15s}'.format('Energy of VBM')} : {input_data.e_vbm}\n")

        f.write("\n")

        if input_data.electron_method == "off":
            f.write("    Not calculating concentration of electrons\n")

        elif input_data.electron_method == "boltzmann":
            f.write("    Calculating concentration of electrons with Boltzmann statistics\n")

        elif input_data.electron_method == "fermi-dirac":
            f.write("    Calculating concentration of electrons with Fermi-Dirac statistics\n")
            f.write(f"    Conduction band integral limits           : {input_data.conductionband_limits[0]} - {input_data.conductionband_limits[1]}\n")
            f.write(f"    Number of functional units per unit cell  : {input_data.fu_unit_cell}\n")

        elif input_data.electron_method == "fixed":
            f.write(f"    Fixed electron concentration of : {input_data.electron_fixed_conc}\n")

        elif input_data.electron_method == "effective_masses":
            f.write(f"    Calculating electron concentration from effective masses\n")
            f.write(f"    Electron effective masses : {input_data.electron_effective_masses}\n")
            f.write(f"    Number of functional units per unit cell  : {input_data.fu_unit_cell}\n")
            f.write(f"    Volume of unit cell: {input_data.volume_unit_cell}\n")

        f.write("\n")

        if input_data.hole_method == "off":
            f.write("    Not calculating concentration of holes\n")

        elif input_data.hole_method == "boltzmann":
            f.write("    Calculating concentration of holes with Boltzmann statistics\n")

        elif input_data.hole_method == "fermi-dirac":
            f.write("    Calculating concentration of holes with Fermi-Dirac statistics\n")
            f.write(f"    Valence band integral limits              : {input_data.valenceband_limits[0]} - {input_data.valenceband_limits[1]}\n")
            f.write(f"    Number of functional units per unit cell  : {input_data.fu_unit_cell}\n")

        elif input_data.hole_method == "fixed":
            f.write(f"    Fixed hole concentration of : {input_data.hole_fixed_conc}\n")

        elif input_data.hole_method == "effective_masses":
            f.write(f"    Calculating hole concentration from effective masses\n")
            f.write(f"    Hole effective masses : {input_data.hole_effective_masses}\n")
            f.write(f"    Number of functional units per unit cell  : {input_data.fu_unit_cell}\n")
            f.write(f"    Volume of unit cell: {input_data.volume_unit_cell}\n")

        f.write("\n\n")

        f.write(">>> Calculation of Chemical Potentials:\n\n")

        if input_data.chem_pot_method == "defined":
            f.write("    Chemical potentials are defined\n")

        elif input_data.chem_pot_method == "rich-poor":
            f.write("    Rich-poor chemical potential method selected\n")

        elif input_data.chem_pot_method == "volatile":
            f.write("    Volatile chemical potential method selected\n")

            if input_data.real_gas == 0:
                f.write("    Assuming ideal gas relations\n")

            elif input_data.real_gas == 1 or input_data.real_gas == 2:
                f.write("    Calculating real gas relations using Shomate Equations\n")

            elif input_data.real_gas == 3:
                f.write("    Calculating real gas relations using PYroMat library (NASA equations)\n")

            f.write("\n")
            f.write(f"    +{25*'-'}+{25*'-'}+\n")
            f.write("    |{:^25s}|{:^25s}|\n".format("Volatile", "Partial Pressure"))
            f.write(f"    +{25*'-'}+{25*'-'}+\n")
            f.write("    |  {:23s}|  {:23s}|\n".format(input_data.constituents["volatile"]["volatile_element"],
                                                       str(input_data.constituents["volatile"]["log_pp"])))
            f.write(f"    +{25 * '-'}+{25 * '-'}+\n")

            f.write("\n")

            f.write(f"    +{25*'-'}+{25*'-'}+{25*'-'}+{25*'-'}+\n")
            f.write("    |{:^25s}|{:^25s}|{:^25s}|{:^25s}|\n".format("Constituent", "Compound Energy", "Metal Energy", "Std Formation Energy"))
            f.write(f"    +{25 * '-'}+{25 * '-'}+{25 * '-'}+{25 * '-'}+\n")
            f.write("    |  {:23s}|  {:23s}|  {:23s}|  {:23s}|\n".format(input_data.constituents["compound"]["formula"],
                                                                         str(input_data.constituents["compound"]["compound_energy_pfu"]),
                                                                         str(input_data.constituents["compound"]["metal_energy_pfu"]),
                                                                         str(input_data.constituents["compound"]["std_formation_energy"])))
            f.write(f"    +{25 * '-'}+{25 * '-'}+{25 * '-'}+{25 * '-'}+\n")



        elif input_data.chem_pot_method == "volatile-reference":
            f.write("    Volatile chemical potential calculated using a reference material\n")

        elif input_data.chem_pot_method == "volatile-rich-poor":
            f.write("    Volatile-rich-poor chemical potential method selected\n")


        f.write("\n\n")

        f.write(">>> Dopants\n\n")

        dopants_fitting = 0
        if input_data.dopants:
            dopant_num = 1

            for dopant_element, dopant_vals in input_data.dopants.items():
                if dopant_vals["fitting_option"] == 1 or dopant_vals["fitting_option"] == 2:
                    f.write(f"    Dopant {dopant_num} :\n")
                    f.write(f"    +{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+\n")
                    f.write("    |{:^28s}|{:^28s}|{:^28s}|{:^28s}|{:^28s}|{:^28s}|\n".format("Dopant", "Reference", "Chemical Potential", "Fitting Option", "Target Concentration pfu", "Chemical Potential Range"))
                    f.write(f"    +{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+\n")
                    f.write("    |  {:26s}|  {:26s}|  {:26s}|  {:26s}|  {:26s}|  {:26s}|\n".format(dopant_element,
                                                                                                   dopant_vals["reference"],
                                                                                                   str(dopant_vals["chemical_potential"]),
                                                                                                   str(dopant_vals["fitting_option"]),
                                                                                                   str(dopant_vals["concentration_pfu"]),
                                                                                                   str(dopant_vals["chem_pot_range"])))
                    f.write(f"    +{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+\n")

                    dopant_num += 1
                    dopants_fitting += 1

                elif dopant_vals["fitting_option"] == 3 or dopant_vals["fitting_option"] == 4:
                    f.write(f"    Dopant {dopant_num} :\n")
                    f.write(f"    +{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+\n")
                    f.write("    |{:^28s}|{:^28s}|{:^28s}|{:^28s}|{:^28s}|\n".format("Dopant", "Reference",
                                                                                     "Chemical Potential",
                                                                                     "Fitting Option",
                                                                                     "log_{10}(P) /atm"))

                    f.write(f"    +{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+\n")
                    f.write("    |  {:26s}|  {:26s}|  {:26s}|  {:26s}|  {:26s}|\n".format(dopant_element,dopant_vals["reference"],
                                                                                                   str(dopant_vals["chemical_potential"]),
                                                                                                   str(dopant_vals["fitting_option"]),
                                                                                                   str(dopant_vals["log_PP_atm"])))
                    f.write(f"    +{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+{28 * '-'}+\n")

                    dopant_num += 1

                f.write("\n")

            f.write(f"    Fitting chemical potential of {dopants_fitting} dopants\n")

        if dopants_fitting == 0:
            f.write("    No dopant chemical potentials being fitted\n")
        elif dopants_fitting == 1:
            f.write("    Using Linear bisection\n")
        else:
            f.write("    Using SLSQP\n")

        if input_data.art_dopant_conc:
            f.write(f"    Artificial dopant concentration  : {input_data.art_dopant_conc}\n")

        if input_data.art_dopant_chg:
            f.write(f"    Artificial dopant charge         : {input_data.art_dopant_chg}\n")

        f.write("\n\n")

        f.write(">>> Vibrational Entropy\n\n")

        if input_data.entropy_data:
            f.write("    Adding entropy contributions to defect formation energies\n")
            f.write(f"    Number of functional units in the supercell for the entropy calculation : {input_data.entropy_units}\n")

        else:
            f.write("    No vibrational entropy contributions\n")

        f.write("\n\n")

        if input_data.gibbs_data:
            f.write(">>> Gibbs Temperature Dependant Energies\n\n")
            f.write("    Using supplied Gibbs temperature dependent energies\n")
            f.write("\n\n")

        f.write(">>> Defect Concentrations\n\n")

        f.write(f"    Defect concentration method : {input_data.defect_conc_method.capitalize()}\n")

        if input_data.conc_units == 0:
            f.write(f"    Defect concentration units  : per formula unit\n")
        elif input_data.conc_units == 1:
            f.write(f"    Defect concentration units  : per cm3\n")

        f.write("\n\n")

        f.write(">>> Defect Energy Corrections\n\n")

        if input_data.coulombic_correction == 0 and not input_data.tab_correction:
            f.write("    No energy corrections being applies\n")

        elif input_data.coulombic_correction == 1:
            f.write("    Adding simple point charge for a cubic system\n")
            f.write(f"    Supercell length     : {input_data.length}\n")
            f.write(f"    Dielectric constant  : {input_data.dielectric_constant}\n")
            f.write("    Madelung constant    : 2.8373\n")

        elif input_data.coulombic_correction == 2:
            f.write("    Adding anisotropic point charge\n")

            if "madelung" not in input_data.tasks:
                f.write(f"    Screened Madelung potential : {input_data.screened_madelung}\n")
            else:
                f.write("    Screened Madelung to be calculated\n")

        if input_data.tab_correction:
            f.write("    Adding tabulated corrections from defects file\n")

        f.write("\n\n")

        f.write(">>> Summary of defects:\n\n")
        f.write(f"    +{20*'-'}+{20*'-'}+{20*'-'}+{20*'-'}+{20*'-'}+{20*'-'}+{20*'-'}+\n")
        f.write("    |{:^20s}|{:^20s}|{:^20s}|{:^20s}|{:^20s}|{:^20s}|{:^20s}|\n".format("Defect", "Group", "Multiplicity",
                                                                                     "Site", "Charge", "Energy",
                                                                                     "Correction"))

        f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")

        for defect_name, defect_vals in defects_data.items():
            group, multiplicity, site, charge, energy, correction, added_removed = defect_vals.values()

            if charge >= 0:
                charge = f" {charge}"

            if correction >= 0:
                correction = f" {correction}"

            f.write("    |  {:18s}|  {:18s}|  {:18s}|  {:18s}|  {:18s}|  {:18s}|  {:18s}|\n".format(defect_name, group,
                                                                                         str(multiplicity), str(site),
                                                                                         str(charge), str(energy),
                                                                                         str(correction)))
        f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n\n")

        f.write(f"    Number of defects : {len(defects_data)}\n")

        f.write("\n\n")


def write_defect_phases_output(f, concentrations, formation_energies, chemical_pots, fermi_level,
                               secondary_phases, loop_type_a, loop_type_b, loop_a_val, loop_b_val,
                               progress_a, progress_b, total_a, total_b):

    elements = chemical_pots.keys()

    if secondary_phases:
        phases_names = secondary_phases.keys()

    f.write(f"    Loop step a: {progress_a} of {total_a} \n")
    f.write(f"    Loop step b: {progress_b} of {total_b} \n\n")

    if loop_type_a == 0:
        f.write(f"    Volatile partial pressure : 10^( {loop_a_val:10.8f} ) atm \n")

    elif loop_type_a == 1:
        f.write(f"    Temperature : {loop_a_val:10.8f} K \n")

    elif loop_type_a == 2:
        f.write(f"    Dopant Concentration : 10^( {loop_a_val:10.8f} ) \n")

    elif loop_type_a == 3:
        f.write(f"    Dopant Partial Pressure : 10^( {loop_a_val:10.8f} ) atm \n")

    if loop_type_b == 0:
        f.write(f"    Volatile partial pressure : 10^( {loop_b_val:10.8f} ) atm \n\n")

    elif loop_type_b == 1:
        f.write(f"    Temperature : {loop_b_val:10.2f} K \n\n")

    elif loop_type_b == 2:
        f.write(f"    Dopant Concentration : 10^( {loop_b_val:10.8f} ) \n\n")

    elif loop_type_b == 3:
        f.write(f"    Dopant Partial Pressure : 10^( {loop_b_val:10.8f} ) atm \n\n")

    f.write("    Calculated chemical potentials: \n\n")

    for element in elements:
        f.write(f"      {'{:2s} :'.format(element)} {'{:12.8f}'.format(chemical_pots[element])}\n")

    f.write("\n")

    f.write(f"    Fermi level : {fermi_level:10.8f}\n\n")

    f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
    f.write("    |{:^20s}|{:^20s}|{:^20s}|\n".format("Defect", "Formation Energy", "log_{10}[D]", ))
    f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
    for defect_name in concentrations.keys():

        if defect_name != "electrons" and defect_name != "holes" and defect_name != "log_stoic":
            f.write("    |  {:18s}|  {:<18.8f}|  {:<18.8f}|\n".format(defect_name,
                                                                      formation_energies[defect_name],
                                                                      concentrations[defect_name]
                                                                      )
                    )
        else:
            f.write("    |  {:18s}|  {:<18s}|  {:<18.8f}|\n".format(defect_name, '-',
                                                                    concentrations[defect_name]
                                                                    )
                    )

    f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n\n")

    if secondary_phases:
        f.write("    Secondary phase stability check:\n")

        f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
        f.write("    |{:^20s}|{:^20s}|{:^20s}|{:^20s}|{:^20s}|\n".format("Phase", "Energy", "Chem Pot Sum",
                                                                         "Difference", "Stable?"))
        f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")

        for phase in secondary_phases:


            phase_eng = secondary_phases[phase]["energy"]
            phase_chem_pot = secondary_phases[phase]["chem_pot_sum"]
            phase_difference = secondary_phases[phase]["difference"]

            phase_stability = secondary_phases[phase]["is_stable"]
            if phase_stability:
                phase_stability = "True"
            else:
                phase_stability = "False"

            f.write("    |  {:<18s}|  {:<18.8f}|  {:<18.8f}|  {:<18.8f}|  {:<18s}|\n".format(phase,
                                                                                             phase_eng,
                                                                                             phase_chem_pot,
                                                                                             phase_difference,
                                                                                             phase_stability
                                                                                             )
                    )

        f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n\n")

    f.write(f"{110 * '-'}\n\n")



def write_brouwer_output(f, concentrations, formation_energies, chemical_pots, fermi_level, secondary_phases, loop_type, loop_val, loop_prog, total_step, grouped_defects):


    elements = chemical_pots.keys()

    f.write(f"    Loop step : {loop_prog} of {total_step}\n\n")

    if loop_type == 0:
        f.write(f"    Volatile partial pressure : 10^( {loop_val:10.8f} ) atm\n\n")

    f.write("    Calculated chemical potentials:\n\n")

    for element in elements:
        f.write(f"      {'{:2s} :'.format(element)} {'{:12.8f}'.format(chemical_pots[element])}\n")

    f.write("\n")

    f.write(f"    Fermi level : {fermi_level:10.8f}\n\n")

    f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
    f.write("    |{:^20s}|{:^20s}|{:^20s}|\n".format("Defect", "Formation Energy", "log_{10}[D]",))
    f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
    for defect_name in concentrations.keys():

        if defect_name != "electrons" and defect_name != "holes" and defect_name != "pm_stoic" and defect_name != "log_stoic" and not grouped_defects:
            f.write("    |  {:18s}|  {:<18.8f}|  {:<18.8f}|\n".format(defect_name, formation_energies[defect_name], concentrations[defect_name]))
        else:
            f.write("    |  {:18s}|  {:<18s}|  {:<18.8f}|\n".format(defect_name, '-', concentrations[defect_name]))

    f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n\n")

    if secondary_phases:
        f.write("    Secondary phase stability check:\n")

        f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
        f.write("    |{:^20s}|{:^20s}|{:^20s}|{:^20s}|{:^20s}|\n".format("Phase", "Energy", "Chem Pot Sum", "Difference", "Stable?"))
        f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")

        for phase in secondary_phases:

            phase_eng = secondary_phases[phase]["energy"]
            phase_chem_pot = secondary_phases[phase]["chem_pot_sum"]
            phase_difference = secondary_phases[phase]["difference"]

            phase_stability = secondary_phases[phase]["is_stable"]
            if phase_stability:
                phase_stability = "True"
            else:
                phase_stability = "False"

            f.write("    |  {:<18s}|  {:<18.8f}|  {:<18.8f}|  {:<18.8f}|  {:<18s}|\n".format(phase, phase_eng, phase_chem_pot, phase_difference, phase_stability))

        f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n\n")

    f.write(f"{110*'-'}\n\n")


def defect_phases_plot(min_val_x, max_val_x, min_val_y, max_val_y, phase_type, loop_a_type, loop_b_type, looping_dopants, art_dopant_chg):

    phases_dataframe = pd.read_csv(f"defect_phases_{phase_type}_data.csv")
    legend_labels = []

    colours = ["forestgreen", "indianred", "#008cf9", "#d163e6", "#00bbad", "#ff9287", "peru", "#878500", "#00c6f8",
               "#00a76c", "#bdbdbd", "#5954d6", "#ebac23", "silver"]

    text_colours = ["#274e13", "darkred", "b", "purple", "#008080", "maroon", "#b24502", "#004225", "steelblue",
                    "darkgreen", "dimgrey", "darkslateblue", "darkgoldenrod", "grey"]

    looping_num = 0

    with open(f"plot_defect_phases_{phase_type}.py", "w") as f:

        f.write('import pandas as pd\n')
        f.write('import numpy as np\n')
        f.write('import matplotlib.pyplot as plt\n')
        f.write('from matplotlib.patches import Polygon\n')
        f.write('plt.close("all")\n\n')

        f.write('plt.rcParams["axes.facecolor"] = "black"\n\n')

        f.write("\nplt.rc('font', size=14)  # controls default text sizes\n")
        f.write("plt.rc('axes', titlesize=14)  # fontsize of the axes title\n")
        f.write("plt.rc('axes', labelsize=14)  # fontsize of the x and y labels\n")
        f.write("plt.rc('xtick', labelsize=12)  # fontsize of the tick labels\n")
        f.write("plt.rc('ytick', labelsize=12)  # fontsize of the tick labels\n")
        f.write("plt.rc('legend', fontsize=12)  # legend fontsize\n\n")

        f.write(f'phases_dataframe = pd.read_csv("defect_phases_{phase_type}_data.csv")\n\n')

        f.write('fig, ax = plt.subplots(1, 1, figsize=(12,7))\n')
        f.write(f"plt.axis({min_val_x, max_val_x, min_val_y, max_val_y})\n")

        if loop_a_type == 0:
            f.write(r"plt.xlabel(r'$\mathrm{log}_{10}\mathrm{P}_{\mathrm{O}_{2}}$ /atm')"+"\n")
        elif loop_a_type == 1:
            f.write("plt.xlabel('Temperature /K')\n")
        elif loop_a_type == 2:
            f.write(f"plt.xlabel('[{looping_dopants[0]}] /pfu')\n")
            looping_num+=1
        elif loop_a_type == 3:
            string = r"plt.xlabel(r'$\mathrm{log}_{10}\mathrm{P}_{\mathrm{" + f"{looping_dopants[0]}" + r"}_{2}}$ /atm')"+"\n"
            f.write(string)
            looping_num+=1
        elif loop_a_type == 4:
            string = r"plt.xlabel(r'$\mu_\mathrm{" + f"{looping_dopants[0]}" +"}$ /eV')" +"\n"
            f.write(string)
        elif loop_a_type == 5:
            string = r"plt.xlabel(r'[$\lambda^{" + f"{art_dopant_chg}" + r"}$] /pfu')" +"\n"
            f.write(string)
        elif loop_a_type == 6:
            string = r"plt.xlabel(r'$s$')" +"\n"
            f.write(string)

        if loop_b_type == 0:
            f.write(r"plt.ylabel(r'$\mathrm{log}_{10}\mathrm{P}_{\mathrm{O}_{2}}$ /atm')" +"\n")
        elif loop_b_type == 1:
            f.write("plt.ylabel('Temperature /K')\n")
        elif loop_b_type == 2:
            f.write(f"plt.ylabel('[{looping_dopants[1]}] /pfu')\n")
        elif loop_b_type == 3:
            string = r"plt.ylabel(r'$\mathrm{log}_{10}\mathrm{P}_{\mathrm{" + f"{looping_dopants[1]}" + r"}_{2}}$ /atm')" +"\n"
            f.write(string)
        elif loop_b_type == 4:
            string = r"plt.ylabel(r'$\mu_\mathrm{" + f"{looping_dopants[1]}" +"}$ /eV')" +"\n"
            f.write(string)
        elif loop_b_type == 5:
            string = r"plt.ylabel(r'[$\lambda^{" + f"{art_dopant_chg}" + r"}$] /pfu')" +"\n"
            f.write(string)
        elif loop_b_type == 6:
            string = r"plt.ylabel(r'$s$')" +"\n"
            f.write(string)

        f.write("\n")

        colour_index = 0

        for phase_index, phase in enumerate(phases_dataframe):
            if phase == "index":
                continue

            else:

                phase_coords_lst = np.array([np.fromstring(val[1:-1], dtype=float, count=2, sep=' ') for val in phases_dataframe[phase] if 'nan' not in val])

                f.write(f"phase_{phase_index}_coords_lst = np.array([np.fromstring(val[1:-1], dtype=float, count=2, sep=' ') for val in phases_dataframe['{phase}'] if 'nan' not in val])\n")
                f.write(f"phase_{phase_index}_polygon = Polygon(phase_{phase_index}_coords_lst, color='{colours[colour_index]}', ec='black')\n")
                f.write(f"ax.add_patch(phase_{phase_index}_polygon)\n")


                cx = phase_coords_lst[:,0].mean(0)
                cy = phase_coords_lst[:, 1].mean(0)

                f.write(f"plt.text(x={cx}, y={cy}, s='{phase}', ha='center', va='center', c='{text_colours[colour_index]}')\n")

                f.write("\n\n")

                if colour_index < (len(colours)-1):
                    colour_index+=1
                else:
                    colour_index = len(colours)-1

                legend_labels.append(phase)

        f.write("\n")
        for location in ['left', 'right', 'top', 'bottom']:
            f.write(f'ax.spines["{location}"].set_linewidth(2)\n')
        f.write('ax.tick_params(width=2, pad=15, direction="in", length=10)\n')

        f.write('plt.tight_layout()\n')
        f.write('plt.show()\n')


def brouwer_plot(axes, defects_data, grouped_defects, loop_type, looping_dopants, art_dopant_chg, x_variable):

    concentrations_dataframe = pd.read_csv("brouwer_data.csv")
    legend_labels = []

    if x_variable == 2:
        x_axis_var = "pm_stoic"
    else:
        x_axis_var = "loop_step"

    with open("plot_brouwer.py", "w") as f:

        colours = ["#006e00", "#b80058", "#008cf9", "#d163e6", "#00bbad", "#ff9287", "#b24502", "#878500", "#00c6f8",
                   "#00a76c", "#bdbdbd", "#000000"]

        linestyles = [(0, ()), (0, (5, 2)), (0, (5, 2, 1, 2)), (0, (1, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1))]

        group_colours = {}

        f.write('import pandas as pd\n')
        f.write('import matplotlib.pyplot as plt\n')
        f.write('plt.close("all")\n\n')
        f.write('concentrations_dataframe = pd.read_csv("brouwer_data.csv")\n\n')

        f.write("\nplt.rc('font', size=20)  # controls default text sizes\n")
        f.write("plt.rc('axes', titlesize=20)  # fontsize of the axes title\n")
        f.write("plt.rc('axes', labelsize=20)  # fontsize of the x and y labels\n")
        f.write("plt.rc('xtick', labelsize=16)  # fontsize of the tick labels\n")
        f.write("plt.rc('ytick', labelsize=16)  # fontsize of the tick labels\n")
        f.write("plt.rc('legend', fontsize=10)  # legend fontsize\n")
        f.write("plt.rc('lines', linewidth=2)\n")
        f.write("plt.rcParams['figure.figsize'] = (12,8)\n\n")

        f.write("fig, ax = plt.subplots()\n")
        f.write(f"plt.axis({axes})\n")

        if x_variable == 2:
            f.write(r"plt.xlabel(r'$\pm x$')" + "\n")
        elif loop_type == 0:
            f.write(r"plt.xlabel(r'$\mathrm{log}_{10}\mathrm{P}_{\mathrm{O}_{2}}$ /atm')"+"\n")
        elif loop_type == 1:
            f.write("plt.xlabel('Temperature /K')\n")
        elif loop_type == 2:
            f.write(f"plt.xlabel('[{looping_dopants[0]}] /pfu')\n")
        elif loop_type == 3:
            string = r"plt.xlabel(r'$\mathrm{log}_{10}\mathrm{P}_{\mathrm{" + f"{looping_dopants[0]}" + r"}_{2}}$ /atm')" +"\n"
            f.write(string)
        elif loop_type == 4:
            string = r"plt.xlabel(r'$\mu_\mathrm{" + f"{looping_dopants[0]}" +"}$ /eV')" +"\n"
            f.write(string)
        elif loop_type == 5:
            string = r"plt.xlabel(r'[$\lambda^{" + f"{art_dopant_chg}" + r"}$] /pfu')" +"\n"
            f.write(string)
        elif loop_type == 6:
            string = r"plt.xlabel(r'$s$')" +"\n"
            f.write(string)

        f.write("plt.ylabel('[D] /pfu')\n\n")

        colour_index = 0
        line_index = 0
        for defect in concentrations_dataframe:

            if defect == "loop_step":
                continue

            elif defect != "electrons" and defect != "holes" and defect != "pm_stoic" and defect != "log_stoic":

                if max(concentrations_dataframe[defect]) > axes[2]:
                    if not grouped_defects:
                        defect_group = defects_data[defect]["group"]
                    else:
                        defect_group = defect

                    if defect_group not in group_colours:
                        #print(colour_index)

                        if colour_index < len(colours):
                            group_colours[defect_group] = colours[colour_index]
                            colour_index+=1
                        else:
                            group_colours[defect_group] = colours[-1]

                        #colour_index += 1 if colour_index < len(colours) - 1 else len(colours) - 1

                        line_index = 0
                    else:
                        line_index += 1 if line_index < 7 else 0

                    f.write(f'plt.plot(concentrations_dataframe["{x_axis_var}"], concentrations_dataframe["{defect}"], "{group_colours[defect_group]}", linestyle={linestyles[line_index]}, label="{defect}")\n')
                    legend_labels.append(defect)

            elif defect == "electrons" and max(concentrations_dataframe[defect]) > axes[2]:
                f.write(f'plt.plot(concentrations_dataframe["{x_axis_var}"], concentrations_dataframe["{defect}"], "#5954d6", linestyle="solid", label="electrons")\n')
                legend_labels.append(defect)

            elif defect == "holes" and max(concentrations_dataframe[defect]) > axes[2]:
                f.write(
                    f'plt.plot(concentrations_dataframe["{x_axis_var}"], concentrations_dataframe["{defect}"], "#ebac23", linestyle="solid", label="holes")\n')
                legend_labels.append(defect)

            elif defect == "log_stoic" and x_variable != 2 and max(concentrations_dataframe[defect]) > axes[2]:
                f.write(f'plt.plot(concentrations_dataframe["{x_axis_var}"], concentrations_dataframe["{defect}"], "#000000", linestyle=(0, (6, 4)), linewidth=3, label="stoic")\n')
                legend_labels.append(defect)


        f.write("\n")
        for location in ['left', 'right', 'top', 'bottom']:
            f.write(f'ax.spines["{location}"].set_linewidth(2)\n')
        f.write('ax.tick_params(width=2, pad=15, direction="in", length=10)\n')

        f.write(f'plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol={int(np.round(len(legend_labels)/3))}, edgecolor="white")\n')
        f.write('plt.subplots_adjust(bottom=0.2)\n')
        f.write('plt.tight_layout()\n')
        f.write('plt.show()\n')


def plot_formation_energies(defects_data, bandgap):

    form_eng_dataframe = pd.read_csv("all_form_eng_data.csv")
    min_form_eng_dataframe = pd.read_csv("all_min_form_eng_data.csv")

    colours = ["#006e00", "#b80058", "#008cf9", "#d163e6", "#00bbad", "#ff9287", "#b24502", "#878500", "#00c6f8",
               "#00a76c", "#bdbdbd", "#5954d6", "#ebac23", "#000000"]

    linestyles = [(0, ()), (0, (5, 2)), (0, (5, 2, 1, 2)), (0, (1, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (5, 1)),
                  (0, (1, 1)), (0, (3, 1, 1, 1))]

    group_colours = {}
    for index, group_name in enumerate(min_form_eng_dataframe):
        if group_name == "fermi_energy":
            continue

        if index < len(colours):
            group_colours[group_name] = colours[index-1]
        else:
            group_colours[group_name] = colours[-1]


    with open("plot_formation_energies.py", "w") as f:
        f.write('import pandas as pd\n')
        f.write('import matplotlib.pyplot as plt\n')
        f.write('plt.close("all")\n\n')

        f.write('form_eng_dataframe = pd.read_csv("all_form_eng_data.csv")\n')
        f.write('min_form_eng_dataframe = pd.read_csv("all_min_form_eng_data.csv")\n\n')

        # plots all defect formation energies on one plot
        f.write("# Plot all defect formation energies\n")
        f.write("plt.figure(figsize=(12, 7))\n")
        f.write("plt.xlabel('Fermi Level /eV')\n")
        f.write("plt.ylabel('Formation Energy /eV')\n")
        f.write(f"plt.xlim(0, {bandgap})\n\n")

        group_line_index = {}

        for defect in form_eng_dataframe:
            if defect == "fermi_energy":
                continue

            defect_group = defects_data[defect]["group"]
            if defect_group not in group_line_index:
                group_line_index[defect_group] = 0


            f.write(f'plt.plot(form_eng_dataframe["fermi_energy"], form_eng_dataframe["{defect}"], "{group_colours[defect_group]}", linestyle={linestyles[group_line_index[defect_group]]}, label="{defect}")\n')

            if group_line_index[defect_group] < len(linestyles) - 1:
                group_line_index[defect_group] += 1
            else:
                group_line_index[defect_group] = 0

        f.write('\nplt.tight_layout()\n')
        f.write(f'plt.legend(loc="lower center", ncol={math.ceil(len(group_colours) / 4)}, fancybox=True)\n')
        f.write('plt.savefig("all_form_eng_plot.png", dpi=200)\n')
        f.write('plt.close()\n\n\n')

        # Plots all defect formation energies in a group
        # loop through defect groups. Min form eng dataframe is grouped by these, so loop through them
        for defect_group in min_form_eng_dataframe:

            if defect_group == "fermi_energy":
                continue

            f.write(f"# Plot {defect_group} defect formation energies\n")

            f.write("plt.figure(figsize=(12, 7))\n")
            f.write("plt.xlabel('Fermi Level /eV')\n")
            f.write("plt.ylabel('Formation Energy /eV')\n")
            f.write(f"plt.xlim(0, {bandgap})\n\n")

            # loop through defects to see which defects are in the current group iteration
            line_index = 0
            for defect in form_eng_dataframe:

                if defect == "fermi_energy":
                    continue

                if defects_data[defect]["group"] == defect_group:
                    f.write(
                        f'plt.plot(form_eng_dataframe["fermi_energy"], form_eng_dataframe["{defect}"], "{group_colours[defect_group]}", linestyle={linestyles[line_index]}, label="{defect}")\n')

                    if line_index < len(linestyles) - 1:
                        line_index += 1
                    else:
                        line_index = 0

            f.write('\nplt.tight_layout()\n')
            f.write(f'plt.legend(loc="lower center", ncol={math.ceil(len(group_colours) / 2)}, fancybox=True)\n')
            f.write(f'plt.savefig("{defect_group}_form_eng_plot.png", dpi=200)\n')
            f.write('plt.close()\n\n\n')

    with open("plot_min_formation_energies.py", "w") as f:
        f.write('import pandas as pd\n')
        f.write('import matplotlib.pyplot as plt\n')
        f.write('plt.close("all")\n\n')

        f.write('form_eng_dataframe = pd.read_csv("all_form_eng_data.csv")\n')
        f.write('min_form_eng_dataframe = pd.read_csv("all_min_form_eng_data.csv")\n\n')

        # Plots all the minimum defect formation energies
        f.write("# Plot all minimum defect formation energies\n")
        f.write("plt.figure(figsize=(12, 7))\n")
        f.write("plt.xlabel('Fermi Level /eV')\n")
        f.write("plt.ylabel('Formation Energy /eV')\n")
        f.write(f"plt.xlim(0, {bandgap})\n\n")

        for defect_group in min_form_eng_dataframe:
            if defect_group == "fermi_energy":
                continue

            f.write(f'plt.plot(min_form_eng_dataframe["fermi_energy"], min_form_eng_dataframe["{defect_group}"], "{group_colours[defect_group]}", linestyle="solid", label="{defect_group}")\n')

        f.write('\nplt.tight_layout()\n')
        f.write(f'plt.legend(loc="lower center", ncol={math.ceil(len(group_colours)/2)}, fancybox=True)\n')
        f.write('plt.savefig("all_min_form_eng_plot.png", dpi=200)\n')
        f.write('plt.close()\n\n\n')

        # plot all min formation energies for each defect group
        for defect_group in min_form_eng_dataframe:
            if defect_group == "fermi_energy":
                continue

            # Plots individual minimum defect formation energies
            f.write(f"# Plot {defect_group} minimum defect formation energies\n")
            f.write("plt.figure(figsize=(12, 7))\n")
            f.write("plt.xlabel('Fermi Level /eV')\n")
            f.write("plt.ylabel('Formation Energy /eV')\n")
            f.write(f"plt.xlim(0, {bandgap})\n\n")

            f.write(f'plt.plot(min_form_eng_dataframe["fermi_energy"], min_form_eng_dataframe["{defect_group}"], "{group_colours[defect_group]}", linestyle="solid", label="{defect_group}")\n')

            f.write('\nplt.tight_layout()\n')
            f.write(f'plt.legend(loc="lower center", ncol={math.ceil(len(group_colours) / 2)}, fancybox=True)\n')
            f.write(f'plt.savefig("{defect_group}_min_form_eng_plot.png", dpi=200)\n')
            f.write('plt.close()\n\n\n')

