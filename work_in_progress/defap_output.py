from subprocess import call
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import ast
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


def write_defect_phases_output(conc_dataframe, form_eng_dataframe, chem_pot_dataframe, fermi_level_dataframe,
                               secondary_phases_dataframe, loop_type_a, loop_type_b, loop_a_vals, loop_b_vals,
                               seedname):

    with open(f"{seedname}.output", "a") as f:
        f.write(">>> Task : Defect Phases\n\n")

        conc_defect_names = conc_dataframe.columns
        elements = chem_pot_dataframe.columns


        if not secondary_phases_dataframe.empty:
            phases_names = secondary_phases_dataframe.columns

        for index_a, loop_step_a in enumerate(loop_a_vals, start=1):
            for index_b, loop_step_b in enumerate(loop_b_vals, start=1):

                total_index = (index_a-1)*(len(loop_b_vals)) + (index_b-1)

                f.write(f"    Loop step : {index_a} of {len(loop_a_vals)} \n")
                f.write(f"    Loop step : {index_b} of {len(loop_b_vals)} \n\n")

                if loop_type_a == 0:
                    f.write(f"    Volatile partial pressure : 10^( {loop_step_a:10.8f} ) atm \n")

                elif loop_type_a == 1:
                    f.write(f"    Temperature : {loop_step_a:10.8f} K \n")

                elif loop_type_a == 2:
                    f.write(f"    Dopant Concentration : 10^( {loop_step_a:10.8f} ) \n")

                elif loop_type_a == 3:
                    f.write(f"    Dopant Partial Pressure : 10^( {loop_step_a:10.8f} ) atm \n")

                if loop_type_b == 0:
                    f.write(f"    Volatile partial pressure : 10^( {loop_step_b:10.8f} ) atm \n\n")

                elif loop_type_b == 1:
                    f.write(f"    Temperature : {loop_step_b:10.8f} K \n\n")

                elif loop_type_b == 2:
                    f.write(f"    Dopant Concentration : 10^( {loop_step_b:10.8f} ) \n\n")

                elif loop_type_b == 3:
                    f.write(f"    Dopant Partial Pressure : 10^( {loop_step_b:10.8f} ) atm \n\n")

                f.write("    Calculated chemical potentials: \n\n")

                for element in elements:
                    f.write(f"      {'{:2s} :'.format(element)} {'{:12.8f}'.format(chem_pot_dataframe.loc[total_index][element])}\n")

                f.write("\n")

                f.write(f"    Fermi level : {fermi_level_dataframe.loc[total_index]['fermi_level']:10.8f}\n\n")

                f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
                f.write("    |{:^20s}|{:^20s}|{:^20s}|\n".format("Defect", "Formation Energy", "log_{10}[D]", ))
                f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
                for defect_name in conc_defect_names:

                    if defect_name != "electrons" and defect_name != "holes" and defect_name != "stoic":
                        f.write("    |  {:18s}|  {:<18.8f}|  {:<18.8f}|\n".format(defect_name,form_eng_dataframe.loc[total_index][defect_name],
                                                                                  conc_dataframe.loc[total_index][defect_name]
                                                                                  )
                                )
                    else:
                        f.write("    |  {:18s}|  {:<18s}|  {:<18.8f}|\n".format(defect_name, '-',
                                                                                conc_dataframe.loc[total_index][defect_name]
                                                                                )
                                )

                f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n\n")

                if not secondary_phases_dataframe.empty:
                    f.write("    Secondary phase stability check:\n")

                    f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
                    f.write("    |{:^20s}|{:^20s}|{:^20s}|{:^20s}|{:^20s}|\n".format("Phase", "Energy", "Chem Pot Sum",
                                                                                     "Difference", "Stable?"))
                    f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")

                    for phase in phases_names:

                        phase_eng = secondary_phases_dataframe.loc[total_index][phase]["energy"]
                        phase_chem_pot = secondary_phases_dataframe.loc[total_index][phase]["chem_pot_sum"]
                        phase_difference = secondary_phases_dataframe.loc[total_index][phase]["difference"]

                        phase_stability = secondary_phases_dataframe.loc[total_index][phase]["is_stable"]
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

            f.write(">>> Finished\n\n")


def write_brouwer_output(conc_dataframe, form_eng_dataframe, chem_pot_dataframe, fermi_level_dataframe,  secondary_phases_dataframe, loop_type, seedname):

    with open(f"{seedname}.output", "a") as f:
        f.write(">>> Task : Brouwer\n\n")

        loop_steps = conc_dataframe.index
        conc_defect_names = conc_dataframe.columns
        form_eng_defect_names = form_eng_dataframe.columns
        elements = chem_pot_dataframe.columns

        if not secondary_phases_dataframe.empty:
            phases_names = secondary_phases_dataframe.columns

        for index, loop_step in enumerate(loop_steps, start=1):

            f.write(f"    Loop step : {index} of {len(loop_steps)}\n\n")

            if loop_type == 0:
                f.write(f"    Volatile partial pressure : 10^( {loop_step:10.8f} ) atm\n\n")

            f.write("    Calculated chemical potentials:\n\n")

            for element in elements:
                f.write(f"      {'{:2s} :'.format(element)} {'{:12.8f}'.format(chem_pot_dataframe.loc[loop_step][element])}\n")

            f.write("\n")

            f.write(f"    Fermi level : {fermi_level_dataframe.loc[loop_step]['fermi_level']:10.8f}\n\n")

            f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
            f.write("    |{:^20s}|{:^20s}|{:^20s}|\n".format("Defect", "Formation Energy", "log_{10}[D]",))
            f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
            for defect_name in conc_defect_names:

                if defect_name != "electrons" and defect_name != "holes" and defect_name != "stoic":
                    f.write("    |  {:18s}|  {:<18.8f}|  {:<18.8f}|\n".format(defect_name, form_eng_dataframe.loc[loop_step][defect_name], conc_dataframe.loc[loop_step][defect_name]))
                else:
                    f.write("    |  {:18s}|  {:<18s}|  {:<18.8f}|\n".format(defect_name, '-', conc_dataframe.loc[loop_step][defect_name]))

            f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+\n\n")


            if not secondary_phases_dataframe.empty:
                f.write("    Secondary phase stability check:\n")

                f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")
                f.write("    |{:^20s}|{:^20s}|{:^20s}|{:^20s}|{:^20s}|\n".format("Phase", "Energy", "Chem Pot Sum", "Difference", "Stable?"))
                f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n")

                for phase in phases_names:

                    phase_eng = secondary_phases_dataframe.loc[loop_step][phase]["energy"]
                    phase_chem_pot = secondary_phases_dataframe.loc[loop_step][phase]["chem_pot_sum"]
                    phase_difference = secondary_phases_dataframe.loc[loop_step][phase]["difference"]

                    phase_stability = secondary_phases_dataframe.loc[loop_step][phase]["is_stable"]
                    if phase_stability:
                        phase_stability = "True"
                    else:
                        phase_stability = "False"

                    f.write("    |  {:<18s}|  {:<18.8f}|  {:<18.8f}|  {:<18.8f}|  {:<18s}|\n".format(phase, phase_eng, phase_chem_pot, phase_difference, phase_stability))

                f.write(f"    +{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+{20 * '-'}+\n\n")

            f.write(f"{110*'-'}\n\n")

        f.write(">>> Finished\n\n")

def defect_phases_plot(min_val_x, max_val_x, min_val_y, max_val_y, phase_type, loop_a_type, loop_b_type, looping_dopants, art_dopant_chg):

    phases_dataframe = pd.read_csv(f"defect_phases_{phase_type}_data.csv")
    legend_labels = []

    group_colours = {}

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
            string = r"plt.xlabel(r'[$\lambda^{" + f"{art_dopant_chg}" + r"}$] /pfu')" +"\n"
            f.write(string)
        elif loop_a_type == 5:
            string = r"plt.xlabel(r'$\mu_\mathrm{" + f"{looping_dopants[0]}" +"}$ /eV')" +"\n"
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
            string = r"plt.ylabel(r'[$\lambda^{" + f"{art_dopant_chg}" + r"}$] /pfu')" +"\n"
            f.write(string)
        elif loop_b_type == 5:
            string = r"plt.ylabel(r'$\mu_\mathrm{" + f"{looping_dopants[1]}" +"}$ /eV')" +"\n"
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

                cx, cy = phase_coords_lst.mean(0)
                f.write(f"plt.text(x={cx}, y={cy}, s='{phase}', c='{text_colours[colour_index]}')\n")

                f.write("\n\n")

                colour_index += 1 if colour_index < len(colours) - 1 else len(colours) - 1
                legend_labels.append(phase)

        f.write(f'\nfig.legend(loc="outside lower center", ncol={len(legend_labels)},fancybox=True, labels={legend_labels})\n')
        f.write('plt.tight_layout()\n')
        f.write('plt.subplots_adjust(bottom=0.125)\n')
        #f.write('plt.savefig("/Users/bedforrt/Desktop/test.png")\n')
        f.write('plt.show()\n')


def brouwer_plot(axes, defects_data, grouped_defects, loop_type, looping_dopants, art_dopant_chg):

    concentrations_dataframe = pd.read_csv("brouwer_data.csv")
    legend_labels = []

    with open("plot_brouwer.py", "w") as f:

        colours = ["#006e00", "#b80058", "#008cf9", "#d163e6", "#00bbad", "#ff9287", "#b24502", "#878500", "#00c6f8",
                   "#00a76c", "#bdbdbd", "#5954d6", "#ebac23", "#000000"]

        linestyles = [(0, ()), (0, (5, 2)), (0, (5, 2, 1, 2)), (0, (1, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1))]

        group_colours = {}

        f.write('import pandas as pd\n')
        f.write('import matplotlib.pyplot as plt\n')
        f.write('plt.close("all")\n\n')
        f.write('concentrations_dataframe = pd.read_csv("brouwer_data.csv")\n\n')
        f.write("plt.figure(figsize=(12, 7))\n")
        f.write(f"plt.axis({axes})\n")

        if loop_type == 0:
            f.write(r"plt.xlabel(r'$\mathrm{log}_{10}\mathrm{P}_{\mathrm{O}_{2}}$ /atm')"+"\n")
        elif loop_type == 1:
            f.write("plt.xlabel('Temperature /K')\n")
        elif loop_type == 2:
            f.write(f"plt.xlabel('[{looping_dopants[0]}] /pfu')\n")
        elif loop_type == 3:
            string = r"plt.xlabel(r'$\mathrm{log}_{10}\mathrm{P}_{\mathrm{" + f"{looping_dopants[0]}" + r"}_{2}}$ /atm')" "+\n"
            f.write(string)
        elif loop_type == 4:
            string = r"plt.xlabel(r'[$\lambda^{" + f"{art_dopant_chg}" + r"}$] /pfu')" +"\n"
            f.write(string)
        elif loop_type == 5:
            string = r"plt.xlabel(r'$\mu_\mathrm{" + f"{looping_dopants[0]}" +"}$ /eV')" +"\n"
            f.write(string)

        f.write("plt.ylabel('[D] /pfu')\n\n")

        colour_index = 0
        line_index = 0
        for defect in concentrations_dataframe:
            if defect == "loop_step":
                continue

            elif defect != "electrons" and defect != "holes" and defect != "stoic":

                if max(concentrations_dataframe[defect]) > axes[2]:
                    if not grouped_defects:
                        defect_group = defects_data[defect]["group"]
                    else:
                        defect_group = defect

                    if defect_group not in group_colours:
                        group_colours[defect_group] = colours[colour_index]
                        colour_index += 1 if colour_index < len(colours) - 1 else len(colours) - 1

                        line_index = 0
                    else:
                        line_index += 1


                    f.write(
                        f'plt.plot(concentrations_dataframe["loop_step"], concentrations_dataframe["{defect}"], "{group_colours[defect_group]}", linestyle={linestyles[line_index]}, label="{defect}")\n')
                    legend_labels.append(defect)

            elif defect == "electrons":
                f.write(f'plt.plot(concentrations_dataframe["loop_step"], concentrations_dataframe["{defect}"], "#5954d6", linestyle="solid", label="electrons")\n')
                legend_labels.append(defect)

            elif defect == "holes":
                f.write(
                    f'plt.plot(concentrations_dataframe["loop_step"], concentrations_dataframe["{defect}"], "#ebac23", linestyle="solid", label="holes")\n')
                legend_labels.append(defect)

            elif defect == "stoic":
                f.write(
                    f'plt.plot(concentrations_dataframe["loop_step"], concentrations_dataframe["{defect}"], "#000000", linestyle=(5, (10, 3)), label="stoic")\n')
                legend_labels.append(defect)


        f.write(f'\nplt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol={len(group_colours)+1}, fancybox=True, labels={legend_labels})\n')
        f.write('plt.subplots_adjust(bottom=0.225)\n')
        f.write('plt.tight_layout()\n')
        f.write('plt.show()\n')


    #call(["python3", "brouwer_plot.py"])

