import sys
from scipy import interpolate
from scipy.optimize import minimize
from scipy import special
import numpy as np
import math
import time
import os
import shutil
from decimal import *
from datetime import datetime
import platform
import csv
import pyromat as pm
import fnmatch

#####################################################
#                                                   #
#           Defect Analysis Package 3               #
#                                                   #
#               by Samuel T. Murphy                 #
#               & William D. Neilson                #
#               & Reece T. Bedford                  #
#                                                   #
#####################################################
#                                                   #
# Script for exploring a material's defect          #
# chemistry, connecting the results of DFT          #
# calculations with bespoke, user-customised        #
# thermodynamics and processing operations that can #
# be tailored to any defect-containing system of    #
# interest                                          #
#                                                   #
# William D. Neilson, Reece T. Bedford and          #
# Samuel T. Murphy                                  #
# DefAP: A Python code for the analysis of point    #
# defects in crystalline solids, Computational      #
# Materials Science 210 (2022) 111434.              #
#                                                   #
#####################################################
#                                                   #
# Last updated :  19/10/23                          #
#                                                   #
#####################################################

version = '3.0'


# Function to print header
def main_wrapper(func):
    def wrapper():
        start_time = time.time()

        print("+-------------------------------------------+")
        print("|  ____        __   _    ____               |")
        print("| |  _ \  ___ / _| / \  |  _ \ _ __  _   _  |")
        print("| | | | |/ _ \ |_ / _ \ | |_) | '_ \| | | | |")
        print("| | |_| |  __/  _/ ___ \|  __/| |_) | |_| | |")
        print("| |____/ \___|_|/_/   \_\_| (_) .__/ \__, | |")
        print("|                             |_|    |___/  |")
        print("|                                           |")
        print("|       Defect Analysis Package : 3.0       |")
        print("|         https://github.com/DefAP          |")
        print("+-------------------------------------------+")
        print("|                                           |")
        print("|             William D. Neilson            |")
        print("|              Reece T. Bedford             |")
        print("|              Samuel T. Murphy             |")
        print("|                                           |")
        print("+-------------------------------------------+\n")

        print(
            "For assistance, please contact the developers at:\n"
            "wneilson@lanl.gov\nr.t.bedford@lancaster.ac.uk\nsamuel.murphy@lancaster.ac.uk\n")

        func()

        end_time = time.time()

        print("\n>>> Tasks complete")

        execute_time = end_time - start_time

        time_format = time.strftime("%H hrs %M minutes %S seconds", time.gmtime(execute_time))

        print(f">>> DefAP executed in {time_format}.")

    return wrapper


# function to read input file
def inputs(seedname):
    filename = str(seedname) + ".input"

    # Define variales and set some sensible defaults
    temperature = None # Temperature in K
    def_statistics = None  # Defect statistics 0 = Boltzmann, 1 = Kasamatsu
    tab_correction = 0  # Include a correction to the defect energy in the filename.defects file (0 = off, 1 = on)
    num_tasks = 0  # Number of tasks to perform
    host_energy = None  # Energy of a formula unit of the host
    host_supercell = None  # Energy of the host perfect host supercell
    chem_pot_method = None  # Definition of the chemical potentials 0 = defined, 1 = rich-poor 2= volatile, 3=volatile-rich-poor.
    use_coul_correction = 0  # Use a Coulombic correction 0 = none, 1 = Makov-Payne, 2 = screened Madelung
    length = None  # Supercell length for the MP correction
    dielectric = None  # Dielectric constant for MP correction
    v_M = None  # Screened Madelung potential
    lattice_vectors = None # for madelung task
    dielectric_tensor = None # for madelung
    E_VBM = 0  # Energy of the valence band maximum
    bandgap = 0  # Bandgap for the host lattice
    condband = None  # Effective conduction band integral
    valband = None  # Effective valence band integral
    electron_method = 0  # Method for calculating the electron concentration 0 = none, 1 = Boltzmann, 2 = Fermi-Dirac, 3 = Fixed
    hole_method = 0  # Method for calculating the hole concentration 0 = none, 1 = Boltzmann, 2 = Fermi-Dirac, 3 = Fixed
    fixed_e_conc = 0  # Fixed electron concentration
    fixed_p_conc = 0  # Fixed hole concentration
    art_dop_conc = 0  # Concentration of an artificial dopant
    art_dop_charge = 0  # Charge on the artifical dopant
    plot_art_dopant_conc = 0
    loop = None  # Property to loop over 0 = partial pressure, 1 = temperature, 2 = dopant conc, 3 = aritifical dopant conc
    min_value = None  # Minimum value used in loop
    max_value = None  # Maximum value using in loop
    iterator = None  # Iterator between minimum and maximum
    loop2 = None # Property2 to loop over 0 = partial pressure, 1 = temperature, 2 = dopant conc, 3 = aritifical dopant conc
    min_value_y = None  # Minimum value used in loop2
    max_value_y = None  # Maximum value using in loop2
    iterator_y = None  # Iterator2 between minimum_y and maximum_y
    gnuplot_version = 0  # Version of gnuplot 0 = v4, 1 = v5
    min_y_range = -20  # Minimum on the y axis for Brouwer plots
    max_y_range = 0  # Maximum on the y axis for Brouwer plots
    host_name = None  # Host name
    cond_band_min = None  # Conduction band minimum
    cond_band_max = None  # Conduction band maximum
    val_band_min = None  # Valence band minimum
    val_band_max = None  # Valence band maximum
    fu_uc = None  # Number of functional units per unit cell.
    electron_mass_eff = 0  # DOS effective mass for electron
    hole_mass_eff = 0  # DOS effective mass for hole
    unit_vol = None  # Volume of unit cell (A^3) Used in DOS effective masses and for y axis unit conversion
    lines = 0  # Not in use
    y_form_min = 0  # Not in use
    y_form_max = 10  # Not in use
    accommodate = None  # The dopant element that the accomodation mechanmism is being tested for.
    entropy_marker = 0  # Use a vibrational entropy contribution (0 = off, 1 = on)
    entropy_units = 1  # The number of functional units the entropy values that are entered in filename.entropy represen
    scheme = 0  # Selection of colour scheme for plots produced by DefAP: 0: DefAP colour scheme (default).1: User customised scheme. Requires the input file, filename.plot
    stoichiometry = 0  # Calculate and plot stoichiometry 1= on, 2= special option that considers dopants and calulates an O/M ratio.
    x_variable = 0  # Selection of x-axis in final defect concentration plots: 0: Plot as function of the property defined with loop (default). 1: Plot as a function of stoichiometry.
    y_variable = 0  # Selection of y-axis units in final defect concentration plots: 0: Per functional unit (default). 1: per cm^-3.
    real_gas = None  # Calculate volatile chemical potetial with real gas parameters (1)
    function_tol = 1e-10  # Sequential Least Squares Programming: Precision goal for the value of function in the stopping criterion.
    maxiter_dop = 100  # Maximum number of iterations to optimise dopant chemical potential(s) (SLSQP)
    SLSQP_dial = 10  # Controls the relaunch guesses of dopant chemical potentials - a relaunch is triggered when SLSQP exits without solution. Increasing value increases success rate at the expense of speed.
    charge_convergence = 0.0000000001  # The stopping criteria for the calculation of the Fermi level. Fermi level deemed satisfactory when the total charge does not exceed this value.
    potential_convergence = 0.001  # Convergence criteria for dopant concentration: the difference between logarithmic target and calculated concentration.
    gibbs_marker = 0  # Selection of 'gibbs' temperature dependence.
    include_temperature_contribution = "false"
    number_of_dopants = 0

    # Data holds
    tasks = []
    constituents = []
    constituents_name_list = []
    dopref_name_list = []
    dopants = [0]
    stability = [0]
    dopant_fitting = 0
    host_array = []  # Host array
    num_elements = 0  # Number of elements in the host

    valid_loop_values = [0, 1, 2, 3, 4, 5]

    print(">>> Reading in parameters from ", filename)

    with open(filename) as file:
        for linenumber, line in enumerate(file):
            fields = line.strip().split()

            if len(fields) != 0:
                name = fields[0]

                # Tasks
                if (name.lower() == "tasks"):
                    num_tasks = len(fields) - 2
                    if (num_tasks == 0):
                        raise Exception("<!> There are no tasks to perform")

                    for i in np.arange(0, num_tasks, 1):
                        task = fields[2 + i]
                        tasks.append(task)

                if (name.lower() == "loop"):
                    loop = int(fields[2])

                    if loop not in valid_loop_values:
                        raise Exception("<!> Error! Invalid loop selection.\n"
                                        f"Your input: Loop = {loop}\n"
                                        f"Valid selections: {valid_loop_values}")

                if (name.lower() == "loop2"):
                    # loop = 0.1 : loop over volatile partial pressure, converted for phase.
                    loop2 = int(fields[2])

                    if loop2 not in valid_loop_values and loop2 != 0.1:
                        raise Exception("<!> Error! Invalid loop2 selection.\n"
                                        f"Your input: Loop2 = {loop2}\n"
                                        f"Valid selections: {valid_loop_values}")

                # Properties for loop
                if (name.lower() == "min_value"):
                    min_value = float(fields[2])
                if (name.lower() == "max_value"):
                    max_value = float(fields[2])
                if (name.lower() == "iterator"):
                    iterator = float(fields[2])

                # Properties for loop2
                if (name.lower() == "min_value_y"):
                    min_value_y = float(fields[2])
                if (name.lower() == "max_value_y"):
                    max_value_y = float(fields[2])
                if (name.lower() == "iterator_y"):
                    iterator_y = float(fields[2])

                # Host formula
                if (name.lower() == "host"):
                    host_name = fields[2]
                    host_array = break_formula(host_name, 0)
                    num_elements = host_array[0]

                # Calculate stoichiometry
                if (name.lower() == "stoichiometry"):
                    stoichiometry = int(fields[2])

                # Host Energy (eV)
                if (name.lower() == "host_energy"):
                    host_energy = float(fields[2])

                if (name.lower() == "host_supercell"):
                    host_supercell = float(fields[2])

                # Energy of the Valence Band Maximum (eV)
                if (name.lower() == "e_vbm"):
                    E_VBM = float(fields[2])

                # Bangap of the host material
                if (name.lower() == "bandgap"):
                    bandgap = float(fields[2])

                # Effective conduction band integral
                if (name.lower() == "conductionband"):
                    condband = float(fields[2])

                # Effective valence band integral
                if (name.lower() == "valenceband"):
                    valband = float(fields[2])

                # Electron calculation method
                if (name.lower() == "electron_method"):
                    if (fields[2].lower() == "off"):
                        electron_method = 0
                    elif (fields[2].lower() == "boltzmann"):
                        electron_method = 1
                    elif (fields[2].lower() == "fermi-dirac"):
                        electron_method = 2
                    elif (fields[2].lower() == "fixed"):
                        electron_method = 3
                        fixed_e_conc = fields[3]
                    elif (fields[2].lower() == "effective_masses"):
                        electron_method = 4
                        electron_mass_eff = eval(fields[3])
                    else:
                        raise ValueError("<!> Error : Undefined method for calculating electron concentration\n"
                                         f"Your input: {name}")

                # Hole calculation method
                if (name.lower() == "hole_method"):
                    if (fields[2].lower() == "off"):
                        hole_method = 0
                    elif (fields[2].lower() == "boltzmann"):
                        hole_method = 1
                    elif (fields[2].lower() == "fermi-dirac"):
                        hole_method = 2
                    elif (fields[2].lower() == "fixed"):
                        hole_method = 3
                        fixed_p_conc = fields[3]
                    elif (fields[2].lower() == "effective_masses"):
                        hole_method = 4
                        hole_mass_eff = eval(fields[3])
                    else:
                        raise ValueError("<!> Error : Undefined method for calculating hole concentration\n"
                                         f"Your input: {name}")

                # Minimum and maximum for the valence and conduction bands
                if (name.lower() == "valence_band_limits"):
                    val_band_min = float(fields[2])
                    val_band_max = float(fields[3])
                if (name.lower() == "conduction_band_limits"):
                    cond_band_min = float(fields[2])
                    cond_band_max = float(fields[3])

                # Unit cell details
                if (name.lower() == "fu_unit_cell"):
                    fu_uc = float(fields[2])
                if (name.lower() == "volume_unit_cell"):
                    unit_vol = float(fields[2])

                # Temperature
                if (name.lower() == "temperature"):
                    temperature = float(fields[2])

                # Chemical potential method
                if (name.lower() == "real_gas"):
                    real_gas = int(fields[2])

                    if int(real_gas) == 2:
                        print("\n!!! WARNING !!!\n"
                              "The coefficients used in this mode have been refitted since the original publication.\n"
                              "Errors may be incurred if used.\n")
                        rg_2_continue = input("Do you still wish to proceed? [Y/N] ")

                        if rg_2_continue.lower() == "no" or rg_2_continue.lower() == "n":
                            print("Please select a new method for calculating real gas relations.")
                            selected_rg = False
                            while not selected_rg:
                                print("0 - ideal gas\n"
                                      "1 - real gas relations\n"
                                      "3 - PYroMat library (NASA equation)\n")
                                real_gas = int(input("Real gas: "))

                                if real_gas == 0 or real_gas == 1 or real_gas == 3:
                                    selected_rg = True
                                else:
                                    print("Invalid selection. Please try again.\n")

                        else:
                            print("Proceeding with real_gas = 2...\n")

                if (name.lower() == "chem_pot_method"):
                    if fields[2].lower() == "defined":
                        chem_pot_method = 0

                    elif fields[2].lower() == "rich-poor":
                        chem_pot_method = 1

                    elif fields[2].lower() == "volatile":
                        chem_pot_method = 2
                    elif fields[2].lower() == "volatile-reference":
                        chem_pot_method = 3
                    elif fields[2].lower() == "volatile-stoic":
                        chem_pot_method = 4
                    elif fields[2].lower() == "volatile-rich-poor":
                        chem_pot_method = 5

                    else:
                        raise ValueError("<!> Error : Unknown chem_pot_method entered\n"
                                         f"Your input: {name}")

                # Gibbs temperature dependence
                if (name.lower() == "gibbs"):
                    gibbs_marker = float(fields[2])

                # Convergence criteria
                if (name.lower() == "charge_convergence"):
                    charge_convergence = float(fields[2])

                # Constituents
                if (name.lower() == "constituents"):

                    defintion_total = 0

                    # Loop over list of constituents
                    for i in np.arange(1, host_array[0] + 1, 1):

                        with open(filename) as file3:
                            for linenumber3, line3 in enumerate(file3):
                                fields3 = line3.strip().split()

                                if linenumber + i == linenumber3:

                                    if (chem_pot_method == 0):  # Use defined chemical potentials

                                        constituent_name = fields3[0]
                                        constituent_energy = float(fields3[1])

                                        constituents.append(constituent_name)
                                        constituents.append(constituent_energy)
                                        constituents_name_list.append(constituent_name)

                                    elif (chem_pot_method == 1):  # Use rich-poor chemical potential method

                                        constituent_name = fields3[0]
                                        constituent_energy = float(fields3[1])
                                        constituent_definition = float(fields3[2])

                                        constituents.append(constituent_name)
                                        constituents.append(constituent_energy)
                                        constituents.append(constituent_definition)
                                        constituents_name_list.append(constituent_name)

                                        if (constituent_definition > 1.0):
                                            raise ValueError(
                                                f"<!> Error : Constituent {constituent_name} has greater definition than 1 ({constituent_definition})")

                                        defintion_total += constituent_definition
                                        if (defintion_total > (host_array[0] - 1)):
                                            raise ValueError(
                                                f"<!> Error : Your total rich-poor balance ({defintion_total}) is greater than possible with this number of constituents ({host_array[0] - 1})")

                                    if (chem_pot_method == 2 or chem_pot_method == 3):  # Use volatile method with a binary system

                                        if i == 1:

                                            gaseous_species = fields3[0]
                                            partial_pressure = float(fields3[1])

                                            constituents.append(gaseous_species)
                                            constituents.append(partial_pressure)

                                        else:

                                            constituent_name = fields3[0]
                                            constituent_energy_DFT = float(fields3[1])
                                            constituent_metal_DFT = float(fields3[2])
                                            constituent_formation = float(fields3[3])

                                            constituents.append(constituent_name)
                                            constituents.append(constituent_energy_DFT)
                                            constituents.append(constituent_metal_DFT)
                                            constituents.append(constituent_formation)
                                            constituents_name_list.append(constituent_name)




                                    if (chem_pot_method == 4):  # Use volatile-stoic method with a binary system

                                        if i == 1:

                                            gaseous_species = fields3[0]
                                            partial_pressure = float(fields3[1])

                                            constituents.append(gaseous_species)
                                            constituents.append(partial_pressure)

                                        else:

                                            constituent_A_name = fields3[0]
                                            constituent_A_stoic = int(fields3[1])
                                            constituent_A_energy_DFT = float(fields3[2])

                                            constituent_B_name = fields3[3]
                                            constituent_B_stoic = int(fields3[4])
                                            constituent_B_energy_DFT = float(fields3[5])

                                            reaction_free_energy = float(fields3[6])

                                            constituents += [constituent_A_name, constituent_A_stoic,
                                                             constituent_A_energy_DFT,
                                                             constituent_B_name, constituent_B_stoic,
                                                             constituent_B_energy_DFT,
                                                             reaction_free_energy]

                                            constituents_name_list += [constituent_A_name, constituent_B_name]



                                    if (chem_pot_method == 5):  # Use rich-poor chemical potential and volatile method

                                        number_bin_oxides = host_array[0] - 1

                                        if i == 1:

                                            gaseous_species = fields3[0]
                                            gaseous_stoichiometry = float(fields3[1])
                                            partial_pressure = float(fields3[2])

                                            constituents.append(gaseous_species)
                                            constituents.append(gaseous_stoichiometry)
                                            constituents.append(partial_pressure)
                                        else:
                                            constituent_name = fields3[0]
                                            constituent_stoich = float(fields3[1])
                                            constituent_energy_DFT = float(fields3[2])
                                            constituent_metal_DFT = float(fields3[3])
                                            constituent_formation = float(fields3[4])
                                            constituent_definition = float(fields3[5])

                                            constituents.append(constituent_name)
                                            constituents.append(constituent_stoich)
                                            constituents.append(constituent_energy_DFT)
                                            constituents.append(constituent_metal_DFT)
                                            constituents.append(constituent_formation)
                                            constituents.append(constituent_definition)
                                            constituents_name_list.append(constituent_name)



                                            # print (constituent_name ,constituent_stoich ,constituent_energy_DFT ,constituent_metal_DFT ,constituent_formation ,constituent_definition)
                                            if (constituent_definition > 1.0):
                                                raise ValueError(
                                                    f"<!> Error : Constituent {constituent_name} has greater definition than 1 ({constituent_definition})")

                                            defintion_total += constituent_definition
                                            if (defintion_total > (host_array[0] - 2)):
                                                raise ValueError(
                                                    f"<!> Error : Your total rich-poor balance ({defintion_total}) is greater than possible with this number of constituents ({host_array[0] - 2})")


                # Dopants
                if (name.lower() == "dopant_table"):
                    number_of_dopants = float(fields[2])
                    dopants[0] = number_of_dopants

                    # Loop over dopant table and fill dopants array

                    for i in np.arange(1, number_of_dopants + 1, 1):
                        with open(filename) as file4:
                            for linenumber4, line4 in enumerate(file4):
                                fields4 = line4.strip().split()

                                if linenumber + i == linenumber4:
                                    dopant_name = fields4[0]
                                    # Break down details of the reference state
                                    reference_state = fields4[1]
                                    temp_array = break_formula(reference_state, 1)
                                    dopants.append(dopant_name)

                                    reference_energy = float(fields4[2])
                                    dopants.append(reference_energy)

                                    fit_chempot = int(fields4[3])
                                    dopants.append(fit_chempot)

                                    if fit_chempot == 0:
                                        dopants.append(0)
                                        dopants.append(temp_array)
                                        dopants.append(0)

                                    if fit_chempot == 1 or fit_chempot == 2:
                                        dopant_fitting += 1
                                        target_conc = float(fields4[4])
                                        dopant_range = float(fields4[5])
                                        dopants.append(target_conc)
                                        dopants.append(temp_array)
                                        dopants.append(dopant_range)

                                    if fit_chempot == 3 or fit_chempot == 4:
                                        dop_partial_pressure = float(fields4[4])
                                        dopants.append(0)
                                        dopants.append(temp_array)
                                        dopants.append(dop_partial_pressure)

                                    dopref_name_list.append(reference_state)

                # Dopant optimise details

                if (name.lower() == "tolerance"):
                    function_tol = float(fields[2])
                if (name.lower() == "max_iteration"):
                    maxiter_dop = float(fields[2])
                if (name.lower() == "potential_convergence"):
                    potential_convergence = float(fields[2])
                if (name.lower() == "slsqp_dial"):
                    SLSQP_dial = float(fields[2])

                # Artificial dopants
                if (name.lower() == "art_dopant_conc"):
                    art_dop_conc = float(fields[2])
                if (name.lower() == "art_dopant_chg"):
                    art_dop_charge = float(fields[2])
                if name.lower() == "plot_art_dopant_conc":
                    plot_art_dopant_conc = int(fields[2])

                # Dopant accomodation mechanism
                if (name.lower() == 'accommodate'):
                    accommodate = fields[2]

                # Stability checks
                if (name.lower() == "stability_check"):
                    number_of_checks = float(fields[2])
                    stability[0] = number_of_checks

                    for i in np.arange(1, number_of_checks + 1, 1):
                        with open(filename) as file5:
                            for linenumber5, line5 in enumerate(file5):
                                fields5 = line5.strip().split()

                                if linenumber + i == linenumber5:
                                    constituent = fields5[0]
                                    reference_energy = float(fields5[1])

                                    try:
                                        include_temperature_contribution = str(fields5[2]).lower()
                                    except:
                                        include_temperature_contribution = "false"

                                    stability.append(constituent)
                                    stability.append(reference_energy)

                                    # Break down details of the reference state
                                    temp_array = break_formula(constituent, 1)

                                    stability.append(temp_array)

                                    valid_temp_cont_selection = ["true", "t", "false", "f"]
                                    valid_gibbs_cont_selection = ["gibbs_true", "gibbs_false"]
                                    if (include_temperature_contribution.lower() not in valid_temp_cont_selection) and (
                                            include_temperature_contribution.lower() not in valid_gibbs_cont_selection):
                                        raise ValueError(
                                            f"<!> Error. Invalid selection for including temperature contribution for secondary phases\n"
                                            f"Valid selections are: {valid_temp_cont_selection}\n"
                                            f"Or for gibbs contribution: {valid_gibbs_cont_selection}\n"
                                            f"Your input is: {constituent} {reference_energy} {include_temperature_contribution}\n"
                                            "Please correct input file")
                                    else:
                                        stability.append(include_temperature_contribution)

                                    if include_temperature_contribution == "true" or include_temperature_contribution == "t":
                                        if real_gas != 3:
                                            real_gas = 3
                                            print("\n> Temperature contributions to secondary phases chemical potentials specified to be added to one or more compounds in 'stability_check'."
                                                f"\n> Setting real_gas = 3 (PYroMat) as required.")


                                    if (include_temperature_contribution.lower() == "gibbs_true") and (constituent not in constituents_name_list):
                                        constituents_name_list.append(constituent)

                # Defect concentration method
                if (name.lower() == "defect_conc_method"):
                    if (fields[2].lower() == "Boltzmann".lower()):
                        def_statistics = 0

                    elif (fields[2].lower() == "Kasamatsu".lower()):
                        def_statistics = 1

                    else:
                        raise ValueError("<!> Error : Unknown defect statistics method entered\n"
                                         f"Your input: {fields[2]}")

                # Use correction schemes
                if (name.lower() == "tab_correction"):
                    tab_correction = int(fields[2])
                if (name.lower() == "coulombic_correction"):
                    use_coul_correction = int(fields[2])

                # point charge correction additional parameters
                if (name.lower() == "dielectric_constant"):
                    dielectric = float(fields[2])
                if (name.lower() == "length"):
                    length = float(fields[2])
                if (name.lower() == "screened_madelung"):
                    v_M = float(fields[2])


                # Formation plot preferences
                if name.lower() == "formation_energy_limits":
                    y_form_min = float(fields[2])
                    y_form_max = float(fields[2])
                if name.lower() == "lines":
                    lines = int(fields[2])

                # Entropy
                if name.lower() == "entropy":
                    entropy_marker = int(fields[2])
                if name.lower() == "entropy_units":
                    entropy_units = int(fields[2])

                # Plotting preferences
                if (name.lower() == "x_variable"):
                    x_variable = int(fields[2])
                if (name.lower() == "y_axis"):
                    y_variable = int(fields[2])
                    if y_variable == 1:
                        max_y_range = 20
                if (name.lower() == "gnuplot_version"):
                    gnuplot_version = fields[2]
                if (name.lower() == "min_y_range"):
                    min_y_range = fields[2]
                if (name.lower() == "max_y_range"):
                    max_y_range = fields[2]
                if name.lower() == "scheme":
                    scheme = int(fields[2])

    # Some error messages

    if len(tasks) == 0:
        raise ValueError("<!> There are no tasks to perform")

    for i in tasks:
        if i not in ['brouwer', 'energy', 'form_plots', 'autodisplay', 'stability', 'madelung', 'bibliography', 'group',
                     'phases', 'defect_phase', 'dopant', 'stability_highest', 'update_reference']:
            raise ValueError("<!> '", i, "' not an optional task")

    if 'form_plots' in tasks:
        if 'energy' not in tasks:
            raise ValueError("<!> 'The 'form_plots' task has no effect without the 'energy' task")

    if 'dopant' in tasks:
        if 'defect_phase' not in tasks:
            raise ValueError("<!> 'The 'dopant' task has no effect without the 'defect_phase' task")

        if accommodate == None:
            raise ValueError("<!> 'The 'dopant' task requires a dopant to be specified with the 'accommodate' tag in the input file")

    for i in tasks:
        if i in ['brouwer', 'energy', 'defect_phase']:

            if host_name is None:
                raise Exception("<!> Error. No host compound has been defined!")

            if host_energy is None:
                raise Exception("<!> Error. Undefined 'Host_energy'")

            if host_supercell is None:
                raise Exception("<!> Error. Undefined 'Host_supercell' energy")

            if chem_pot_method is None:
                raise Exception("<!> Error. No method for calculating chemical potentials has been defined!")

            if temperature is None and (loop != 1 or loop2 != 1):
                temperature = 1000
                print(f"\n> No temperature defined.\n> Setting temperature = {temperature} K as default.")


    # check loop has been defined for loop tasks
    if loop is None:
        if "brouwer" in tasks or "defect_phase" in tasks:
            raise Exception("\n<!> Error. 'loop' not defined for the 'brouwer' or 'defect_phase' tasks")

    else:
        # set defaults for loop if not defined
        if min_value is None:
            print(f"\n> 'min_value' for 'loop' not defined.")

            # default values
            if loop == 0:
                min_value = -40
            elif loop == 1:
                min_value = 500
            elif loop == 2 or loop == 3 or loop == 4:
                min_value = -20

            print(f"> Setting min_value = {min_value} as default for loop = {loop}")


        if max_value is None:
            print(f"\n> 'max_value' for 'loop' not defined.")

            # default values
            if loop == 0:
                max_value = 0
            elif loop == 1:
                max_value = 1000
            elif loop == 2 or loop == 3 or loop == 4:
                max_value = 0

            print(f"> Setting max_value = {max_value} as default for loop = {loop}")


        if min_value >= max_value:
            raise Exception("<!> Error: Incompatible min_value and max_value\n"
                            f"min_value: {min_value}\n"
                            f"max_value: {max_value}")

        if iterator is None:
            print("\n> 'iterator' not defined for loop.")

            default_steps = 20
            iterator = (max_value - min_value) / default_steps

            print(f"> Setting iterator = {iterator} ({default_steps} steps) as default.")

    if loop2 is None:
        if "defect_phase" in tasks:
            raise Exception("<!> Error. 'loop2' not defined for the 'defect_phase' task")

    else:
        # set defaults for loop2 if not defined
        if min_value_y is None:
            print(f"\n> 'min_value_y' for 'loop2' not defined.")

            # default values
            if loop2 == 0:
                min_value_y = -40
            elif loop2 == 1:
                min_value_y = 500
            elif loop2 == 2 or loop2 == 3 or loop2 == 4:
                min_value_y = -20

            print(f"> Setting min_value_y = {min_value_y} as default for loop = {loop2}")

        if max_value_y is None:
            print(f"\n> 'max_value_y' for 'loop2' not defined.")

            # default values
            if loop2 == 0:
                max_value_y = 0
            elif loop2 == 1:
                max_value_y = 1000
            elif loop2 == 2 or loop2 == 3 or loop2 == 4:
                max_value_y = 0

            print(f"> Setting max_value_y = {max_value_y} as default for loop = {loop2}")

        if min_value_y >= max_value_y:
            raise Exception("<!> Error: Incompatible min_value_y and max_value_y for loop2\n"
                            f"min_value_y: {min_value_y}\n"
                            f"max_value_y: {max_value_y}")

        if iterator_y is None:
            print("\n> 'iterator_y' not defined for loop2.")

            default_steps_y = 20
            iterator_y = (max_value_y - min_value_y) / default_steps_y

            print(f"> Setting iterator_y = {iterator_y} ({default_steps_y} steps) as default.")


    # check parameters needed to convert defect concentration units into cm-3 are defined
    if y_variable == 1:
        if fu_uc is None:
            raise Exception("<!> Error: 'fu_unit_cell' parameter must be defined for defect concentrations to be in units of cm^-3")

        if unit_vol is None:
            raise Exception("<!> Error: 'volume_unit_cell' parameter must be defined for defect concentrations to be in units of cm^-3")


    # check paramters for boltzmann electron/hole method are defined
    if electron_method == 1:
        if condband is None:
            raise Exception(f"<!> Error. 'boltzmann' electron_method requires the 'conductionband' input.\n"
                  "Please consult the input for help.")

    if hole_method == 1:
        if valband is None:
            raise Exception(f"<!> Error. 'boltzmann' hole_method requires the 'valenceband' input.\n"
                            "Please consult the input for help")


    # ensure all values for fermi-dirac electron/hole method are defined
    if electron_method == 2:
        if fu_uc is None:
            raise Exception(f"<!> Error. 'fermi-dirac' electron_method requires the 'fu_unit_cell' parameter.\n"
                            "Please consult the manuel for help")

        if cond_band_min is None or cond_band_max is None:
            raise Exception(f"<!> Error. 'fermi-dirac' electron_method requires the minimum and maximum in the 'conduction_band_limits' input to be defined for the integration.\n"
                            f"Your inputs are:\n"
                            f"> Min: {cond_band_min}\n"
                            f"> Max: {cond_band_max}\n"
                            f"Please consult the manuel for help.")

    if hole_method == 2:
        if fu_uc is None:
            raise Exception(f"<!> Error. 'fermi-dirac' hole_method requires the 'fu_unit_cell' parameter.\n"
                            "Please consult the manuel for help")

        if val_band_min is None or val_band_max is None:
            raise Exception(
                f"<!> Error. 'fermi-dirac' hole_method requires the minimum and maximum in the 'valence_band_limits' input to be defined for the integration.\n"
                f"Your inputs are:\n"
                f"> Min: {val_band_min}\n"
                f"> Max: {val_band_max}\n"
                f"Please consult the manuel for help.")


    # ensure all values for effective-masses electron/hole method are defined
    if electron_method == 4:
        if fu_uc is None:
            raise Exception(f"<!> Error. 'effective-masses' electron_method requires the 'fu_unit_cell' parameter.\n"
                            "Please consult the manuel for help")

        if unit_vol is None:
            raise Exception(f"<!> Error. 'effective-masses' electron_method requires the 'unit_vol' parameter.\n"
                            "Please consult the manuel for help")

    if hole_method == 4:
        if fu_uc is None:
            raise Exception(f"<!> Error. 'effective-masses' hole_method requires the 'fu_unit_cell' parameter.\n"
                            "Please consult the manuel for help")

        if unit_vol is None:
            raise Exception(f"<!> Error. 'effective-masses' hole_method requires the 'unit_vol' parameter.\n"
                            "Please consult the manuel for help")


    # check not looping over dopant conc. or partial pressure if no dopants defined
    if (loop == 2 or loop2 == 2) and number_of_dopants == 0:
        raise Exception("<!> Error. No dopants defined to loop over dopant concentration")

    if (loop == 4 or loop2 == 4) and number_of_dopants == 0:
        raise Exception("<!> Error. No dopants defined to loop over volatile dopant partial pressure")


    if def_statistics is None:
        def_statistics = 0
        print("\n> Method to calculate defect concentrations not specified."
              "\n> Calculating using Boltzmann statistics as default")

    # adding vibrational entropy can not be used with kasamatsu statistics
    if (entropy_marker == 1 and def_statistics == 1):
        raise Exception("<!> Error: Entropy cannot be used with the Kasamatsu statistics")

    # adding vibrational entropy can not be used with gibbs energy task
    if (entropy_marker == 1 and gibbs_marker == "gibbs_true"):
        raise Exception(
            "<!> Error: Vibrational entropy contribution and the Gibbs energy function can not be applied simultaneously")

    volatile_chem_pot_methods = [2, 3, 4, 5]
    # real_gas default is ideal gas method
    if real_gas is None and chem_pot_method in volatile_chem_pot_methods:
        real_gas = 0
        print("\n> 'real_gas' method not defined for volatile chemical potential methods."
              f"\n> Setting real_gas = {real_gas} as default.")

    # check parameters for adding point charge correction
    if use_coul_correction == 1:
        if length is None:
            raise Exception(f"<!> Error. Cubic point charge correction needs the 'length' of the suppercell used"
                            f" to be defined.")
        if dielectric is None:
            raise Exception(f"<!> Error. Cubic point charge correction needs the 'dielectric_constant' of the material"
                            f" to be defined.")

    elif use_coul_correction == 2:
        if v_M is None and "madelung" not in tasks:
            raise Exception(f"<!> Error. Anisotropic point charge correction requires the 'screened_madelung' potential.\n"
                            f"'screened_madelung' can also be calculated with task = madelung.")


    # Output file construction

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:
        print('DefAP', version, file=f)
        print("Executed on", dt_string, "\n", file=f)
        print("-------------------------------------------------------", "\n", file=f)
        for i in tasks:
            if i in ['brouwer', 'energy', 'form_plots', 'autodisplay', 'stability', 'group', 'defect_phase']:
                print(">>> Reading in parameters from ", filename, "\n", file=f)
                print("   Number of tasks :", num_tasks, file=f)
                for i in np.arange(1, num_tasks + 1, 1):
                    print("   Task", i, ":", tasks[i - 1], file=f)
                print("\n   Host :", host_name, file=f)
                print("   Number of elements in host :", num_elements, file=f)
                print("   DFT energy of host pfu:", host_energy, 'eV', file=f)
                print("   DFT energy of host supercell:", host_supercell, 'eV', file=f)

                print("   Energy of the valence band maximum:", E_VBM, 'eV', file=f)
                print("   Stoichiometry method:", stoichiometry, file=f)

                print("\n>>> Electronic properties\n", file=f)
                print("   Bandgap of host material:", bandgap, 'eV\n', file=f)

                if (electron_method == 0):
                    print("   Not calculating electron concentrations\n", file=f)
                elif (electron_method == 1):
                    print("   Using Boltzmann statistics for electron concentrations", file=f)
                    print("   Effective conduction band integral:", condband, 'eV pfu\n', file=f)
                elif (electron_method == 2):
                    print("   Using Fermi-Dirac statistics for the electron concentration", file=f)
                    print("   Conduction_band_limits:", cond_band_min, '-', cond_band_max, 'eV', file=f)
                    print("   Number of functional units per unit cell:", fu_uc, '\n', file=f)
                elif (electron_method == 3):
                    print("   Using fixed electron concentration of ", fixed_e_conc, "\n", file=f)
                elif (electron_method == 4):
                    print("   Using electron density of states effective masses", file=f)
                    print("   Electron density of states effective mass", electron_mass_eff, file=f)
                    print("   Number of functional units per unit cell:", fu_uc, file=f)
                    print("   Volume of the unit cell:", unit_vol, "Angstroms^3\n", file=f)

                if (hole_method == 0):
                    print("   Not calculating hole concentrations\n", file=f)
                elif (hole_method == 1):
                    print("   Using Boltzmann statistics for hole concentrations", file=f)
                    print("   Effective valence band integral:", valband, 'eV pfu\n', file=f)
                elif (hole_method == 2):
                    print("   Using Fermi-Dirac statistics for the hole concentration", file=f)
                    print("   Valence_band_limits:", val_band_min, '-', val_band_max, 'eV', file=f)
                    print("   Number of functional units per unit cell:", fu_uc, '\n', file=f)
                elif (hole_method == 3):
                    print("   Using fixed hole concentration of ", fixed_p_conc, "\n", file=f)
                elif (hole_method == 4):
                    print("   Using hole density of states effective masses", file=f)
                    print("   Hole density of states effective mass", hole_mass_eff, file=f)
                    print("   Number of functional units per unit cell:", fu_uc, file=f)
                    print("   Volume of the unit cell:", unit_vol, "Angstroms^3\n", file=f)

                print(">>> Chemical potentials\n", file=f)
                if (chem_pot_method == 0):
                    print("   Chemical potentials defined\n", file=f)
                    print("   Defining chemical potentials of elements in host:", file=f)
                    print("   +---------+------------------------+", file=f)
                    print("   | Element | Chemical potential (eV)|", file=f)
                    print("   +---------+------------------------+", file=f)
                    for i in np.arange(0, len(constituents) / 2, 1):
                        i = int(i)
                    print("   | %7s | %22f |" % (constituents[2 * i], constituents[2 * i + 1]), file=f)
                    print("   +---------+------------------------+\n", file=f)
                if (chem_pot_method == 1):
                    print("   Rich-poor method selected\n", file=f)
                    print("   Defining chemical potentials of elements in host:", file=f)
                    print("   +---------+------------------------+-----------+", file=f)
                    print("   | Element | Chemical potential (eV)|  fraction |", file=f)
                    print("   +---------+------------------------+-----------+", file=f)
                    for i in np.arange(0, len(constituents) / 3, 1):
                        i = int(i)
                    print("   | %7s | %22f | %8f |" % (
                        constituents[3 * i], constituents[2 * i + 1], constituents[3 * i + 2]), file=f)
                    print("   +---------+------------------------+-----------+\n", file=f)

                if (chem_pot_method == 2 or chem_pot_method == 3):
                    if chem_pot_method == 2:
                        print("   Volatile method selected\n", file=f)
                    else:
                        print("   Volatile-reference method selected\n", file=f)
                    if (real_gas == 0):
                        print("   Using ideal gas specific heat\n", file=f)
                    elif (real_gas == 1) or (real_gas == 2):
                        print("   Using real gas relations for specific heat\n", file=f)
                    elif real_gas == 3:
                        print("   Using PYroMat library (NASA equation) for real gas relations\n", file=f)
                    print("   Defining volatile species:", file=f)
                    print("   +-----------------+------------------+", file=f)
                    print("   | Gaseous species | Partial pressure |", file=f)
                    print("   +-----------------+------------------+", file=f)
                    print("   | %15s | %16f |" % (constituents[0], constituents[1]), file=f)
                    print("   +-----------------+------------------+\n", file=f)

                    print("   Defining properties of binary species:", file=f)
                    print("   +----------------+----------------+------------------------+-----------------------+",
                          file=f)
                    print("   | Binary species | DFT energy (eV)| Cation DFT energy (eV) | Formation energy (eV) |",
                          file=f)
                    print("   +----------------+----------------+------------------------+-----------------------+",
                          file=f)
                    print("   | %14s | %14f | %22f | %21f |" % (
                        constituents[2], constituents[3], constituents[4], constituents[5]), file=f)
                    print("   +----------------+----------------+------------------------+-----------------------+\n",
                          file=f)

                if (chem_pot_method == 4):
                    print("   Volatile-Stoichiometry method selected\n", file=f)
                    if (real_gas == 0):
                        print("   Using ideal gas specific heat\n", file=f)
                    elif (real_gas == 1) or (real_gas == 2):
                        print("   Using real gas relations for specific heat\n", file=f)
                    elif real_gas == 3:
                        print("   Using PYroMat library (NASA equation) for real gas relations\n", file=f)
                    print("   Defining volatile species:", file=f)
                    print("   +-----------------+------------------+", file=f)
                    print("   | Gaseous species | Partial pressure |", file=f)
                    print("   +-----------------+------------------+", file=f)
                    print("   | %15s | %16f |" % (constituents[0], constituents[1]), file=f)
                    print("   +-----------------+------------------+\n", file=f)

                    print("   Defining properties of binary species:", file=f)
                    print(
                        "   +------------------+---------------------+------------------+---------------------+---------------------------+",
                        file=f)
                    print(
                        "   | Binary species A | DFT energy (eV/pfu) | Binary species B | DFT energy (eV/pfu) | Reaction Free Energy (eV) |",
                        file=f)
                    print(
                        "   +------------------+---------------------+------------------+---------------------+---------------------------+",
                        file=f)
                    print("   | %16s | %19f | %16s | %19f | %25f |" % (
                        constituents[2], constituents[4], constituents[5], constituents[7], constituents[8]), file=f)
                    print(
                        "   +------------------+---------------------+------------------+---------------------+---------------------------+\n",
                        file=f)


                if (chem_pot_method == 5):
                    print("   Volatile-Rich-Poor method selected\n", file=f)
                    if (real_gas == 0):
                        print("   Using ideal gas specific heat", file=f)
                    elif (real_gas == 1) or (real_gas == 2):
                        print("   Using real gas relations for specific heat", file=f)
                    elif real_gas == 3:
                        print("   Using PYroMat library (NASA equation) for real gas relations", file=f)

                    print("   Defining volatile species:", file=f)
                    print("   +-----------------+---------------+------------------+", file=f)
                    print("   | Gaseous species | Stoichiometry | Partial pressure |", file=f)
                    print("   +-----------------+---------------+------------------+", file=f)
                    print("   | %15s | %13f | %16f |" % (constituents[0], constituents[1], constituents[2]), file=f)
                    print("   +-----------------+---------------+------------------+\n", file=f)

                    print("   Defining constituents of host:", file=f)
                    print(
                        "   +------------------+---------------+----------------+------------------------+--------------------------------------+----------+",
                        file=f)
                    print(
                        "   | Constituent name | Stoichiometry | DFT energy (eV)| Cation DFT energy (eV) | Formation energy of constituent (eV) | fraction |",
                        file=f)
                    print(
                        "   +------------------+---------------+----------------+------------------------+--------------------------------------+----------+",
                        file=f)

                    for i in np.arange(0, number_bin_oxides, 1):
                        i = int(i)
                        constituent_name = constituents[6 * i + 3]
                        constituent_stoich = float(constituents[6 * i + 4])
                        constituent_energy_DFT = float(constituents[6 * i + 5])
                        constituent_metal_DFT = float(constituents[6 * i + 6])
                        constituent_formation = float(constituents[6 * i + 7])
                        constituent_definition = float(constituents[6 * i + 8])

                        print("   | %16s | %13f | %14f | %22f | %36f | %6f |" % (
                        constituent_name, constituent_stoich, constituent_energy_DFT, constituent_metal_DFT,
                        constituent_formation, constituent_definition), file=f)

                    print(
                        "   +------------------+---------------+----------------+------------------------+--------------------------------------+----------+\n",
                        file=f)



                # Dopants
                print(">>> Dopants\n", file=f)
                print("   Number of dopants :", int(dopants[0]), file=f)
                if (dopants[0] != 0):
                    for i in np.arange(0, dopants[0], 1):
                        i = int(i)
                        print("\n   Dopant", i + 1, ":", file=f)
                        fit_chempot = int(dopants[6 * i + 3])
                        if fit_chempot == 0:
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+",
                                file=f)
                            print(
                                "   | Dopant element | Dopant reference | DFT energy of reference (eV)| Fitting option |",
                                file=f)
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+",
                                file=f)
                            dopant_name = dopants[6 * i + 1]
                            reference_state = dopref_name_list[i]
                            reference_energy = float(dopants[6 * i + 2])
                            fit_chempot = int(dopants[6 * i + 3])
                            print("   | %14s | %16s | %27f | %14i |" % (
                                dopant_name, reference_state, reference_energy, fit_chempot), file=f)
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+\n",
                                file=f)
                        elif fit_chempot == 1 or fit_chempot == 2:
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+--------------------------+-------------------------------+",
                                file=f)
                            print(
                                "   | Dopant element | Dopant reference | DFT energy of reference (eV)| Fitting option | Target concentration pfu | Chemical potential range (eV) |",
                                file=f)
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+--------------------------+-------------------------------+",
                                file=f)
                            dopant_name = dopants[6 * i + 1]
                            reference_state = dopref_name_list[i]
                            reference_energy = float(dopants[6 * i + 2])
                            target_conc = float(dopants[6 * i + 4])
                            dopant_range = float(dopants[6 * i + 6])
                            print("   | %14s | %16s | %27f | %14i | %24s | %29f |" % (
                                dopant_name, reference_state, reference_energy, fit_chempot,
                                "{:.10f}".format(target_conc),
                                dopant_range), file=f)
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+--------------------------+-------------------------------+\n",
                                file=f)
                        elif fit_chempot == 3 or fit_chempot == 4:
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+------------------+",
                                file=f)
                            print(
                                "   | Dopant element | Dopant reference | DFT energy of reference (eV)| Fitting option | Partial pressure |",
                                file=f)
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+------------------+",
                                file=f)
                            dopant_name = dopants[6 * i + 1]
                            reference_state = dopref_name_list[i]
                            reference_energy = float(dopants[6 * i + 2])
                            fit_chempot = int(dopants[6 * i + 3])
                            partial_pressure = float(dopants[6 * i + 6])
                            print("   | %14s | %16s | %27f | %14i | %16i |" % (
                                dopant_name, reference_state, reference_energy, fit_chempot, partial_pressure), file=f)
                            print(
                                "   +----------------+------------------+-----------------------------+----------------+------------------+\n",
                                file=f)

                if loop != 3:
                    print("   Artificial dopant concentration:", art_dop_conc, file=f)
                    print("   Artificial dopant charge:", art_dop_charge, file=f)

                if (dopant_fitting >= 1):
                    print("\n   Fitting chemical potential of", dopant_fitting, "dopants", file=f)
                    if dopant_fitting == 1:
                        print("   Using Linear Bisection", file=f)
                    else:
                        print("   Using Seqential Least Squares Programming", file=f)
                        print("   Convergence criteria for logarithmic dopant concentration : ", potential_convergence,
                              file=f)
                        print("   SLSQP precision goal : ", function_tol, file=f)
                        print("   SLSQP maximum iterations : ", maxiter_dop, file=f)
                        print("   SLSQP_dial: ", SLSQP_dial, file=f)
                else:
                    print("\n   No fitting of dopant chemical potentials selected", file=f)

                # Entropy
                print("\n>>> Entropy\n", file=f)
                if entropy_marker == 1:
                    print("   Entropy contribution ON", file=f)
                    print("   Number of functional units in supercell used to calculate entropy:", entropy_units,
                          file=f)
                else:
                    print("   Entropy contribution OFF", file=f)

                print("\n>>> Defect methodology\n", file=f)
                if def_statistics == 0:
                    print("   Defect concentration method : Boltzmann", file=f)
                elif def_statistics == 1:
                    print("   Defect concentration method : Kasamatsu", file=f)

                # Use correction schemes
                if (tab_correction == 1):
                    print("   Tab correction ON, modifier will be read for each defect from column 7 of defects file.",
                          file=f)
                else:
                    print("   Tab correction OFF", file=f)

                if (use_coul_correction == 1) and ('madelung' not in tasks):

                    print("   Makov-Payne correction ON", file=f)
                    print("   Supercell length:", length, 'Angstroms', file=f)
                    print("   Dielectric constant:", dielectric, file=f)
                    print("   Madelung constant: 2.8373", file=f)

                elif use_coul_correction == 2 and ('madelung' not in tasks):
                    print("   Screened Madelung correction ON", file=f)
                    print("   Screened Madelung potential:", v_M, file=f)

                elif ('madelung' in tasks):
                    # use_coul_correction == 2
                    print("   Screened Madelung correction ON", file=f)
                    print("   Screened Madelung potential to be calculated", file=f)

                else:
                    print("   Makov-Payne and Screened Madelung corrections OFF", file=f)

                if ('brouwer' in tasks):
                    print("\n>>> Instructions for: Task = brouwer", file=f)
                    print('\n   loop =', int(loop), file=f)
                    if (loop == 0):
                        print("   Looping over volatile partial pressure\n", file=f)

                        print("   Temperature :", temperature, "K", file=f)
                        print("   Volatile partial pressure range :", min_value, "-", max_value, "\n", file=f)
                    if (loop == 1):
                        print("   Looping over temperature\n", file=f)

                        if (chem_pot_method in volatile_chem_pot_methods):
                            print("   Volatile partial pressure :", partial_pressure, file=f)
                        print("   Temerature range :", min_value, "-", max_value, "K\n", file=f)
                    if (loop == 2):
                        print("   Looping over dopant concentration\n", file=f)

                        print("   Temperature :", temperature, "K", file=f)
                        if (chem_pot_method in volatile_chem_pot_methods):
                            print("   Volatile partial pressure :", partial_pressure, file=f)
                        print("   Target dopant concentration range :", min_value, "-", max_value, "pfu\n", file=f)
                    if (loop == 3):
                        print("   Looping over artificial dopant concentration\n", file=f)

                        print("   Temperature :", temperature, "K", file=f)
                        if (chem_pot_method in volatile_chem_pot_methods):
                            print("   Volatile partial pressure :", partial_pressure, file=f)
                        print("   Artificial dopant concentration range :", min_value, "-", max_value, "pfu\n", file=f)
                        print("   Artificial dopant charge:", art_dop_charge, file=f)
                        if plot_art_dopant_conc == 1:
                            print("   Plotting artificial dopant concentration", file=f)

                    if (loop == 4):
                        print("   Looping over dopant partial pressure\n", file=f)

                        print("   Temperature :", temperature, "K", file=f)
                        print("   Dopant partial pressure range :", min_value, "-", max_value, "\n", file=f)

                    if (loop == 5):
                        print("   Looping over rich-poor fraction\n", file=f)

                        print("   Temperature :", temperature, "K", file=f)
                        if (chem_pot_method == 5):
                            print("   Volatile partial pressure :", partial_pressure, file=f)

                    print("\n>>> Plotting preferences", file=f)

                    if y_variable == 0:
                        print('\n   Units of y axis set at \"concentration per functional unit\"', file=f)
                    if y_variable == 1:
                        print('\n   Units of y axis set at \"concentration per cm^-3\"', file=f)
                        print('   Conversion parameters:', file=f)
                        print('   Unit cell volume:', unit_vol, "Angstroms^3", file=f)
                        print('   Number of functional units in unit cell:', fu_uc, file=f)
                    print('   Minimum of y-axis set at', min_y_range, file=f)
                    print('   Maximum of y-axis set at', max_y_range, file=f)
                    if x_variable == 1:
                        print('   Plotting as a function of stoichiometery; default range -0.1 to +0.1', file=f)
                    if scheme == 0:
                        print('   Default coulour scheme will be used', file=f)
                    if scheme == 1:
                        print('   User defeined coulour scheme will be used from file.plot', file=f)
                if ('defect_phase' in tasks):

                    print("\n>>> Instructions for: Task = defect_phase", file=f)
                    print('\n   x-axis loop =', int(loop), file=f)
                    print('   y-axis loop =', int(loop2), "\n", file=f)
                    if (loop == 0):
                        print("   Looping over volatile partial pressure on x-axis", file=f)
                        print("   Volatile partial pressure range :", min_value, "-", max_value, "\n", file=f)
                    if (loop == 1):
                        print("   Looping over temperature on x-axis", file=f)
                        print("   Temerature range :", min_value, "-", max_value_y, "K\n", file=f)
                    if (loop == 2):
                        print("   Looping over dopant concentration on x-axis", file=f)
                        print("   Target dopant concentration range :", min_value, "-", max_value, "pfu\n", file=f)
                    if (loop == 3):
                        print("   Looping over artificial dopant concentration on x-axis", file=f)
                        print("   Artificial dopant concentration range :", min_value, "-", max_value, "pfu\n", file=f)
                        print("   Artificial dopant charge:", art_dop_charge, file=f)
                        if plot_art_dopant_conc == 1:
                            print("   Plotting artificial dopant concentration", file=f)
                    if (loop == 4):
                        print("   Looping over dopant partial pressure on x-axis", file=f)
                        print("   Dopant partial pressure range :", min_value, "-", max_value, "\n", file=f)
                    if (loop == 5):
                        print("   Looping over rich-poor fraction\n", file=f)
                        print("   Temperature :", temperature, "K", file=f)
                        if (chem_pot_method == 5):
                            print("   Volatile partial pressure :", partial_pressure, file=f)

                    if (loop2 == 0):
                        print("   Looping over volatile partial pressure on y-axis", file=f)
                        print("   Volatile partial pressure range :", min_value_y, "-", max_value_y, "\n", file=f)
                    if (loop2 == 1):
                        print("   Looping over temperature on y-axis", file=f)
                        print("   Temerature range :", min_value_y, "-", max_value_y, "K\n", file=f)
                    if (loop2 == 2):
                        print("   Looping over dopant concentration on y-axis", file=f)
                        print("   Target dopant concentration range :", min_value_y, "-", max_value_y, "pfu\n", file=f)
                    if (loop2 == 3):
                        print("   Looping over artificial dopant concentration on y-axis", file=f)
                        print("   Artificial dopant concentration range :", min_value_y, "-", max_value_y, "pfu\n",
                              file=f)
                        print("   Artificial dopant charge:", art_dop_charge, file=f)
                        if plot_art_dopant_conc == 1:
                            print("   Plotting artificial dopant concentration", file=f)
                    if (loop2 == 4):
                        print("   Looping over dopant partial pressure on y-axis", file=f)
                        print("   Dopant partial pressure range :", min_value_y, "-", max_value_y, "\n", file=f)
                    if (loop2 == 5):
                        print("   Looping over rich-poor fraction\n", file=f)
                        print("   Temperature :", temperature, "K", file=f)
                        if (chem_pot_method == 5):
                            print("   Volatile partial pressure :", partial_pressure, file=f)

                    if ('dopant' in tasks):
                        print("\n..> Also plotting the accommodation mechanism of", accommodate, file=f)

                if ('stability' in tasks):
                    print("\n>>> Instructions for: Task = stability", file=f)
                    print("\n   Checking the stability of", int(number_of_checks), "compounds", file=f)
                    print("\n   Compounds:", file=f)
                    print("   +------------------+-----------------+-----------------------------------+", file=f)
                    print("   |     Compound     | DFT energy (eV) | Include temperature contributions |", file=f)
                    print("   +------------------+-----------------+-----------------------------------+", file=f)
                    for i in np.arange(0, number_of_checks, 1):
                        i = int(i)
                        compound = stability[4 * i + 1]
                        compound_energy = float(stability[4 * i + 2])
                        include_heat_capacity = stability[4 * i + 4]
                        print("   | %16s | %15f | %33s |" % (compound, compound_energy, include_heat_capacity), file=f)
                    print("   +------------------+-----------------+-----------------------------------+", file=f)
                break

    print("\n..> Input file read successfully")

    return (
        host_array, dopants, tasks, constituents, constituents_name_list, temperature, def_statistics,
        tab_correction,
        host_energy, chem_pot_method, host_supercell, use_coul_correction, length, dielectric, v_M, E_VBM, bandgap,
        condband, valband, electron_method, hole_method, fixed_e_conc, fixed_p_conc, art_dop_conc, art_dop_charge, loop,
        min_value, max_value, iterator, gnuplot_version, min_y_range, max_y_range, host_name, val_band_min,
        val_band_max,
        cond_band_min, cond_band_max, y_form_min, y_form_max, lines, entropy_marker, entropy_units, fu_uc,
        electron_mass_eff, hole_mass_eff, unit_vol, charge_convergence, potential_convergence, stability, scheme,
        stoichiometry, x_variable, real_gas, function_tol, SLSQP_dial, maxiter_dop, y_variable, loop2, min_value_y,
        max_value_y, iterator_y, accommodate, gibbs_marker, plot_art_dopant_conc)


# function for breaking down chemical formula
def break_formula(formula, index):
    temp_array = []

    # Split the host definition on a hyphen
    splithost = formula.split('-')

    # Determine how many elements there are in the new array
    num_elements = len(splithost)
    if (index == 0):
        pass
    else:
        # print("Number of elements in subsystem",index,num_elements)
        pass
    temp_array.append(num_elements)

    # Now loop over the number of elements in the formula
    for i in np.arange(0, num_elements, 1):
        splitelement = splithost[i].split('_')
        element = splitelement[0]
        if (len(splitelement) == 2):
            stoich_number = float(splitelement[1])
        else:
            stoich_number = 1

        # Push details into temp_array
        temp_array.append(element)
        temp_array.append(stoich_number)

    return (temp_array)


# function to read defects file
def read_defects(seedname, elements, defects, dopants):
    total_species = int(dopants[0] + elements[0])
    charged_system = 0

    # Determine the  minimum number of columns required for each defect
    num_columns = 7 + total_species

    # Print header for defect summary table
    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:
        print("\n>>> Summary of defects:", file=f)
        print("   +------------+-----------+--------------+------+--------+-------------+------------+",
              "{0}".format('------+' * (int(total_species))), sep="", file=f)

        element_print = ''
        for i in np.arange(0, total_species, 1):
            if (i < elements[0]):
                element_print_i = (" n %2s |" % (elements[int(2 * i + 1)]))
            elif (i >= elements[0]):
                element_print_i = (" n %2s |" % dopants[int(6 * (i - elements[0]) + 1)])
            element_print += element_print_i

        print("   |   Defect   |   Group   | Multiplicity | Site | Charge | Energy /eV  | Correction |", element_print,
              sep="", file=f)

        print("   +------------+-----------+--------------+------+--------+-------------+------------+",
              "{0}".format('------+' * (int(total_species))), sep="", file=f)

        if os.path.exists(f"./{seedname}.defects"):
            defectfile = str(seedname) + ".defects"
            print("\n>>> Reading in parameters from ", defectfile)

            # Open file containing all the defect information
            file = open(defectfile)
            total_defects = 0

            for defect in file:

                fields = defect.strip().split()
                if len(fields) == 0:
                    raise Exception("<!> Blank line detected in", defectfile)

                # Prevent dopant defects inclusion if not requested in input file
                skip = 0
                if dopants[0] == 0:
                    if len(fields) > num_columns:
                        excess_columns = len(fields) - num_columns
                        for i in np.arange(0, excess_columns, 1):
                            i = int(i + 1)
                            if fields[-i] != '0':
                                skip = 1
                if skip == 1:
                    continue

                total_defects += 1
                defects.append(fields)
                if len(fields) < num_columns:
                    raise Exception(f"<!> Error : Number of columns insufficient for {fields[0]}\n"
                                    f"Did you remember to specify a column for all your dopants?")

                else:
                    defect_name = fields[0]
                    defect_group = fields[1]
                    multiplicity = float(fields[2])
                    site = int(fields[3])
                    charge = float(fields[4])
                    energy = float(fields[5])
                    tabulated_correction = float(fields[6])

                    # Quick check to see if overall system is charged
                    if (charge != 0):
                        charged_system = 1

                    # Loop over elements and dopants in the host
                    element_prints = ''
                    for i in np.arange(0, total_species, 1):
                        if (i < elements[0]):
                            element_prints_i = (" %4s |" % fields[int(7 + i)])
                        elif (i >= elements[0]):
                            element_prints_i = (" %4s |" % fields[int(7 + i)])
                        element_prints += element_prints_i

                    print("   | %10s | %9s | %12f | %4i | %6i | %11.3f | %10.3f |%2s " % (
                        defect_name, defect_group, multiplicity, site, charge, energy, tabulated_correction,
                        element_prints),
                          file=f)

            print("   +------------+-----------+--------------+------+--------+-------------+------------+",
                  "{0}".format('------+' * (int(total_species))), sep="", file=f)
            print("   Number of defects :", total_defects, file=f)


        elif os.path.exists(f"./{seedname}_defects.csv"):
            defectfile = str(seedname) + "_defects.csv"

            print("\n>>> Reading in parameters from ", defectfile)

            with open(defectfile) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                total_defects = 0
                for line_count, defect_row in enumerate(csv_reader):
                    if line_count == 0:
                        continue

                    else:

                        if len(defect_row) == 0:
                            raise Exception("<!> Blank line detected in", defectfile)

                        # Prevent dopant defects inclusion if not requested in input file
                        skip = 0
                        if dopants[0] == 0:
                            if len(defect_row) > num_columns:
                                excess_columns = len(defect_row) - num_columns
                                for i in np.arange(0, excess_columns, 1):
                                    i = int(i + 1)
                                    if defect_row[-i] != '0':
                                        skip = 1

                        if skip == 1:
                            continue

                        total_defects += 1
                        defects.append(defect_row)

                        if len(defect_row) < num_columns:
                            raise Exception(f"<!> Error : Number of columns insufficient for {defect_row[0]}\n"
                                            f"Did you remember to specify a column for all your dopants?")
                        else:

                            defect_name = defect_row[0]
                            defect_group = defect_row[1]
                            multiplicity = float(defect_row[2])
                            site = int(defect_row[3])
                            charge = float(defect_row[4])
                            energy = float(defect_row[5])
                            tabulated_correction = float(defect_row[6])

                            # Quick check to see if overall system is charged
                            if (charge != 0):
                                charged_system = 1

                            # Loop over elements and dopants in the host
                            element_prints = ''
                            for i in np.arange(0, total_species, 1):
                                if (i < elements[0]):
                                    element_prints_i = (" %4s |" % defect_row[int(7 + i)])
                                elif (i >= elements[0]):
                                    element_prints_i = (" %4s |" % defect_row[int(7 + i)])
                                element_prints += element_prints_i

                            print("   | %10s | %9s | %12f | %4i | %6i | %11.3f | %10.3f |%2s " % (
                            defect_name, defect_group, multiplicity, site, charge, energy, tabulated_correction,
                            element_prints), file=f)

                print("   +------------+-----------+--------------+------+--------+-------------+------------+",
                      "{0}".format('------+' * (int(total_species))), sep="", file=f)
                print("   Number of defects :", total_defects, file=f)

        else:
            raise Exception("ERROR! No defects file is present.\nPlease provide a .defects or _defects.csv file")

        if (charged_system == 0):
            print("   Treating system as charge neutral", file=f)

        if (charged_system == 1):
            print("   Treating system as charged", file=f)

        print("..> Defect file read successfully (", total_defects, "defects )")

    return (defects, total_defects, total_species, charged_system)


# function to read entropy data
def read_entropy(seedname, defect_data, total_defects, constituents_name_list, chem_pot_method):
    entropy_data = []

    # check if entropy data is in .entropy text file
    if os.path.exists(f"./{seedname}.entropy"):
        entropy_file = str(seedname) + ".entropy"

        # read entropy file, split and strip lines and append to list
        file = open(entropy_file)
        for line in file:
            fields = line.strip().split()
            entropy_data.append(fields)

    # check if entropy data has been provided as a csv file
    elif os.path.exists(f"./{seedname}_entropy.csv"):
        entropy_file = str(seedname) + "_entropy.csv"

        with open(entropy_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for rows in csv_reader:
                entropy_data.append(rows)

    else:
        raise Exception("<!> ERROR. No entropy file provided to include entropy contributions.\n"
                        "Please provide a .entropy or _entropy.csv file.")

    print(f"\n>>> Performing checks on {entropy_file}...")

    if chem_pot_method == 5:
        i = 0
        while i < len(constituents_name_list):
            constituent = constituents_name_list[i]
            entropy_constituent = entropy_data[0][i + 2]
            if (entropy_constituent != constituent):
                raise ValueError(f"<!> ERROR: There is a mismatch in the name for the constituent {constituent} and "
                                 f"the entropy {entropy_constituent}\n"
                                 f"Recommend you go back to ensure constituents occur in the same order in the "
                                 f".input and .entropy files and spellings are identical")

            i += 1

        i = 0
        while i < total_defects:
            defect = defect_data[i][0]
            entropy_defect = entropy_data[0][i + len(constituents_name_list) + 2]
            if (entropy_defect != defect):
                raise ValueError(f"<!> ERROR: There is a mismatch in the name for the defect formation energy "
                                 f"{defect} and the entropy {entropy_defect}\n"
                                 f"Recommend you go back to ensure defects occur in the same order in the .dat and .entropy files and spellings are identical")
            i += 1

    else:
        i = 0
        while i < total_defects:
            defect = defect_data[i][0]
            entropy_defect = entropy_data[0][i + 2]
            if (entropy_defect != defect):
                raise ValueError(f"<!> ERROR: There is a mismatch in the name for the defect formation energy "
                                 f"{defect} and the entropy {entropy_defect}\n"
                                 f"Recommend you go back to ensure defects occur in the same order in the .dat and .entropy files and spellings are identical")
            i += 1

    print("Entropy data okay!\n")

    return entropy_data


# function to read gibbs energy data file
def read_gibbs(seedname):
    gibbs_data = []

    if os.path.exists(f"./{seedname}.gibbs"):
        gibbsfile = str(seedname) + ".gibbs"

        file = open(gibbsfile)
        for line in file:
            fields = line.strip().split()
            gibbs_data.append(fields)

        return gibbs_data

    elif os.path.exists(f"./{seedname}_gibbs.csv"):
        gibbsfile = str(seedname) + "_gibbs.csv"

        with open(gibbsfile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for rows in csv_reader:
                gibbs_data.append(rows)

        return gibbs_data

    else:
        raise Exception("ERROR. No Gibbs file provided.\n"
                        "Please provide either a .gibbs or _gibbs.csv file.")


# function to calculate entropy contributions
def calc_entropy(entropy_data, temperature, total_defects, constituents_name_list, chem_pot_method, seedname, prnt):
    # Print table header to the output file

    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:
        if prnt == 1:
            print("   Vibrational entropy at", temperature, "K", file=f)
            print("   +---------+---------------------+---------------+", file=f)
            print("   |  System |   Entropy /eV K^-1  |  ds /eV K^-1  |", file=f)
            print("   +---------+---------------------+---------------+", file=f)

        i = 0
        entropies = []
        constituent_entropies = []
        num_lines = len(entropy_data)

        if chem_pot_method == 5:
            length = total_defects + len(constituents_name_list) + 1
            length2 = len(constituents_name_list) + 1
        else:
            length = total_defects + 1
            length2 = 1

        while i < length:
            # Get the system name
            system_name = entropy_data[0][i + 1]
            x = []
            y = []
            j = 0
            # extract the data as a function of temperature
            while j < num_lines - 1:
                current_temp = float(entropy_data[j + 1][0])
                if (j == 0):
                    min_temp = current_temp
                if (j == (num_lines - 2)):
                    max_temp = current_temp
                current_entropy = float(entropy_data[j + 1][i + 1])
                x.append(current_temp)
                y.append(current_entropy)
                j += 1

            # Run a quick check to see whether temperature falls in the range of the data
            if (temperature < min_temp or temperature > max_temp):
                print("<!> WARNING Temperature is outside of range with entropy data provided\n")

            # Now use extrapolation to determine entropy of the perfect/defect system
            tck = interpolate.splrep(x, y)
            final_entropy = interpolate.splev(temperature, tck)
            if system_name == "host":
                perfect_entropy = final_entropy
            if i < length2:
                ds = ' '
                if prnt == 1:
                    print("   | %7s | %19f | %13s |" % (system_name, final_entropy, ds), file=f)

                constituent_entropies.append(final_entropy * 1)

            else:
                ds = final_entropy - perfect_entropy
                entropies.append(ds)
                if prnt == 1:
                    print("   | %7s | %19f | %13f |" % (system_name, final_entropy, ds), file=f)
            i += 1
        if prnt == 1:
            print("   +---------+---------------------+---------------+\n", file=f)

    return (entropies, constituent_entropies)


# function to acquire energies from the gibbs energy data
def calc_gibbs(gibbs_data, temperature, constituents_name_list, chem_pot_method, host_energy, constituents, stability,seedname,prnt):
    # dict to add gibbs energies to for each specie
    gibbs_energies = {}

    # Print table header to the output file

    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:
        if prnt == 1:
            print("\n   Gibbs energy at", temperature, "K", file=f)
            print("   +---------------------+---------------------+", file=f)
            print("   |        System       |      Energy /eV     |", file=f)
            print("   +---------------------+---------------------+", file=f)

        i = 0
        gibbs = []

        num_lines = len(gibbs_data)

        if chem_pot_method == 5:
            length = len(constituents_name_list) + 1
        else:
            length = 1

        while i < length:
            # Get the system name
            system_name = gibbs_data[0][i + 1]
            x = []
            y = []
            j = 0
            # extract the data as a function of temperature
            while j < num_lines - 1:
                current_temp = float(gibbs_data[j + 1][0])
                if (j == 0):
                    min_temp = current_temp
                if (j == (num_lines - 2)):
                    max_temp = current_temp
                current_gibbs = float(gibbs_data[j + 1][i + 1])
                x.append(current_temp)
                y.append(current_gibbs)
                j += 1

            # Run a quick check to see whether temperature falls in the range of the data
            if (temperature < min_temp or temperature > max_temp):
                raise Exception("<!> ERROR. Temperature is outside of range of Gibbs data provided\n")

            # Now use extrapolation to determine gibbs energy at T
            tck = interpolate.splrep(x, y)
            final_gibbs = interpolate.splev(temperature, tck)

            if prnt == 1:
                print("   | %19s | %19f |" % (system_name, final_gibbs), file=f)

            if system_name.lower() == "host":
                gibbs_energies["host"] = final_gibbs * 1
                if chem_pot_method == 2:
                    # constituents[3] == final_gibbs*1
                    gibbs_energies[constituents[3]] = final_gibbs * 1

            elif system_name in constituents:
                gibbs_energies[system_name] = final_gibbs * 1
            elif system_name in stability:
                stability_eng_index = stability.index(system_name) + 1
                stability[stability_eng_index] = final_gibbs * 1
            else:
                raise ValueError("ERROR. Invalid species in gibbs file\n"
                                 f"Invalid species: {system_name}\n"
                                 f"If this is the host material, please rename to 'host'")

            i += 1
        if prnt == 1:
            print("   +---------------------+---------------------+\n", file=f)

    return (gibbs_energies)


def madelung_input(seedname):
    filename = str(seedname) + ".input"
    print("\n>>> Reading in parameters from ", filename)


    # Initialise some variables
    gamma = 0.3
    real_space = 20
    total_charge = 1
    debug = 0
    num_atoms = 1
    motif = [0, 0, 0, 1]

    lattice = None
    dielectric = None

    with open(filename) as file:
        for linenumber, line in enumerate(file):
            fields = line.strip().split()

            if len(fields) != 0:
                name = fields[0]

                # Gamma
                if (name.lower() == "gamma"):
                    gamma = float(fields[2])

                # Cutoff
                if (name.lower() == "cutoff"):
                    real_space = float(fields[2])

                # debug
                if (name.lower() == "debug"):
                    debug = float(fields[2])

                # Lattice
                if (name.lower() == "lattice"):
                    lattice = []  # Array to contain lattice parrallelpiped
                    for i in np.arange(1, 4, 1):
                        with open(filename) as file2:
                            for linenumber2, line2 in enumerate(file2):
                                fields2 = line2.strip().split()

                                if linenumber + i == linenumber2:
                                    col1 = float(fields2[0])
                                    col2 = float(fields2[1])
                                    col3 = float(fields2[2])
                                    lattice.append(col1)
                                    lattice.append(col2)
                                    lattice.append(col3)

                # Dielectric
                if (name.lower() == "dielectric"):
                    dielectric = []  # Array to contain the dielectric tensor
                    for i in np.arange(1, 4, 1):
                        with open(filename) as file4:
                            for linenumber4, line4 in enumerate(file4):
                                fields4 = line4.strip().split()

                                if linenumber + i == linenumber4:
                                    col1 = float(fields4[0])
                                    col2 = float(fields4[1])
                                    col3 = float(fields4[2])
                                    dielectric.append(col1)
                                    dielectric.append(col2)
                                    dielectric.append(col3)

                # Motif
                if (name.lower() == "motif"):
                    motif = []  # Array containing the motif
                    num_atoms = float(fields[1])
                    for i in np.arange(1, num_atoms + 1, 1):
                        with open(filename) as file3:
                            for linenumber3, line3 in enumerate(file3):
                                fields3 = line3.strip().split()

                                if linenumber + i == linenumber3:
                                    motif_x = float(fields3[0])
                                    motif_y = float(fields3[1])
                                    motif_z = float(fields3[2])
                                    charge = float(fields3[3])
                                    motif.append(motif_x)
                                    motif.append(motif_y)
                                    motif.append(motif_z)

                                    if (num_atoms == 1):
                                        charge = 1
                                        # print("Treating as a point charge, therefore, charge defined in motif is being ignored\n")

                                    else:
                                        print(motif_x, motif_y, motif_z, charge, "\n")

                                    motif.append(charge)

                                    # Calculate the total charge
                                    total_charge += charge

    if lattice is None:
        raise Exception("<!> Error. The 'lattice' parameter is not defined for the madelung task")
    if dielectric is None:
        raise Exception("<!> Error. The 'dielectric' parameter is not defined for the madelung task")

    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:
        print(
            "\n-------------------------------------------------------------------------------------------------------------------",
            "\n", file=f)
        print(">>> Task = madelung", "\n", file=f)
        print("   Real space lattice:", file=f)
        print("   %.6f  %.6f  %.6f" % (lattice[0], lattice[1], lattice[2]), file=f)
        print("   %.6f  %.6f  %.6f" % (lattice[3], lattice[4], lattice[5]), file=f)
        print("   %.6f  %.6f  %.6f" % (lattice[6], lattice[7], lattice[8]), file=f)
        print("\n   Dielectric tensor:", file=f)
        print("   %.6f  %.6f  %.6f" % (dielectric[0], dielectric[1], dielectric[2]), file=f)
        print("   %.6f  %.6f  %.6f" % (dielectric[3], dielectric[4], dielectric[5]), file=f)
        print("   %.6f  %.6f  %.6f" % (dielectric[6], dielectric[7], dielectric[8]), file=f)
        print("\n   gamma =", gamma, file=f)
        print("   Real space cutoff set to", real_space, "* longest lattice parameter", file=f)
        if (debug == 1):
            print("   Debugging settings enabled", file=f)
        if (total_charge != 0):
            print("   System has an overall charge of", total_charge, file=f)
            print("   Applying charge neutralising background jellium", file=f)

    return (dielectric, lattice, motif, gamma, real_space, num_atoms, total_charge, debug)


def calc_chemical_defined(host_array, constituents, chemical_potentials, host_energy, temperature, entropy_marker,
                          constituent_entropies, entropy_units):
    # Define a limit for the discrepancy in chemcial potentials for numerical reasons
    error = 0.001

    total_potential = 0

    # Loop over host_array and match chemical potentials
    for i in np.arange(0, host_array[0], 1):
        i = int(i)

        current_element = host_array[2 * i + 1]
        stoichiometric_number = host_array[2 * i + 2]

        # Loop over constituents and match potential
        for j in np.arange(0, host_array[0], 1):
            j = int(j)

            if (constituents[2 * j] == current_element):
                chemical_potential = float(constituents[2 * j + 1])

                chemical_potentials.append(current_element)
                chemical_potentials.append(chemical_potential)

                total_potential += (stoichiometric_number * chemical_potential)

    # Compare the total chemical potential of the constituents with the host
    difference = math.sqrt((host_energy - total_potential) ** 2)
    if (difference == 0):
        pass
    elif (difference <= error):
        print("<!> Warning : There is a small difference (<", error,
              ") between the sum of the chemical potentials and that of the host")
    else:
        raise ValueError(
            f"<!> Error : The chemical potentials for the constituents do not add up to that for the host system\n"
            f"Host: {host_energy}\n"
            f"Sum of constituents: {total_potential}")

    return (chemical_potentials)


def calc_chemical_rich_poor(host_array, constituents, chemical_potentials, host_energy, temperature, entropy_marker,
                            constituent_entropies, entropy_units):
    # Loop over host_array and match chemical potentials
    for i in np.arange(0, host_array[0], 1):
        i = int(i)

        current_element = host_array[2 * i + 1]
        stoichiometric_number = float(host_array[2 * i + 2])

        running_total = 0

        # Loop over constituents to calculate the checmial potential
        for j in np.arange(0, host_array[0], 1):
            j = int(j)

            if (constituents[3 * j] == current_element):

                rich_potential = float(constituents[3 * j + 1])
                xxx = float(constituents[3 * j + 2])
                # print("Rich potential",rich_potential,xxx)

            else:

                other = constituents[3 * j]

                other_rich = float(constituents[3 * j + 1])

                # Find the stoichiometric number for the 'other' constituent

                for k in np.arange(0, host_array[0], 1):
                    k = int(k)

                    if (other == host_array[2 * k + 1]):
                        other_stoich = host_array[2 * k + 2]

                contribution = other_stoich * other_rich
                running_total += contribution

        chemical_potential = xxx * rich_potential + (1 - xxx) * (
                (host_energy - running_total) / stoichiometric_number)
        chemical_potentials.append(current_element)
        chemical_potentials.append(chemical_potential)

    return (chemical_potentials)


def calc_chemical_volatile(host_array, constituents_name_list, constituents, chemical_potentials, host_energy, temperature, entropy_marker, constituent_entropies, entropy_units, real_gas):

    # get volatile element and partial pressure
    volatile_species = constituents[0]
    volatile_pressure = constituents[1]

    # loop through the host array and get the stoichiometries and the metal element | host_array[0] = number elements in compound
    for host_element_index in range(host_array[0]):
        element = host_array[2 * host_element_index + 1]

        if element == volatile_species:
            volatile_stoichiometry = host_array[2 * host_element_index + 2]
        else:
            metal_element = element
            metal_stoichiometry = host_array[2 * host_element_index + 2]

    # calculate std volatile chemical potential using dft energies of the compound, metal and the standard formation energy of the compound
    # 6 values for the constituent should be defined: [V, VPP, host, host_energy, metal_energy, std_form_eng]
    if len(constituents) == 6:

        constituent_array = break_formula(constituents[2], 1)

        # if using another material as a reference
        if constituent_array != host_array:
            # loop through the host array and get the stoichiometries and the metal element | host_array[0] = number elements in compound
            for constituent_element_index in range(constituent_array[0]):
                element = constituent_array[2 * constituent_element_index + 1]

                if element == volatile_species:
                    const_volatile_stoichiometry = constituent_array[2 * constituent_element_index + 2]
                else:
                    const_metal_stoichiometry = constituent_array[2 * constituent_element_index + 2]

        else:
            # use values from host otherwise
            const_volatile_stoichiometry = volatile_stoichiometry
            const_metal_stoichiometry = metal_stoichiometry

        # get host energies from the constituents list
        constituent_DFT_energy = float(constituents[3])
        metal_DFT_energy = float(constituents[4])
        constituent_std_form_energy = float(constituents[5])

        # calculate chemical potential of the volatile under standard conditions
        nu_volatile_std = (constituent_DFT_energy - (const_metal_stoichiometry * metal_DFT_energy) - constituent_std_form_energy) / const_volatile_stoichiometry

    # calculate std volatile chemical potential using dft energies of two volatile compounds with different stoichiometries and the reaction energy to form one from the other
    # 9 values should be defined: [V, VPP, compound_A, stoic_A, energy_A, compound_B, stoic_B, energy_B, reaction_eng]
    elif len(constituents) == 9:
        # get details for compound A
        compound_A = constituents[2]
        compound_A_formula = break_formula(compound_A, 1)
        compound_A_stoic = int(constituents[3])
        compound_A_energy = float(constituents[4])

        # get details of compound B
        compound_B = constituents[5]
        compound_B_formula = break_formula(compound_B, 1)
        compound_B_stoic = int(constituents[6])
        compound_B_energy = float(constituents[7])

        reaction_free_energy = constituents[8]

        # check that the stoichiometries balance
        if compound_A_stoic * compound_A_formula[2] != compound_B_stoic * compound_B_formula[2]:
            raise ValueError(f"ERROR! Cation species in Volatile-Stoichiometry method do not balance!\n"
                             f"Total cations in compound A: {compound_A_stoic}x{compound_A_formula[2]} = {compound_A_stoic * compound_A_formula[2]}\n"
                             f"Total cations in compound B: {compound_B_stoic}x{compound_B_formula[2]} = {compound_B_stoic * compound_B_formula[2]}")

        elif compound_A_stoic * compound_A_formula[4] + 1 != compound_B_stoic * compound_B_formula[4]:
            raise ValueError("ERROR! Volatile species in Volatile-Stoichiometry method do not balance!\n"
                             f"Total volatile atoms in compound A + 1/2V2: {compound_A_stoic}x{compound_A_formula[4]} + 1 = {compound_A_stoic * compound_A_formula[4] + 1}\n"
                             f"Total volatile atoms in compound B: {compound_B_stoic}x{compound_B_formula[4]} = {compound_B_stoic * compound_B_formula[4] }")
        else:
            # calculate standard chemical potential of the volatile
            nu_volatile_std = compound_B_stoic * compound_B_energy - compound_A_stoic * compound_A_energy - reaction_free_energy

    else:
        # error if 6 or 9 values not defined in the constituents
        raise TypeError("ERROR! Constituents for the volatile chem_pot_method improperly defined.\n"
                        "Please consult the manuel for help.\n"
                        f"You have {len(constituents)} values defined in the constituents. There should be 6 or 9 depending on the method you choose.")


    # calculate temperature contribution to the volatile chemical potential
    # 0.5x as temperature contribution is calculated for a diatomic gas molecule
    temperature_contribution = 0.5 * temperature_cont(volatile_species, temperature, real_gas)

    # calculate pressure contribution to the volatile chemical potential
    pressure_cont = pressure_contribution(volatile_pressure, temperature)

    # calculate volatile chemical potential at desired conditions
    nu_volatile = nu_volatile_std + temperature_contribution + pressure_cont

    # now time to calculate the chemical potential of the metal at desired conditions

    # modify the host dft energy if entropy is being included
    # constituent_entropies[0] is the entropy modification of the host
    if entropy_marker == 1:
        modification = (constituent_entropies[0] * temperature) / entropy_units
    else:
        modification = 0

    host_energy -= modification

    # final metal chemical potential
    metal_chemical_potential = (host_energy - (volatile_stoichiometry * nu_volatile)) / metal_stoichiometry

    # add element symbols and chemical potentials to the chemical potentials list
    chemical_potentials = [metal_element, metal_chemical_potential, volatile_species, nu_volatile]

    return chemical_potentials


def calc_chemical_volatile_rich_poor(host_array, constituents_name_list, constituents, chemical_potentials, host_energy,
                                     temperature,
                                     entropy_marker, constituent_entropies, entropy_units, real_gas, gibbs_energies):


    # change host_energy to the one calculated using the gibbs function if specified
    if gibbs_energies is not None:
        host_energy = gibbs_energies["host"]

    # get info on the volatile species
    volatile_species = constituents[0]
    host_volatile_stoichiometry = constituents[1]
    volatile_partial_pressure = constituents[2]

    # variable to store the sum of the rich poor fraction of the compounds that make up the host
    constituents_rich_poor_total = 0

    # variable to store the sum of the contributions of each compound to the volatile std chemical potential
    nu_volatile_std = 0


    # loop through the constituent compounds that make up the host
    for compound in constituents_name_list:
        # host: Li2TiO3 | constituents_name_list: [Li2O, TiO2]

        # skip stability check compounds
        if compound not in constituents:
            continue

        else:
            # get the index of the compound in the constituents
            compound_constituents_index = constituents.index(compound)

            # break the compound down into elements and the stoichiometries
            compound_array = break_formula(compound, 1)

            # loop through elements in this compound
            # compound_array[0] is number of elements in the compound
            for compound_element_index in range(compound_array[0]):
                element = compound_array[2 * compound_element_index + 1]

                # get stoichiometry of the volatile and metal in this compound
                if element == volatile_species:
                    compound_volatile_stoic = compound_array[2 * compound_element_index + 2]
                else:
                    metal_element = element
                    compound_metal_stoic = compound_array[2 * compound_element_index + 2]

            # get the energies of this compound and store them into variables
            compound_DFT_energy = constituents[compound_constituents_index + 2]
            compound_metal_DFT_energy = constituents[compound_constituents_index + 3]
            compound_std_form_energy = constituents[compound_constituents_index + 4]
            compound_rich_poor_frac = constituents[compound_constituents_index + 5]

            constituents_rich_poor_total += compound_rich_poor_frac

            # calculate the contribution of the compound to the volatile chemical potential
            vol_chem_pot_contribution = compound_rich_poor_frac * ((compound_DFT_energy - (
                        compound_metal_stoic * compound_metal_DFT_energy) - compound_std_form_energy) / compound_volatile_stoic)

            nu_volatile_std += vol_chem_pot_contribution

    # calc final standard volatile chemical potential
    nu_volatile_std /= constituents_rich_poor_total

    # calculate temperature contribution to the volatile chemical potential
    # 0.5x as temperature contribution is calculated for a diatomic gas molecule
    temperature_contribution = 0.5 * temperature_cont(volatile_species, temperature, real_gas)

    # calculate pressure contribution to the volatile chemical potential
    pressure_cont = pressure_contribution(volatile_partial_pressure, temperature)

    # calculate volatile chemical potential at desired conditions
    nu_volatile = nu_volatile_std + temperature_contribution + pressure_cont

    # time to calculate chemical potentials of any remaining elements

    # Modify the energy of the host to include vibrational if entropy contribution is specified
    if (entropy_marker == 1):
        modification = (constituent_entropies[0] * temperature) / entropy_units
    else:
        modification = 0

    host_energy -= modification

    # loop over remaining elements in the host
    for host_element_index in range(host_array[0] - 1):
        host_element = host_array[2 * host_element_index + 1]

        other_constituents_contribution = 0

        # loop over the constituent compounds of the host to find which compound this element is in
        for compound_index, compound in enumerate(constituents_name_list):

            # skip stability check compounds
            if compound not in constituents:
                continue
            else:
                # break formula of compound and get its index in the constituents
                compound_array = break_formula(compound, 1)
                compound_constituents_index = constituents.index(compound)

                # store energies and ratio about the compound in variables
                compound_stoichiometry = constituents[compound_constituents_index + 1]
                compound_energy = constituents[compound_constituents_index + 2]

                # add entropy contribution to compound energy if specified
                entropy_modificiation = 0
                if entropy_marker == 1:
                    # host is index zero in entropies list
                    entropy_modificiation = (constituent_entropies[compound_index + 1] * temperature) / entropy_units
                    compound_energy = constituents[compound_constituents_index + 1] - entropy_modificiation

                # change compound energy to that calculated by the gibbs function if specified
                elif gibbs_energies is not None:
                    compound_energy = gibbs_energies[compound]

                # check if current element is in this compound
                if host_element in compound_array:

                    # hold energy, stoichiometry and rich poor frac of the compound containing the element
                    constituent_energy = compound_energy
                    constituent_stoic = compound_stoichiometry
                    compound_rich_poor_frac = constituents[compound_constituents_index + 5]

                    # loop over compound elements to get element stoichiometries in this compound
                    for element_index in range(compound_array[0]):
                        compound_element = compound_array[2 * element_index + 1]

                        if compound_element == host_element:
                            element_stoichiometry = compound_array[2 * element_index + 2]
                        else:
                            compound_volatile_stoichiometry = compound_array[2 * element_index + 2]
                else:
                    # sum the contributions of compounds that do not contain this element
                    other_constituents_contribution += (compound_stoichiometry * compound_energy)

        # calculate the chemical potential of this element
        element_chemical_potential = compound_rich_poor_frac * ((constituent_energy - (compound_volatile_stoichiometry * nu_volatile)) / element_stoichiometry) + (1 - compound_rich_poor_frac) * (((host_energy - other_constituents_contribution - (host_volatile_stoichiometry * nu_volatile)) / constituent_stoic - (compound_volatile_stoichiometry * nu_volatile)) / element_stoichiometry)

        chemical_potentials.append(host_element)
        chemical_potentials.append(element_chemical_potential)

    chemical_potentials.append(volatile_species)
    chemical_potentials.append(nu_volatile)

    return chemical_potentials


def gas_thermo_values(volatile_species, temperature, real_gas):

    # ideal gas
    if real_gas == 0:
        ideal_gas_entropies = {"H": 0.00135436,
                               "N": 0.00198589,
                               "O": 0.00212622,
                               "F": 0.00210186,
                               "Cl": 0.00231205
                               }

        ideal_gas_heat_capacities = {"H": 0.000298891,
                                     "N": 0.00030187,
                                     "O": 0.000304546,
                                     "F": 0.000324774,
                                     "Cl": 0.000351828
                                     }

        return ideal_gas_entropies[volatile_species], ideal_gas_heat_capacities[volatile_species]

    # real gas relations
    elif real_gas == 1:

        if volatile_species == "H":
            if 100 <= temperature <= 1000:
                coefficients = {"aaa": 0.000342734,
                                "bbb": -0.000117783,
                                "ccc": 0.000118502,
                                "ddd": -2.87411E-05,
                                "eee": -1.64347E-06,
                                "fff": -0.000103452,
                                "ggg": 0.001790133
                                }

            elif 1000 < temperature <= 2500:
                coefficients = {"aaa": 0.000192408,
                                "bbb": 0.000127049,
                                "ccc": -2.96419E-05,
                                "ddd": 2.78031E-06,
                                "eee": 2.0502E-05,
                                "fff": -1.18933E-05,
                                "ggg": 0.00161994
                                }

            elif 2500 < temperature <= 6000:
                coefficients = {"aaa": 0.000449985,
                                "bbb": -4.44981E-05,
                                "ccc": 1.31888E-05,
                                "ddd": -1.00413E-06,
                                "eee": -0.000212835,
                                "fff": -0.000399213,
                                "ggg": 0.001679987
                                }
            else:
                raise ValueError(f"<!> Cannot use real gas parameters at temperature of {temperature} K\n"
                                 f"The temperature range for hydrogen is between 100 and 6000 K.")

        elif volatile_species == "N":
            if 100 <= temperature <= 500:
                coefficients = {"aaa": 0.000300447,
                                "bbb": 1.92166E-05,
                                "ccc": -9.99967E-05,
                                "ddd": 0.000172427,
                                "eee": 1.21271E-09,
                                "fff": -8.98851E-05,
                                "ggg": 0.002346829
                                }

            elif 500 < temperature <= 2000:
                coefficients = {"aaa": 0.00020218,
                                "bbb": 0.000206131,
                                "ccc": -8.91245E-05,
                                "ddd": 1.41979E-05,
                                "eee": 5.46863E-06,
                                "fff": -5.11538E-05,
                                "ggg": 0.00220144
                                }

            elif 2000 < temperature <= 6000:
                coefficients = {"aaa": 0.000368155,
                                "bbb": 1.16994E-05,
                                "ccc": -2.03262E-06,
                                "ddd": 1.51973E-07,
                                "eee": -4.72001E-05,
                                "fff": -0.000196635,
                                "ggg": 0.002331947
                                }
            else:
                raise ValueError(f"<!> Cannot use real gas parameters at temperature of {temperature} K\n"
                                 f"The temperature range for nitrogen is between 100 and 6000 K.")

        elif volatile_species == "F":
            if 298 <= temperature <= 6000:
                coefficients = {"aaa": 0.000325931,
                                "bbb": 8.72101E-05,
                                "ccc": -2.8803E-05,
                                "ddd": 2.26067E-06,
                                "eee": -2.18885E-06,
                                "fff": -0.000108135,
                                "ggg": 0.002459396
                                }

            else:
                raise ValueError(f"<!> Cannot use real gas parameters at temperature of {temperature} K\n"
                                 f"The temperature range for fluorine is between 298 and 6000 K.")

        elif volatile_species == "Cl":
            if 298 <= temperature <= 1000:
                coefficients = {"aaa": 0.000342572,
                                "bbb": 0.000126759,
                                "ccc": -0.000125056,
                                "ddd": 4.54543E-05,
                                "eee": -1.65317E-06,
                                "fff": -0.000112304,
                                "ggg": 0.002684858
                                }

            elif 1000 < temperature <= 3000:
                coefficients = {"aaa": 0.000442354,
                                "bbb": -5.19246E-05,
                                "ccc": 1.97416E-05,
                                "ddd": -1.71688E-06,
                                "eee": -2.17509E-05,
                                "fff": -0.00017921,
                                "ggg": 0.002796914
                                }

            elif 3000 < temperature <= 6000:
                coefficients = {"aaa": -0.000441071,
                                "bbb": 0.000432076,
                                "ccc": -7.38702E-05,
                                "ddd": 4.01998E-06,
                                "eee": 0.001048366,
                                "fff": 0.00137611,
                                "ggg": 0.002744529
                                }
            else:
                raise ValueError(f"<!> Cannot use real gas parameters at temperature of {temperature} K\n"
                                 f"The temperature range for nitrogen is between 100 and 6000 K.")

        elif volatile_species == "O":
            if 100 <= temperature <= 700:
                coefficients = {"aaa": 0.000324659,
                                "bbb": -0.000209741,
                                "ccc": 0.000599791,
                                "ddd": -0.00037839,
                                "eee": -7.64321e-08,
                                "fff": -9.22852e-05,
                                "ggg": 0.002558046
                                }

            elif 700 < temperature <= 2000:
                coefficients = {"aaa": 0.000311288,
                                "bbb": 9.09326E-05,
                                "ccc": -4.13373E-05,
                                "ddd": 8.17093E-06,
                                "eee": -7.68674E-06,
                                "fff": -0.000117381,
                                "ggg": 0.002447884
                                }

            elif 2000 < temperature <= 6000:
                coefficients = {"aaa": 0.000216745,
                                "bbb": 0.000111121,
                                "ccc": -2.09426E-05,
                                "ddd": 1.51796E-06,
                                "eee": 9.58327E-05,
                                "fff": 5.53252E-05,
                                "ggg": 0.002462936
                                }

            else:
                raise ValueError(f"<!> Cannot use real gas parameters at temperature of {temperature} K\n"
                                 f"The temperature range for oxygen is between 100 and 6000 K.")

        else:
            raise Exception(f"{volatile_species} is not available for real_gas method 1.\n"
                            f"Please consult the manual for help.")

        return coefficients

    # johnston et al for oxygen
    elif real_gas == 2:

        coefficients = {"aaa": 3.074E-4,
                        "bbb": 6.36066E-8,
                        "ccc": -1.22974E-11,
                        "ddd": 9.927E-16,
                        "eee": -2.2766,
                        "fff": -0.1022061,
                        "ggg": 0.0024661578656}

        return coefficients


def calc_volatile_gibbs_free_energy(temperature, real_gas, aaa, bbb, ccc, ddd, eee, fff, ggg):
    if real_gas == 1:
        t = temperature / 1000
        enthalpy = 1000 * (aaa * t +
                           (1 / 2) * bbb * (t ** 2) +
                           (1 / 3) * ccc * (t ** 3) +
                           (1 / 4) * ddd * (t ** 4) -
                           (eee / t) +
                           fff)

        entropy = (aaa * math.log(t) +
                   bbb * t +
                   (1 / 2) * ccc * (t ** 2) +
                   (1 / 3) * ddd * (t ** 3) -
                   eee / (2 * (t ** 2)) +
                   ggg)

        Gibbs = enthalpy - (temperature * entropy)
        return Gibbs

    elif real_gas == 2:

        Gibbs = (aaa * (temperature - (temperature * math.log(temperature / 1000))) -
                 (1 / 2) * bbb * (temperature ** 2) -
                 (1 / 6) * ccc * (temperature ** 3) -
                 (1 / 12) * ddd * (temperature ** 4) -
                 (eee / (2 * temperature)) +
                 fff -
                 ggg * temperature)

        return Gibbs


def temperature_cont(gas_species, temperature, real_gas):

    std_temp = 298.15

    # ideal gas
    if real_gas == 0:

        volatile_entropy, volatile_Cp = gas_thermo_values(gas_species, temperature, real_gas)

        temp_cont = -(volatile_entropy - volatile_Cp) * (temperature - std_temp) + volatile_Cp * temperature * math.log(temperature / std_temp)
        return temp_cont

    # real gas relations
    elif real_gas == 1 or real_gas == 2:

        shomate_coefficients = gas_thermo_values(gas_species, temperature, real_gas)

        gibbs_std = calc_volatile_gibbs_free_energy(std_temp, real_gas, **shomate_coefficients)
        gibbs = calc_volatile_gibbs_free_energy(temperature, real_gas, **shomate_coefficients)

        temp_cont = gibbs - gibbs_std
        return temp_cont

    # pyromat library
    elif real_gas == 3:

        # min temperature for many species is 300 K, so this is set as std temp
        std_temp = 300

        # define units
        pm.config['unit_energy'] = "eV"
        pm.config["unit_temperature"] = "K"
        pm.config["unit_matter"] = "n"

        molecular_vol_species_list = ["H", "N", "O", "Cl", "F"]

        # ensure constituent volatile species is defined properly for pyromat
        if gas_species in molecular_vol_species_list:
            gas_species += "_2"

        # break formula of gas molecule
        gas_array = break_formula(gas_species, 1)

        # create string input for pyromat libray
        pyromat_input = "ig."
        for index, item in enumerate(gas_array):
            if index == 0:
                continue
            elif str(item) == "1":
                continue
            else:
                if isinstance(item, float):
                    pyromat_input += str(int(item))
                else:
                    pyromat_input += item

        # eg: gas_array = [1, 'N', 1, 'O', 2]
        #     pyromat_input = "ig.NO2"
        pyromat_species_thermo_data = pm.get(pyromat_input)

        # G = H - TS
        pyromat_species_Gibbs = pyromat_species_thermo_data.h(T=temperature) - (temperature * pyromat_species_thermo_data.s(T=temperature))

        pyromat_species_Gibbs_std = pyromat_species_thermo_data.h(T=std_temp) - (std_temp * pyromat_species_thermo_data.s(T=std_temp))

        temp_cont = (pyromat_species_Gibbs - pyromat_species_Gibbs_std)

        return temp_cont[0]


def pressure_contribution(volatile_PP, temperature):
    std_pressure_atm = 1
    boltzmann = 0.000086173324

    # Change partial pressure from a log to atm
    partial_pressure_atm = 1 / (10 ** -volatile_PP)
    pressure_cont = (1 / 2) * boltzmann * temperature * math.log(partial_pressure_atm / std_pressure_atm)

    return pressure_cont


def stability_check(stability, chemical_potentials, indicator, temperature, real_gas):
    stability_printout = []

    # Loop over all stability check constituents
    for i in np.arange(0, stability[0], 1):
        i = int(i)
        constituent = stability[4 * i + 1]
        supplied_energy = stability[4 * i + 2]
        constituent_breakdown = stability[4 * i + 3]
        temperature_contribution_tag = stability[4 * i + 4]
        contribution = 0
        stability_printout_i = []

        # Loop over elements in each constituent
        for j in np.arange(0, constituent_breakdown[0], 1):
            j = int(j)

            element = constituent_breakdown[2 * j + 1]
            stoic = float(constituent_breakdown[2 * j + 2])

            # Search chemical potentials for matching element
            for k in np.arange(0, len(chemical_potentials) / 2, 1):
                k = int(k)
                element_i = chemical_potentials[2 * k]
                pot = chemical_potentials[2 * k + 1]

                if element == element_i:
                    contribution += (pot * stoic)

        # add temperature contributions to supplied energy if specified
        if temperature_contribution_tag.lower() == "true" or temperature_contribution_tag.lower() == "t":
            calc_temperature_contribution = temperature_cont(constituent, temperature, real_gas)
            supplied_energy += calc_temperature_contribution


        # secondary phase thermodynamically feasible if the sum of chemical potentials is greater than the supplied energy
        if contribution > supplied_energy:
            entry = "True"
            indicator = 1
        else:
            entry = "False"

        stability_printout_i.append(constituent)
        stability_printout_i.append(supplied_energy)
        stability_printout_i.append(contribution)
        stability_printout_i.append(contribution - supplied_energy)
        stability_printout_i.append(entry)
        stability_printout.append(stability_printout_i)

    return stability_printout, indicator


def dopant_chemical(dopants, chemical_potentials, temperature, real_gas):
    # Some constants
    std_pressue = 0.2
    boltzmann = 0.000086173324

    number_dopants = dopants[0]
    opt_chem_pot = 0
    # Loop over all dopants

    for i in np.arange(0, number_dopants, 1):

        running_pot_total = 0

        target = dopants[int((6 * i) + 1)]
        reference_state_energy = float(dopants[int((6 * i) + 2)])
        potential_method = int(dopants[int((6 * i) + 3)])
        reference_breakdown = dopants[int((6 * i) + 5)]
        num_element_ref = reference_breakdown[0]

        # identify if optimise of dopant chemical potential is requested.
        if potential_method == 1 or potential_method == 2:
            opt_chem_pot = 1

        if potential_method == 3 or potential_method == 4:
            partial_pressure = float(dopants[int((6 * i) + 6)])

        if potential_method != 3 or potential_method != 4:
            # Loop over elements in reference state
            for j in np.arange(0, num_element_ref, 1):

                element = reference_breakdown[int((2 * j) + 1)]

                if (element == target):

                    denominator = float(reference_breakdown[int((2 * j) + 2)])

                else:

                    stoich_number = float(reference_breakdown[int((2 * j) + 2)])

                    # Find the chemical potential for element in chemical_potentials
                    elements_in_list = (len(chemical_potentials)) / 2
                    for w in np.arange(0, elements_in_list, 1):
                        ref_element = chemical_potentials[int(2 * w)]

                        if (ref_element == element):
                            contribution = stoich_number * float(chemical_potentials[int(2 * w + 1)])
                            running_pot_total += contribution

            final_chemical = (reference_state_energy - running_pot_total) / denominator

        if potential_method == 3 or potential_method == 4:
            # Loop over elements in reference state
            for j in np.arange(0, num_element_ref, 1):

                element = reference_breakdown[int((2 * j) + 1)]

                if (element == target):

                    denominator = float(reference_breakdown[int((2 * j) + 2)])

                else:

                    stoich_number = float(reference_breakdown[int((2 * j) + 2)])

                    # Find the chemical potential for element in chemical_potentials
                    elements_in_list = (len(chemical_potentials)) / 2
                    for w in np.arange(0, elements_in_list, 1):
                        ref_element = chemical_potentials[int(2 * w)]

                        if (ref_element == element):
                            contribution = stoich_number * float(chemical_potentials[int(2 * w + 1)])
                            running_pot_total += contribution

            nu_volatile_std = (reference_state_energy - running_pot_total) / denominator

            temp_cont = 0.5 * temperature_cont(target, temperature, real_gas)

            # Change partial pressure from a log to atm
            pres_cont = pressure_contribution(partial_pressure, temperature)

            # Calcate volatile element chemical potential under desired conditions
            final_chemical = nu_volatile_std + temp_cont + pres_cont


        chemical_potentials.append(target)
        chemical_potentials.append(final_chemical)

    return chemical_potentials, opt_chem_pot


def calc_opt_chem_pot(b, loop, defects, dopants, chemical_potentials, number_of_defects, host_supercell, tab_correction,
                      E_VBM, total_species, use_coul_correction, length, dielectric, v_M, bandgap, temperature,
                      def_statistics, nu_e, condband, valband, electron_method, hole_method, fixed_e_conc, fixed_p_conc,
                      art_dop_conc, art_dop_charge, charge_convergence, val_band_min, val_band_max, cond_band_min,
                      cond_band_max, seedname, entropies, entropy_marker, fu_uc, electron_mass_eff, hole_mass_eff,
                      unit_vol,
                      charged_sys, log_diff_conv, function_tol, maxiter_dop, environment, prog_meter, prog_bar,
                      num_iter, real_gas, SLSQP_dial, constituents_name_list, dos_data_lst):

    number_dopants = int(dopants[0])
    host_atoms = total_species - number_dopants

    # Determine number and postion of elements that are to be fitted
    position = []

    for i in np.arange(0, number_dopants, 1):

        fit_potential = float(dopants[int((6 * i) + 3)])
        if fit_potential != 0:
            position.append(i + host_atoms)

    # Determine number of fitted dopant(s) defects
    if len(defects[0]) == 7 + total_species:

        signals_master = []

        for j in np.arange(0, number_of_defects, 1):
            signals = []
            signal_ii = 0
            for k in np.arange(0, number_dopants, 1):
                signal_i = float(defects[int(j)][int(7 + host_atoms + k)])
                if signal_i != 0:
                    signal = -1
                    signal_ii = -signal_i

                else:
                    signal = 0

                signals.append(signal)

            # Check to see whether this signal has been found before
            if (signals in signals_master):
                defects[int(j)].append(signals_master.index(signals))  # Defects with the same 'signal' summed later
                defects[int(j)].append(signals)  # Used to retrieve correct dopant sum.
                defects[int(j)].append(signal_ii)
            else:
                signals_master.append(signals)
                defects[int(j)].append(signals_master.index(signals))
                defects[int(j)].append(signals)
                defects[int(j)].append(signal_ii)

    dp_list = []
    for w in np.arange(0, len(position), 1):
        w = int(w)
        dp = chemical_potentials[int(2 * (position[w]))]
        dp_list.append(dp)

    # One dopant to optimise
    if len(position) == 1:
        optimiser = 1

        # Extract dopant to optimise, target conc and range
        nudp = chemical_potentials[int(2 * (position[0]) + 1)]

        target_conc = float(dopants[int((6 * (position[0] - host_atoms)) + 4)])
        dopant_range = float(dopants[int((6 * (position[0] - host_atoms)) + 6)])

        # Create 'key' that corresponds to the dopant defects in .defect file.
        key = number_dopants * [0]
        key[(int(position[0] - host_atoms))] = -1

        i = nudp - dopant_range
        j = nudp + dopant_range
        bnds = [(i, j)]
        conc_diff = 1
        log_conc_diff = 1
        iteration = 1

        # check root lies within bounds
        while True:
            chemical_potentials[int(2 * (position[0]) + 1)] = i
            defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction, E_VBM,
                                           total_species, use_coul_correction, length, dielectric, v_M, 1)

            try:
                (nu_e_final, concentrations, dopant_concentration_sums, fail) = calc_fermi_dopopt(b, loop, defects,
                                                                                                  defects_form,
                                                                                                  number_of_defects, bandgap,
                                                                                                  temperature, def_statistics,
                                                                                                  nu_e, condband, valband,
                                                                                                  electron_method, hole_method,
                                                                                                  fixed_e_conc, fixed_p_conc,
                                                                                                  art_dop_conc, art_dop_charge,
                                                                                                  charge_convergence,
                                                                                                  val_band_min, val_band_max,
                                                                                                  cond_band_min, cond_band_max,
                                                                                                  seedname, entropies,
                                                                                                  entropy_marker, fu_uc,
                                                                                                  electron_mass_eff,
                                                                                                  hole_mass_eff, unit_vol,
                                                                                                  charged_sys, number_dopants,
                                                                                                  bnds, dp_list, optimiser,
                                                                                                  log_diff_conv, function_tol,
                                                                                                  maxiter_dop, environment,
                                                                                                  prog_meter, prog_bar,
                                                                                                  num_iter, real_gas,
                                                                                                  constituents_name_list, 1, dos_data_lst)
            except:
                i += 0.05

            else:
                break

        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key) - 1)]
        lower = dopant_concentration_sum - target_conc


        while True:
            chemical_potentials[int(2 * (position[0]) + 1)] = j
            defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction, E_VBM,
                                           total_species, use_coul_correction, length, dielectric, v_M, 1)
            try:
                (nu_e_final, concentrations, dopant_concentration_sums, fail) = calc_fermi_dopopt(b, loop, defects,
                                                                                                  defects_form,
                                                                                                  number_of_defects, bandgap,
                                                                                                  temperature, def_statistics,
                                                                                                  nu_e, condband, valband,
                                                                                                  electron_method, hole_method,
                                                                                                  fixed_e_conc, fixed_p_conc,
                                                                                                  art_dop_conc, art_dop_charge,
                                                                                                  charge_convergence,
                                                                                                  val_band_min, val_band_max,
                                                                                                  cond_band_min, cond_band_max,
                                                                                                  seedname, entropies,
                                                                                                  entropy_marker, fu_uc,
                                                                                                  electron_mass_eff,
                                                                                                  hole_mass_eff, unit_vol,
                                                                                                  charged_sys, number_dopants,
                                                                                                  bnds, dp_list, optimiser,
                                                                                                  log_diff_conv, function_tol,
                                                                                                  maxiter_dop, environment,
                                                                                                  prog_meter, prog_bar,
                                                                                                  num_iter, real_gas,
                                                                                                  constituents_name_list, 1, dos_data_lst)
            except:
                j -= 0.05

            else:
                break


        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key) - 1)]

        upper = dopant_concentration_sum - target_conc

        if lower * upper > 0:
            raise ValueError(
                f"<!> No chemical potential in the specific range can give the requested defect concentration!\n"
                f"You may need to increase the dopant chemical potential range from its current value of {dopant_range}\n"
                f"Or, you may need to may need to shift your reference chemical potential from {float(dopants[int((6 * (position[0] - host_atoms)) + 2)])}\n"
                f"if this fails you may need to revisit the chemical potential from which your dopant chemical potential is derived\n\n"
                f"The difference between the calculated dopant concentration and the target concentration at the upper and lower bounds of your chemical potential range are:\n"
                f"> Upper ({j:.3f} eV): {upper}\n"
                f"> Lower ({i:.3f} eV): {lower}")

        # Perform linear biesction search to find the chemical potential that gives the desired dopant concentration
        while (log_conc_diff > log_diff_conv):
            midpoint = (i + j) / 2
            chemical_potentials[int(2 * (position[0]) + 1)] = midpoint
            x = [midpoint]
            defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction, E_VBM,
                                           total_species, use_coul_correction, length, dielectric, v_M, 1)

            (nu_e_final, concentrations, dopant_concentration_sums, fail) = calc_fermi_dopopt(b, loop, defects,
                                                                                              defects_form,
                                                                                              number_of_defects,
                                                                                              bandgap, temperature,
                                                                                              def_statistics, nu_e,
                                                                                              condband, valband,
                                                                                              electron_method,
                                                                                              hole_method, fixed_e_conc,
                                                                                              fixed_p_conc,
                                                                                              art_dop_conc,
                                                                                              art_dop_charge,
                                                                                              charge_convergence,
                                                                                              val_band_min,
                                                                                              val_band_max,
                                                                                              cond_band_min,
                                                                                              cond_band_max, seedname,
                                                                                              entropies, entropy_marker,
                                                                                              fu_uc, electron_mass_eff,
                                                                                              hole_mass_eff, unit_vol,
                                                                                              charged_sys,
                                                                                              number_dopants, bnds,
                                                                                              dp_list, optimiser,
                                                                                              log_diff_conv,
                                                                                              function_tol, maxiter_dop,
                                                                                              environment, prog_meter,
                                                                                              prog_bar, num_iter,
                                                                                              real_gas, constituents_name_list,
                                                                                              1, dos_data_lst)

            dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key) - 1)]

            conc_diff = dopant_concentration_sum - target_conc

            log_conc_diff = ((math.log(target_conc) - math.log(dopant_concentration_sum)) ** 2) ** 0.5
            if (lower * conc_diff < 0):
                j = midpoint
                upper = conc_diff
            if (upper * conc_diff < 0):
                i = midpoint
                lower = conc_diff

            # print(iteration,i,j,dopant_concentration_sum, conc_diff)
            iteration += 1

        # Result:
        nudp_final = midpoint

    # Multiple dopant elements to optimise
    else:
        optimiser = 0

        completion = 0
        prior_fail_con = 0
        iteration_relaunch = 0
        while completion < 1:

            # Extract dopant to optimise, target conc and range

            relaunch_triggered = 0
            nudp_list = []
            target_conc_list = []
            dopant_range_list = []
            key_list = []
            bnds = []
            global iteration_slsqp
            iteration_slsqp = 0
            for k in np.arange(0, len(position), 1):
                k = int(k)

                nudp = chemical_potentials[int(2 * (position[k]) + 1)]
                target_conc = float(dopants[int((6 * (position[k] - host_atoms)) + 4)])
                dopant_range = float(dopants[int((6 * (position[k] - host_atoms)) + 6)])
                key = number_dopants * [0]
                key[(int(position[k] - host_atoms))] = -1
                lower = nudp - dopant_range
                upper = nudp + dopant_range
                bnd = (lower, upper)
                nudp_list.append(nudp)

                target_conc_list.append(target_conc)
                dopant_range_list.append(dopant_range)
                key_list.append(key)
                bnds.append(bnd)

            # Initial guess
            x0 = nudp_list

            # Impose constraints
            cons = ({'type': 'ineq', 'fun': constraint,
                     'args': [target_conc_list, b, loop, number_of_defects, host_supercell, tab_correction, E_VBM, total_species,
               use_coul_correction, length, dielectric, v_M, bandgap, temperature, def_statistics, nu_e, condband,
               valband, electron_method, hole_method, fixed_e_conc, fixed_p_conc, art_dop_conc, art_dop_charge,
               charge_convergence, val_band_min, val_band_max, cond_band_min, cond_band_max, seedname, entropies,
               entropy_marker, fu_uc,
               electron_mass_eff, hole_mass_eff, unit_vol, charged_sys, number_dopants, position, key_list, bnds,
               dp_list, optimiser, environment, prog_meter, prog_bar, num_iter, log_diff_conv, function_tol,
               maxiter_dop, real_gas, chemical_potentials, defects, constituents_name_list, dos_data_lst]})

            # Minimise function. (Minimising the difference between each dopant concentration and its target)
            sol = minimize(calc_opt_chem_pot_multidim, x0, args=(
                target_conc_list, b, loop, number_of_defects, host_supercell, tab_correction, E_VBM,
                total_species, use_coul_correction, length, dielectric, v_M, bandgap, temperature,
                def_statistics, nu_e, condband, valband, electron_method, hole_method, fixed_e_conc,
                fixed_p_conc, art_dop_conc, art_dop_charge, charge_convergence, val_band_min,
                val_band_max, cond_band_min, cond_band_max, seedname, entropies, entropy_marker, fu_uc,
                electron_mass_eff, hole_mass_eff, unit_vol, charged_sys, number_dopants, position,
                key_list, bnds, dp_list, optimiser, environment, prog_meter, prog_bar, num_iter,
                log_diff_conv, function_tol, maxiter_dop, real_gas, iteration_relaunch,
                chemical_potentials, defects, constituents_name_list, dos_data_lst), method='SLSQP', bounds=bnds, constraints=cons,
                           options={'ftol': function_tol, 'disp': False, 'maxiter': maxiter_dop})

            relaunch_triggered = globals()['relaunch_triggered']

            if relaunch_triggered == 1:
                chemical_potentials, dopants, iteration_relaunch, prior_fail_con = relaunch(b, loop, defects, dopants,
                                                                                            chemical_potentials,
                                                                                            number_of_defects,
                                                                                            host_supercell,
                                                                                            tab_correction, E_VBM,
                                                                                            total_species,
                                                                                            use_coul_correction, length,
                                                                                            dielectric, v_M, bandgap,
                                                                                            temperature, def_statistics,
                                                                                            nu_e, condband, valband,
                                                                                            electron_method,
                                                                                            hole_method, fixed_e_conc,
                                                                                            fixed_p_conc, art_dop_conc,
                                                                                            art_dop_charge,
                                                                                            charge_convergence,
                                                                                            val_band_min, val_band_max,
                                                                                            cond_band_min,
                                                                                            cond_band_max, seedname,
                                                                                            entropies, fu_uc,
                                                                                            electron_mass_eff,
                                                                                            hole_mass_eff, unit_vol,
                                                                                            charged_sys, log_diff_conv,
                                                                                            function_tol, maxiter_dop,
                                                                                            environment, prog_meter,
                                                                                            prog_bar, num_iter,
                                                                                            real_gas, number_dopants,
                                                                                            x0, bnds, dp_list,
                                                                                            iteration_relaunch, 1, 0,
                                                                                            x0, prior_fail_con,
                                                                                            position, SLSQP_dial)

            if relaunch_triggered == 1:
                pass
            else:

                # Solution
                xOpt = sol.x

                # Check the output of the optimiser that concentrations are correct.
                for w in np.arange(0, len(position), 1):
                    w = int(w)
                    chemical_potentials[int(2 * (position[w]) + 1)] = xOpt[w]

                defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction, E_VBM,
                                               total_species, use_coul_correction, length, dielectric, v_M, 1)
                (nu_e_final, concentrations, dopant_concentration_sums, fail) = calc_fermi_dopopt(b, loop, defects,
                                                                                                  defects_form,
                                                                                                  number_of_defects,
                                                                                                  bandgap, temperature,
                                                                                                  def_statistics, nu_e,
                                                                                                  condband, valband,
                                                                                                  electron_method,
                                                                                                  hole_method,
                                                                                                  fixed_e_conc,
                                                                                                  fixed_p_conc,
                                                                                                  art_dop_conc,
                                                                                                  art_dop_charge,
                                                                                                  charge_convergence,
                                                                                                  val_band_min,
                                                                                                  val_band_max,
                                                                                                  cond_band_min,
                                                                                                  cond_band_max,
                                                                                                  seedname, entropies,
                                                                                                  entropy_marker, fu_uc,
                                                                                                  electron_mass_eff,
                                                                                                  hole_mass_eff,
                                                                                                  unit_vol, charged_sys,
                                                                                                  number_dopants, bnds,
                                                                                                  dp_list, optimiser,
                                                                                                  log_diff_conv,
                                                                                                  function_tol,
                                                                                                  maxiter_dop,
                                                                                                  environment,
                                                                                                  prog_meter, prog_bar,
                                                                                                  num_iter, real_gas,
                                                                                                  constituents_name_list, xOpt, dos_data_lst)

                minimise_array = len(position) * [
                    1]  # This array will hold the difference between each dopants current concentration and its target concentration

                for j in np.arange(0, len(position), 1):
                    j = int(j)
                    key = key_list[j]

                    if dopant_concentration_sums == 0:
                        pass
                    else:
                        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key) - 1)]
                        target_conc = target_conc_list[j]
                        # minimise_array[j] = (((target_conc)-(dopant_concentration_sum))**2)
                        minimise_array[j] = ((math.log(target_conc) - math.log(dopant_concentration_sum)) ** 2) ** 0.5

                product = max(minimise_array)

                if product > (2 * log_diff_conv):

                    chemical_potentials, dopants, iteration_relaunch, prior_fail_con = relaunch(b, loop, defects,
                                                                                                dopants,
                                                                                                chemical_potentials,
                                                                                                number_of_defects,
                                                                                                host_supercell,
                                                                                                tab_correction, E_VBM,
                                                                                                total_species,
                                                                                                use_coul_correction,
                                                                                                length, dielectric, v_M,
                                                                                                bandgap, temperature,
                                                                                                def_statistics, nu_e,
                                                                                                condband, valband,
                                                                                                electron_method,
                                                                                                hole_method,
                                                                                                fixed_e_conc,
                                                                                                fixed_p_conc,
                                                                                                art_dop_conc,
                                                                                                art_dop_charge,
                                                                                                charge_convergence,
                                                                                                val_band_min,
                                                                                                val_band_max,
                                                                                                cond_band_min,
                                                                                                cond_band_max, seedname,
                                                                                                entropies, fu_uc,
                                                                                                electron_mass_eff,
                                                                                                hole_mass_eff, unit_vol,
                                                                                                charged_sys,
                                                                                                log_diff_conv,
                                                                                                function_tol,
                                                                                                maxiter_dop,
                                                                                                environment, prog_meter,
                                                                                                prog_bar, num_iter,
                                                                                                real_gas,
                                                                                                number_dopants, xOpt,
                                                                                                bnds, dp_list,
                                                                                                iteration_relaunch, 2,
                                                                                                product, x0,
                                                                                                prior_fail_con,
                                                                                                position, SLSQP_dial)

                else:
                    completion = 1

                if environment == 'energy':
                    print("\n")  # Improving printout

    # Return chemical potential array, with optimised dopant chemical potential now included.

    return chemical_potentials


def calc_opt_chem_pot_multidim(x, target_conc_list, b, loop, number_of_defects, host_supercell, tab_correction, E_VBM,
                               total_species, use_coul_correction, length, dielectric, v_M, bandgap, temperature,
                               def_statistics, nu_e, condband, valband, electron_method, hole_method, fixed_e_conc,
                               fixed_p_conc, art_dop_conc, art_dop_charge, charge_convergence, val_band_min,
                               val_band_max, cond_band_min, cond_band_max, seedname, entropies, entropy_marker, fu_uc,
                               electron_mass_eff, hole_mass_eff, unit_vol, charged_sys, number_dopants, position,
                               key_list, bnds, dp_list, optimiser, environment, prog_meter, prog_bar, num_iter,
                               log_diff_conv, function_tol, maxiter_dop, real_gas, iteration_relaunch,
                               chemical_potentials, defects, constituents_name_list, dos_data_lst):
    global relaunch_triggered
    relaunch_triggered = 0
    for i in np.arange(0, len(position), 1):  # Update Chemical potentials with new trial
        i = int(i)
        chemical_potential = x[i]

        chemical_potentials[int(2 * (position[i]) + 1)] = chemical_potential

    defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction,
                                   E_VBM, total_species, use_coul_correction, length, dielectric, v_M, 1)
    (nu_e_final, concentrations, dopant_concentration_sums, fail) = calc_fermi_dopopt(b, loop, defects, defects_form,
                                                                                      number_of_defects, bandgap,
                                                                                      temperature, def_statistics, nu_e,
                                                                                      condband, valband,
                                                                                      electron_method, hole_method,
                                                                                      fixed_e_conc, fixed_p_conc,
                                                                                      art_dop_conc, art_dop_charge,
                                                                                      charge_convergence, val_band_min,
                                                                                      val_band_max, cond_band_min,
                                                                                      cond_band_max, seedname,
                                                                                      entropies, entropy_marker, fu_uc,
                                                                                      electron_mass_eff, hole_mass_eff,
                                                                                      unit_vol, charged_sys,
                                                                                      number_dopants, bnds, dp_list,
                                                                                      optimiser, log_diff_conv,
                                                                                      function_tol, maxiter_dop,
                                                                                      environment, prog_meter, prog_bar,
                                                                                      num_iter, real_gas,
                                                                                      constituents_name_list, x, dos_data_lst)

    if fail == 'relaunch':
        # global relaunch_triggered
        relaunch_triggered = 1

        return 0
    minimise_array = len(position) * [
        0]  # This array will hold the difference between each dopants current concentration and its target concentration
    minimise_array2 = len(position) * [0]

    for j in np.arange(0, len(position), 1):
        j = int(j)
        key = key_list[j]
        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key) - 1)]
        target_conc = target_conc_list[j]
        minimise_array[j] = (abs((target_conc) - (
            dopant_concentration_sum)))  # Will minimise the maximum value in this array, aiming for all to be zero.
        minimise_array2[j] = ((math.log(target_conc) - math.log(dopant_concentration_sum)) ** 2) ** 0.5

    global iteration_slsqp
    iteration_slsqp += 1

    if environment == 'energy':
        print("..> SLSQP:", iteration_relaunch + 1, ",", iteration_slsqp, ", max(log10([target])-log10([present])):",
              max(minimise_array2), '        ', end="\r", flush=True)

    else:
        print("..> Calculating defect concentrations for", environment, prog_meter, "of", num_iter,
              " [{0}]   ".format('#' * (prog_bar) + ' ' * (25 - prog_bar)), "SLSQP:", iteration_relaunch + 1, ",",
              iteration_slsqp, ",", max(minimise_array2), '       ', end="\r", flush=True)

    return max(minimise_array)


def constraint(x, target_conc_list, b, loop, number_of_defects, host_supercell, tab_correction, E_VBM, total_species,
               use_coul_correction, length, dielectric, v_M, bandgap, temperature, def_statistics, nu_e, condband,
               valband, electron_method, hole_method, fixed_e_conc, fixed_p_conc, art_dop_conc, art_dop_charge,
               charge_convergence, val_band_min, val_band_max, cond_band_min, cond_band_max, seedname, entropies,
               entropy_marker, fu_uc,
               electron_mass_eff, hole_mass_eff, unit_vol, charged_sys, number_dopants, position, key_list, bnds,
               dp_list, optimiser, environment, prog_meter, prog_bar, num_iter, log_diff_conv, function_tol,
               maxiter_dop, real_gas, chemical_potentials, defects, constituents_name_list, dos_data_lst):
    global relaunch_triggered
    relaunch_triggered = 0
    for i in np.arange(0, len(position), 1):  # Update Chemical potentials with new trial
        i = int(i)
        chemical_potential = x[i]
        chemical_potentials[int(2 * (position[i]) + 1)] = chemical_potential

    defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction,
                                   E_VBM, total_species, use_coul_correction, length, dielectric, v_M, 1)
    (nu_e_final, concentrations, dopant_concentration_sums, fail) = calc_fermi_dopopt(b, loop, defects, defects_form,
                                                                                      number_of_defects, bandgap,
                                                                                      temperature, def_statistics, nu_e,
                                                                                      condband, valband,
                                                                                      electron_method, hole_method,
                                                                                      fixed_e_conc, fixed_p_conc,
                                                                                      art_dop_conc, art_dop_charge,
                                                                                      charge_convergence, val_band_min,
                                                                                      val_band_max, cond_band_min,
                                                                                      cond_band_max, seedname,
                                                                                      entropies, entropy_marker, fu_uc,
                                                                                      electron_mass_eff, hole_mass_eff,
                                                                                      unit_vol, charged_sys,
                                                                                      number_dopants, bnds, dp_list,
                                                                                      optimiser, log_diff_conv,
                                                                                      function_tol, maxiter_dop,
                                                                                      environment, prog_meter, prog_bar,
                                                                                      num_iter, real_gas,
                                                                                      constituents_name_list, x, dos_data_lst)

    if fail == 'relaunch':
        # global relaunch_triggered
        relaunch_triggered = 1

        return 0
    minimise_array = len(position) * [
        0]  # This array will hold the difference between each dopants current concentration and its target concentration

    for j in np.arange(0, len(position), 1):
        j = int(j)
        key = key_list[j]
        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key) - 1)]
        target_conc = target_conc_list[j]
        minimise_array[j] = ((math.log(target_conc) - math.log(dopant_concentration_sum)) ** 2) ** 0.5
        product = sum(minimise_array)  # Will minimise the sum of the array, aiming for all to be zero.

    return (log_diff_conv - (number_dopants * sum(minimise_array)))


def relaunch(b, loop, defects, dopants, chemical_potentials, number_of_defects, host_supercell, tab_correction, E_VBM,
             total_species, use_coul_correction, length, dielectric, v_M, bandgap, temperature, def_statistics, nu_e,
             condband, valband, electron_method, hole_method, fixed_e_conc, fixed_p_conc, art_dop_conc, art_dop_charge,
             charge_convergence, val_band_min, val_band_max, cond_band_min, cond_band_max, seedname, entropies, fu_uc,
             electron_mass_eff, hole_mass_eff, unit_vol, charged_sys, log_diff_conv, function_tol, maxiter_dop,
             environment, prog_meter, prog_bar, num_iter, real_gas, number_dopants, x, bnds, dp_list,
             iteration_relaunch, fail_con, product, prior_chemical_potentials, prior_fail_con, position, SLSQP_dial):
    # This function returns new initial guesses to the SLSQP optimser, based upon the exit mode of the previous attempt.
    iteration_relaunch += 1
    if iteration_relaunch == 1000:
        raise Exception(dopant_fail(x, bnds, dp_list, fail_con))
    if iteration_relaunch == 1:
        global ex_product
        ex_product = product
    if fail_con == 1:
        host_atoms = total_species - number_dopants
        del chemical_potentials[-2 * number_dopants:]
        chemical_potentials, opt_chem_pot = dopant_chemical(dopants, chemical_potentials, temperature, real_gas)

    if (iteration_relaunch % 2) == 0:
        multi_num = 1
    else:
        multi_num = -1

    for i in np.arange(0, len(position), 1):
        i = int(i)
        bound_indicator = 0

        if (iteration_relaunch == 2) and (product > ex_product):
            chemical_potentials[int(2 * (position[i]) + 1)] = chemical_potentials[
                                                                  int(2 * (position[i]) + 1)] - 0.1 * abs(
                bnds[i][1] - bnds[i][0])

        else:

            if (fail_con == 1) and (prior_fail_con != 2):
                if (x[i] == bnds[i][0]):

                    dopants[6 * i + 6] = 0.9 * dopants[6 * i + 6]
                elif (x[i] == bnds[i][1]):
                    dopants[6 * i + 6] = 0.9 * dopants[6 * i + 6]


                elif (x[i] > chemical_potentials[int(2 * (position[i]) + 1)]) or (
                        x[i] < chemical_potentials[int(2 * (position[i]) + 1)]):
                    chemical_potentials[int(2 * (position[i]) + 1)] = chemical_potentials[
                                                                          int(2 * (position[i]) + 1)] + multi_num * (
                                                                              iteration_relaunch / SLSQP_dial)

            elif (fail_con == 1) and (prior_fail_con == 2):

                if x[i] > chemical_potentials[int(2 * (position[i]) + 1)]:
                    chemical_potentials[int(2 * (position[i]) + 1)] = chemical_potentials[
                                                                          int(2 * (position[i]) + 1)] + 1.1 * abs(
                        x[i] - chemical_potentials[int(2 * (position[i]) + 1)])
                elif x[i] < chemical_potentials[int(2 * (position[i]) + 1)]:
                    chemical_potentials[int(2 * (position[i]) + 1)] = chemical_potentials[
                                                                          int(2 * (position[i]) + 1)] - 1.1 * abs(
                        x[i] - chemical_potentials[int(2 * (position[i]) + 1)])

            elif fail_con == 2:
                if (x[i] == bnds[i][0]):
                    bound_indicator = 1
                    dopants[6 * i + 6] = 1.1 * dopants[6 * i + 6]
                elif (x[i] == bnds[i][1]):
                    bound_indicator = 1

                    dopants[6 * i + 6] = 1.1 * dopants[6 * i + 6]

                elif x[i] > prior_chemical_potentials[i]:
                    chemical_potentials[int(2 * (position[i]) + 1)] = prior_chemical_potentials[i] + 0.8 * abs(
                        x[i] - prior_chemical_potentials[i])

                elif x[i] < prior_chemical_potentials[i]:
                    chemical_potentials[int(2 * (position[i]) + 1)] = prior_chemical_potentials[i] - 0.8 * abs(
                        x[i] - prior_chemical_potentials[i])

            if (x[i] == chemical_potentials[int(2 * (position[i]) + 1)]) and (bound_indicator == 0):
                chemical_potentials[int(2 * (position[i]) + 1)] = chemical_potentials[
                                                                      int(2 * (position[i]) + 1)] + multi_num * (
                                                                          iteration_relaunch / 5)

    if fail_con == 2:
        prior_fail_con = 2
    else:
        prior_fail_con = 1

    return chemical_potentials, dopants, iteration_relaunch, prior_fail_con


def calc_fermi_dopopt(b, loop, defects, defects_form, number_of_defects, bandgap, temperature, def_statistics, nu_e,
                      condband, valband, electron_method, hole_method, fixed_e_conc, fixed_p_conc, art_dop_conc,
                      art_dop_charge, charge_convergence, val_band_min, val_band_max, cond_band_min, cond_band_max,
                      seedname, entropies, entropy_marker, fu_uc, electron_mass_eff, hole_mass_eff, unit_vol,
                      charged_sys, dopants_opt,
                      bnds, dp_list, optimiser, log_diff_conv, function_tol, maxiter_dop, environment, prog_meter,
                      prog_bar, num_iter, real_gas, constituents_name_list, x, dos_data_lst):
    fail = 0
    # Check that the point at which charge neutrality occurs falls in the bandgap

    # VBM
    total_charge, concentrations, dopant_concentration_sum = calc_charge(defects_form, defects, number_of_defects, 0,
                                                                         bandgap, condband, valband, temperature,
                                                                         art_dop_conc, art_dop_charge, def_statistics,
                                                                         electron_method, fixed_e_conc, hole_method,
                                                                         fixed_p_conc, entropy_marker, entropies,
                                                                         seedname, cond_band_min, cond_band_max,
                                                                         val_band_min, val_band_max, fu_uc,
                                                                         electron_mass_eff, hole_mass_eff, unit_vol,
                                                                         dopants_opt, constituents_name_list, dos_data_lst)

    # This checks if a math error has occured (i.e. def form eng too low)
    if dopant_concentration_sum == 'flag':
        fail = 1
        return 1, concentrations, dopant_concentration_sum, fail

    if (total_charge < 0):

        if optimiser == 1:
            fail = 1
            return 1, concentrations, dopant_concentration_sum, fail

        else:
            return 0, 0, 0, 'relaunch'

    # CBM
    total_charge, concentrations, dopant_concentration_sum = calc_charge(defects_form, defects, number_of_defects,
                                                                         bandgap, bandgap, condband, valband,
                                                                         temperature, art_dop_conc, art_dop_charge,
                                                                         def_statistics, electron_method, fixed_e_conc,
                                                                         hole_method, fixed_p_conc, entropy_marker,
                                                                         entropies, seedname, cond_band_min,
                                                                         cond_band_max, val_band_min, val_band_max,
                                                                         fu_uc, electron_mass_eff, hole_mass_eff,
                                                                         unit_vol, dopants_opt, constituents_name_list, dos_data_lst)

    # This checks if a math error has occured (i.e. def form eng too low)
    if dopant_concentration_sum == 'flag':
        fail = 1
        return 1, concentrations, dopant_concentration_sum, fail

    if (total_charge > 0):

        if optimiser == 1:
            fail = 1
            return 1, concentrations, dopant_concentration_sum, fail

        else:
            return 0, 0, 0, 'relaunch'

    total_charge = 1
    i = 0
    j = bandgap
    counter = 0

    while (total_charge > charge_convergence or total_charge < -charge_convergence):

        midpoint = (i + j) / 2
        total_charge, concentrations, dopant_concentration_sum = calc_charge(defects_form, defects, number_of_defects,
                                                                             midpoint, bandgap, condband, valband,
                                                                             temperature, art_dop_conc, art_dop_charge,
                                                                             def_statistics, electron_method,
                                                                             fixed_e_conc, hole_method, fixed_p_conc,
                                                                             entropy_marker, entropies, seedname,
                                                                             cond_band_min, cond_band_max, val_band_min,
                                                                             val_band_max, fu_uc, electron_mass_eff,
                                                                             hole_mass_eff, unit_vol, dopants_opt,
                                                                             constituents_name_list, dos_data_lst)

        if (total_charge > 0):
            i = midpoint
            counter += 1
        elif (total_charge < 0):
            j = midpoint
            counter += 1

        # print(midpoint,total_charge, charge_convergence)
        if (counter > 100):

            if optimiser == 1:
                fail = 1
                return 1, concentrations, dopant_concentration_sum, fail
            else:

                break

    if charged_sys == 0:
        nu_e_final = 0
    else:
        nu_e_final = midpoint

    return nu_e_final, concentrations, dopant_concentration_sum, fail


def dopant_fail(x, bnds, dp_list, fail):
    if fail == 1:
        print("\n<!> Optimisation of dopant chemical potential(s) terminated unsucessfully")
        print("    Could not calculate satisfactory Fermi level at following conditions:")
        print("   +--------+--------------------------------+-----------------+-----------------+")
        print("   | Dopant | Current chemical potential (eV)| Lower bound (eV)| Upper bound (eV)|")
        print("   +--------+--------------------------------+-----------------+-----------------+")
        for i in np.arange(0, len(x), 1):
            i = int(i)
            print("   | %6s | %30f | %15f | %15f |" % (dp_list[i], x[i], bnds[i][0], bnds[i][1]))

        print("   +--------+--------------------------------+-----------------+-----------------+")
        print(
            "    Bounds or dopant reference energy should be altered to make these chemical potential(s) unattainable.")

    elif fail == 2:
        print("\n<!> Unable to calculate requested concentration of dopant(s)")
        print(
            "    The following dopant chemical potential(s) represent the closest optimiser could achieve to request:")
        print("   +--------+--------------------------------+-----------------+-----------------+")
        print("   | Dopant | Current chemical potential (eV)| Lower bound (eV)| Upper bound (eV)|")
        print("   +--------+--------------------------------+-----------------+-----------------+")
        for i in np.arange(0, len(x), 1):
            i = int(i)
            print("   | %6s | %30f | %15f | %15f |" % (dp_list[i], x[i], bnds[i][0], bnds[i][1]))

        print("   +--------+--------------------------------+-----------------+-----------------+")
        print(
            "    If printout shows that current chemical potential is at a boundary, consider increasing bounds or changing dopant reference energy.")
        print(
            "    If not at boundary, recommend decreasing 'Tolerence' or increasing 'max_iteration'/'Potential_convergence'.")
        print("    Alternatively, no solution may be possible at the current conditions.")


def defect_energies(defects_list, chemical_potentials, host_supercell, use_tab_correction, E_VBM,
                    total_species, use_coul_correction, length, dielectric, v_M, opt_chem):
    defect_formation_energies = []

    # Define constants for adding point charge correction
    alpha = 2.8373
    electro_static_conv = 14.39942

    # loop through defects provided
    for defect in defects_list:

        # assign values of defect to variables
        defect_name = defect[0]
        defect_group = defect[1]
        multiplicity = float(defect[2])
        lattice_site_num = int(defect[3])
        defect_charge = int(defect[4])
        defect_DFT_energy = float(defect[5])
        defect_correction_energy = float(defect[6])
        compound_energy = float(host_supercell)

        # calculate defect formation energy at the valence band maximum
        defect_form_eng = defect_DFT_energy - compound_energy + (defect_charge * E_VBM)


        # add chemical potential contributions to the defect formation energy
        for element_index in range(total_species):
            # elements being/added removed in defect are specified from index 7 onwards in the defect row
            chem_pot_contribution = int(defect[7 + element_index]) * chemical_potentials[2 * element_index + 1]
            defect_form_eng += chem_pot_contribution

        # add cubic point charge correction if specified
        if use_coul_correction == 1:
            point_charge_correction = electro_static_conv * ((defect_charge ** 2 * alpha) / (2 * length * dielectric))

        # add anisotropic point charge correction if specified
        elif use_coul_correction == 2:
            point_charge_correction = electro_static_conv * ((defect_charge ** 2 * v_M) / 2)

        else:
            point_charge_correction = 0

        defect_form_eng += point_charge_correction

        # add own tabulated correction energy if specified
        if use_tab_correction == 1:
            defect_form_eng += defect_correction_energy

        # add defect to defect form energies list
        if opt_chem == 1: # for fitting dopant chemical potentials
            defect_formation_energies.append(
                [defect_name, defect_group, multiplicity, lattice_site_num, defect_charge, defect_form_eng,
                 float(defect[-3]), defect[-2], float(defect[-1])])
        else:
            defect_formation_energies.append(
                [defect_name, defect_group, multiplicity, lattice_site_num, defect_charge, defect_form_eng])

    return defect_formation_energies

def get_dos_data(seedname):
    # Open the $seedname.dos file
    dosfile = str(seedname) + ".dos"
    dos_data = [[], []]
    file = open(dosfile)
    for line in file:
        fields = line.strip().split()
        if len(fields) == 0:
            raise Exception("\n<!> Blank line detected in", dosfile)

        x, y = fields

        dos_data[0].append(x)
        dos_data[1].append(y)

    return dos_data

def fermi_dirac(dos_data_lst, nu_e, temperature, elec_or_hole, minimum, maximum, fu_uc):
    # Determine the number of records in $seedname.dos
    num_records = len(dos_data_lst[0])

    boltzmann = 0.000086173324

    # Determine the spacing
    step1 = float(dos_data_lst[0][0])
    step2 = float(dos_data_lst[0][1])

    dE = step2 - step1

    running_total = 0

    # Now loop through file and calculate
    i = 0
    while (i < num_records):
        energy = float(dos_data_lst[0][i])
        states = (float(dos_data_lst[1][i])) / fu_uc

        # Determine whether this is within the range defined
        if (energy >= minimum and energy <= maximum):
            # Calculate the contribution to the electron concentration
            contribution = 0
            if (elec_or_hole == 0):  # Electrons
                contribution = states * (dE / (1 + math.exp(((energy - nu_e) / (boltzmann * temperature)))))

            if (elec_or_hole == 1):  # Holes
                contribution = states * (dE / (1 + math.exp(((nu_e - energy) / (boltzmann * temperature)))))

            running_total += contribution
        i += 1

    return running_total


def eff_mass(temperature, mass_eff):
    if isinstance(mass_eff, float):
        pass
    else:
        x, y = [], []
        for i in np.arange(0, len(mass_eff), 1):
            i = int(i)
            x.append(mass_eff[i][0])
            y.append(mass_eff[i][1])

        tck = interpolate.splrep(x, y)
        mass_eff = interpolate.splev(temperature, tck)

    return (mass_eff)


def calc_charge(defects_form, defects, number_of_defects, nu_e, bandgap, condband, valband, temperature, art_dop_conc,
                art_dop_charge, def_statistics, electron_method, fixed_e_conc, hole_method, fixed_p_conc,
                entropy_marker, entropies, seedname, cond_band_min, cond_band_max, val_band_min, val_band_max, fu_uc,
                electron_mass_eff, hole_mass_eff, unit_vol, dopants_opt, constituents_name_list, dos_data_lst):

    # Some constants
    boltzmann = 0.000086173324

    # SI constants for working with electron/hole method 4.
    boltzmann_SI = 1.380649E-23
    planck_SI = 6.62607015E-34

    concentrations = []
    dopant_concentration_sum = [0] * 2 * (int(dopants_opt + 1))

    # Calculate electron and hole contributions to the total charge
    # electrons
    if (electron_method == 0):  # Off
        electrons = -100
    elif (electron_method == 1):  # Boltzmann
        electrons = condband * math.exp(-((bandgap - nu_e) / (temperature * boltzmann)))
    elif (electron_method == 2):  # Fermi-Dirac
        electrons = fermi_dirac(dos_data_lst ,nu_e, temperature, 0, cond_band_min, cond_band_max, fu_uc)
    elif (electron_method == 3):  # Fixed concentration
        electrons = fixed_e_conc
    elif (electron_method == 4):  # Effective masses

        electron_mass_eff = eff_mass(temperature, electron_mass_eff)

        unit_vol_SI = unit_vol * 1E-30
        N_c = (2 * ((2 * math.pi * electron_mass_eff * 9.11E-31 * boltzmann_SI * temperature) / (planck_SI ** 2)) ** (
                    3 / 2))
        electrons = ((N_c * unit_vol_SI) / fu_uc) * math.exp(-((bandgap - nu_e) / (temperature * boltzmann)))

    # holes
    if (hole_method == 0):  # Off
        holes = -100
    elif (hole_method == 1):  # Boltzmann
        holes = valband * math.exp(-((nu_e) / (temperature * boltzmann)))
    elif (hole_method == 2):  # Fermi-Dirac
        holes = fermi_dirac(dos_data_lst ,nu_e, temperature, 1, val_band_min, val_band_max, fu_uc)
    elif (hole_method == 3):  # Fixed concentration
        holes = fixed_p_conc
    elif (hole_method == 4):  # Effective masses

        hole_mass_eff = eff_mass(temperature, hole_mass_eff)

        unit_vol_SI = unit_vol * 1E-30
        N_v = (2 * ((2 * math.pi * hole_mass_eff * 9.11E-31 * boltzmann_SI * temperature) / (planck_SI ** 2)) ** (
                    3 / 2))
        holes = ((N_v * unit_vol_SI) / fu_uc) * math.exp(-((nu_e) / (temperature * boltzmann)))

    total_charge = -1 * electrons + holes

    # Convert electron and hole concentrations to log values
    if (electron_method != 0):
        electrons = math.log(electrons) / math.log(10)
    if (hole_method != 0):
        holes = math.log(holes) / math.log(10)

    concentrations.append(electrons)
    concentrations.append(holes)

    # Add the contribution from an aritificial dopant
    total_charge += (art_dop_conc * art_dop_charge)


    # Loop over all defects and calculate concentration and contribution to the total charge
    for i in np.arange(0, number_of_defects, 1):
        # Read in details of the defect from the defects.dat file
        defect_name = defects_form[int(i)][0]
        multiplicity = float(defects_form[int(i)][2])
        site = int(defects_form[int(i)][3])
        charge = int(defects_form[int(i)][4])
        form_energy_vbm = float(defects_form[int(i)][5])

        # Calculate formation energy at nu_e
        def_form_energy = form_energy_vbm + (charge * nu_e)

        # Check to see whether the calculated defect formation energies are reasonable
        if (def_form_energy > 100 or def_form_energy < -100):
            raise Exception(f"<!> Error: Defect formation energy falls outside reasonable limits\n"
                            f"{defect_name} {charge} has formation energy of {def_form_energy}\n"
                            f"Check whether the host lattice has been defined correctly if so then you may need to revisit your DFT energies")

        # Prevent math error:
        if (dopants_opt == 1) and ((-def_form_energy / (temperature * boltzmann)) > 705):
            return 'flag', 'flag', 'flag'

            # Calculate the concentration and consequent contribution to total charge
        if (def_statistics == 0):  # Simple Boltzmann statistics

            concentration = multiplicity * math.exp(-def_form_energy / (temperature * boltzmann))

            if (entropy_marker == 1):
                concentration = concentration * math.exp(entropies[i] / boltzmann)

        if (def_statistics == 1):  # Kasamatsu statistics
            competing = 0
            for j in np.arange(0, number_of_defects, 1):
                # Loop over all defects and determine if competing for the same site

                site2 = int(defects_form[int(j)][3])
                if (site == site2):
                    if (i == j):
                        pass
                        # This is the target defect and cannot compete with itself
                    else:

                        defect_name2 = defects_form[int(j)][0]
                        multiplicity2 = float(defects_form[int(j)][2])
                        charge2 = float(defects_form[int(j)][4])
                        form_energy_vbm2 = float(defects_form[int(j)][5])

                        def_form_energy2 = form_energy_vbm2 + (charge2 * nu_e)

                        # Using this defect formation energy as to the sum in the denominator
                        competing += math.exp(-1 * def_form_energy2 / (temperature * boltzmann))

            concentration = multiplicity * (math.exp(-def_form_energy / (temperature * boltzmann))) / (1 + competing)

        charge_contribution = concentration * charge
        total_charge += charge_contribution


        # Sum concentrations for use in dopant chemical potential optimisation
        if dopants_opt != 0:
            for k in np.arange(0, dopants_opt + 1, 1):
                marker = int(defects_form[int(i)][-3])
                signal = defects_form[int(i)][-2]
                multiply = defects_form[int(i)][-1]


                if int(k) == marker:
                    dopant_concentration_sum[2 * marker] = dopant_concentration_sum[2 * marker] + (
                                multiply * concentration)
                    dopant_concentration_sum[2 * marker + 1] = signal

        if concentration < 10e-200:
            concentration = 10e-200
        concentration = math.log(concentration) / math.log(10)
        concentrations.append(concentration)

    return total_charge, concentrations, dopant_concentration_sum


def calc_fermi(b, loop, defects, defects_form, number_of_defects, bandgap, temperature, def_statistics, nu_e, condband,
               valband, electron_method, hole_method, fixed_e_conc, fixed_p_conc, art_dop_conc, art_dop_charge,
               charge_convergence, val_band_min, val_band_max, cond_band_min, cond_band_max, seedname, entropies,
               entropy_marker, fu_uc, electron_mass_eff, hole_mass_eff, unit_vol, charged_sys, dopants_opt,
               constituents_name_list, dos_data_lst):

    # check that a root lies somewhere in the bandgap. Therefore, positive total charge at the valence band and negative total charge at the conduction band

    # calculate the charge at the valence band maximum (Fermi level = 0)
    total_charge, concentrations, dopant_concentration_sum = calc_charge(defects_form, defects, number_of_defects, 0,
                                                                         bandgap, condband, valband, temperature,
                                                                         art_dop_conc, art_dop_charge, def_statistics,
                                                                         electron_method, fixed_e_conc, hole_method,
                                                                         fixed_p_conc, entropy_marker, entropies,
                                                                         seedname, cond_band_min, cond_band_max,
                                                                         val_band_min, val_band_max, fu_uc,
                                                                         electron_mass_eff, hole_mass_eff, unit_vol,
                                                                         dopants_opt, constituents_name_list, dos_data_lst)

    # error if negative total charge at the VBM
    if (total_charge < 0):
        raise Exception("<!> Error: Charge neutrality occurs outside of the band gap (nu_e < 0)\n"
                        f"Total charge calculated at the valence band maximum: {total_charge}\n"
                        f"You may need to:\n"
                        f"> Modify the range of your loop parameters\n"
                        f"> Review your DFT energies\n"
                        f"> Review how the concentration of electron/holes are calculated")

    # calculate total charge at the conduction band minimum (Fermi level = bandgap)
    total_charge, concentrations, dopant_concentration_sum = calc_charge(defects_form, defects, number_of_defects,
                                                                         bandgap, bandgap, condband, valband,
                                                                         temperature, art_dop_conc, art_dop_charge,
                                                                         def_statistics, electron_method, fixed_e_conc,
                                                                         hole_method, fixed_p_conc, entropy_marker,
                                                                         entropies, seedname, cond_band_min,
                                                                         cond_band_max, val_band_min, val_band_max,
                                                                         fu_uc, electron_mass_eff, hole_mass_eff,
                                                                         unit_vol, dopants_opt, constituents_name_list, dos_data_lst)

    # error if positive total charge at the CBM
    if (total_charge > 0):
        raise Exception("<!> Error: Charge neutrality occurs outside of the band gap (nu_e > Bandgap)\n"
                        f"Total charge calculated at the conduction band minimum:: {total_charge}\n"
                        f"You may need to:\n"
                        f"> Modify the range of your loop parameters\n"
                        f"> Review your DFT energies\n"
                        f"> Review how the concentration of electron/holes are calculated")

    i = 0
    j = bandgap
    counter = 0


    # find the root Fermi level using linear bisection
    while (total_charge > charge_convergence or total_charge < -charge_convergence):

        midpoint = (i + j) / 2

        total_charge, concentrations, dopant_concentration_sum = calc_charge(defects_form, defects, number_of_defects,
                                                                             midpoint, bandgap, condband, valband,
                                                                             temperature, art_dop_conc, art_dop_charge,
                                                                             def_statistics, electron_method,
                                                                             fixed_e_conc, hole_method, fixed_p_conc,
                                                                             entropy_marker, entropies, seedname,
                                                                             cond_band_min, cond_band_max, val_band_min,
                                                                             val_band_max, fu_uc, electron_mass_eff,
                                                                             hole_mass_eff, unit_vol, dopants_opt,
                                                                             constituents_name_list, dos_data_lst)


        if (total_charge > 0):
            i = midpoint
            counter += 1
        if (total_charge < 0):
            j = midpoint
            counter += 1

        # after 100 iterations, unlikely to reach convergence
        if (counter > 100):
            raise Exception("<!> Could not determine the Fermi level that gives charge neutrality."
                            f"\nTotal charge calculated: {total_charge}\n"
                            f"Recommend to decrease charge convergence from {charge_convergence}.")


    if charged_sys == 0:
        nu_e_final = 0
    else:
        nu_e_final = midpoint

    return nu_e_final, concentrations, dopant_concentration_sum


def group(final_concentrations, number_of_defects, defects, num_iter, stoichiometry):
    print("..> Summing defect concentrations according to group assigned")

    new_concs = []
    final_group_list = []

    # Loop over the number of records
    for i in np.arange(0, len(final_concentrations), 1):
        i = int(i)
        group_list = []
        grouped_concs = []

        # Extract the iterator condition and the electron and hole concentrations
        iterator = final_concentrations[i][0]
        fermi = final_concentrations[i][1]
        electron = final_concentrations[i][2]
        hole = final_concentrations[i][3]
        if stoichiometry != 0:
            stoic = final_concentrations[i][-1]

        # Now loop over the defects in each record
        for j in np.arange(0, number_of_defects, 1):
            j = int(j)
            # Extract the defect group
            group = defects[j][1]
            concentration = final_concentrations[i][j + 4]
            unlogged_conc = 10 ** concentration

            # Check to see whether this group has been found before
            if (group in group_list):

                # Loop over group_list and determine the cell were this concentration should go
                group_list_length = len(group_list)
                for k in np.arange(0, group_list_length, 1):
                    k = int(k)
                    if (group == group_list[k]):
                        # Add the concentration to this grouping
                        grouped_concs[k] = grouped_concs[k] + unlogged_conc

            else:
                group_list.append(group)
                group_list_length = len(group_list)
                grouped_concs.append(unlogged_conc)

            final_group_list = group_list

            size = len(grouped_concs)

        # Quick loop to relog everything
        group_list_length = len(group_list)
        new_concs.append([iterator, fermi, electron, hole])

        for w in np.arange(0, group_list_length, 1):
            w = int(w)
            log_conc = math.log(grouped_concs[w]) / math.log(10)
            new_concs[-1].append(log_conc)
        if stoichiometry != 0:
            new_concs[-1].append(stoic)

    return (new_concs, final_group_list)


def stoich(concentrations, defects, host_array, number_of_defects, dopants, x_variable, stoichiometry, phase):
    # Function that finds deviation in stoichiometry for volatile species
    # Volatile species must be the last element in the host.
    # Whether a defect contributes to hyper/hypo stoic is determined by input in .defects file.

    stoic_sum = 0
    numerator = 0
    denominator = 0
    # loop over atoms in host
    for i in np.arange(0, host_array[0], 1):
        i = int(i)
        stoic = float(host_array[2 * i + 2])
        contribution = 0
        # loop over defecs
        for j in np.arange(0, number_of_defects, 1):
            j = int(j)

            element_change = float(defects[j][7 + i])

            if element_change != 0:
                contribution += (10 ** float(concentrations[j + 2])) * (-1 * element_change)

        # contribution = contribution
        contribution += stoic

        if i == host_array[0] - 1:
            numerator += contribution

        else:
            denominator += contribution
            stoic_sum += stoic

    # Two options for dopants:
    # Stoichiometry = 1 calculates stoichiometry with original cations, considers the cation/volatile species leaving the system in a substitution, but not the dopant added.
    # Stoichiometry = 2 calculates a volatile to metal ratio, where any dopant added is treated as a metal.

    if stoichiometry == 2:
        # loop over dopant atoms
        if (dopants[0] > 0):
            for i in np.arange(0, dopants[0], 1):
                i = int(i)
                contribution = 0
                # loop over defecs
                for j in np.arange(0, number_of_defects, 1):
                    j = int(j)

                    element_change = float(defects[j][7 + host_array[0] + i])

                    if element_change != 0:
                        contribution += (10 ** float(concentrations[j + 2])) * (-1 * element_change)

                contribution = (contribution)

                denominator += contribution


    final_stoic = -1 * ((numerator / (denominator / stoic_sum)) - stoic)

    if (x_variable == 1):  # Plotting as a function of stoichiometery

        new_stoichiometry = -final_stoic

        concentrations.insert(0, new_stoichiometry)

    elif (phase == 1):
        new_stoichiometry = -final_stoic

        concentrations.append(new_stoichiometry)

    else:
        # This function reflects the value of x so under hyperstoichiometry it is MO2+x and MO2-x for hypostoichiometry
        if (final_stoic < 0):

            new_stoichiometry = -1 * final_stoic
        else:
            new_stoichiometry = final_stoic


        if (new_stoichiometry == 0):
            log_stoichiometry = math.log(1e-20) / math.log(10)
            concentrations.append(log_stoichiometry)
        else:
            log_stoichiometry = math.log(new_stoichiometry) / math.log(10)
            concentrations.append(log_stoichiometry)


    return concentrations, final_stoic


def print_results(results, seedname):
    resultfile = str(seedname) + ".res"
    print("..> Printing defect concentrations in", resultfile)
    with open(resultfile, 'w') as f:
        for i in results:
            print(*i, file=f)


def print_stoic_data(stoic_dict):
    for stoic in stoic_dict:
        data_file = f"xhat={stoic}.dat"
        with open(data_file, "w") as f:
            for data in stoic_dict[stoic]:
                print(*data, file=f)


def print_fermi(fermi, seedname):
    fermifile = str(seedname) + ".fermi"
    print("..> Printing fermi energies in", fermifile)
    with open(fermifile, 'w') as f:
        for i in fermi:
            print(*i, file=f)


def print_formation(master_list, seedname):
    formationfile = str(seedname) + ".formation"

    with open(formationfile, 'w') as f:
        for i in master_list:
            print(*i, file=f)


def print_phases(final_phases, seedname):
    phasesfile = str(seedname) + ".phases"

    with open(phasesfile, 'w') as f:
        for i in final_phases:
            print(*i, file=f)


def print_defect_phases(plot_master, nametags, tag, min_value, max_value, iterator):
    
    
    if tag == 1:
        addendum = ''

        # make an empty directory to store defect phase output files in
        if os.path.exists("defect_phase_defects"):
            os.chdir("defect_phase_defects")

            for file in os.listdir("./"):
                if fnmatch.fnmatch(file, f'*'):
                    os.remove(file)

        else:
            os.mkdir("defect_phase_defects")
            os.chdir("defect_phase_defects")


    elif tag == 2:
        addendum = '_dpt'

        # make an empty directory to store dopant defect phase output files in
        if os.path.exists("defect_phase_dopant"):
            os.chdir("defect_phase_dopant")

            for file in os.listdir("./"):
                if fnmatch.fnmatch(file, f'*'):
                    os.remove(file)
        else:
            os.mkdir("defect_phase_dopant")
            os.chdir("defect_phase_dopant")


    for i in np.arange(0, len(nametags), 1):
        i = int(i)
        defect_phasesfile = nametags[i] + addendum + ".dat"
        with open(defect_phasesfile, 'w') as f:
            for j in plot_master[i]:
                print(*j, file=f)

    # make sure to move back out of the output directory
    os.chdir("..")

def graphical_inputs(seedname):
    filename = str(seedname) + ".plot"

    file = open(filename)
    for line in file:
        fields = line.strip().split()
        if len(fields) > 2:
            name = fields[0]
            if name == "concentration_colour":
                conc_colour = fields[2:]
            if name == "formation_colour":
                form_colour = fields[2:]
            if name == "electron_colour":
                electron_colour = fields[2]
            if name == "hole_colour":
                hole_colour = fields[2]

    return conc_colour, form_colour, electron_colour, hole_colour


def graphical_output(number_of_defects, min_value, max_value, final_concentrations, seedname, loop, gnuplot_version,
                     min_y_range, host_name, defects, electron_method, hole_method, dopants, host_array, entry_marker,
                     conc_colour, electron_colour, hole_colour, scheme, dopant_xvar, stoichiometry, x_variable,
                     total_species, volatile_element, charged_sys, y_variable, max_y_range, plot_art_dop_conc,
                     art_dop_charge):
    print("..> Plotting defect concentrations")

    # Improve presentation of the host name
    host_name = host_name.replace("-", "")

    # colour choice

    if scheme == 0:

        colours = ["#006e00", "#b80058", "#008cf9", "#d163e6", "#00bbad", "#ff9287", "#b24502", "#878500", "#00c6f8",
                   "#00a76c", "#bdbdbd"]
        colour_electron = "#5954d6"
        colour_hole = "#ebac23"
    elif scheme == 1:
        colours = conc_colour
        colour_electron = electron_colour
        colour_hole = hole_colour

    colourx = "black"
    defect_colours = []
    defect_lines = []
    line_marker = 0
    colour_marker = 0

    # Assign colours and lines, for all defects. Each defect of same 'type' assigned same colour, with each given different line type.
    if entry_marker == 0:
        key_master = []
        for i in np.arange(0, number_of_defects, 1):
            i = int(i)
            key = ''
            for j in np.arange(0, total_species, 1):
                j = int(j) + 7
                key_i = defects[i][j]
                key += key_i
            key_master.append(key)

        assigner = []
        j = 0

        for i in np.arange(0, number_of_defects, 1):
            i = int(i)
            key = key_master[i]
            if key in assigner:
                colour = assigner[assigner.index(key) + 1]
                line = assigner[assigner.index(key) + 2] + 1
                assigner[assigner.index(key) + 2] = line

                if line > 8:
                    line = 1
                    if line_marker == 0:
                        print(
                            "<!> Unable to assign unique line dashes to all defects, due to a large number of defects of the same 'type'. Consider task = 'group'")
                        line_marker = 1

            else:
                if j > len(colours) - 1:
                    colour = colourx
                    if colour_marker == 0:
                        print(
                            "<!> Colour list exceeded: some defects assigned colour black. Specify colour by increasing colours in 'concentration_colour' in",
                            seedname, ".plot file")
                        colour_marker = 1
                else:
                    colour = colours[j]
                line = 1
                j += 1
                assigner.append(key)
                assigner.append(colour)
                assigner.append(line)
            defect_colours.append(colour)
            defect_lines.append(line)

    # Assign colours and line types, if task = group.
    if entry_marker == 1:
        i = 1
        while i < 9:
            j = 0
            while j < len(colours):
                defect_colours.append(colours[j])
                defect_lines.append(i)
                j += 1
            i += 1

    graphfile = str(seedname) + ".p"
    outputfile = str(seedname) + ".eps"
    resultfile = str(seedname) + ".res"
    fermifile = str(seedname) + ".fermi"
    fermiplotfile = str(seedname) + "_fermi.eps"

    with open(graphfile, 'w') as f:
        # Print header to file
        print("#GNUPLOT script for showing defect concentrations\n", file=f)
        print("set terminal postscript eps enhanced color font 'Helvetica,20'", file=f)
        print("set output \"", outputfile, "\"", sep="", file=f)
        print("set encoding iso_8859_1", file=f)

        if x_variable == 1:
            if host_array[-1] == 1:
                print("set xlabel \"x in ", host_name, "_{1+x}\"", sep="", file=f)
            else:
                print("set xlabel \"x in ", host_name, "_{+x}\"", sep="", file=f)
        else:
            if (loop == 0):
                print("set xlabel 'log_{10}P_{", volatile_element, "_{2}} /atm'", sep="", file=f)
            elif (loop == 1):
                print("set xlabel 'Temperature /K'", file=f)
            elif (loop == 2):
                print("set xlabel 'log_{10}[", dopant_xvar[0], "] (per ", host_name, ")'", sep="", file=f)
            elif (loop == 3):
                print("set xlabel 'log_{10}[artificial dopant conc] (per ", host_name, ")'", sep="", file=f)
            elif (loop == 4):
                print("set xlabel 'log_{10}P_{", dopant_xvar[0], "_{2}} /atm'", sep="", file=f)
            elif (loop == 5):
                print("set xlabel 'rich-poor fraction (s)'", sep="", file=f)
        if y_variable == 1:
            print("set ylabel 'log_{10}[D] (per cm^{-3})'\n", sep="", file=f)
        else:
            print("set ylabel 'log_{10}[D] (per ", host_name, ")'\n", sep="", file=f)
        if x_variable == 1:
            print("set xrange [-0.1:0.1]", sep="", file=f)
        else:
            if loop == 5:
                print("set xrange [", final_concentrations[0][0], ":", final_concentrations[-1][0], "]", sep="", file=f)
            else:
                print("set xrange [", min_value, ":", max_value, "]", sep="", file=f)
        print("set yrange [", min_y_range, ":", max_y_range, "]\n", sep="", file=f)
        # Dashtype
        print("set linetype 2 dt \"_\"", file=f)
        print("set linetype 3 dt 2", file=f)
        print("set linetype 4 dt 4", file=f)
        print("set linetype 5 dt 5", file=f)
        print("set linetype 6 dt 6", file=f)
        print("set linetype 7 dt 7", file=f)
        print("set linetype 8 dt 8", file=f)
        print("set linetype 9 dt 9", file=f)

        print("set key outside", file=f)
        print("set key center below", file=f)
        print("set key horizontal\n", file=f)
        print("set key font 'Helvetica,14'", file=f)

        if electron_method != 0:
            print("plot \"./", resultfile, "\" using 1:3 with lines lt 1 lw 2 lc rgb \"", colour_electron,
                  "\" ti \"Electrons\",\\", sep="", file=f)

        if hole_method != 0 and electron_method != 0:
            print("\"./", resultfile, "\" using 1:4 with lines lt 1 lw 2 lc rgb \"", colour_hole, "\" ti \"Holes\",\\",
                  sep="", file=f)
        if hole_method != 0 and electron_method == 0:
            print("plot \"./", resultfile, "\" using 1:4 with lines lt 1 lw 2 lc rgb \"", colour_hole,
                  "\" ti \"Holes\",\\", sep="", file=f)

        # Plot concentration of every defect. each charge state assigned different 'dash type'
        if entry_marker == 0:
            i = 0
            while i < number_of_defects:
                defect = defects[i][0]
                group = defects[i][1]
                charge = int(defects[i][4])
                colour = defect_colours[i]
                line_type = defect_lines[i]
                if charged_sys == 1:
                    if i == 0 and electron_method == 0 and hole_method == 0:
                        print("plot \"./", resultfile, "\" using 1:", i + 5, " with lines lt ", line_type,
                              " lw 2 lc rgb \"", colour, "\" ti \"", defect, " ", charge, "\",\\", sep="", file=f)
                    else:
                        print("\"./", resultfile, "\" using 1:", i + 5, " with lines lt ", line_type, " lw 2 lc rgb \"",
                              colour, "\" ti \"", defect, " ", charge, "\",\\", sep="", file=f)
                else:
                    if i == 0 and electron_method == 0 and hole_method == 0:
                        print("plot \"./", resultfile, "\" using 1:", i + 5, " with lines lt ", line_type,
                              " lw 2 lc rgb \"", colour, "\" ti \"", defect, " \",\\", sep="", file=f)
                    else:
                        print("\"./", resultfile, "\" using 1:", i + 5, " with lines lt ", line_type, " lw 2 lc rgb \"",
                              colour, "\" ti \"", defect, " \",\\", sep="", file=f)
                i += 1
            if stoichiometry != 0 and x_variable == 0:
                pm = r"\261"
                if host_array[-1] == 1:
                    print("\"./", resultfile, "\" using 1:", i + 5, " with lines lt 2 lw 2 lc rgb \"", colourx,
                          "\" ti \"x in ", host_name, "_{1", pm, "x}\",\\", sep="", file=f)
                else:
                    print("\"./", resultfile, "\" using 1:", i + 5, " with lines lt 2 lw 2 lc rgb \"", colourx,
                          "\" ti \"x in ", host_name, "_{", pm, "x}\",\\", sep="", file=f)

        # Plot sum of concentrations, based on group.
        elif entry_marker == 1:
            i = 0
            while i < len(defects):
                defect = defects[i]
                colour = defect_colours[i]
                line_type = defect_lines[i]
                if i == 0 and electron_method == 0 and hole_method == 0:
                    print("plot \"./", resultfile, "\" using 1:", i + 5, " with lines lt ", line_type,
                          " lw 2 lc rgb \"", colour, "\" ti \"", defect, "\",\\", sep="", file=f)
                else:
                    print("\"./", resultfile, "\" using 1:", i + 5, " with lines lt ", line_type, " lw 2 lc rgb \"",
                          colour, "\" ti \"", defect, "\",\\", sep="", file=f)
                i += 1

            if stoichiometry != 0 and x_variable == 0:
                pm = r"\261"
                if host_array[-1] == 1:
                    print("\"./", resultfile, "\" using 1:", i + 5, " with lines lt 2 lw 2 lc rgb \"", colourx,
                          "\" ti \"x in ", host_name, "_{1", pm, "x}\",\\", sep="", file=f)
                else:
                    print("\"./", resultfile, "\" using 1:", i + 5, " with lines lt 2 lw 2 lc rgb \"", colourx,
                          "\" ti \"x in ", host_name, "_{", pm, "x}\",\\", sep="", file=f)

        if plot_art_dop_conc == 1:
            print(
                f'"./{resultfile}" using 1:{i + 6} with lines lt 4 lw 2 lc rgb "#000000" ti "Art Dopant {int(art_dop_charge)}"\\',
                sep="", file=f)

        # Plot Fermi energy
        if charged_sys == 1:

            print("\n\n#GNUPLOT script for showing Fermi energy\n", file=f)
            print("set output \"", fermiplotfile, "\"", sep="", file=f)
            if (loop == 0):
                print("set xlabel 'log_{10}P_{", volatile_element, "_{2}} /atm'", sep="", file=f)
            elif (loop == 1):
                print("set xlabel 'Temperature /K'", file=f)
            elif (loop == 2):
                print("set xlabel 'log_{10}[", dopant_xvar[0], "] (per ", host_name, ")'", sep="", file=f)
            elif (loop == 3):
                print("set xlabel 'log_{10}[artificial_dopant_conc] (per ", host_name, ")'", sep="", file=f)
            elif (loop == 4):
                print("set xlabel 'log_{10}P_{", dopant_xvar[0], "_{2}} /atm'", sep="", file=f)
            elif (loop == 5):
                print("set xlabel 'rich-poor fraction (s)'", sep="", file=f)
            print("set autoscale y", sep="", file=f)
            print("set key off", sep="", file=f)
            print("set ylabel 'Fermi level (eV)'\n", sep="", file=f)
            print("plot \"./", fermifile, "\" using 1:2 with lines lt 1 lw 2 lc rgb \"#008cf9\" \\", sep="", file=f)


def form_energies(defects_form, number_of_defects, tasks, bandgap, seedname, entropies, entropy_marker, temperature):
    # Some constants
    boltzmann = 0.000086173324


    defect_types = []
    lowest_formation = []
    formation = []

    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:

        # Print header for the formation energies
        print("\n-----------------------------------------------------------------------------------------", "\n",
              file=f)
        print(">>> Formation energies\n", file=f)
        print("   +----------------+--------+----------------------+", file=f)
        print("   |     Defect     | Charge | Formation energy /eV |", file=f)
        print("   +----------------+--------+----------------------+", file=f)

        # Search through defects_form and print output
        for i in np.arange(0, number_of_defects, 1):
            i = int(i)
            defect_name = defects_form[i][0]
            defect_group = defects_form[i][1]
            charge = defects_form[i][4]
            form_energy = defects_form[i][5]
            if (entropy_marker == 1):
                form_energy += (-entropies[i] * temperature)

            if ("form_plots" in tasks):

                if (defect_group in defect_types):
                    pass
                else:
                    defect_types.append(defect_group)

            print("   | %14s | %6s | %20f |" % (defect_name, charge, form_energy), file=f)

        print("   +----------------+--------+----------------------+\n", file=f)

    print("..> Defect formation energies tabulated in", outputfile)

    if ("form_plots" in tasks):

        # Find lowest formation energy for each class of defect across bandgap
        increment_fermi = 0.001
        i = 0
        while i <= bandgap:
            j = 0
            defect_form_list = [i]
            while j < len(defect_types):
                group = defect_types[j]
                defect_group_form_list = []
                for w in np.arange(0, number_of_defects, 1):
                    w = int(w)
                    group_i = defects_form[w][1]
                    charge = defects_form[w][4]
                    form_energy = defects_form[w][5]

                    if group == group_i:
                        if (entropy_marker == 1):
                            form_energy += (-entropies[w] * temperature)
                        defect_form = charge * i + form_energy
                        defect_group_form_list.append(defect_form)

                defect_form_list.append(min(defect_group_form_list))

                j += 1
            lowest_formation.append(defect_form_list)
            i += increment_fermi

        # Find formation energy for every defect across bandgap
        increment_fermi = 0.01
        i = 0

        while i <= bandgap:
            defect_form_list = [i]
            for w in np.arange(0, number_of_defects, 1):
                w = int(w)
                charge = defects_form[w][4]
                form_energy = defects_form[w][5]
                if (entropy_marker == 1):
                    form_energy += (-entropies[w] * temperature)
                defect_form = charge * i + form_energy
                defect_form_list.append(defect_form)

            formation.append(defect_form_list)
            i += increment_fermi

    formationfile = str(seedname) + ".formation_grouped"
    formationfile2 = str(seedname) + ".formation"

    with open(formationfile, 'w') as f:
        for i in lowest_formation:
            print(*i, file=f)

    with open(formationfile2, 'w') as f:
        for i in formation:
            print(*i, file=f)

    return defect_types


def formation_graphical_output(seedname, bandgap, defects, y_form_min, y_form_max, form_colour, scheme,
                               number_of_defects, total_species, defect_types):
    graphfile = "formation_plot.p"
    outputfile1 = "formation_minimum.eps"
    outputfile2 = "formation.eps"
    resultfile1 = str(seedname) + ".formation_grouped"
    resultfile2 = str(seedname) + ".formation"

    if scheme == 0:
        colours = ["#006e00", "#b80058", "#008cf9", "#d163e6", "#00bbad", "#ff9287", "#b24502", "#878500", "#00c6f8",
                   "#00a76c", "#bdbdbd", "#ebac23", "#5954d6"]

    elif scheme == 1:
        colours = form_colour
    colourx = "black"

    defect_colours = []
    defect_lines = []
    defect_colours2 = []
    defect_lines2 = []
    line_marker = 0
    colour_marker = 0

    # Assign colours and lines, for all defects. Each defect of same 'type' assigned same colour, with each given different line type.

    key_master = []
    for i in np.arange(0, number_of_defects, 1):
        i = int(i)
        key = ''
        for j in np.arange(0, total_species, 1):
            j = int(j) + 7
            key_i = defects[i][j]
            key += key_i
        key_master.append(key)

    assigner = []
    j = 0

    for i in np.arange(0, number_of_defects, 1):
        i = int(i)
        key = key_master[i]
        if key in assigner:
            colour = assigner[assigner.index(key) + 1]
            line = assigner[assigner.index(key) + 2] + 1
            assigner[assigner.index(key) + 2] = line

            if line > 8:
                line = 1
                if line_marker == 0:
                    print(
                        "<!> Unable to assign unique line dashes to all defects, due to a large number of defects of the same 'type'. Consider task = 'group'")
                    line_marker = 1

        else:
            if j > len(colours) - 1:
                colour = colourx
                if colour_marker == 0:
                    print(
                        "<!> Colour list exceeded: some defects assigned colour black. Specify colour by increasing colours in 'formation_colour' in",
                        seedname, ".plot file")
                    colour_marker = 1
            else:
                colour = colours[j]
            line = 1
            j += 1
            assigner.append(key)
            assigner.append(colour)
            assigner.append(line)
        defect_colours.append(colour)
        defect_lines.append(line)

    # Assign colours and line types for grouped formation energies

    i = 1
    while i < 9:
        j = 0
        while j < len(colours):
            defect_colours2.append(colours[j])
            defect_lines2.append(i)
            j += 1
        i += 1

    with open(graphfile, 'w') as f:
        # Print header to file
        print("#GNUPLOT script for formation energies of defects\n", file=f)
        print("set terminal postscript eps enhanced color font 'Helvetica,20'", file=f)
        print("set xlabel 'Fermi level (eV)'", file=f)
        print("set ylabel 'Formation enegy (eV)'\n", file=f)
        print("set xrange [", 0, ":", bandgap, "]", sep="", file=f)
        # print("set yrange [",y_form_min,":",y_form_max,"]\n",sep="", file=f)
        # Dashtype
        print("set linetype 2 dt \"_\"", file=f)
        print("set linetype 3 dt 2", file=f)
        print("set linetype 4 dt 4", file=f)
        print("set linetype 5 dt 5", file=f)
        print("set linetype 6 dt 6", file=f)
        print("set linetype 7 dt 7", file=f)
        print("set linetype 8 dt 8", file=f)
        print("set linetype 9 dt 9", file=f)

        print("set key outside\n", file=f)
        print("set key font 'Helvetica,14'", file=f)
        # print("set key center below", file=f)
        # print("set key horizontal\n", file=f)

        # Print one line for each defect class
        print('..> Plotting minimum formation energy for each group')
        print("set output \"", outputfile1, "\"", sep="", file=f)
        i = 2
        for group in defect_types:
            colour = defect_colours2[i - 2]
            line_type = defect_lines2[i - 2]
            if i == 2:
                print("plot \"./", resultfile1, "\" using 1:", i, " with lines lt ", line_type, " lw 2 lc rgb \"",
                      colour, "\" ti \"", group, "\",\\", sep="", file=f)
            else:
                print("\"./", resultfile1, "\" using 1:", i, " with lines lt ", line_type, " lw 2 lc rgb \"", colour,
                      "\" ti \"", group, "\",\\", sep="", file=f)

            i += 1

        # Print every defect, assigning different dash type for each charge

        print('..> Plotting formation energy for every defect')
        print("\n set output \"", outputfile2, "\"", sep="", file=f)

        i = 0
        while i < number_of_defects:
            defect = defects[i][0]
            charge = int(defects[i][4])
            colour = defect_colours[i]
            line_type = defect_lines[i]
            if i == 0:
                print("plot \"./", resultfile2, "\" using 1:", i + 2, " with lines lt ", line_type, " lw 2 lc rgb \"",
                      colour, "\" ti \"", defect, " ", charge, "\",\\", sep="", file=f)
            else:
                print("\"./", resultfile2, "\" using 1:", i + 2, " with lines lt ", line_type, " lw 2 lc rgb \"",
                      colour, "\" ti \"", defect, " ", charge, "\",\\", sep="", file=f)
            i += 1

        # Print individual groups on individual figures

        print('..> Plotting minimum formation energy for each group, with seperate figures for each group')

        i = 2
        for group in defect_types:
            colour = defect_colours2[i - 2]
            outputfile = str(group) + "_min.eps"
            print("\n set output \"", outputfile, "\"", sep="", file=f)
            print("plot \"./", resultfile1, "\" using 1:", i, " with lines lt 1 lw 2 lc rgb \"", colour, "\" ti \"",
                  group, "\",\\", sep="", file=f)

            i += 1

        # Print every defect on individual figures

        print('..> Plotting formation energy for every defect, with seperate figures for each group')
        group_position = []  # A log of the group positions
        for group in defect_types:
            outputfile = str(group) + ".eps"
            print("\n set output \"", outputfile, "\"", sep="", file=f)
            i = 1
            j = 0
            group_position_i = []
            while i < number_of_defects + 1:
                group_i = defects[i - 1][1]
                defect = defects[i - 1][0]
                charge = int(defects[i - 1][4])
                colour = defect_colours[i - 1]
                line_type = defect_lines[i - 1]
                if group == group_i:
                    group_position_i.append(i)
                    if j == 0:
                        print("plot \"./", resultfile2, "\" using 1:", i + 1, " with lines lt ", line_type,
                              " lw 2 lc rgb \"", colour, "\" ti \"", defect, " ", charge, "\",\\", sep="", file=f)
                    else:
                        print("\"./", resultfile2, "\" using 1:", i + 1, " with lines lt ", line_type,
                              " lw 2 lc rgb \"", colour, "\" ti \"", defect, " ", charge, "\",\\", sep="", file=f)
                    j += 1
                i += 1
            group_position.append(group_position_i)


def y_convert(final_concentrations, fu_uc, uc_volume, stoichiometry):
    # Nummber of Angstrom^3 in cm^3
    A3_2_cm3 = 1E24

    conversion = fu_uc * (1 / uc_volume) * A3_2_cm3

    inc = 0  # Do not want to convert the final column if stoichiometry has been calculated
    if stoichiometry != 0:
        inc = 1

    for i in np.arange(0, (len(final_concentrations)), 1):
        for j in np.arange(2, (len(final_concentrations[0]) - inc), 1):
            concentration = 10 ** final_concentrations[i][j]
            concentration = concentration * conversion
            final_concentrations[i][j] = math.log(concentration) / math.log(10)

    return final_concentrations


def defect_phases_graphical_output(min_value, max_value, min_value_y, max_value_y, seedname, loop, loop2, host_name,
                                   dopant_xvar, volatile_element, stability, nametags, increment, tag):
    print("..> Plotting maximum defect concentrations")

    # Improve presentation of the host name
    host_name = host_name.replace("-", "")

    # colour choice
    colours = ["#006e00", "#b80058", "#008cf9", "#d163e6", "#00bbad", "#ebac23", "#5954d6", "#ff9287", "#b24502",
               "#878500", "#00c6f8", "#00a76c", "#bdbdbd"]
    colourx = "black"
    defect_colours = []
    defect_lines = []

    if tag == 1:
        output_dir = "defect_phase_defects"
        graphfile = str(seedname) + "_defect_phases.p"
        outputfile = str(seedname) + "_defect_phases.eps"
        addendum = ''
    elif tag == 2:
        output_dir = "defect_phase_dopant"
        graphfile = str(seedname) + "_dopant_phases.p"
        outputfile = str(seedname) + "_dopant_phases.eps"
        addendum = '_dpt'

    os.chdir(output_dir)

    with open(graphfile, 'w') as f:
        # Print header to file
        print("#GNUPLOT script for plotting defect phase diagram\n", file=f)
        print("set terminal postscript eps enhanced color font 'Helvetica,20'", file=f)
        print("set output \"", outputfile, "\"", sep="", file=f)
        print("set encoding iso_8859_1", file=f)

        if (loop == 0):
            print("set xlabel 'log_{10}P_{", volatile_element, "_{2}} /atm'", sep="", file=f)
        elif (loop == 1):
            print("set xlabel 'Temperature /K'", file=f)
        elif (loop == 2):
            print("set xlabel 'log_{10}[", dopant_xvar[0], "] (per ", host_name, ")'", sep="", file=f)
        elif (loop == 3):
            print("set xlabel 'log_{10}[artificial dopant conc] (per ", host_name, ")'", sep="", file=f)
        elif (loop == 4):
            print("set xlabel 'log_{10}P_{", dopant_xvar[0], "_{2}} /atm'", sep="", file=f)
        elif (loop == 5):
            print("set xlabel 'rich-poor fraction (s)'", sep="", file=f)


        if (loop2 == 0):
            print("set ylabel 'log_{10}P_{", volatile_element, "_{2}} /atm'", sep="", file=f)
        elif (loop2 == 1):
            print("set ylabel 'Temperature /K'", file=f)
        elif (loop2 == 2):
            # check if plotting as a func of one or two dopants
            if loop == 2:
                print("set ylabel 'log_{10}[", dopant_xvar[1], "] (per ", host_name, ")'", sep="", file=f)
            else:
                print("set ylabel 'log_{10}[", dopant_xvar[0], "] (per ", host_name, ")'", sep="", file=f)
        elif (loop2 == 3):
            print("set ylabel 'log_{10}[artificial dopant conc] (per ", host_name, ")'", sep="", file=f)
        elif (loop2 == 4):
            if loop == 4:
                print("set ylabel 'log_{10}P_{", dopant_xvar[1], "_{2}} /atm'", sep="", file=f)
            else:
                print("set ylabel 'log_{10}P_{", dopant_xvar[0], "_{2}} /atm'", sep="", file=f)
        elif (loop2 == 5):
            print("set ylabel 'rich-poor fraction (s)'", sep="", file=f)

        print("set xrange [", min_value, ":", max_value, "]", sep="", file=f)
        print("set yrange [", min_value_y, ":", max_value_y, "]", sep="", file=f)
        print("set xtics out", file=f)
        print("set ytics out", file=f)
        print("set key outside", file=f)
        print("set key center below", file=f)
        print("set key horizontal\n", file=f)
        print("set key font 'Helvetica,18'", file=f)
        print("set linetype 2 dt \"_\"", file=f)
        print("set linetype 3 dt 2", file=f)
        print("set linetype 4 dt 4", file=f)
        print("set linetype 5 dt 5", file=f)
        print("set linetype 6 dt 6", file=f)
        print("set linetype 7 dt 7", file=f)
        print("set linetype 8 dt 8", file=f)
        print("set linetype 9 dt 9", file=f)
        print("set style fill  transparent solid 0.35 noborder", file=f)
        print("set style circle radius", increment / 2, file=f)

        i = 0
        while i < len(nametags):

            name = nametags[i]
            resultfile = nametags[i] + addendum + ".dat"
            if i < 13:
                colour = colours[i]
            else:
                colour = colourx
            if i == 0:
                print("plot \"./", resultfile, "\" using 1:2 with circles lc rgb \"", colour, "\" ti \"", name, "\",\\",
                      sep="", file=f)
            else:
                print("\"./", resultfile, "\" using 1:2 with circles lc rgb \"", colour, "\" ti \"", name, "\",\\",
                      sep="", file=f)
            i += 1

    os.chdir("..")


def invert_matrix(input_mat, marker):  # Function for inverting a matrix

    # marker : Factor 0 for inverting dielectric and 1 for inverting lattice
    output_mat = []
    adjoint = []

    if (marker == 0):
        factor = 1
    elif (marker == 1):
        factor = 2 * math.pi

    # Calulate determinant of input matrix
    determinant = det(input_mat)

    # Calculate adjoint matrix
    adjoint.append(input_mat[4] * input_mat[8] - input_mat[7] * input_mat[5])
    adjoint.append(input_mat[3] * input_mat[8] - input_mat[6] * input_mat[5])
    adjoint.append(input_mat[3] * input_mat[7] - input_mat[6] * input_mat[4])
    adjoint.append(input_mat[1] * input_mat[8] - input_mat[7] * input_mat[2])
    adjoint.append(input_mat[0] * input_mat[8] - input_mat[6] * input_mat[2])
    adjoint.append(input_mat[0] * input_mat[7] - input_mat[6] * input_mat[1])
    adjoint.append(input_mat[1] * input_mat[5] - input_mat[4] * input_mat[2])
    adjoint.append(input_mat[0] * input_mat[5] - input_mat[3] * input_mat[2])
    adjoint.append(input_mat[0] * input_mat[4] - input_mat[3] * input_mat[1])

    # Calculate inverse
    output_mat.append((factor * adjoint[0]) / determinant)
    output_mat.append(-(factor * adjoint[1]) / determinant)
    output_mat.append((factor * adjoint[2]) / determinant)
    output_mat.append(-(factor * adjoint[3]) / determinant)
    output_mat.append((factor * adjoint[4]) / determinant)
    output_mat.append(-(factor * adjoint[5]) / determinant)
    output_mat.append((factor * adjoint[6]) / determinant)
    output_mat.append(-(factor * adjoint[7]) / determinant)
    output_mat.append((factor * adjoint[8]) / determinant)

    return (output_mat)


# Subroutine for calculating the determinant of a matrix
def det(input_mat):
    # Calculate determinant
    determinant = input_mat[0] * (input_mat[4] * input_mat[8] - input_mat[7] * input_mat[5]) - input_mat[1] * (
            input_mat[3] * input_mat[8] - input_mat[6] * input_mat[5]) + input_mat[2] * (
                          input_mat[3] * input_mat[7] - input_mat[6] * input_mat[4])

    return (determinant)


# Determine the longest lattice parameter and define limits for the real space
def limits_real(lattice, factor, seedname):
    real_limits = []

    # Calculate the cell lattice parameters
    latt_a = math.sqrt((lattice[0] ** 2 + lattice[1] ** 2 + lattice[2] ** 2))
    latt_b = math.sqrt((lattice[3] ** 2 + lattice[4] ** 2 + lattice[5] ** 2))
    latt_c = math.sqrt((lattice[6] ** 2 + lattice[7] ** 2 + lattice[8] ** 2))

    # Determine which of the lattice parameters is the largest
    if (latt_a >= latt_b) and (latt_a >= latt_c):
        longest = latt_a
    if (latt_b >= latt_a) and (latt_b >= latt_c):
        longest = latt_b
    if (latt_c >= latt_a) and (latt_c >= latt_b):
        longest = latt_c

    # Calculate real space cutoff
    r_c = factor * longest

    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:
        print("   Supercell parameters %.6f  %.6f  %.6f" % (latt_a, latt_b, latt_c), file=f)
        print("   Longest lattice parameter =", longest, file=f)
        print("   Realspace cutoff =", r_c, file=f)

    # Estimate the number of boxes required in each direction to ensure r_c is contained (the tens are added to ensure the number of cells contains $r_c)
    a_range = r_c / latt_a + 10
    b_range = r_c / latt_b + 10
    c_range = r_c / latt_c + 10
    a_range_final = round(a_range)
    b_range_final = round(b_range)
    c_range_final = round(c_range)

    # This defines the size of the supercell in which the real space section is performed, however only atoms within rc will be conunted
    real_limits.append(a_range_final)
    real_limits.append(b_range_final)
    real_limits.append(c_range_final)

    return (real_limits, r_c)


# Function to calculate the real and reciprocal space contributions
def real_recip(lattice, inv_dielectric, motif, real_limits, r_c, gamma, num_atoms, debug, determinant, recip_lattice,
               dielectric, volume, seedname):
    # lattice =		Lattice parallelpiped
    # inv_dielectric = Inverse of the dielectric tensor
    # motif = Motif
    # real_limits = Limits to the real space cell
    # r_c = Real space cutoff
    # gamma = gamma parameter
    # num_atoms = Number of defects in motif
    # debug = Debug flag
    # determinant = Determinant of the dielectric tensor
    # recip_lattice = Reciprocal lattice
    # dielectric = Dielectric tensor
    # volume = Volume of the supercell

    # Calculate superlattice lattice parallelpiped
    superlattice = []
    real_space = 0
    recip_superlattice = []
    reciprocal = 0
    incell = 0

    # Calculate supercell parrallelpiped
    superlattice.append(real_limits[0] * lattice[0])
    superlattice.append(real_limits[0] * lattice[1])
    superlattice.append(real_limits[0] * lattice[2])
    superlattice.append(real_limits[1] * lattice[3])
    superlattice.append(real_limits[1] * lattice[4])
    superlattice.append(real_limits[1] * lattice[5])
    superlattice.append(real_limits[2] * lattice[6])
    superlattice.append(real_limits[2] * lattice[7])
    superlattice.append(real_limits[2] * lattice[8])

    # Calculate the reciprocal space parrallelpiped
    recip_superlattice.append(real_limits[0] * recip_lattice[0])
    recip_superlattice.append(real_limits[0] * recip_lattice[1])
    recip_superlattice.append(real_limits[0] * recip_lattice[2])
    recip_superlattice.append(real_limits[1] * recip_lattice[3])
    recip_superlattice.append(real_limits[1] * recip_lattice[4])
    recip_superlattice.append(real_limits[1] * recip_lattice[5])
    recip_superlattice.append(real_limits[2] * recip_lattice[6])
    recip_superlattice.append(real_limits[2] * recip_lattice[7])
    recip_superlattice.append(real_limits[2] * recip_lattice[8])

    # Print the real space superlattice
    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:

        # Print the real space superlattice
        print("\n   Real space superlattice", file=f)
        print("   %.6f  %.6f  %.6f" % (superlattice[0], superlattice[1], superlattice[2]), file=f)
        print("   %.6f  %.6f  %.6f" % (superlattice[3], superlattice[4], superlattice[5]), file=f)
        print("   %.6f  %.6f  %.6f" % (superlattice[6], superlattice[7], superlattice[8]), file=f)

        # Print the real space superlattice
        print("\n   Reciprocal space superlattice", file=f)
        print("   %.6f  %.6f  %.6f" % (recip_superlattice[0], recip_superlattice[1], recip_superlattice[2]), file=f)
        print("   %.6f  %.6f  %.6f" % (recip_superlattice[3], recip_superlattice[4], recip_superlattice[5]), file=f)
        print("   %.6f  %.6f  %.6f" % (recip_superlattice[6], recip_superlattice[7], recip_superlattice[8]), file=f)

    ###########################
    # Real space contribution #
    ###########################
    print("..> Calcualting real space contribution")

    with open('REAL_SPACE', 'a') as f:
        # Loop over all atoms in the motif and calculate contributions
        for i in np.arange(0, num_atoms, 1):
            i = int(i)

            # Convert fractional motif co-ordinates to cartesian
            motif_charge = motif[4 * i + 3]
            motif_cart_x = motif[4 * i] * lattice[0] + motif[4 * i + 1] * lattice[3] + motif[4 * i + 2] * lattice[6]
            motif_cart_y = motif[4 * i] * lattice[1] + motif[4 * i + 1] * lattice[4] + motif[4 * i + 2] * lattice[7]
            motif_cart_z = motif[4 * i] * lattice[2] + motif[4 * i + 1] * lattice[5] + motif[4 * i + 2] * lattice[8]
            # printf ("Cartesian defect co-ordinates %.6f  %.6f  %.6f %.6f\n\n", $motif_cart_x, $motif_cart_y, $motif_cart_z, $motif_charge);
            # printf ("Fractional defect co-ordinates %.6f  %.6f  %.6f %.6f\n\n", $motif[4*$i], $motif[4*$i+1], $motif[4*$i+2], $motif_charge);

            # Convert fractional co-ordinates to reciprocal space
            motif_recip_x = motif[4 * i] * recip_lattice[0] + motif[4 * i + 1] * recip_lattice[3] + motif[4 * i + 2] * \
                            recip_lattice[6]
            motif_recip_y = motif[4 * i] * recip_lattice[1] + motif[4 * i + 1] * recip_lattice[4] + motif[4 * i + 2] * \
                            recip_lattice[7]
            motif_recip_z = motif[4 * i] * recip_lattice[2] + motif[4 * i + 1] * recip_lattice[5] + motif[4 * i + 2] * \
                            recip_lattice[8]
            # printf ("Reciprocal space defect co-ordinates %.6f  %.6f  %.6f %.6f\n\n", $motif_recip_x, $motif_recip_y, $motif_recip_z, $motif_charge);

            # Loop over all other atoms in the motif
            for j in np.arange(0, num_atoms, 1):
                j = int(j)

                incell_contribution = 0

                image_charge = motif[4 * j + 3]

                # Loop over all points in the superlattice
                for m in np.arange(-real_limits[0], real_limits[0], 1):
                    m = int(m)

                    for n in np.arange(-real_limits[1], real_limits[1], 1):
                        n = int(n)

                        for o in np.arange(-real_limits[2], real_limits[2], 1):
                            o = int(o)

                            real_contribution = 0
                            recip_contribution = 0

                            # Calculate the defect's fractional position in the extended supercell
                            x_super = 1 / (real_limits[0]) * m + motif[4 * j + 0] / (real_limits[0])
                            y_super = 1 / (real_limits[1]) * n + motif[4 * j + 1] / (real_limits[1])
                            z_super = 1 / (real_limits[2]) * o + motif[4 * j + 2] / (real_limits[2])

                            # Convert these fractional co-ordinates to cartesian
                            x_cart = x_super * superlattice[0] + y_super * superlattice[3] + z_super * superlattice[6]
                            y_cart = x_super * superlattice[1] + y_super * superlattice[4] + z_super * superlattice[7]
                            z_cart = x_super * superlattice[2] + y_super * superlattice[5] + z_super * superlattice[8]

                            # Test to see whether the new atom coordinate falls within r_c and then solve
                            seperation = math.sqrt((x_cart - motif_cart_x) ** 2 + (y_cart - motif_cart_y) ** 2 + (
                                    z_cart - motif_cart_z) ** 2)

                            if ((i == j) and (m == 0) and (n == 0) and (
                                    o == 0)):  # Setting separation == 0 does not always work for numerical reasons

                                # print("Found the central atom", m,n,o,  motif_cart_x, motif_cart_y, motif_cart_z,"\n")
                                incell += 0

                            elif (seperation < r_c):

                                mod_x = (x_cart - motif_cart_x) * inv_dielectric[0] + (y_cart - motif_cart_y) * \
                                        inv_dielectric[3] + (z_cart - motif_cart_z) * inv_dielectric[6]
                                mod_y = (x_cart - motif_cart_x) * inv_dielectric[1] + (y_cart - motif_cart_y) * \
                                        inv_dielectric[4] + (z_cart - motif_cart_z) * inv_dielectric[7]
                                mod_z = (x_cart - motif_cart_x) * inv_dielectric[2] + (y_cart - motif_cart_y) * \
                                        inv_dielectric[5] + (z_cart - motif_cart_z) * inv_dielectric[8]

                                dot_prod = mod_x * (x_cart - motif_cart_x) + mod_y * (y_cart - motif_cart_y) + mod_z * (
                                        z_cart - motif_cart_z)

                                # This section calculates the Coulombic interactions inside the defect supercell
                                if (m == 0) and (n == 0) and (o == 0):
                                    incell_contribution = motif_charge * image_charge * (
                                            1 / (math.sqrt(determinant))) * (1 / (math.sqrt(dot_prod)))
                                    incell += incell_contribution
                                    # print("Calculating inner energy",motif_cart_x, motif_cart_y ,motif_cart_z ,x_cart ,y_cart ,z_cart ,incell_contribution ,incell)

                                real_contribution = (motif_charge * image_charge) * (1 / (math.sqrt(determinant)) * (
                                    special.erfc(gamma * math.sqrt(dot_prod))) / (math.sqrt(dot_prod)))

                                if (debug == 1):
                                    print(x_cart, y_cart, z_cart, seperation, dot_prod, real_contribution, file=f)

                                real_space += real_contribution

    #################################
    # Reciprocal space contribution #
    #################################
    print("..> Calcualting reciprocal space contribution")

    with open('RECIPROCAL_SPACE', 'a') as f:
        # Loop over all k-points
        recip_contribution = 0
        for s in np.arange(-real_limits[0], real_limits[0], 1):
            s = int(s)

            for t in np.arange(-real_limits[1], real_limits[1], 1):
                t = int(t)

                for u in np.arange(-real_limits[2], real_limits[2], 1):
                    u = int(u)

                    # Determine which k-point to calculate
                    x_recip_super = 1 / (real_limits[0]) * s
                    y_recip_super = 1 / (real_limits[1]) * t
                    z_recip_super = 1 / (real_limits[2]) * u

                    # Convert to reciprocal space
                    x_recip = x_recip_super * recip_superlattice[0] + y_recip_super * recip_superlattice[
                        3] + z_recip_super * recip_superlattice[6]
                    y_recip = x_recip_super * recip_superlattice[1] + y_recip_super * recip_superlattice[
                        4] + z_recip_super * recip_superlattice[7]
                    z_recip = x_recip_super * recip_superlattice[2] + y_recip_super * recip_superlattice[
                        5] + z_recip_super * recip_superlattice[8]

                    # my $recip_seperation = sqrt(($x_recip-$motif_recip_x)**2 + ($y_recip-$motif_recip_y)**2 + ($z_recip-$motif_recip_z)**2);
                    if (s == 0) and (t == 0) and (u == 0):
                        recip_contribution += 0
                        # print("Found image in reciprocal space", x_recip, y_recip,z_recip)

                    else:

                        recip_mod_x = x_recip * dielectric[0] + y_recip * dielectric[3] + z_recip * dielectric[6]
                        recip_mod_y = x_recip * dielectric[1] + y_recip * dielectric[4] + z_recip * dielectric[7]
                        recip_mod_z = x_recip * dielectric[2] + y_recip * dielectric[5] + z_recip * dielectric[8]
                        recip_dot_prod = recip_mod_x * x_recip + recip_mod_y * y_recip + recip_mod_z * z_recip

                        structure_factor = ((4 * math.pi) / volume) * (1 / recip_dot_prod) * (
                            math.exp(-recip_dot_prod / (4 * (gamma ** 2))))

                        cos_cumulative = 0
                        sin_cumulative = 0

                        # Loop over all atoms in the motif
                        for w in np.arange(0, num_atoms, 1):
                            w = int(w)

                            # Convert fractional motif co-ordinates to cartesian
                            motif_charge = motif[4 * w + 3]
                            motif_cart_x = motif[4 * w] * lattice[0] + motif[4 * w + 1] * lattice[3] + motif[
                                4 * w + 2] * lattice[6]
                            motif_cart_y = motif[4 * w] * lattice[1] + motif[4 * w + 1] * lattice[4] + motif[
                                4 * w + 2] * lattice[7]
                            motif_cart_z = motif[4 * w] * lattice[2] + motif[4 * w + 1] * lattice[5] + motif[
                                4 * w + 2] * lattice[8]

                            rdotG = motif_cart_x * x_recip + motif_cart_y * y_recip + motif_cart_z * z_recip

                            cos_term = motif_charge * math.cos(rdotG)
                            sin_term = motif_charge * math.sin(rdotG)

                            cos_cumulative += cos_term
                            sin_cumulative += sin_term

                        recip_contribution = structure_factor * (cos_cumulative ** 2 + sin_cumulative ** 2)
                        # $recip_contribution = ($motif_charge*$image_charge)*(((4*pi)/$volume) * exp(-$rdotG) * ((exp(-$recip_dot_prod/(4*($gamma**2))))/$recip_dot_prod));
                        # print "$s $t $u $recip_cont\n";
                        # $current_atm_recip += $recip_contribution;
                        reciprocal += recip_contribution
                        if (debug == 1):
                            print(s, t, u, x_recip, y_recip, z_recip, recip_contribution, file=f)

    return (real_space, reciprocal, incell, 1, 1)


# Subroutine for calculating the self interaction term
def self_interaction(motif, gamma, determinant, num_atoms):
    summation = 0
    for k in np.arange(0, num_atoms, 1):
        k = int(k)

        defect_charge = motif[4 * k + 3]
        # $summation += ($defect_charge**2) * (sqrt($gamma/(3.141592654*$determinant)));
        summation += (defect_charge ** 2)

    self_interaction = -((2 * gamma) / math.sqrt(3.141592654 * determinant)) * summation
    # my $self_interaction = -$summation;

    return (self_interaction)


# Subroutine for calculating the background contribution to the Madelung potential
def background(volume, gamma, total_charge):
    background_term = -3.141592654 / (volume * gamma ** 2) * total_charge ** 2

    return (background_term)


# Function for printing the final results
def madelung_results(real_space, reciprocal, self_interaction, background_contribution, incell, num_atoms, seedname):
    # Unit conversion factor
    conversion = 14.39942

    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:
        # Print the results based on the number of atoms
        if (num_atoms == 1):

            final_madelung = real_space + reciprocal + self_interaction + background_contribution - incell
            final_madelung_eV = (final_madelung * conversion) / 2

            print("\n   --------------------------------------------------", file=f)
            print("   Results                      ", file=f)
            print("   --------------------------------------------------", file=f)
            print("   Real space contribution    =", real_space, file=f)
            print("   Reciprocal space component =", reciprocal, file=f)
            print("   Self interaction           =", self_interaction, file=f)
            print("   Neutralising background    =", background_contribution, file=f)
            print("   --------------------------------------------------", file=f)
            print("   Final Madelung potential   =", final_madelung, file=f)
            print("   --------------------------------------------------\n", file=f)

            # Print final point charge correction
            print("   Example corrections using the calculated Madelung potential:", file=f)
            print("   +--------+------------------+-----------------+", file=f)
            print("   | Charge | Point charge /eV | Lany-Zunger /eV |", file=f)
            print("   +--------+------------------+-----------------+", file=f)

            for chge_state in np.arange(1, 7, 1):
                chge_state = int(chge_state)

                makov_payne = 1 / 2 * final_madelung * chge_state ** 2 * conversion
                lany = 0.65 * makov_payne

                print("   |   %i    |     %.10s   |    %.10s   |" % (chge_state, makov_payne, lany), file=f)

            print("   +--------+------------------+-----------------+\n", file=f)

        elif (num_atoms > 1):

            real_space_eV = (real_space * conversion) / 2
            reciprocal_eV = (reciprocal * conversion) / 2
            self_interaction_eV = (self_interaction * conversion) / 2
            background_eV = (background_contribution * conversion) / 2
            incell_eV = (incell * conversion) / 2

            total = real_space_eV + reciprocal_eV + self_interaction_eV + background_eV
            final_madelung_eV = -(total - incell_eV)

            print("   ----------------------------------------------", file=f)
            print("   Results                        Energy /eV ", file=f)
            print("   ----------------------------------------------", file=f)
            print("   Real space contribution    =", real_space_eV, file=f)
            print("   Reciprocal space component =", reciprocal_eV, file=f)
            print("   Self interaction           =", self_interaction_eV, file=f)
            print("   Neutralising background    =", background_eV, file=f)
            print("   Total                      =", total, file=f)
            print("   ----------------------------------------------", file=f)
            print("   Internal interaction       =", incell_eV, file=f)
            print("   ----------------------------------------------", file=f)
            print("   Final correction           =", final_madelung_eV, file=f)
            print("   ----------------------------------------------\n", file=f)

    return final_madelung


def madelung(seedname):
    ####################################################################################
    # Madelung potential for a periodic system with anisotropic dielectric properties.
    ####################################################################################

    # Read in information from input file
    dielectric, lattice, motif, gamma, real_cutoff, num_atoms, total_charge, debug = madelung_input(seedname)

    # Calculate reciprocal lattice
    recip_lattice = invert_matrix(lattice, 1)

    # Calculate inverse of the dielectric
    inv_dielectric = invert_matrix(dielectric, 0)

    # Calculate volume
    volume = det(lattice)
    if (volume < 0):  # Check to make sure determinant (and hence volume) isn't negative
        volume = -volume

    # Calculate the determinant of the inverse dielectric
    determinant = det(dielectric)

    outputfile = str(seedname) + ".output"
    with open(outputfile, 'a') as f:
        print("\n   Reciprocal space lattice:", file=f)
        print("   %.6f  %.6f  %.6f" % (recip_lattice[0], recip_lattice[1], recip_lattice[2]), file=f)
        print("   %.6f  %.6f  %.6f" % (recip_lattice[3], recip_lattice[4], recip_lattice[5]), file=f)
        print("   %.6f  %.6f  %.6f" % (recip_lattice[6], recip_lattice[7], recip_lattice[8]), file=f)
        print("\n   Inverse dielectric tensor:", file=f)
        print("   %.6f  %.6f  %.6f" % (inv_dielectric[0], inv_dielectric[1], inv_dielectric[2]), file=f)
        print("   %.6f  %.6f  %.6f" % (inv_dielectric[3], inv_dielectric[4], inv_dielectric[5]), file=f)
        print("   %.6f  %.6f  %.6f" % (inv_dielectric[6], inv_dielectric[7], inv_dielectric[8]), file=f)
        print("\n   Volume of the cell =", volume, "A^3", file=f)
        print("   Determinant of the dielectric tensor =", determinant, "\n", file=f)

    # Calculate limits for the real and reciprocal space sums
    real_limits, r_c = limits_real(lattice, real_cutoff, seedname)

    # Calculate real space term
    real_space, reciprocal, incell, real_duration, recip_duration = real_recip(lattice, inv_dielectric, motif,
                                                                               real_limits, r_c, gamma, num_atoms,
                                                                               debug, determinant, recip_lattice,
                                                                               dielectric, volume, seedname)

    # Calculate the self interaction term
    print("..> Calculating self interaction term")
    self_interaction_contribution = self_interaction(motif, gamma, determinant, num_atoms)

    # Calculate contribution to energy due to interaction with background potential
    print("..> Calculating background contribution")
    if (total_charge != 0):
        background_contribution = background(volume, gamma, total_charge)
        # print("Background contribution =",background_contribution,"eV")

    else:
        background_contribution = 0

    # Print results
    print("..> Printing final results and Madelung potential in", outputfile)

    v_M = madelung_results(real_space, reciprocal, self_interaction_contribution, background_contribution, incell,
                           num_atoms, seedname)

    return v_M


def bibliography(tasks, chem_pot_method, real_gas, entropy_marker):
    # Printing bibliograhy for processes used.
    print("..> Writing bibliography for processes used, in DefAP.bib")
    with open('DefAP.bib', 'w') as f:

        print("%DefAP Publication", file=f)
        print("@article{DefAP2022,", file=f)
        print("author = {Neilson, William D and Murphy, Samuel T },", file=f)
        print("doi = {https://doi.org/10.1016/j.commatsci.2022.111434},", file=f)
        print("issn = {0927-0256},", file=f)
        print("journal = {Comput. Mater. Sci.},", file=f)
        print("month = {feb},", file=f)
        print("pages = {111434},", file=f)
        print("title = {DefAP: A Python code for the analysis of point defects in crystalline solids},", file=f)
        print("volume = {210},", file=f)
        print("year = {2022}", file=f)
        print("}", file=f)

        if ('brouwer' in tasks) or ('energy' in tasks):
            print("\n%Defect formation energy", file=f)
            print("@article{Zhang1991,", file=f)
            print(
                "title = {Chemical potential dependence of defect formation energies in GaAs: Application to Ga self-diffusion},",
                file=f)
            print("author = {Zhang, S. B. and Northrup, John E.},", file=f)
            print("journal = {Phys. Rev. Lett.},", file=f)
            print("volume = {67},", file=f)
            print("issue = {17},", file=f)
            print("pages = {2339--2342},", file=f)
            print("year = {1991},", file=f)
            print("month = {Oct},", file=f)
            print("publisher = {American Physical Society},", file=f)
            print("doi = {10.1103/PhysRevLett.67.2339},", file=f)
            print("}", file=f)

        if (chem_pot_method >= 2):

            print("\n%Volatile chemical potential method.", file=f)
            print("@article{Finnis2005,", file=f)
            print("annote = {doi: 10.1146/annurev.matsci.35.101503.091652},", file=f)
            print("author = {Finnis, M W and Lozovoi, A Y and Alavi, A},", file=f)
            print("doi = {10.1146/annurev.matsci.35.101503.091652},", file=f)
            print("issn = {1531-7331},", file=f)
            print("journal = {Annu. Rev. Mater. Res.},", file=f)
            print("month = {jun},", file=f)
            print("pages = {167--207},", file=f)
            print("publisher = {Annual Reviews},", file=f)
            print("title = {{The Oxidation Of NiAl: What Can We Learn from Ab Initio Calculations?}},", file=f)
            print("volume = {35},", file=f)
            print("year = {2005}", file=f)
            print("}", file=f)

            if real_gas == 2:

                print("\n%Volatile chemical potential temperature dependence parameters.", file=f)
                print("@article{Johnston2004,", file=f)
                print(
                    "author = {Johnston, Karen and Castell, Martin R. and Paxton, Anthony T. and Finnis, Michael W.},",
                    file=f)
                print("doi = {10.1103/PhysRevB.70.085415},", file=f)
                print("issn = {1098-0121},", file=f)
                print("journal = {Phys. Rev. B},", file=f)
                print("month = {aug},", file=f)
                print("pages = {085415},", file=f)
                print("publisher = {American Physical Society},", file=f)
                print(
                    "title = {{SrTiO$_3$ (001) (2 $\times$ 1) reconstructions: First-principles calculations of surface energy and atomic structure compared with scanning tunneling microscopy images}},",
                    file=f)
                print("volume = {70},", file=f)
                print("year = {2004}", file=f)
                print("}", file=f)

            else:

                print("\n%Volatile chemical potential temperature dependence parameters.", file=f)
                print("@book{NIST1,", file=f)
                print("address = {National Institute of Standards and Technology, Gaithersburg MD, 20899},", file=f)
                print("editor = {Linstrom, P.J and Mallard, W.J},", file=f)
                print("title = {{NIST Chemistry WebBook, NIST Standard Reference Database Number 69}},", file=f)
                print("url = {https://doi.org/10.18434/T4D303}", file=f)
                print("}", file=f)

        if (entropy_marker == 1):
            print("\n%Vibrational entropy method.", file=f)
            print("@article{Soulie2018,", file=f)
            print(
                "author = {Souli{\'{e}}, Aur{\'{e}}lien and Bruneval, Fabien and Marinica, Mihai-Cosmin and Murphy, Samuel and Crocombette, Jean-Paul},",
                file=f)
            print("doi = {10.1103/PhysRevMaterials.2.083607},", file=f)
            print("issn = {2475-9953},", file=f)
            print("journal = {Phys. Rev. Mater.},", file=f)
            print("month = {aug},", file=f)
            print("pages = {083607},", file=f)
            print("publisher = {American Physical Society},", file=f)
            print(
                "title = {{Influence of vibrational entropy on the concentrations of oxygen interstitial clusters and uranium vacancies in nonstoichiometric UO$_2$}},",
                file=f)
            print("volume = {2},", file=f)
            print("year = {2018}", file=f)
            print("}", file=f)

            print("\n%Vibrational entropy method.", file=f)
            print("@article{Cooper2018,", file=f)
            print("author = {Cooper, M. W.D. and Murphy, S. T. and Andersson, D. A.},", file=f)
            print("doi = {10.1016/j.jnucmat.2018.02.034},", file=f)
            print("issn = {00223115},", file=f)
            print("journal = {J. Nucl. Mater.},", file=f)
            print("pages = {251--260},", file=f)
            print("title = {{The defect chemistry of UO$_{2\pm x }$ from atomistic simulations}},", file=f)
            print("volume = {504},", file=f)
            print("year = {2018}", file=f)
            print("}", file=f)

        if ('madelung' in tasks):
            print("\n%Screened Madelung potential", file=f)
            print("@article{Murphy2013,", file=f)
            print("title = {Anisotropic charge screening and supercell size convergence of defect formation energies},",
                  file=f)
            print("author = {Murphy, Samuel T. and Hine, Nicholas D. M.},", file=f)
            print("journal = {Phys. Rev. B},", file=f)
            print("volume = {87},", file=f)
            print("issue = {9},", file=f)
            print("pages = {094111},", file=f)
            print("numpages = {6},", file=f)
            print("year = {2013},", file=f)
            print("month = {Mar},", file=f)
            print("publisher = {American Physical Society},", file=f)
            print("doi = {10.1103/PhysRevB.87.094111},", file=f)
            print("}", file=f)


def looper(loop, b, constituents, chem_pot_method, temperature, entropies, constituent_entropies, entropy_data,
           number_of_defects, constituents_name_list, seedname, dopants, dopant_xvar, plot_second_dopant_flag,
           art_dop_conc, volatile_element,
           gibbs_marker, gibbs_data, host_energy):

    if (loop == 0):  # Volatile partial pressure
        volatile_element = constituents[0]
        if (chem_pot_method == 2 or chem_pot_method == 3 or chem_pot_method == 4):
            constituents[1] = b
        elif (chem_pot_method == 5):
            constituents[2] = b
        environment = "partial pressure"

    if (loop == 1):  # Temperature
        temperature = b
        environment = "temperature"

    if (loop == 2):  # Dopant concentration
        number_dopants = int(dopants[0])

        fit_counter = 0
        for i in np.arange(0, number_dopants, 1):
            fit_potential = float(dopants[int((6 * i) + 3)])
            if fit_potential == 2:

                if plot_second_dopant_flag == True:
                    # if plotting second dopant on y-axis, skip first instance of dopant with fitting option = 2
                    plot_second_dopant_flag = False
                    continue

                else:
                    dopants[int((6 * i) + 4)] = (10 ** b)

                    dopant_name = dopants[(6 * i) + 1]
                    if len(dopant_xvar) < 2:
                        dopant_xvar.append(dopant_name)

        environment = "dopant concentration"

    if (loop == 3):  # Artificial charge
        art_dop_conc = (10 ** b)
        environment = "artificial dopant concentration"

    if (loop == 4):  # Dopant partial pressure
        number_dopants = int(dopants[0])

        fitting_dopants_index = []
        fit_counter = 0
        for i in np.arange(0, number_dopants, 1):
            fit_potential = float(dopants[int((6 * i) + 3)])
            if fit_potential == 4:
                dopants[int((6 * i) + 6)] = b
                dopant_xvar.append(dopants[int((6 * i) + 1)])
                fit_counter += 1
                fitting_dopants_index.append(i)

        if fit_counter == 0:
            raise Exception("<!> No dopant selected as independent variable for loop = 4. Review input file")
        elif fit_counter > 1:
            newline = "\n"  # \escapes not allowed in f-strings
            raise Exception("<!> Too many dopants selected as independent variable for loop = 4. Review input file\n"
                            f"{newline.join(f'{dopants[6 * i + 1:6 * i + 7]}' for i in fitting_dopants_index)}")  # prints the dopants with the fitting option set to 4

        environment = "dopant partial pressure"

    if loop == 5:
        if chem_pot_method == 1:
            constituents[5] = (1-b)
            constituents[2] = b
        elif chem_pot_method == 5:
            constituents[14] = (1-b)
            constituents[8] = b
        else:
            raise Exception(
                f"<!> Error. Loop = 5 (Rich-poor fraction) can only be used with the rich-poor and volatile-rich-poor chem_pot_methods")
        environment = "rich-poor fraction"


    return (
        constituents, temperature, entropies, constituent_entropies, environment, dopants, dopant_xvar, art_dop_conc,
        volatile_element, host_energy)


@main_wrapper
def main():
    if len(sys.argv) != 2:
        raise Exception("No input file has been provided, remember to include")

    seedname = sys.argv[1]

    outputfile = str(seedname) + ".output"
    if os.path.exists(outputfile):
        os.remove(outputfile)
    if os.path.exists('REAL_SPACE'):
        os.remove('REAL_SPACE')
    if os.path.exists('RECIPROCAL_SPACE'):
        os.remove('RECIPROCAL_SPACE')

    # Create some arrays to store the data

    details = []
    defects = []
    final_concentrations = []
    fermi = []
    stoichiometry_list = []
    entropy_data = []
    all_plot_master = []
    A_nametags = []
    dopant_plot_master = []
    D_nametags = []
    # Defaults
    indicator = 0
    dopant_xvar = []
    volatile_element = ''
    concentration_check = 0
    gibbs_data = ''
    plot_second_dopant_flag = False

    # Read in data
    host_array, dopants, tasks, constituents, constituents_name_list, temperature, def_statistics, tab_correction, host_energy, chem_pot_method, host_supercell, use_coul_correction, length, dielectric, v_M, E_VBM, bandgap, condband, valband, electron_method, hole_method, fixed_e_conc, fixed_p_conc, art_dop_conc, art_dop_charge, loop, min_value, max_value, iterator, gnuplot_version, min_y_range, max_y_range, host_name, val_band_min, val_band_max, cond_band_min, cond_band_max, y_form_min, y_form_max, lines, entropy_marker, entropy_units, fu_uc, electron_mass_eff, hole_mass_eff, unit_vol, charge_convergence, potential_convergence, stability, scheme, stoichiometry, x_variable, real_gas, function_tol, SLSQP_dial, maxiter_dop, y_variable, loop2, min_value_y, max_value_y, iterator_y, accommodate, gibbs_marker, plot_art_dopant_conc = inputs(
        seedname)


    # Read in data from defects file
    for i in tasks:
        if i in ['brouwer', 'energy', 'form_plots', 'autodisplay', 'stability', 'group', 'phases', 'defect_phase']:
            defects, number_of_defects, total_species, charged_sys = read_defects(seedname, host_array, defects,
                                                                                  dopants)
            break

    # Determine whether entropy is being used
    if (entropy_marker == 1):
        entropy_data = read_entropy(seedname, defects, number_of_defects, constituents_name_list, chem_pot_method)
        if (loop == 1):
            entropies, constituent_entropies = calc_entropy(entropy_data, min_value, number_of_defects,
                                                            constituents_name_list, chem_pot_method, seedname, 1)
            entropies, constituent_entropies = calc_entropy(entropy_data, max_value, number_of_defects,
                                                            constituents_name_list, chem_pot_method, seedname, 1)
        else:
            entropies, constituent_entropies = calc_entropy(entropy_data, temperature, number_of_defects,
                                                            constituents_name_list, chem_pot_method, seedname, 1)
    else:
        entropies, constituent_entropies = 0, 0

    # read gibbs file if specified
    if (gibbs_marker == 1):
        gibbs_data = read_gibbs(seedname)
        # Perform a check on input data
        gibbs_energies = calc_gibbs(gibbs_data, temperature, constituents_name_list, chem_pot_method, host_energy,
                                    constituents, stability, seedname, 1)
    else:
        gibbs_energies = None

    # Read in plotting customisation
    if scheme == 1:
        conc_colour, form_colour, electron_colour, hole_colour = graphical_inputs(seedname)
    else:
        conc_colour, form_colour, electron_colour, hole_colour = 0, 0, 0, 0


    # get density of states date if fermi-dirac electron or hole method
    if electron_method == 2 or hole_method == 2:
        dos_data_lst = get_dos_data(seedname)
    else:
        dos_data_lst = None

    # Calculate madelung potential task
    if ('madelung' in tasks):
        print("\n>>> Task:'madelung':")
        v_M = madelung(seedname)
    # elif chem_pot_method == 4:
    #    print("<!> Error : Unknown chem_pot_method")
    #    exit()

    # Formation energy task
    if ('energy' in tasks):
        print("\n>>> Task:'energy':")


        # Calculate chemical potentials of host atoms
        chemical_potentials = []

        if (chem_pot_method == 0):
            chemical_potentials = calc_chemical_defined(host_array, constituents, chemical_potentials, host_energy,
                                                        temperature, entropy_marker, constituent_entropies,
                                                        entropy_units)
            pp = 'x'
        elif (chem_pot_method == 1):
            chemical_potentials = calc_chemical_rich_poor(host_array, constituents, chemical_potentials, host_energy,
                                                          temperature, entropy_marker, constituent_entropies,
                                                          entropy_units)
            pp = 'x'
        elif (chem_pot_method == 2 or chem_pot_method == 3 or chem_pot_method == 4):
            chemical_potentials = calc_chemical_volatile(host_array, constituents_name_list, constituents, chemical_potentials,
                                                         host_energy,
                                                         temperature, entropy_marker, constituent_entropies,
                                                         entropy_units,
                                                         real_gas)
            pp = constituents[1]
        elif (chem_pot_method == 5):
            chemical_potentials = calc_chemical_volatile_rich_poor(host_array, constituents_name_list, constituents,
                                                                   chemical_potentials,
                                                                   host_energy, temperature, entropy_marker,
                                                                   constituent_entropies, entropy_units, real_gas,
                                                                   gibbs_energies)
            pp = constituents[2]



        opt_chem_pot = 0
        # Calculate the dopant chemical potentials
        if (dopants[0] > 0):
            chemical_potentials, opt_chem_pot = dopant_chemical(dopants, chemical_potentials, temperature, real_gas)

        nu_e = 1
        # Optimise the dopant checmial potentials, if requsted
        if opt_chem_pot == 1:
            chemical_potentials = calc_opt_chem_pot(pp, loop, defects, dopants, chemical_potentials, number_of_defects,
                                                    host_supercell, tab_correction, E_VBM, total_species,
                                                    use_coul_correction, length, dielectric, v_M, bandgap, temperature,
                                                    def_statistics, nu_e, condband, valband, electron_method,
                                                    hole_method,
                                                    fixed_e_conc, fixed_p_conc, art_dop_conc, art_dop_charge,
                                                    charge_convergence, val_band_min, val_band_max, cond_band_min,
                                                    cond_band_max, seedname, entropies, entropy_marker, fu_uc,
                                                    electron_mass_eff,
                                                    hole_mass_eff, unit_vol, charged_sys, potential_convergence,
                                                    function_tol, maxiter_dop, 'energy', 0, 0, 0, real_gas, SLSQP_dial,
                                                    constituents_name_list, dos_data_lst)

            # Calclate the defect formation energies
        defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction, E_VBM,
                                       total_species, use_coul_correction, length, dielectric, v_M, 0)

        # Print formation energies
        defect_types = form_energies(defects_form, number_of_defects, tasks, bandgap, seedname, entropies,
                                     entropy_marker,
                                     temperature)

        # Perform stability check, if requested.
        if ('stability' in tasks):
            stability_printout, indicator = stability_check(stability, chemical_potentials, indicator, temperature,
                                                            real_gas)
            outputfile = str(seedname) + ".output"
            with open(outputfile, 'a') as f:
                print("\n   Stability check results", file=f)
                print(
                    "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                    file=f)
                print(
                    "   |     Compound     | Total energy (eV) | Sum of chemical potentials (eV) | Difference (eV) |  Thermodynamically Stable  |",
                    file=f)
                print(
                    "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                    file=f)
                for i in np.arange(0, stability[0], 1):
                    i = int(i)
                    compound = stability_printout[i][0]
                    compound_energy = float(stability_printout[i][1])
                    chem_pot_sum = float(stability_printout[i][2])
                    diff = float(stability_printout[i][3])
                    message = stability_printout[i][4]
                    print("   | %16s | %17f | %31f | %15f | %26s |" % (
                        compound, compound_energy, chem_pot_sum, diff, message), file=f)
                print(
                    "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                    file=f)

        if ('form_plots' in tasks):
            print("\n>>> Task:'form_plots':")

            # Print defect formation energies
            formation_graphical_output(seedname, bandgap, defects, y_form_min, y_form_max, form_colour, scheme,
                                       number_of_defects, total_species, defect_types)

            # Create formation energy figures, in new directroy

            # Due to number of plots, a new directory is made to store plots.
            directory = "mkdir " + str(seedname) + "_formation_plots"
            mv_graphfile = "mv formation_plot.p " + str(seedname) + ".formation " + str(
                seedname) + ".formation_grouped " + str(seedname) + "_formation_plots"
            directory_i = str(seedname) + "_formation_plots"
            form_graphfile = "gnuplot formation_plot.p"

            if os.path.exists(directory_i):
                shutil.rmtree(directory_i)

            os.system(directory)
            os.system(mv_graphfile)
            os.chdir(directory_i)
            os.system(form_graphfile)
            os.chdir('../')

            print("..> Successfully plotted formation energies. Plots found in", directory_i, )

    # Initialise progress meter
    prog_meter = 1

    # Brouwer diagram task
    if ('brouwer' in tasks):
        print("\n>>> Task:'brouwer':")

        # Calculate the number of iterations in the loop
        num_iter = ((max_value - min_value) / iterator) + 1
        # print("Number of iterations in the loop =",num_iter)

        with open(outputfile, 'a') as f:

            # Loop over the requested range
            for b in np.arange(min_value, max_value + (iterator / 2), iterator):

                print(
                    "\n-------------------------------------------------------------------------------------------------------------------",
                    "\n", file=f)
                print(">>> Task = brouwer, condition", prog_meter, "of", num_iter, "\n", file=f)

                if loop == 1:
                    if (entropy_marker == 1):
                        entropies, constituent_entropies = calc_entropy(entropy_data, b, number_of_defects,
                                                                        constituents_name_list, chem_pot_method, seedname,
                                                                        0)
                    if (gibbs_marker == 1):
                        gibbs_energies = calc_gibbs(gibbs_data, b, constituents_name_list, chem_pot_method, host_energy,
                                                    constituents, stability, seedname, 1)



                # Loop assignment
                constituents, temperature, entropies, constituent_entropies, environment, dopants, dopant_xvar, art_dop_conc, volatile_element, host_energy = looper(
                    loop, b, constituents, chem_pot_method, temperature, entropies, constituent_entropies, entropy_data,
                    number_of_defects, constituents_name_list, seedname, dopants, dopant_xvar, plot_second_dopant_flag,
                    art_dop_conc, volatile_element, gibbs_marker, gibbs_data, host_energy)

                if (x_variable == 1):  # Stoichiometry
                    stoichiometry = 1

                prog_bar = round((prog_meter / num_iter) * 25)
                print("..> Calculating defect concentrations for", environment, prog_meter, "of", num_iter,
                      " [{0}]   ".format('#' * (prog_bar) + ' ' * (25 - prog_bar)), end="\r", flush=True)

                # Calculate chemical potentials of host atoms

                chemical_potentials = []
                if (chem_pot_method == 0):
                    chemical_potentials = calc_chemical_defined(host_array, constituents, chemical_potentials, host_energy,
                                                                temperature, entropy_marker, constituent_entropies,
                                                                entropy_units)
                elif (chem_pot_method == 1):
                    chemical_potentials = calc_chemical_rich_poor(host_array, constituents, chemical_potentials,
                                                                  host_energy, temperature, entropy_marker,
                                                                  constituent_entropies, entropy_units)
                elif (chem_pot_method == 2 or chem_pot_method == 3 or chem_pot_method == 4):
                    chemical_potentials = calc_chemical_volatile(host_array, constituents_name_list, constituents,
                                                                 chemical_potentials, host_energy, temperature,
                                                                 entropy_marker, constituent_entropies, entropy_units,
                                                                 real_gas)
                    pp = constituents[1]
                elif (chem_pot_method == 5):
                    chemical_potentials = calc_chemical_volatile_rich_poor(host_array, constituents_name_list, constituents,
                                                                           chemical_potentials, host_energy, temperature,
                                                                           entropy_marker, constituent_entropies,
                                                                           entropy_units, real_gas, gibbs_energies)
                    pp = constituents[2]

                opt_chem_pot = 0
                # Calculate the dopant checmical potentials
                if (dopants[0] > 0):

                    chemical_potentials, opt_chem_pot = dopant_chemical(dopants, chemical_potentials, temperature, real_gas)

                    # Update reference for dopant chemical potential: only works for one dopant specie.
                    if ('update_reference' in tasks) and (dopants[3] == 0):
                        indicator = 0
                        if loop == 1:
                            stability_printout, indicator = stability_check(stability, chemical_potentials, indicator, b,
                                                                            real_gas)
                        else:
                            stability_printout, indicator = stability_check(stability, chemical_potentials, indicator,
                                                                            temperature, real_gas)

                        while indicator == 1:
                            max_diff = 0
                            for i in np.arange(0, stability[0], 1):
                                i = int(i)
                                compound = stability_printout[i][0]
                                compound_energy = float(stability_printout[i][1])
                                chem_pot_sum = float(stability_printout[i][2])
                                diff = float(stability_printout[i][3])
                                message = stability_printout[i][4]
                                if diff > max_diff:
                                    max_diff = diff
                                    ref_compound = compound
                                    ref_energy = compound_energy

                            dopants[2] = ref_energy
                            dopants[5] = break_formula(ref_compound, 1)

                            dopant_position = chemical_potentials.index(dopants[1])
                            del chemical_potentials[dopant_position:dopant_position + 2:1]

                            chemical_potentials, opt_chem_pot = dopant_chemical(dopants, chemical_potentials, temperature,
                                                                                real_gas)
                            indicator = 0
                            stability_printout, indicator = stability_check(stability, chemical_potentials, indicator, b,
                                                                            real_gas)

                nu_e = 1
                # Optimise the dopant chemical potentials, if requsted
                if opt_chem_pot == 1:
                    chemical_potentials = calc_opt_chem_pot(b, loop, defects, dopants, chemical_potentials,
                                                            number_of_defects, host_supercell, tab_correction, E_VBM,
                                                            total_species, use_coul_correction, length, dielectric, v_M,
                                                            bandgap, temperature, def_statistics, nu_e, condband, valband,
                                                            electron_method, hole_method, fixed_e_conc, fixed_p_conc,
                                                            art_dop_conc, art_dop_charge, charge_convergence, val_band_min,
                                                            val_band_max, cond_band_min, cond_band_max, seedname, entropies,
                                                            entropy_marker, fu_uc, electron_mass_eff, hole_mass_eff,
                                                            unit_vol, charged_sys, potential_convergence, function_tol,
                                                            maxiter_dop, environment, prog_meter, prog_bar, num_iter,
                                                            real_gas, SLSQP_dial, constituents_name_list, dos_data_lst)



                    # Perform stability check, if requested.
                if ('stability' in tasks) or ('stability_highest' in tasks):
                    if loop == 1:
                        stability_printout, indicator = stability_check(stability, chemical_potentials, indicator, b,
                                                                        real_gas)
                    else:
                        stability_printout, indicator = stability_check(stability, chemical_potentials, indicator,
                                                                        temperature, real_gas)

                    # Calclate the defect formation energies

                defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction, E_VBM,
                                               total_species, use_coul_correction, length, dielectric, v_M, 0)

                # Calculate final Fermi level and concentrations of defects.
                (nu_e_final, concentrations, dopant_concentration_sum) = calc_fermi(b, loop, defects, defects_form,
                                                                                    number_of_defects, bandgap, temperature,
                                                                                    def_statistics, nu_e, condband, valband,
                                                                                    electron_method, hole_method,
                                                                                    fixed_e_conc, fixed_p_conc,
                                                                                    art_dop_conc, art_dop_charge,
                                                                                    charge_convergence, val_band_min,
                                                                                    val_band_max, cond_band_min,
                                                                                    cond_band_max, seedname, entropies,
                                                                                    entropy_marker, fu_uc,
                                                                                    electron_mass_eff, hole_mass_eff,
                                                                                    unit_vol, charged_sys, 0,
                                                                                    constituents_name_list, dos_data_lst)
                fermi.append([b, nu_e_final])

                # Perform check to determine if a very high conctration has been calcualted
                if concentration_check == 0:
                    max_concentration = max(concentrations)
                    if max_concentration > 0:
                        print(
                            "<!> Very high concentrations predicted, exceeding 1 p.f.u.: This will not be visable on default Brouwer diagram.")
                        concentration_check = 1

                # Calculate the stoichiometry, if requested

                if (stoichiometry == 1 or stoichiometry == 2):
                    concentrations, calced_stoic = stoich(concentrations, defects, host_array, number_of_defects, dopants, x_variable,stoichiometry, 0)

                if x_variable == 1:
                    concentrations.insert(1, nu_e_final)
                    concentrations.append(b)
                    stoichiometry_list.append(concentrations[0])

                else:
                    concentrations.insert(0, nu_e_final)
                    concentrations.insert(0, b)

                if plot_art_dopant_conc == 1 and loop != 3:
                    concentrations.append(art_dop_conc)

                final_concentrations.append(concentrations)

                # Output file printing

                if (loop == 0):
                    print("   Volatile partial pressure : 10^(", b, ") atm", file=f)
                    print("   Temperature :", temperature, "K", file=f)
                if (loop == 1):
                    if (chem_pot_method == 2 or chem_pot_method == 3 or chem_pot_method == 4 or chem_pot_method == 5):
                        print("   Volatile partial pressure : 10^(", pp, ") atm", file=f)
                    print("   Temperature :", temperature, "K", file=f)
                if (loop == 2):
                    print("   Temperature :", temperature, "K", file=f)
                    if (chem_pot_method == 2 or chem_pot_method == 3 or chem_pot_method == 4 or chem_pot_method == 5):
                        print("   Volatile partial pressure : 10^(", pp, ") atm", file=f)
                    print("   Dopant concentration : 10^(", b, ") per f.u.", file=f)
                if (loop == 3):
                    print("   Temperature :", temperature, "K", file=f)
                    if (chem_pot_method == 2 or chem_pot_method == 3 or chem_pot_method == 4 or chem_pot_method == 5):
                        print("   Volatile partial pressure : 10^(", pp, ") atm", file=f)
                    print("   Artificial dopant concentration : 10^(", b, ") per f.u.", file=f)
                    print("   Artificial dopant charge:", art_dop_charge, file=f)
                if (loop == 4):
                    print("   Dopant partial pressure : 10^(", b, ") atm", file=f)
                    print("   Temperature :", temperature, "K", file=f)
                if (loop == 5):
                    print("   Rich-poor fraction : ", b, "", file=f)
                    print("   Temperature :", temperature, "K", file=f)
                    if (chem_pot_method == (5)):
                        print("   Volatile partial pressure : 10^(", pp, ") atm", file=f)

                print("\n   Calculated chemical potentials:", "\n", file=f)
                for i in np.arange(0, len(chemical_potentials) / 2, 1):
                    i = int(i)
                    print("   ", chemical_potentials[2 * i], ":", chemical_potentials[2 * i + 1], "eV", file=f)
                print("\n   Fermi level:", nu_e_final, "eV", file=f)
                print("\n   Concentrations:", file=f)
                print("   +----------------+--------+----------------------------------------+", file=f)
                print("   |     Defect     | Charge | log_{10}[Concentration] (per f.u.) /eV |", file=f)
                print("   +----------------+--------+----------------------------------------+", file=f)
                print("   | %14s | %6s | %38f |" % ('Electrons', '-1', concentrations[2]), file=f)
                print("   | %14s | %6s | %38f |" % ('Holes', '1', concentrations[3]), file=f)

                # Search through defects_form and print output
                for i in np.arange(0, number_of_defects, 1):
                    i = int(i)
                    defect_name = defects_form[i][0]
                    charge = defects_form[i][4]
                    concentration = concentrations[i + 4]

                    print("   | %14s | %6s | %38f |" % (defect_name, charge, concentration), file=f)

                print("   +----------------+--------+----------------------------------------+", file=f)

                if ('stability' in tasks) or ('stability_highest' in tasks):
                    print("\n   Stability check results", file=f)
                    print(
                        "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                        file=f)
                    print(
                        "   |     Compound     | Total energy (eV) | Sum of chemical potentials (eV) | Difference (eV) |  Thermodynamically Stable  |",
                        file=f)
                    print(
                        "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                        file=f)
                    for i in np.arange(0, stability[0], 1):
                        i = int(i)
                        compound = stability_printout[i][0]
                        compound_energy = float(stability_printout[i][1])
                        chem_pot_sum = float(stability_printout[i][2])
                        diff = float(stability_printout[i][3])
                        message = stability_printout[i][4]
                        print("   | %16s | %17f | %31f | %15f | %26s |" % (
                            compound, compound_energy, chem_pot_sum, diff, message), file=f)
                    print(
                        "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                        file=f)
                prog_meter += 1

        print("\n..> Loop successfully executed")

        # print the seedname.fermi file
        if charged_sys == 1:
            print_fermi(fermi, seedname)

            # Obtain new x range, if plotting as function as stoichiometry
        if x_variable == 1:
            min_value = min(stoichiometry_list)
            max_value = max(stoichiometry_list)

        # Group defect concentrations

        if ('group' in tasks):
            print("\n>>> Task:'group':")

            (final_grouped_concs, group_list) = group(final_concentrations, number_of_defects, defects, num_iter,
                                                      stoichiometry)

            # Convert concentrations to cm^-3, if requested
            if y_variable == 1:
                final_grouped_concs = y_convert(final_grouped_concs, fu_uc, unit_vol, stoichiometry)

            # Print the seedname.res file
            print_results(final_grouped_concs, seedname)

            # Generate Brouwer diagram
            graphical_output(number_of_defects, min_value, max_value, final_concentrations, seedname, loop,
                             gnuplot_version,
                             min_y_range, host_name, group_list, electron_method, hole_method, dopants, host_array, 1,
                             conc_colour, electron_colour, hole_colour, scheme, dopant_xvar, stoichiometry, x_variable,
                             total_species, volatile_element, charged_sys, y_variable, max_y_range,
                             plot_art_dopant_conc, art_dop_charge)

        else:
            # Convert concentrations to cm^-3, if requested
            if y_variable == 1:
                final_concentrations = y_convert(final_concentrations, fu_uc, unit_vol, stoichiometry)

            print_results(final_concentrations, seedname)

            # Generate Brouwer diagram
            graphical_output(number_of_defects, min_value, max_value, final_concentrations, seedname, loop,
                             gnuplot_version,
                             min_y_range, host_name, defects, electron_method, hole_method, dopants, host_array, 0,
                             conc_colour, electron_colour, hole_colour, scheme, dopant_xvar, stoichiometry, x_variable,
                             total_species, volatile_element, charged_sys, y_variable, max_y_range,
                             plot_art_dopant_conc, art_dop_charge)

    # Initialise progress meter
    prog_meter = 1

    # defect_phase diagram task
    if ('defect_phase' in tasks):
        print("\n>>> Task:'defect_phase':")

        # Calculate the number of iterations in the loop
        num_iter = ((max_value - min_value) / iterator) + 1
        num_iter2 = ((max_value_y - min_value_y) / iterator_y) + 1

        with open(outputfile, 'a') as f:
            # Loop over the requested range
            for a in np.arange(min_value, max_value + (iterator / 2), iterator):
                plot_second_dopant_flag = False

                # Loop assignment
                constituents, temperature, entropies, constituent_entropies, environment, dopants, dopant_xvar, art_dop_conc, volatile_element, host_energy = looper(
                    loop, a, constituents, chem_pot_method, temperature, entropies, constituent_entropies, entropy_data,
                    number_of_defects, constituents_name_list, seedname, dopants, dopant_xvar, plot_second_dopant_flag,
                    art_dop_conc, volatile_element, gibbs_marker, gibbs_data, host_energy)

                prog_bar = round((prog_meter / num_iter) * 25)

                prog_meter2 = 1

                # Loop over the second requested range
                for b in np.arange(min_value_y, max_value_y + (iterator_y / 2), iterator_y):

                    prog_bar2 = round((prog_meter2 / num_iter2) * 25)

                    print(f"..> Calculating defect concentrations for: x-axis value {prog_meter} of {num_iter} {'#' * (prog_bar) + ' ' * (25 - prog_bar)}"
                          f"\t// y-axis value {prog_meter2} of {num_iter2} {'#' * (prog_bar2) + ' ' * (25 - prog_bar2)}", end="\r", flush=True)

                    if loop == 2 and loop2 == 2:
                        plot_second_dopant_flag = True

                    # Loop assignment
                    constituents, temperature, entropies, constituent_entropies, environment2, dopants, dopant_xvar, art_dop_conc, volatile_element, host_energy = looper(
                        loop2, b, constituents, chem_pot_method, temperature, entropies, constituent_entropies,
                        entropy_data, number_of_defects, constituents_name_list, seedname, dopants, dopant_xvar,
                        plot_second_dopant_flag, art_dop_conc, volatile_element, gibbs_marker, gibbs_data, host_energy)



                    if loop == 1:
                        if (entropy_marker == 1):
                            entropies, constituent_entropies = calc_entropy(entropy_data, a, number_of_defects,
                                                                            constituents_name_list, chem_pot_method,
                                                                            seedname, 0)
                        if (gibbs_marker == 1):
                            gibbs_energies = calc_gibbs(gibbs_data, a, constituents_name_list, chem_pot_method, host_energy,
                                                        constituents, stability, seedname, 1)
                    elif loop2 == 1:
                        if (entropy_marker == 1):
                            entropies, constituent_entropies = calc_entropy(entropy_data, b, number_of_defects,
                                                                            constituents_name_list, chem_pot_method,
                                                                            seedname, 0)
                        if (gibbs_marker == 1):
                            gibbs_energies = calc_gibbs(gibbs_data, b, constituents_name_list, chem_pot_method, host_energy,
                                                        constituents, stability, seedname, 1)

                    # Calculate chemical potentials of host atoms

                    chemical_potentials = []
                    if (chem_pot_method == 0):
                        chemical_potentials = calc_chemical_defined(host_array, constituents, chemical_potentials,
                                                                    host_energy,
                                                                    temperature, entropy_marker, constituent_entropies,
                                                                    entropy_units)
                    elif (chem_pot_method == 1):
                        chemical_potentials = calc_chemical_rich_poor(host_array, constituents, chemical_potentials,
                                                                      host_energy, temperature, entropy_marker,
                                                                      constituent_entropies, entropy_units)
                    elif (chem_pot_method == 2 or chem_pot_method == 3 or chem_pot_method == 4):
                        chemical_potentials = calc_chemical_volatile(host_array, constituents_name_list, constituents,
                                                                     chemical_potentials, host_energy,
                                                                     temperature, entropy_marker, constituent_entropies,
                                                                     entropy_units, real_gas)
                        pp = constituents[1]
                    elif (chem_pot_method == 5):
                        chemical_potentials = calc_chemical_volatile_rich_poor(host_array, constituents_name_list, constituents,
                                                                               chemical_potentials,
                                                                               host_energy, temperature, entropy_marker,
                                                                               constituent_entropies, entropy_units,
                                                                               real_gas, gibbs_energies)
                        pp = constituents[2]

                    opt_chem_pot = 0
                    # Calculate the dopant checmical potentials
                    if (dopants[0] > 0):

                        chemical_potentials, opt_chem_pot = dopant_chemical(dopants, chemical_potentials, temperature,
                                                                            real_gas)

                        # Update reference for dopant chemical potential: only works for one dopant specie.
                        if ('update_reference' in tasks) and (dopants[3] == 0):
                            indicator = 0
                            if loop == 1:
                                stability_printout, indicator = stability_check(stability, chemical_potentials, indicator,
                                                                                a, real_gas)
                            elif loop2 == 1:
                                stability_printout, indicator = stability_check(stability, chemical_potentials, indicator,
                                                                                b, real_gas)
                            else:
                                stability_printout, indicator = stability_check(stability, chemical_potentials, indicator,
                                                                                temperature, real_gas)

                            while indicator == 1:
                                max_diff = 0
                                for i in np.arange(0, stability[0], 1):
                                    i = int(i)
                                    compound = stability_printout[i][0]
                                    compound_energy = float(stability_printout[i][1])
                                    chem_pot_sum = float(stability_printout[i][2])
                                    diff = float(stability_printout[i][3])
                                    message = stability_printout[i][4]
                                    if diff > max_diff:
                                        max_diff = diff
                                        ref_compound = compound
                                        ref_energy = compound_energy

                                dopants[2] = ref_energy
                                dopants[5] = break_formula(ref_compound, 1)

                                dopant_position = chemical_potentials.index(dopants[1])
                                del chemical_potentials[dopant_position:dopant_position + 2:1]

                                chemical_potentials, opt_chem_pot = dopant_chemical(dopants, chemical_potentials,
                                                                                    temperature,
                                                                                    real_gas)
                                indicator = 0
                                stability_printout, indicator = stability_check(stability, chemical_potentials, indicator,
                                                                                b, real_gas)

                    nu_e = 1
                    # Optimise the dopant chemical potentials, if requsted
                    if opt_chem_pot == 1:
                        chemical_potentials = calc_opt_chem_pot(b, loop, defects, dopants, chemical_potentials,
                                                                number_of_defects, host_supercell, tab_correction, E_VBM,
                                                                total_species, use_coul_correction, length, dielectric, v_M,
                                                                bandgap, temperature, def_statistics, nu_e, condband,
                                                                valband,
                                                                electron_method, hole_method, fixed_e_conc, fixed_p_conc,
                                                                art_dop_conc, art_dop_charge, charge_convergence,
                                                                val_band_min,
                                                                val_band_max, cond_band_min, cond_band_max, seedname,
                                                                entropies, entropy_marker,
                                                                fu_uc, electron_mass_eff, hole_mass_eff, unit_vol,
                                                                charged_sys,
                                                                potential_convergence, function_tol, maxiter_dop,
                                                                environment,
                                                                prog_meter, prog_bar, num_iter, real_gas, SLSQP_dial,
                                                                constituents_name_list, dos_data_lst)

                        # Perform stability check, if requested.
                    if ('stability' in tasks) or ('stability_highest' in tasks):
                        indicator = 0
                        if loop == 1:
                            stability_printout, indicator = stability_check(stability, chemical_potentials, indicator, a,
                                                                            real_gas)
                        elif loop2 == 1:
                            stability_printout, indicator = stability_check(stability, chemical_potentials, indicator, b,
                                                                            real_gas)
                        else:
                            stability_printout, indicator = stability_check(stability, chemical_potentials, indicator,
                                                                            temperature, real_gas)



                        # Calclate the defect formation energies
                    defects_form = defect_energies(defects, chemical_potentials, host_supercell, tab_correction, E_VBM,
                                                   total_species, use_coul_correction, length, dielectric, v_M, 0)

                    # Calculate final Fermi level and concentrations of defects.
                    (nu_e_final, concentrations, dopant_concentration_sum) = calc_fermi(b, loop, defects, defects_form,
                                                                                        number_of_defects, bandgap,
                                                                                        temperature, def_statistics, nu_e,
                                                                                        condband, valband, electron_method,
                                                                                        hole_method, fixed_e_conc,
                                                                                        fixed_p_conc, art_dop_conc,
                                                                                        art_dop_charge, charge_convergence,
                                                                                        val_band_min, val_band_max,
                                                                                        cond_band_min, cond_band_max,
                                                                                        seedname, entropies, entropy_marker,
                                                                                        fu_uc, electron_mass_eff,
                                                                                        hole_mass_eff, unit_vol,
                                                                                        charged_sys, 0, constituents_name_list, dos_data_lst)

                    # Perform check to determine if a very high conctration has been calcualted
                    if concentration_check == 0:
                        max_concentration = max(concentrations)
                        if max_concentration > 0:
                            print(
                                "<!> Very high concentrations predicted, exceeding 1 p.f.u.: This will not be visable on default Brouwer diagram.")
                            concentration_check = 1

                    # For plotting the two defects predicted to have the highest concentration.
                    max_conc = -100
                    max_conc2 = -100
                    max_i = 0
                    max_i2 = 0
                    for i in np.arange(0, number_of_defects + 2, 1):
                        i = int(i)

                        if concentrations[i] > max_conc:  # Highest defect concentration
                            if max_conc != -100:
                                if (max_i > 1) and (defects[max_i - 2][4] == '0'):
                                    pass
                                else:
                                    max_i2, max_conc2 = max_i, max_conc

                            max_i = i
                            max_conc = concentrations[i]

                        elif concentrations[i] > max_conc2:  # 2nd highest defect concentration
                            if (i > 1) and (defects[i - 2][4] == '0'):
                                pass
                            else:
                                max_i2 = i
                                max_conc2 = concentrations[i]

                    if max_i == 0:
                        nametag_i = 'Electrons'
                    elif max_i == 1:
                        nametag_i = 'Holes'
                    else:
                        nametag_i = defects[max_i - 2][0] + str(defects[max_i - 2][4])

                    if max_i2 == 0:
                        nametag_ii = 'Electrons'
                    elif max_i2 == 1:
                        nametag_ii = 'Holes'
                    else:
                        nametag_ii = defects[max_i2 - 2][0] + str(defects[max_i2 - 2][4])

                    if (max_i > 1) and (defects[max_i - 2][4] == '0'):
                        nametag = nametag_i
                    else:
                        nametag = nametag_i + '+' + nametag_ii

                    nametag2 = nametag_ii + '+' + nametag_i

                    if nametag2 in A_nametags:
                        nametag = nametag2

                    max_stable_i = 0
                    if ('stability' in tasks):
                        if indicator == 1:
                            for i in np.arange(0, len(stability_printout), 1):
                                i = int(i)
                                if stability_printout[i][3] > max_stable_i:
                                    max_i = i
                                    max_stable_i = stability_printout[i][3]

                            nametag = stability_printout[max_i][0]

                    max_stable_i = -100
                    if ('stability_highest' in tasks):

                        for i in np.arange(0, len(stability_printout), 1):
                            i = int(i)
                            if stability_printout[i][3] > max_stable_i:
                                max_i = i
                                max_stable_i = stability_printout[i][3]

                        nametag = stability_printout[max_i][0]

                    if (stoichiometry == 1 or stoichiometry == 2):
                        concentrations, calced_stoic = stoich(concentrations, defects, host_array, number_of_defects, dopants, x_variable, stoichiometry, 1)

                    all_plot = []
                    all_plot.append(a)
                    all_plot.append(b)
                    all_plot.append(max_conc)
                    all_plot.append(max_conc2)

                    if nametag in A_nametags:
                        all_plot_master[(A_nametags.index(nametag))].append(all_plot)

                    else:
                        A_nametags.append(nametag)
                        all_plot_master.append([])
                        all_plot_master[-1].append(all_plot)

                    if ('dopant' in tasks):
                        # For plotting accommodation mechanism of 1 dopant that is assigned in input file with 'accommodate' tag.

                        for i in np.arange(0, dopants[0], 1):
                            i = int(i)
                            if dopants[6 * i + 1] == accommodate:
                                signal_i = i

                        max_conc = -100

                        for i in np.arange(0, number_of_defects, 1):
                            i = int(i)

                            signal = defects[i][host_array[0] + 7 + signal_i]

                            if float(signal) < 0:

                                if concentrations[i + 2] > max_conc:
                                    max_i = i
                                    max_conc = concentrations[i + 2]

                        dopant_nametag = defects[max_i][0] + str(defects[max_i][4])

                        max_stable = 0
                        indicator_accommodate = 0
                        if ('stability' in tasks):
                            if indicator == 1:
                                for i in np.arange(0, len(stability_printout), 1):
                                    i = int(i)

                                    if not accommodate in stability[4 * i + 3]:
                                        continue
                                    elif stability_printout[i][3] > max_stable:
                                        indicator_accommodate = 1
                                        max_i = i
                                        max_stable = stability_printout[i][3]

                                if indicator_accommodate == 1:
                                    dopant_nametag = stability_printout[max_i][0]


                        max_stable = -100
                        indicator_accommodate = 0
                        if ('stability_highest' in tasks):

                            for i in np.arange(0, len(stability_printout), 1):
                                i = int(i)

                                if not accommodate in stability[4 * i + 3]:
                                    continue
                                elif stability_printout[i][3] > max_stable:
                                    indicator_accommodate = 1
                                    max_i = i
                                    max_stable = stability_printout[i][3]

                            if indicator_accommodate == 1:
                                dopant_nametag = stability_printout[max_i][0]

                        dopant_plot = []
                        dopant_plot.append(a)
                        dopant_plot.append(b)
                        dopant_plot.append(max_conc)

                        if dopant_nametag in D_nametags:
                            dopant_plot_master[(D_nametags.index(dopant_nametag))].append(dopant_plot)

                        else:
                            D_nametags.append(dopant_nametag)
                            dopant_plot_master.append([])
                            dopant_plot_master[-1].append(dopant_plot)

                    # Output file printing
                    print(
                        "\n-------------------------------------------------------------------------------------------------------------------",
                        "\n", file=f)
                    print(">>> Task = defect_phase, x-axis value", prog_meter, "of", num_iter, file=f)
                    print("                         y-axis value", prog_meter2, "of", num_iter2, "\n", file=f)
                    if (loop == 0):
                        print("   Volatile partial pressure : 10^(", a, ") atm", file=f)
                    if (loop == 1):
                        print("   Temperature :", temperature, "K", file=f)
                    if (loop == 2):
                        print("   Dopant concentration : 10^(", a, ") per f.u.", file=f)
                    if (loop == 3):
                        print("   Artificial dopant concentration : 10^(", a, ") per f.u.", file=f)
                        print("   Artificial dopant charge:", art_dop_charge, file=f)
                    if (loop == 4):
                        print("   Dopant partial pressure : 10^(", a, ") atm", file=f)
                    if (loop == 5):
                        print("   Rich-poor fraction (s) : ", a, "", file=f)
                    if (loop2 == 0):
                        print("   Volatile partial pressure : 10^(", b, ") atm", file=f)
                    if (loop2 == 1):
                        print("   Temperature :", temperature, "K", file=f)
                    if (loop2 == 2):
                        print("   Dopant concentration : 10^(", b, ") per f.u.", file=f)
                    if (loop2 == 3):
                        print("   Artificial dopant concentration : 10^(", b, ") per f.u.", file=f)
                        print("   Artificial dopant charge:", art_dop_charge, file=f)
                    if (loop2 == 4):
                        print("   Dopant partial pressure : 10^(", b, ") atm", file=f)
                    if (loop2 == 5):
                        print("   Rich-poor fraction (s) : ", b, "", file=f)

                    print("\n   Calculated chemical potentials:", "\n", file=f)
                    for i in np.arange(0, len(chemical_potentials) / 2, 1):
                        i = int(i)
                        print("   ", chemical_potentials[2 * i], ":", chemical_potentials[2 * i + 1], "eV", file=f)
                    print("\n   Fermi level:", nu_e_final, "eV", file=f)
                    print("\n   Concentrations:", file=f)
                    print("   +----------------+--------+----------------------------------------+", file=f)
                    print("   |     Defect     | Charge | log_{10}[Concentration] (per f.u.) /eV |", file=f)
                    print("   +----------------+--------+----------------------------------------+", file=f)
                    print("   | %14s | %6s | %38f |" % ('Electrons', '-1', concentrations[0]), file=f)
                    print("   | %14s | %6s | %38f |" % ('Holes', '1', concentrations[1]), file=f)

                    # Search through defects_form and print output
                    for i in np.arange(0, number_of_defects, 1):
                        i = int(i)
                        defect_name = defects_form[i][0]
                        charge = defects_form[i][4]
                        concentration = concentrations[i + 2]

                        print("   | %14s | %6s | %38f |" % (defect_name, charge, concentration), file=f)

                    print("   +----------------+--------+----------------------------------------+", file=f)

                    if ('stability' in tasks) or ('stability_highest' in tasks):
                        print("\n   Stability check results", file=f)
                        print(
                            "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                            file=f)
                        print(
                            "   |     Compound     | Total energy (eV) | Sum of chemical potentials (eV) | Difference (eV) |  Thermodynamically Stable  |",
                            file=f)
                        print(
                            "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                            file=f)
                        for i in np.arange(0, stability[0], 1):
                            i = int(i)
                            compound = stability_printout[i][0]
                            compound_energy = float(stability_printout[i][1])
                            chem_pot_sum = float(stability_printout[i][2])
                            diff = float(stability_printout[i][3])
                            message = stability_printout[i][4]
                            print("   | %16s | %17f | %31f | %15f | %26s |" % (
                                compound, compound_energy, chem_pot_sum, diff, message), file=f)
                        print(
                            "   +------------------+-------------------+---------------------------------+-----------------+----------------------------+",
                            file=f)
                    prog_meter2 += 1

                prog_meter += 1

        print("\n..> Loops successfully executed")

        # Print the defect_phases files
        print_defect_phases(all_plot_master, A_nametags, 1, min_value, max_value, iterator)

        # Generate defect phases diagram
        defect_phases_graphical_output(min_value, max_value, min_value_y, max_value_y, seedname, loop, loop2, host_name,
                                       dopant_xvar, volatile_element, stability, A_nametags, iterator, 1)

        if ('dopant' in tasks):
            # Print the defect_phases files
            print_defect_phases(dopant_plot_master, D_nametags, 2, min_value, max_value, iterator)

            # Generate defect phases diagram
            defect_phases_graphical_output(min_value, max_value, min_value_y, max_value_y, seedname, loop, loop2,
                                           host_name,
                                           dopant_xvar, volatile_element, stability, D_nametags, iterator, 2)

    # print stability readout, if requested
    if ('stability' in tasks) or ('stability_highest' in tasks):
        print("\n>>> Task:'stability':")
        if indicator == 1:
            print("<!> WARNING: Stability check for supplied compounds has found conditions where secondary phases are possibly stable. See", " ", seedname,
                  ".output for details", sep="")
        else:
            print("..> Stability check for supplied compounds complete. No stable secondary phases are indicated, see", " ", seedname,
                  ".output for details", sep="")

    if ('bibliography' in tasks):
        print("\n>>> Task:'bibliography':")
        bibliography(tasks, chem_pot_method, real_gas, entropy_marker)

    # Launch gnuplot

    if ('brouwer' in tasks):
        outputfile = str(seedname) + ".eps"
        graphfile = "gnuplot " + str(seedname) + ".p"
        if os.path.exists(outputfile):
            os.remove(outputfile)
        print("\n..> gnuplot messages:")
        os.system(graphfile)

    if ('defect_phase' in tasks):
        os.chdir("defect_phase_defects")

        outputfile = str(seedname) + "_defect_phases.eps"
        graphfile = "gnuplot " + str(seedname) + "_defect_phases.p"

        if os.path.exists(outputfile):
            os.remove(outputfile)

        print("\n..> gnuplot messages:")
        os.system(graphfile)

        os.chdir("..")

        if ('dopant' in tasks):
            os.chdir("defect_phase_dopant")

            outputfile2 = str(seedname) + "_dopant_phases.eps"
            graphfile2 = "gnuplot " + str(seedname) + "_dopant_phases.p"

            if os.path.exists(outputfile2):
                os.remove(outputfile2)

            print("\n..> gnuplot messages:")
            os.system(graphfile2)

            os.chdir("..")


    # Plot and visulise Brouwer diagram.
    if ('autodisplay' in tasks):

        osys = platform.system()

        if ('brouwer' in tasks):
            print("\n>>> Task: 'autodisplay':")
            print("..> Displaying defect concentration figure")

            if osys == 'Linux':
                command = "gv " + outputfile
                os.system(command)
            elif osys == 'Darwin':
                command = "open " + outputfile
                os.system(command)
            else:
                print("<!> Unable to open", outputfile, "on this system")


if __name__ == "__main__":
    main()
