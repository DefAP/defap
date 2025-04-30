import math
import os
import time

import numpy as np
import pandas as pd
from scipy import special

from shapely import MultiPoint
from shapely.geometry import CAP_STYLE, JOIN_STYLE

import thermodynamics
import defap_defects
from defap_chem_pots import ChemicalPotentials
from defap_output import brouwer_plot
from defap_output import defect_phases_plot
from defap_output import plot_formation_energies
from defap_output import write_brouwer_output
from defap_output import write_defect_phases_output
from defap_misc import break_formula


class Tasks:

    def __init__(self, data, seedname, high_conc_check=False, dopant_warning=False, looping_dopants=None,
                 grouped_defects=False, all_defect_concentrations=None, all_defect_phases=None,
                 all_chemical_potentials=None,
                 all_fermi_levels=None, all_formation_energies=None, all_min_formation_energies=None,
                 all_secondary_phase_vals=None,
                 ):

        # pulls in parsed input data
        if all_defect_concentrations is None:
            all_defect_concentrations = {}

        if all_defect_phases is None:
            all_defect_phases = {}

        if all_chemical_potentials is None:
            all_chemical_potentials = {}

        if all_fermi_levels is None:
            all_fermi_levels = {"fermi_level": []}

        if all_formation_energies is None:
            all_formation_energies = {}

        if all_min_formation_energies is None:
            all_min_formation_energies = {}

        if all_secondary_phase_vals is None:
            all_secondary_phase_vals = {}

        if looping_dopants is None:
            looping_dopants = [None, None]

        self.data = data
        self.seedname = seedname

        self.high_conc_check = high_conc_check
        self.dopant_warning = dopant_warning
        self.looping_dopants = looping_dopants
        self.grouped_defects = grouped_defects

        # for plotting
        self.all_defect_concentrations = all_defect_concentrations
        self.all_defect_phases = all_defect_phases
        self.all_chemical_potentials = all_chemical_potentials
        self.all_fermi_levels = all_fermi_levels
        self.all_formation_energies = all_formation_energies
        self.all_min_formation_energies = all_min_formation_energies
        self.all_secondary_phase_vals = all_secondary_phase_vals

        # calculate initial entropy values at specified temperature
        # recalculated if looping over temperature

        self.entropy_vals = thermodynamics.calc_entropies(entropy_data=self.data.entropy_data,
                                                          temperature=self.data.temperature
                                                          )

        self.gibbs_vals = thermodynamics.calc_gibbs_function(gibbs_data=self.data.gibbs_data,
                                                             temperature=self.data.temperature
                                                             )

    def loop_assignment(self, loop_type, loop_step, loop_a_or_b):


        # assign loop step to the volatile constituent partial pressure
        if loop_type == 0:
            self.data.constituents["volatile"]["log_pp"] = loop_step

        # assign loop step to the temperature of the system
        elif loop_type == 1:
            self.data.temperature = loop_step

        # assign loop step to the specified dopant concentration
        elif loop_type == 2:

            if loop_a_or_b == "a":
                # loop through items in dopant dict to get dopant element and values
                for dopant_element, dopant_values in self.data.dopants.items():

                    # dopant concentration looped over if fitting option == 2. Selects first dopant with fitting option == 2 here
                    if dopant_values["fitting_option"] == 2:

                        self.data.dopants[dopant_element]["concentration_pfu"] = 10 ** loop_step
                        self.looping_dopants[0] = dopant_element
                        break

                # for else
                else:
                    raise ValueError("ERROR<!> No dopant selected to loop over its concentration! Please add fitting option = 2 to a dopant")

            elif loop_a_or_b == "b":
                for dopant_element, dopant_values in self.data.dopants.items():
                    # extra check to see if looping over two dopants in defect phase or not.
                    # self.looping_dopants[0] = None if looping over one dopant on y-axis
                    if dopant_values["fitting_option"] == 2 and dopant_element != self.looping_dopants[0]:
                        self.data.dopants[dopant_element]["concentration_pfu"] = 10 ** loop_step
                        self.looping_dopants[1] = dopant_element
                        break

                # for else
                else:
                    if self.looping_dopants[0]:
                        message = "ERROR<!> Only one dopant selected to loop over its concentration. Please set fitting option = 2 for another dopant"
                    else:
                        message = "ERROR<!> No dopant selected to loop over its concentration! Please add fitting option = 2 to a dopant"

                    raise ValueError(message)

            else:
                if loop_a_or_b == "a":
                    self.data.dopants[self.looping_dopants[0]]["concentration_pfu"] = 10 ** loop_step
                elif loop_a_or_b == "b":
                    self.data.dopants[self.looping_dopants[1]]["concentration_pfu"] = 10 ** loop_step

        # assign loop step to specified dopant partial pressure
        elif loop_type == 3:

            if loop_a_or_b == "a":
                # loop through items in dopant dict to get dopant element and values
                for dopant_element, dopant_values in self.data.dopants.items():

                    # dopant partial pressure looped over if fitting option == 4
                    if dopant_values["fitting_option"] == 4:
                        self.data.dopants[dopant_element]["log_PP_atm"] = loop_step
                        self.looping_dopants[0] = dopant_element
                        break

                # for else
                else:
                    raise ValueError("ERROR<!> No dopant selected to loop over its partial pressure! Please add fitting option = 4 to a dopant")

            elif loop_a_or_b == "b":
                for dopant_element, dopant_values in self.data.dopants.items():
                    if dopant_values["fitting_option"] == 4 and dopant_element != self.looping_dopants[0]:
                        self.data.dopants[dopant_element]["log_PP_atm"] = loop_step
                        self.looping_dopants[1] = dopant_element
                        break

                # for else
                else:
                    if self.looping_dopants[0]:
                        message = "ERROR<!> Only one dopant selected to loop over its partial pressure. Please set fitting option = 4 for another dopant"
                    else:
                        message = "ERROR<!> No dopant selected to loop over its partial pressure! Please add fitting option = 4 to a dopant"

                    raise ValueError(message)

            else:
                if loop_a_or_b == "a":
                    self.data.dopants[self.looping_dopants[0]]["log_PP_atm"] = loop_step
                elif loop_a_or_b == "b":
                    self.data.dopants[self.looping_dopants[1]]["log_PP_atm"] = loop_step


        # direct chemical potential of a dopant
        elif loop_type == 4:

            if loop_a_or_b == "a":
                # loop through items in dopant dict to get dopant element and values
                for dopant_element, dopant_values in self.data.dopants.items():

                    # dopant chemical potential looped over if fitting option == 5
                    if dopant_values["fitting_option"] == 5:
                        self.data.dopants[dopant_element]["chemical_potential"] = loop_step
                        self.looping_dopants[0] = dopant_element
                        break

                # for else
                else:
                    raise ValueError(
                        "ERROR<!> No dopant selected to loop over its chemical potential! Please add fitting option = 5 to a dopant")

            elif loop_a_or_b == "b":
                for dopant_element, dopant_values in self.data.dopants.items():
                    if dopant_values["fitting_option"] == 5 and dopant_element != self.looping_dopants[0]:
                        self.data.dopants[dopant_element]["chemical_potential"] = loop_step
                        self.looping_dopants[1] = dopant_element
                        break

                # for else
                else:
                    if self.looping_dopants[0]:
                        message = "ERROR<!> Only one dopant selected to loop over! Please set fitting option = 5 for another dopant"
                    else:
                        message = "ERROR<!> No dopants selected to loop over its chemical potential! Please add fitting option = 5 to a dopant"

                    raise ValueError(message)

            else:
                if loop_a_or_b == "a":
                    self.data.dopants[self.looping_dopants[0]]["chemical_potential"] = loop_step
                elif loop_a_or_b == "b":
                    self.data.dopants[self.looping_dopants[1]]["chemical_potential"] = loop_step

        # artificial dopant charge
        elif loop_type == 5:
            self.data.art_dopant_conc = 10 ** loop_step


        # loop over rich poor fraction
        elif loop_type == 6:
            other_rich_poor_fracs = 0
            total_fractions = self.data.constituents["total_fraction"]

            for index, constituent in enumerate(self.data.constituents["constituent_compounds"]):

                # first compound gets assigned loop step
                if index == 0:
                    self.data.constituents["constituent_compounds"][constituent]["fraction"] = loop_step

                # hold second consituent to assign rich poor frac later
                elif index == 1:
                    hold_constituent_key = constituent

                # sum fractions of other constituents
                else:
                    other_rich_poor_fracs += self.data.constituents["constituent_compounds"][constituent]["fraction"]

            # assign balanceing rich poor frac to second constiuent
            self.data.constituents["constituent_compounds"][hold_constituent_key]["fraction"] = total_fractions - other_rich_poor_fracs - loop_step

            if loop_step + other_rich_poor_fracs > total_fractions + 0.01:  # account for binary fraction overspill
                raise ValueError(f"ERROR<!> Total rich poor fraction from loop exceeds {total_fractions}!\n"
                                 f"Please modify the range of your loop")


        else:
            raise ValueError(f"ERROR<!> '{loop_type}' is not a valid method for a loop iteration!")


    def brouwer(self):

        # check if defects are to be grouped together
        if "group" in self.data.tasks:
            self.grouped_defects = True

        # determine number of loop iterations there will be
        number_of_iterations = ((self.data.max_value - self.data.min_value) / self.data.iterator)

        with open(f"{self.seedname}.output", "a") as f:
            f.write(">>> Task : Brouwer\n\n")

            # begin loop
            for progress_meter, loop_step in enumerate(np.around(np.arange(self.data.min_value, self.data.max_value + self.data.iterator, self.data.iterator), decimals=5)):

                # assign loop value being iterated over to the desired property
                self.loop_assignment(loop_type=self.data.loop, loop_step=loop_step, loop_a_or_b="a")

                print(f"..> Calculating defect concentrations: {progress_meter} of {number_of_iterations}", end="\r", flush=True)

                # if looping over temperature, ensure that entropy values are recalculated
                if self.data.loop == 1 and self.data.entropy_data:
                    self.entropy_vals = thermodynamics.calc_entropies(entropy_data=self.data.entropy_data,
                                                                      temperature=self.data.temperature
                                                                      )

                if self.data.loop == 1 and self.data.gibbs_data:
                    self.gibbs_vals = thermodynamics.calc_gibbs_function(gibbs_data=self.data.gibbs_data,
                                                                         temperature=self.data.temperature
                                                                         )

                # initialise class for calculting chemical potentials
                chemical_potentials = ChemicalPotentials(data=self.data,
                                                         entropy_vals=self.entropy_vals,
                                                         gibbs_vals=self.gibbs_vals
                                                         )


                # calculate chemical potentials depending on the method specified on the input
                chemical_potentials.call_chemical_pot_method()


                # add any dopants to the chem pots dictionary and determine how many require chemical potentials to be fitted
                if self.data.dopants:
                    chemical_potentials.add_dopants()

                # initialise Defects class to calculate defect concentrations
                defects = defap_defects.Defects(data=self.data,
                                                chemical_potentials=chemical_potentials.chem_pots,
                                                entropy_vals=self.entropy_vals)

                # optimise chemical potential of one dopant
                if self.data.dopants_to_fit == 1:

                    defects.optimise_single_dopant(fitting_dopant=chemical_potentials.fitting_dopants)

                    # adjust fitting dopant chemical potentials on the fly, helps with wide loop ranges

                    # this changes in the chemical potential of the dopant in the dopants dictionary to the value just calculated
                    self.data.dopants[chemical_potentials.fitting_dopants]["chemical_potential"] = chemical_potentials.chem_pots[chemical_potentials.fitting_dopants]

                    # reduce the chem pot range, speeds up calculation
                    self.data.dopants[chemical_potentials.fitting_dopants]["chem_pot_range"] = 2

                # optimise chemical potentials of multiple dopants
                elif self.data.dopants_to_fit > 1:

                    defects.optimise_multi_dopant()

                    # adjust fitting dopant chemical potentials on the fly, helps with wide loop ranges
                    for dopant in chemical_potentials.fitting_dopants:
                        self.data.dopants[dopant]["chemical_potential"] = chemical_potentials.chem_pots[dopant]
                        self.data.dopants[dopant]["chem_pot_range"] = 2

                # no dopants to fit
                else:

                    # calculate formation energies at valence band maximum then optimise fermi level to determine concentrations
                    defects.calc_form_eng_vbm()
                    defects.optimise_fermi_level()

                # warning message for defect concentrations > 1 pfu (indicates a -ve form eng)
                if max(list(defects.concentrations.values())) > 0 and not self.high_conc_check:
                    print("\n<!> Very high concentrations predicted, exceeding 1 p.f.u.: "
                          "This will not be visible on default Brouwer diagram\n")

                    self.high_conc_check = True

                # calculate the stoichiometry of a volatile system if indicated
                if self.data.stoichiometry == 1 or self.data.stoichiometry == 2:
                    log_stoic, stoic = defects.calc_stoichiometry()

                    defects.concentrations["log_stoic"] = log_stoic
                    defects.concentrations["pm_stoic"] = float(stoic)

                # calculate chemical potential sums of secondary phases
                if self.data.secondary_phases:
                    chemical_potentials.calc_secondary_phase_chem_pots()

                if self.grouped_defects:

                    # if defects are to be grouped, determine defect groups and sum total concentration for each group
                    grouped_defect_sums = {}
                    for defect_name, defect_conc in defects.concentrations.items():
                        # defect_group = defect_name for electrons, holes and stoich as they dont have a group assigned in the defects dict
                        if defect_name != "electrons" and defect_name != "holes" and defect_name != "pm_stoic" and defect_name != "log_stoic":
                            defect_group = self.data.defects_data[defect_name]["group"]
                        else:
                            defect_group = defect_name

                        # sum defect concentrations
                        if defect_group not in grouped_defect_sums:
                            grouped_defect_sums[defect_group] = defect_conc
                        else:
                            grouped_defect_sums[defect_group] = math.log10(10 ** grouped_defect_sums[defect_group] + 10 ** defect_conc)

                    # now loop through grouped defects and add them to the all_defect_concentrations dict which is used
                    # to create a pandas dataframe for plotting
                    for defect_group, group_conc in grouped_defect_sums.items():
                        if defect_group not in self.all_defect_concentrations:
                            self.all_defect_concentrations[defect_group] = [group_conc]
                        else:
                            self.all_defect_concentrations[defect_group].append(group_conc)

                # if not grouping defects, just add defects to the all_defect_concentrations dict which is used for plotting
                else:
                    for defect_name, defect_conc in defects.concentrations.items():
                        if defect_name not in self.all_defect_concentrations:
                            self.all_defect_concentrations[defect_name] = [defect_conc]
                        else:
                            self.all_defect_concentrations[defect_name].append(defect_conc)

                # add final calculated chemical potentials to the all_chemical_potentials dict,
                # used to create a dataframe to view chemical potentials for each loop step
                for element, chem_pot in chemical_potentials.chem_pots.items():
                    if element not in self.all_chemical_potentials:
                        self.all_chemical_potentials[element] = [chem_pot]
                    else:
                        self.all_chemical_potentials[element].append(chem_pot)

                self.all_fermi_levels["fermi_level"].append(defects.fermi_level)

                for defect_name, form_eng in defects.formation_energies.items():
                    if defect_name not in self.all_formation_energies:
                        self.all_formation_energies[defect_name] = [form_eng]
                    else:
                        self.all_formation_energies[defect_name].append(form_eng)

                if self.data.secondary_phases:
                    for phase_name, phase_vals in chemical_potentials.phase_chem_pots.items():
                        if phase_name not in self.all_secondary_phase_vals:
                            self.all_secondary_phase_vals[phase_name] = [phase_vals]
                        else:
                            self.all_secondary_phase_vals[phase_name].append(phase_vals)

                write_brouwer_output(f=f, concentrations=defects.concentrations, formation_energies=defects.formation_energies,
                                     chemical_pots=chemical_potentials.chem_pots, fermi_level=defects.fermi_level,
                                     secondary_phases=chemical_potentials.phase_chem_pots, loop_type=self.data.loop,
                                     loop_val=loop_step, loop_prog=progress_meter, total_step=number_of_iterations,
                                     grouped_defects=self.grouped_defects)

            f.write(">>> Finished\n\n")

        if not os.path.exists("brouwer_output"):
            os.mkdir("brouwer_output")
            os.chdir("brouwer_output")
        else:
            os.chdir("brouwer_output")

            for file in os.listdir('.'):
                if not file[:2] == '._':
                    os.remove(file)

        # create dataframe to plot brouwer results with, indexed using each loop step value
        if self.data.x_variable == 2:
            brouwer_axis_range = (self.all_defect_concentrations["pm_stoic"][0], self.all_defect_concentrations["pm_stoic"][-1], self.data.y_axis_min, self.data.y_axis_max)
        else:
            brouwer_axis_range = (self.data.min_value, self.data.max_value, self.data.y_axis_min, self.data.y_axis_max)

        brouwer_df = pd.DataFrame(self.all_defect_concentrations,
                                  index=np.arange(self.data.min_value, self.data.max_value + self.data.iterator,
                                                  self.data.iterator))
        brouwer_df.index.name = "loop_step"
        brouwer_df.to_csv(f"brouwer_data.csv")


        ### uncomment "*.to_csv" lines to generate csv containing property as a function of the loop variable

        # create dataframe for chemical potentials, indexed using each loop step
        chem_pot_df = pd.DataFrame(self.all_chemical_potentials,
                                   index=np.arange(self.data.min_value, self.data.max_value + self.data.iterator,
                                                   self.data.iterator))
        chem_pot_df.index.name = "loop_step"
        #chem_pot_df.to_csv("chem_pot_data.csv")

        # dataframe of optimised fermi levels
        fermi_level_df = pd.DataFrame(self.all_fermi_levels,
                                      index=np.arange(self.data.min_value, self.data.max_value + self.data.iterator,
                                                      self.data.iterator))
        fermi_level_df.index.name = "loop_step"
        #fermi_level_df.to_csv("fermi_level_data.csv")

        form_eng_df = pd.DataFrame(self.all_formation_energies,
                                   index=np.arange(self.data.min_value, self.data.max_value + self.data.iterator,
                                                   self.data.iterator))
        form_eng_df.index.name = "loop_step"
        #form_eng_df.to_csv("all_form_eng_data.csv")

        if self.data.secondary_phases:
            phases_df = pd.DataFrame(self.all_secondary_phase_vals,
                                     index=np.arange(self.data.min_value, self.data.max_value + self.data.iterator,
                                                     self.data.iterator))

            phases_df.index.name = "loop_step"
            # phases_df.to_csv("secondary_phases.csv")
        else:
            phases_df = pd.DataFrame()

        # create brouwer plot
        brouwer_plot(axes=brouwer_axis_range,
                     defects_data=self.data.defects_data,
                     grouped_defects=self.grouped_defects,
                     loop_type=self.data.loop,
                     looping_dopants=self.looping_dopants,
                     art_dopant_chg=self.data.art_dopant_chg,
                     x_variable=self.data.x_variable
                     )

        os.chdir("..")

    def formation_energies(self):

        # determine entropy for defects
        if self.data.entropy_data:
            self.entropy_vals = thermodynamics.calc_entropies(entropy_data=self.data.entropy_data,
                                                              temperature=self.data.temperature
                                                              )

        if self.data.loop == 1 and self.data.gibbs_data:
            self.gibbs_vals = thermodynamics.calc_gibbs_function(gibbs_data=self.data.gibbs_data,
                                                                 temperature=self.data.temperature
                                                                 )

        # initialise class for calculating chemical potentials
        chemical_potentials = ChemicalPotentials(data=self.data,
                                                 entropy_vals=self.entropy_vals,
                                                 gibbs_vals=self.gibbs_vals
                                                 )

        # calculate chemical potentials depending on the method specified on the input
        chemical_potentials.call_chemical_pot_method()

        # add any dopants to the chem pots dictionary and determine how many require chemical potentials to be fitted
        if self.data.dopants:
            chemical_potentials.add_dopants()

        # initialise Defects class to calculate defect concentrations
        defects = defap_defects.Defects(data=self.data,
                                        chemical_potentials=chemical_potentials.chem_pots,
                                        entropy_vals=self.entropy_vals)

        # optimise chemical potential of one dopant
        if self.data.dopants_to_fit == 1:

            defects.optimise_single_dopant(fitting_dopant=chemical_potentials.fitting_dopants)

        # optimise chemical potentials of multiple dopants
        elif self.data.dopants_to_fit > 1:

            defects.optimise_multi_dopant()

        # no dopants to fit
        else:

            # calculate formation energies at valence band maximum
            defects.calc_form_eng_vbm()


        if self.data.charged_system:
            # loop over fermi energies between valence and conduction bands
            for fermi_energy in np.arange(0, self.data.bandgap, 0.001):
                # calculate formation energies at each point in the bandgap
                defects.calc_form_eng_at_fermi(fermi_level=fermi_energy)

                # temporary dict to hold minimum form eng for each defect group at each point in the bandgap
                min_group_form_energies = {}

                # loop through defects and calculated formation energies
                for defect_name, form_eng in defects.formation_energies.items():

                    # add all defects and all formation energies to the master dictionary
                    if defect_name not in self.all_formation_energies:
                        self.all_formation_energies[defect_name] = [form_eng]
                    else:
                        self.all_formation_energies[defect_name].append(form_eng)

                    # get defect group
                    defect_group = self.data.defects_data[defect_name]["group"]

                    # check if this form energy is the lowest for this defect group
                    if defect_group not in min_group_form_energies:
                        min_group_form_energies[defect_group] = form_eng
                    elif form_eng < min_group_form_energies[defect_group]:
                        min_group_form_energies[defect_group] = form_eng

                # loop through determined min form energies of each group and append them to the all_min_form_eng_dict
                for group_name, min_form_eng in min_group_form_energies.items():
                    if group_name not in self.all_min_formation_energies:
                        self.all_min_formation_energies[group_name] = [min_form_eng]
                    else:
                        self.all_min_formation_energies[group_name].append(min_form_eng)

            if not os.path.exists("form_eng_output"):
                os.mkdir("form_eng_output")
                os.chdir("form_eng_output")
            else:
                os.chdir("form_eng_output")

                for file in os.listdir('.'):
                    if not file[:2] == '._':
                        os.remove(file)


            form_eng_df = pd.DataFrame(self.all_formation_energies,
                                       index=np.arange(0, self.data.bandgap, 0.001))

            form_eng_df.index.name = "fermi_energy"
            form_eng_df.to_csv("all_form_eng_data.csv")

            min_form_eng_df = pd.DataFrame(self.all_min_formation_energies,
                                       index=np.arange(0, self.data.bandgap, 0.001))

            min_form_eng_df.index.name = "fermi_energy"
            min_form_eng_df.to_csv("all_min_form_eng_data.csv")

            plot_formation_energies(defects_data=self.data.defects_data,
                                    bandgap=self.data.bandgap
                                    )

            os.chdir("..")

        else:
            with open("form_engs.out", "w") as f:
                for defect_name, form_eng in defects.formation_energies_vbm.items():
                    f.write(f"{defect_name}\t{form_eng:.5f}\n")


    def defect_phases(self):

        defect_phases = {}
        dopant_phases = {}

        # check if defects are to be grouped together
        if "group" in self.data.tasks:
            self.grouped_defects = True

        time_flag = False

        # determine number of loop iterations there will be
        number_of_iterations_a = ((self.data.max_value - self.data.min_value) / self.data.iterator)
        number_of_iterations_b = ((self.data.max_value_y - self.data.min_value_y) / self.data.iterator_y)

        # hold dopant chem pots over loop b
        # used to readjust dopant chem pots on the fly
        dopant_chem_pots = {}

        with open(f"{self.seedname}.output", "a") as f:
            f.write(">>> Task : Defect Phases\n\n")

            # begin loop_a
            for progress_meter_a, loop_step_a in enumerate(
                    np.arange(self.data.min_value, self.data.max_value + self.data.iterator, self.data.iterator)):

                loop_start_time = time.time()

                # assign loop_a value to desired property
                self.loop_assignment(loop_type=self.data.loop, loop_step=loop_step_a, loop_a_or_b="a")

                previous_highest_conc_loop_b = None
                previous_highest_conc_loop_b_dopants = {}

                loop_b_stoics = []
                current_loop_vals = []


                # begin loop_b
                for progress_meter_b, loop_step_b in enumerate(
                        np.arange(self.data.min_value_y, self.data.max_value_y + self.data.iterator_y,
                                  self.data.iterator_y)):

                    # assign loop_b value to desired property
                    self.loop_assignment(loop_type=self.data.loop_y, loop_step=loop_step_b, loop_a_or_b="b")

                    print(f"..> Calculating defect concentrations for x: {progress_meter_a} of {number_of_iterations_a}"
                          f"  ||  y: {progress_meter_b} of {number_of_iterations_b}", end="\r", flush=True)

                    # calculate entropy if requested and looping over temperature
                    if self.data.entropy_data and (self.data.loop == 1 or self.data.loop_y == 1):
                        self.entropy_vals = thermodynamics.calc_entropies(entropy_data=self.data.entropy_data,
                                                                          temperature=self.data.temperature
                                                                          )

                    # calculate gibbs if requested and looping over temperature
                    if self.data.gibbs_data and (self.data.loop == 1 or self.data.loop_y == 1):
                        self.gibbs_vals = thermodynamics.calc_gibbs_function(gibbs_data=self.data.gibbs_data,
                                                                             temperature=self.data.temperature
                                                                             )



                    # initialise class for calculating chemical potentials
                    chemical_potentials = ChemicalPotentials(data=self.data,
                                                             entropy_vals=self.entropy_vals,
                                                             gibbs_vals=self.gibbs_vals
                                                             )


                    # calculate chemical potentials depending on the method specified on the input
                    chemical_potentials.call_chemical_pot_method()


                    # add any dopants to the chem pots dictionary and determine how many require chemical potentials to be fitted
                    if self.data.dopants:
                        chemical_potentials.add_dopants()

                        if chemical_potentials.fitting_dopants:
                            for dopant in chemical_potentials.fitting_dopants:
                                # refit dopant chem pot to be centred on its chemical potential at [loop_step_a - 1, loop_step_b]
                                if dopant in dopant_chem_pots:
                                    if loop_step_b in dopant_chem_pots[dopant]:
                                        self.data.dopants[dopant]['chemical_potential'] = dopant_chem_pots[dopant][loop_step_b][0]
                                        self.data.dopants[dopant]['chem_pot_range'] = dopant_chem_pots[dopant][loop_step_b][1] + 1

                                else:
                                    dopant_chem_pots[dopant] = {}

                                dopant_chem_pots[dopant][loop_step_b] = []

                    # uncomment to view chemical potentials
                    # print(chemical_potentials.chem_pots)

                    # initialise Defects class to calculate defect concentrations
                    defects = defap_defects.Defects(data=self.data,
                                                    chemical_potentials=chemical_potentials.chem_pots,
                                                    entropy_vals=self.entropy_vals)

                    # optimise chemical potential of one dopant
                    if self.data.dopants_to_fit == 1:
                        defects.optimise_single_dopant(fitting_dopant=chemical_potentials.fitting_dopants)

                        # add optimised dopant chem pot to the "refit" dict
                        dopant_chem_pots[chemical_potentials.fitting_dopants][loop_step_b].append(chemical_potentials.chem_pots[chemical_potentials.fitting_dopants])


                    # optimise chemical potentials of multiple dopants
                    elif self.data.dopants_to_fit > 1:

                        defects.optimise_multi_dopant()

                        # add optimised dopant chem pots to the "refit" dict
                        for dopant in chemical_potentials.fitting_dopants:
                            dopant_chem_pots[dopant][loop_step_b].append(chemical_potentials.chem_pots[dopant])

                    # no dopants to fit
                    else:

                        # calculate formation energies at valence band maximum then optimise fermi level to determine concentrations
                        defects.calc_form_eng_vbm()
                        defects.optimise_fermi_level()

                    # warning message for defect concentrations > 1 pfu (indicates a -ve form eng)
                    if max(list(defects.concentrations.values())) > 0 and not self.high_conc_check:
                        print("\n<!> Very high concentrations predicted, exceeding 1 p.f.u.: "
                              "This will not be visible on default Brouwer diagram\n")

                        self.high_conc_check = True


                    # calculate chemical potential sums of secondary phases
                    if self.data.secondary_phases:
                        chemical_potentials.calc_secondary_phase_chem_pots()
                        stable_sec_phase, sorted_phases = self.most_stable_phase(phase_chemical_potentials=chemical_potentials.phase_chem_pots)
                    else:
                        stable_sec_phase = False

                    # create an ordered list of defects from highest to lowest concentration
                    ordered_defect_concs = sorted(defects.concentrations.items(), key=lambda x: x[1], reverse=True)

                    if self.grouped_defects:
                        # if defects are to be grouped, determine defect groups and sum total concentration for each group
                        grouped_defect_sums = self.get_grouped_defect_concs(concentrations=defects.concentrations)

                        # order the grouped defects by total summed concentration
                        ordered_defect_concs_grouped = sorted(grouped_defect_sums.items(), key=lambda x: x[1], reverse=True)

                    # calculate the stoichiometry of a volatile system if indicated
                    if self.data.stoichiometry == 1 or self.data.stoichiometry == 2:
                        log_stoic, stoic = defects.calc_stoichiometry()
                        # print(loop_step_a, loop_step_b, log_stoic, stoic)
                        loop_b_stoics.append(abs(stoic))
                        current_loop_vals.append([loop_step_a, loop_step_b])
                        defects.concentrations["log_stoic"] = log_stoic


                    # secondary phases overrule defect phases
                    if stable_sec_phase:
                        main_phase = stable_sec_phase

                    elif self.grouped_defects:
                        # create nametag the "main defect phase" in this region. Tries to determine charge compensating defects
                        main_phase = self.get_main_defect_phases(ordered_defects=ordered_defect_concs_grouped, is_grouped=True)

                    else:
                        main_phase = self.get_main_defect_phases(ordered_defects=ordered_defect_concs, is_grouped=False)


                    # add coords to relevant phases
                    defect_phases = self.defect_phases_add_coords(phases_dict=defect_phases,
                                                                  current_phase=main_phase,
                                                                  loop_step_x=loop_step_a,
                                                                  loop_step_y=loop_step_b,
                                                                  secondary_phases=self.data.secondary_phases
                                                                  )

                    # variables to hold the current phase as the highest conc phase for the next loop step
                    previous_highest_conc_loop_b = main_phase

                    # to create separate phase plots for each dopant
                    if self.data.dopants:

                        # loop through dopants
                        for dopant in self.data.dopants:

                            # append dopant to dopant phases dict
                            if dopant not in dopant_phases:
                                dopant_phases[dopant] = {}

                            # variables will determine whether dopant is a defect or secondary phase
                            highest_conc_dopant_defect = None
                            stable_dopant_phase = None

                            if self.grouped_defects:
                                for defect_group, group_conc in ordered_defect_concs_grouped:

                                    if not (defect_group == "electrons" or defect_group == "holes"):

                                        for defect in self.data.defects_data:

                                            if defect_group == self.data.defects_data[defect]["group"]:
                                                defect_elements = self.data.defects_data[defect]["added/removed"]
                                                break

                                        if defect_elements[dopant] < 0:
                                            highest_conc_dopant_defect = defect_group
                                            break

                            else:

                                # loop through order defect concentrations list and determine the highest conc defect of this defect
                                for defect_name, defect_conc in ordered_defect_concs:

                                    if not (defect_name == "electrons" or defect_name == "holes"):

                                        defect_elements = self.data.defects_data[defect_name]["added/removed"]
                                        # dopant "added" to a defect if val is negative
                                        if defect_elements[dopant] < 0:
                                            highest_conc_dopant_defect = defect_name
                                            break

                            # loop through secondary phases if any provided
                            if self.data.secondary_phases:
                                for secondary_phase in sorted_phases:

                                    # determine constituent elements of the phases
                                    phase_elements = break_formula(secondary_phase[0])

                                    # first phase to contain the dopant element and "is_stable"
                                    if dopant in phase_elements and secondary_phase[1]["is_stable"]:
                                        stable_dopant_phase = secondary_phase[0]
                                        break

                            # if no stable secondary phases, then dopant is accommodated as a defect
                            if stable_dopant_phase:
                                main_dopant_phase = stable_dopant_phase
                            else:
                                main_dopant_phase = highest_conc_dopant_defect

                            # get main phase of this dopant from the previous loop b iteration
                            if dopant in previous_highest_conc_loop_b_dopants:
                                previous_dopant_phase = previous_highest_conc_loop_b_dopants[dopant]
                            else:
                                previous_dopant_phase = None

                            # add coords to relevant dopant phases
                            dopant_phases[dopant] = self.defect_phases_add_coords(phases_dict=dopant_phases[dopant],
                                                                                  current_phase=main_dopant_phase,
                                                                                  loop_step_x=loop_step_a,
                                                                                  loop_step_y=loop_step_b,
                                                                                  secondary_phases=self.data.secondary_phases
                                                                                  )

                            # variables to hold the current phase as the highest conc phase for the next loop step
                            previous_highest_conc_loop_b_dopants[dopant] = main_dopant_phase


                            if self.data.dopants[dopant]["fitting_option"] == 1 or self.data.dopants[dopant]["fitting_option"] == 2:
                                avg_chem_pot = np.mean(dopant_chem_pots[dopant][loop_step_b])
                                rng_chem_pot = (np.max(dopant_chem_pots[dopant][loop_step_b]) - np.min(dopant_chem_pots[dopant][loop_step_b])) / 2

                                dopant_chem_pots[dopant][loop_step_b] = [avg_chem_pot, rng_chem_pot]


                    # add concentrations to dict. For creating dataframe
                    for defect_name, defect_conc in defects.concentrations.items():

                        if defect_name not in self.all_defect_concentrations:
                            self.all_defect_concentrations[defect_name] = [defect_conc]
                        else:
                            self.all_defect_concentrations[defect_name].append(defect_conc)


                    # add chemical potentials to dict. For creating dataframe
                    for element, chem_pot in chemical_potentials.chem_pots.items():
                        if element not in self.all_chemical_potentials:
                            self.all_chemical_potentials[element] = [chem_pot]
                        else:
                            self.all_chemical_potentials[element].append(chem_pot)

                    # add fermi level to dict. For creating dataframe
                    self.all_fermi_levels["fermi_level"].append(defects.fermi_level)

                    # add formation energies to dict. For creating dataframe
                    for defect_name, form_eng in defects.formation_energies.items():
                        if defect_name not in self.all_formation_energies:
                            self.all_formation_energies[defect_name] = [form_eng]
                        else:
                            self.all_formation_energies[defect_name].append(form_eng)

                    # add secondary phases to dict if any provided. For creating dataframe
                    if self.data.secondary_phases:
                        for phase_name, phase_vals in chemical_potentials.phase_chem_pots.items():
                            if phase_name not in self.all_secondary_phase_vals:
                                self.all_secondary_phase_vals[phase_name] = [phase_vals]
                            else:
                                self.all_secondary_phase_vals[phase_name].append(phase_vals)

                    # write output file during run
                    # can be waiting awhile for it to be written at the end...
                    write_defect_phases_output(f=f, concentrations=defects.concentrations,
                                               formation_energies=defects.formation_energies, chemical_pots=chemical_potentials.chem_pots,
                                               fermi_level=defects.fermi_level, secondary_phases=chemical_potentials.phase_chem_pots,
                                               loop_type_a=self.data.loop, loop_type_b=self.data.loop_y,
                                               loop_a_val=loop_step_a, loop_b_val=loop_step_b, progress_a=progress_meter_a,
                                               progress_b=progress_meter_b, total_a=number_of_iterations_a,
                                               total_b=number_of_iterations_b
                                               )

                loop_end_time = time.time()
                loop_total_time = loop_end_time - loop_start_time
                est_total_time_s = loop_total_time * (number_of_iterations_a - 1)

                if not time_flag and est_total_time_s >= 60:

                    if est_total_time_s < 3600:
                        est_total_time = est_total_time_s / 60
                        time_units = "mins"
                    else:
                        est_total_time = est_total_time_s / 3600
                        time_units = "hrs"

                    print(f"\nTime to complete one loop: {loop_total_time:.1f} seconds")
                    print(f"Estimated run time: {est_total_time:.1f} {time_units}\n")

                time_flag = True


            f.write(">>> Finished\n\n")


        if not os.path.exists("defect_phases_output"):
            os.mkdir("defect_phases_output")
            os.chdir("defect_phases_output")
        else:
            os.chdir("defect_phases_output")

            for file in os.listdir('.'):
                if not file[:2] == '._':
                    os.remove(file)


        print(f"\nGenerating main defect phase boundaries...")
        # refit arrays, so they are all same length for dataframe; orders coords so can be plotted as a polygon
        self.all_defect_phases = self.defect_phases_refit_arrays(phases_dict=defect_phases, xloop_step=self.data.iterator, yloop_step=self.data.iterator_y, xmin=self.data.min_value, xmax=self.data.max_value, ymin=self.data.min_value_y, ymax=self.data.max_value_y)

        # create defect phases dataframe
        defect_phases_df = pd.DataFrame(self.all_defect_phases)
        defect_phases_df.index.name = "index"
        defect_phases_df.to_csv("defect_phases_main_data.csv")

        # create main plot
        defect_phases_plot(self.data.min_value, self.data.max_value, self.data.min_value_y, self.data.max_value_y,
                           "main",  self.data.loop, self.data.loop_y, self.looping_dopants, self.data.art_dopant_chg)

        # create plots for each dopant
        if self.data.dopants:
            for dopant in self.data.dopants:

                print(f"\nGenerating {dopant} defect phase boundaries...")
                # refit arrays so they are all same length, also shifts x-values so plotted polygons tessellate the diagram
                dopant_phases[dopant] = self.defect_phases_refit_arrays(phases_dict=dopant_phases[dopant], xloop_step=self.data.iterator, yloop_step=self.data.iterator_y,  xmin=self.data.min_value, xmax=self.data.max_value, ymin=self.data.min_value_y, ymax=self.data.max_value_y)

                # create dataframe
                dopant_phases_df = pd.DataFrame(dopant_phases[dopant])
                dopant_phases_df.index.name = "index"
                dopant_phases_df.to_csv(f"defect_phases_dopant_{dopant}_data.csv")

                # create plot
                defect_phases_plot(self.data.min_value, self.data.max_value, self.data.min_value_y, self.data.max_value_y,
                                   f"dopant_{dopant}", self.data.loop, self.data.loop_y, self.looping_dopants, self.data.art_dopant_chg)


        # create concentrations dataframe
        concentrations_df = pd.DataFrame(self.all_defect_concentrations)
        concentrations_df.index.name = "index"
        #concentrations_df.to_csv("conc_data.csv")

        # create chem pots dataframe
        chem_pot_df = pd.DataFrame(self.all_chemical_potentials)
        chem_pot_df.index.name = "index"
        #chem_pot_df.to_csv("chem_pot_data.csv")

        # create fermi level dataframe
        fermi_level_df = pd.DataFrame(self.all_fermi_levels)
        fermi_level_df.index.name = "index"
        #fermi_level_df.to_csv("fermi_data.csv")

        # create form eng dataframe
        form_eng_df = pd.DataFrame(self.all_formation_energies)
        form_eng_df.index.name = "index"
        #form_eng_df.to_csv("formation_eng_data.csv")

        # create secondary phases dataframe
        if self.data.secondary_phases:
            phases_df = pd.DataFrame(self.all_secondary_phase_vals)

            phases_df.index.name = "index"
            #phases_df.to_csv("secondary_phases_data.csv")

        else:
            phases_df = pd.DataFrame()

        os.chdir("..")

    def get_grouped_defect_concs(self, concentrations):
        # if defects are to be grouped, determine defect groups and sum total concentration for each group
        grouped_defect_sums = {}
        for defect_name, defect_conc in concentrations.items():
            # defect_group = defect_name for electrons, holes and stoich as they dont have a group assigned in the defects dict
            defect_group = self.data.defects_data[defect_name]["group"] if defect_name != "electrons" and defect_name != "holes" and defect_name != "stoic" else defect_name

            # sum defect concentrations
            if defect_group not in grouped_defect_sums:
                grouped_defect_sums[defect_group] = defect_conc
            else:
                grouped_defect_sums[defect_group] = math.log10(10 ** grouped_defect_sums[defect_group] + 10 ** defect_conc)

        return grouped_defect_sums

    def most_stable_phase(self, phase_chemical_potentials):
        # sort phases from the largest diff in (chem_pot - energy) to smallest
        sorted_phases = sorted(phase_chemical_potentials.items(), key=lambda x: x[1]["difference"], reverse=True)

        # only interested in the phase with the largest difference
        if sorted_phases[0][1]["is_stable"]:
            stable_phase = sorted_phases[0][0]
        else:
            stable_phase = False

        return stable_phase, sorted_phases

    def get_main_defect_phases(self, ordered_defects, is_grouped):
        # hold states of charge states of previous and current defect
        previous_chg = None
        current_chg = None
        main_defects = []

        # loop through ordered list of defects in terms of concentration
        for defect, defect_conc in ordered_defects:

            # assign charges to electrons and holes
            if defect == "electrons":
                defect_chg = -1
            elif defect == "holes":
                defect_chg = 1

            # check if defects have been grouped or not
            # ASSUMES DEFECTS HAVE BEEN GROUPED WITH SAME CHARGE STATE
            elif is_grouped:

                # loop through defects dict, find first defect of this group, and assign this group this charge state
                for defect_name in self.data.defects_data:
                    if defect == self.data.defects_data[defect_name]["group"]:
                        defect_chg = self.data.defects_data[defect_name]["defect_charge"]
                        break
            else:
                # get charge of this defect
                defect_chg = self.data.defects_data[defect]["defect_charge"]

            # stipulate if current defect is positively or negatively charged or neutral
            if defect_chg < 0:
                current_chg = "-ve"
            elif defect_chg > 0:
                current_chg = "+ve"
            else:
                current_chg = 0

            main_defects.append(defect)

            # break loop if charge compensating defects found.
            # Assumes the highest concentrated defects with opposing charges are the charge compensating modes
            if (current_chg == "+ve" and previous_chg == "-ve") or (current_chg == "-ve" and previous_chg == "+ve"):
                break

            # only break for neutral if it has the highest concentration
            elif current_chg == 0 and defect == ordered_defects[0][0]:
                break

            elif len(main_defects) > 2:
                break

            previous_chg = current_chg

        # returns joined string nametag for this defect
        return '+'.join(sorted(main_defects))

    def defect_phases_add_coords(self, phases_dict, current_phase, loop_step_x, loop_step_y, secondary_phases):

        # check if current phase is already in the phases_dict
        if current_phase not in phases_dict:

            # check if current phase is a secondary phase (not a defect) and is not the first phase to be added
            if current_phase in secondary_phases.keys() and len(phases_dict.keys()) > 0:

                # create temp empty dict for the phase then merge it with the phases dict
                # this ensures sec phases are plotted "first" in matlplotlib with defects overlayed
                temp_dict = {current_phase: []}
                phases_dict = temp_dict | phases_dict


            else:
                # empty list to append coords where the current phase is the most favoured phase
                phases_dict[current_phase] = []

        # add coords for this phase
        phases_dict[current_phase].append([loop_step_x, loop_step_y])

        return phases_dict

    def defect_phases_refit_arrays(self, phases_dict, xloop_step, yloop_step, xmin, xmax, ymin, ymax):

        # create a copy of the dict containing defect phases and coords
        copy_phases = phases_dict.copy()

        # loop through phases in the dict
        phases_ordered_coords = {}
        for phase, phase_vals in copy_phases.items():

            # shapely package gets all the exterior points that forms the boundary of each phase
            ob = MultiPoint(phase_vals)

            boundary = ob.buffer(max(xloop_step, yloop_step), cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre)
            #                    ^^^^^^^^^^^^^^^^^^^^^^^^^^ buffer 'distance' needs to be, at the minimum, the largest iterator step


            if boundary.geom_type == "MultiPolygon":
                fst_poly = list(boundary.geoms)[0]
                phases_ordered_coords[phase] = np.array(fst_poly.exterior.coords)

            else:
                # convert ordered coords to a numpy array and append this phase and coords to the dict
                phases_ordered_coords[phase] = np.array(boundary.exterior.coords)


        # boundary buffer can cause issues with labels, so shift these back to the plot boundaries
        for phase, phase_vals in phases_ordered_coords.items():

            x_min_indices = np.where(phase_vals[:, 0] < xmin)
            x_max_indices = np.where(phase_vals[:, 0] > xmax)
            y_min_indices = np.where(phase_vals[:, 1] < ymin)
            y_max_indices = np.where(phase_vals[:, 1] > ymax)

            phase_vals[:, 0][x_min_indices] = xmin
            phase_vals[:, 0][x_max_indices] = xmax
            phase_vals[:, 1][y_min_indices] = ymin
            phase_vals[:, 1][y_max_indices] = ymax


            #phase_shift_boundary_coords[phase] = new_coords


        #print("\nGetting poylgons boundaries...\n")
        phases_arrays_equal_length = {}

        # determine the phase with the most coords and the number of coords
        longest_array_length = max(len(coords) for coords in phases_ordered_coords.values())

        for phase, phase_vals in phases_ordered_coords.items():

            # get number of coords in current phase
            num_coords = len(phase_vals)

            # create a copy of the coords to append to
            new_vals = list(phase_vals.copy())

            # append np.nan coords until array length matches the longest array
            # matplotlib ignores np.nan values when plotting, but pandas need all arrays as the same length in the dataframe
            while num_coords < longest_array_length:
                new_vals.append([np.nan, np.nan])
                num_coords += 1

            # create dict of phases with coords all equal in length
            phases_arrays_equal_length[phase] = [np.array(item) for item in new_vals]

        return phases_arrays_equal_length


    def madelung(self):

        np.set_printoptions(formatter={'float': "{0:0.3f}".format})

        # get real and reciprocal lattice parameters
        # reciprocal parameters are multiplied by 2*np.pi
        R = self.data.lattice_array
        G = 2*np.pi*np.linalg.inv(R)
        sc_volume = np.linalg.det(R)

        # get real and inverse of dielectric array
        epsilon = self.data.dielectric_array
        inv_epsilon = np.linalg.inv(epsilon)
        det_epsilon = np.linalg.det(epsilon)

        # get defect coords in motif
        motif = [[0.625, 0.392, 0.375, -1], [0.625, 0.625, 0.375, 2], [0.76, 0.76, 0.5, -1], [0.25, 0.75, 0.5, -1]]

        # calculate lattice parameter lengths
        vector_lengths = [np.linalg.norm(R[0]),
                          np.linalg.norm(R[1]),
                          np.linalg.norm(R[2])]

        print(f"Lattice Parameters:\n{R}\n")
        print(f"Vector lengths: {vector_lengths[0]:.3f} {vector_lengths[1]:.3f} {vector_lengths[2]:.3f}")

        # calculates lattice angles for users to check right geometry
        r0_u = R[0] / vector_lengths[0]
        r1_u = R[1] / vector_lengths[1]
        r2_u = R[2] / vector_lengths[2]
        ang_a = np.degrees(np.arccos(np.clip(np.dot(r0_u, r1_u), -1.0, 1.0)))
        ang_b = np.degrees(np.arccos(np.clip(np.dot(r0_u, r2_u), -1.0, 1.0)))
        ang_c = np.degrees(np.arccos(np.clip(np.dot(r1_u, r2_u), -1.0, 1.0)))

        print(f"Cell angles: \t   {ang_a:.3f} {ang_b:.3f}, {ang_c:.3f}\n\n")

        print(f"Reciprocal lattice Parameters (times 2pi):\n{G}\n")
        print(f"Supercell volume: {sc_volume}\n\n")
        print(f"Dielectric tensor:\n{epsilon}\n\n")
        print(f"Reciprocal dielectric tensor:\n{inv_epsilon}\n")
        print(f"Dielectric determinate:{det_epsilon}\n\n")
        print(f"Gamma convergence parameter: {self.data.gamma}\n\n")

        # determine real space cut off by multiplying cutoff factor by the longest lattice parameter
        real_cutoff = self.data.cutoff * max(vector_lengths)

        # set real space limits by dividing the cutoff factor by each lattice parameter
        # + 10 to ensure cutoff is within a cell
        real_limits = np.array([round(real_cutoff / v + 10) for v in vector_lengths]).astype(np.float64)

        # create real and reciprocal superlattices
        superlattice = []
        recip_superlattice = []
        for i in range(3):
            superlattice.append(R[i]*real_limits[i])
            recip_superlattice.append(G[i]*real_limits[i])

        superlattice = np.array(superlattice)
        recip_superlattice = np.array(recip_superlattice)

        real_sum = 0
        recip_sum = 0
        incell_sum = 0

        for motif_atom in motif:

            motif_atom_array = np.array(motif_atom[:-1])
            motif_charge = motif_atom[-1]

            motif_cart_coords = np.array([np.sum(motif_atom_array * R[:, i]) for i in range(3)])

            for other_motif_atom in motif:

                other_motif_atom_array = np.array(other_motif_atom[:-1])
                image_chg = other_motif_atom[-1]

                # loop through superlattice limits
                for m in np.arange(-real_limits[0], real_limits[0], 1):
                    for n in np.arange(-real_limits[1], real_limits[1], 1):
                        for o in np.arange(-real_limits[2], real_limits[2], 1):

                            real_array = np.array([m + other_motif_atom_array[0],
                                                   n + other_motif_atom_array[1],
                                                   o + other_motif_atom_array[2]]
                                                  )

                            real_array /= real_limits

                            # get cartesian real coords for this superlattice point. Then calculate the length of this vector
                            # superlattice[:,i] gets the column of index 'i' in superlattice
                            real_coords = np.array([np.sum(real_array * superlattice[:, i]) for i in range(3)])
                            distance = np.linalg.norm(real_coords-motif_cart_coords)


                            if motif_atom == other_motif_atom and m == n == o == 0:
                                continue

                            else:
                                # only calculate real space contribution of points withing the cutoff distance
                                if distance < real_cutoff:
                                    # dot product of R, inverse dielectric and R
                                    Ri_ieps_Ri = np.linalg.multi_dot([real_coords-motif_cart_coords, inv_epsilon, real_coords-motif_cart_coords])

                                    if m == n == o == 0:
                                        incell_sum += motif_charge * image_chg * ((1/np.sqrt(det_epsilon)) * (1/np.sqrt(Ri_ieps_Ri)))


                                    # From Murphy, Hine anisotropic charge correction
                                    real_sum += (motif_charge * image_chg) * (1 / (np.sqrt(det_epsilon)) * (special.erfc(self.data.gamma * np.sqrt(Ri_ieps_Ri))) / (np.sqrt(Ri_ieps_Ri)))


        # loop through superlattice limits
        for s in np.arange(-real_limits[0], real_limits[0], 1):
            for t in np.arange(-real_limits[1], real_limits[1], 1):
                for u in np.arange(-real_limits[2], real_limits[2], 1):

                    recip_array = np.array([s, t, u]) / real_limits

                    # get cartesian reciprocal coords
                    recip_coords = np.array([np.sum(recip_array * recip_superlattice[:, i]) for i in range(3)])

                    # creates numerical error at the centre, therefore, skip this point
                    if s == t == u == 0:
                        continue
                    else:
                        # dot product of G, dielectric, G
                        Gi_eps_Gi = np.linalg.multi_dot([recip_coords, epsilon, recip_coords])

                        cos_cumulative = 0
                        sin_cumulative = 0

                        for motif_atom in motif:
                            motif_atom_array = np.array(motif_atom[:-1])
                            motif_charge = motif_atom[-1]

                            motif_cart_coords = np.array([np.sum(motif_atom_array * R[:, i]) for i in range(3)])
                            rdotG = np.dot(motif_cart_coords, recip_coords)

                            cos_cumulative += motif_charge * np.cos(rdotG)
                            sin_cumulative += motif_charge * np.sin(rdotG)

                        recip_sum += (cos_cumulative**2 + sin_cumulative**2) * (4 * np.pi / sc_volume) * np.exp(-Gi_eps_Gi / (4 * self.data.gamma ** 2)) / Gi_eps_Gi


        # calculate self interaction and background contributions
        chg_sq_sum = 0
        total_chg = 0
        for motif_atom in motif:
            motif_charge = motif_atom[-1]
            chg_sq_sum += motif_charge**2
            total_chg += motif_charge

        self_interaction = -((2*self.data.gamma)/np.sqrt(np.pi * det_epsilon)) * chg_sq_sum
        background = (-np.pi / (sc_volume * self.data.gamma**2)) * total_chg**2

        if len(motif) < 2:
            # final screened madelung
            # remember, this val is NOT in eV!
            screened_madelung = -(real_sum + recip_sum + self_interaction + background)

            #print(f"Screened_madelung: {screened_madelung}")
            print(f"real contribution: {real_sum}")
            print(f"reciprocal: {recip_sum}")
            print(f"self interaction: {self_interaction}")
            print(f"background: {background}")
            print(f"madelung pot: {screened_madelung}")

        else:

            conversion = 14.39942

            real_sum_eV = real_sum*conversion/2
            recip_sum_eV = recip_sum*conversion/2
            self_interaction_eV = self_interaction*conversion/2
            background_eV = background*conversion/2
            incell_eV = incell_sum*conversion/2
            total_eV = real_sum_eV + recip_sum_eV + self_interaction_eV + background_eV
            madelung_eV = -(total_eV - incell_eV)

            print(f"real contribution: {real_sum_eV}")
            print(f"reciprocal: {recip_sum_eV}")
            print(f"self interaction: {self_interaction_eV}")
            print(f"background: {background_eV}")
            print(f"total: {total_eV}")
            print(f"incell: {incell_eV}")
            print(f"madelung pot: {madelung_eV}")