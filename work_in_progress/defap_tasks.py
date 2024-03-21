import math

import numpy as np
import pandas as pd
import thermodynamics
from defap_chem_pots import ChemicalPotentials
import defap_defects
from defap_output import brouwer_plot
from defap_output import defect_phases_plot
from defap_output import write_brouwer_output
from defap_output import write_defect_phases_output
from defap_misc import break_formula
from defap_misc import refit_defect_phase_arrays

class Tasks:

    def __init__(self, data, seedname, high_conc_check=False, dopant_warning=False, looping_dopants=None,
                 grouped_defects=False, all_defect_concentrations=None, all_defect_phases=None,
                 all_chemical_potentials=None,
                 all_fermi_levels=None, all_formation_energies=None, all_secondary_phase_vals=None
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
        self.all_secondary_phase_vals = all_secondary_phase_vals

        # calculate initial entropy values at specified temperature
        # recalculated if looping over temperature
        self.entropy_vals = thermodynamics.calc_entropies(entropy_data=self.data.entropy_data,
                                                          temperature=self.data.temperature)

    def loop_assignment(self, loop_type, loop_step, loop_a_or_b):

        match loop_type:

            # assign loop step to the volatile constituent partial pressure
            case 0:
                self.data.constituents["volatile"]["log_pp"] = loop_step

            # assign loop step to the temperature of the system
            case 1:
                self.data.temperature = loop_step

            # assign loop step to the specified dopant concentration
            case 2:

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
            case 3:

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

            # artificial dopant charge
            case 4:
                self.data.art_dopant_conc = 10**loop_step


            # direct chemical potential of a dopant
            case 5:

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


            # loop over rich poor fraction
            case 6:
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

                if loop_step + other_rich_poor_fracs > total_fractions + 0.0001:  # account for binary fraction overspill
                    raise ValueError(f"ERROR<!> Total rich poor fraction from loop exceeds {total_fractions}!\n"
                                     f"Please modify the range of your loop")


            case _:
                raise ValueError(f"ERROR<!> '{loop_type}' is not a valid method for a loop iteration!")


    def defect_phases_add_coords(self, phases_dict, current_phase, loop_step_x, loop_step_y, prev_loop_step_y_phase):

        if current_phase not in phases_dict:
            phases_dict[current_phase] = [[], []]

        if float(f"{loop_step_y:.1f}") == self.data.min_value_y:
            phases_dict[current_phase][0].append([loop_step_x, loop_step_y])

        if float(f"{loop_step_y:.1f}") == self.data.max_value_y:
            phases_dict[current_phase][1].insert(0, [loop_step_x, loop_step_y])

        if current_phase != prev_loop_step_y_phase and prev_loop_step_y_phase is not None:
            y_mid = float(f"{loop_step_y - (self.data.iterator_y / 2):.3f}")

            phases_dict[current_phase][0].append([loop_step_x, y_mid])
            phases_dict[prev_loop_step_y_phase][1].insert(0, [loop_step_x, y_mid])

        return phases_dict

    def defect_phases_refit_arrays(self, phases_dict):

        copy_phases = phases_dict.copy()
        for phase, phase_vals in copy_phases.items():
            if phase_vals[0][0][0] != self.data.min_value:
                phases_dict[phase][0][0][0] -= (self.data.iterator / 2)

            if phase_vals[0][-1][0] != self.data.max_value:
                phases_dict[phase][0][-1][0] += (self.data.iterator / 2)

            if phase_vals[1][0][0] != self.data.max_value:
                phases_dict[phase][1][0][0] += (self.data.iterator / 2)

            if phase_vals[1][-1][0] != self.data.min_value:
                phases_dict[phase][1][-1][0] -= (self.data.iterator / 2)

            phases_dict[phase] = phase_vals[0] + phase_vals[1]


        phases_arrays_equal_length = {}
        longest_array_length = max(len(coords) for coords in phases_dict.values())
        for phase, phase_vals in phases_dict.items():
            nan_lst = []
            for i in range(longest_array_length - len(phase_vals)):
                nan_lst.append([np.nan, np.nan])

            if nan_lst != []:
                phases_arrays_equal_length[phase] = list(np.concatenate([phases_dict[phase], np.array(nan_lst)]))
            else:
                phases_arrays_equal_length[phase] = [np.array(item) for item in phases_dict[phase]]

        return phases_arrays_equal_length

    def defect_phases(self):

        defect_phases = {}
        dopant_phases = {}

        # check if defects are to be grouped together
        if "group" in self.data.tasks:
            self.grouped_defects = True

        # variables to determine where perfect stoic occurs for each x-axis values
        min_stoic_coords = None
        min_stoic = None

        # determine number of loop iterations there will be
        number_of_iterations_a = ((self.data.max_value - self.data.min_value) / self.data.iterator)
        number_of_iterations_b = ((self.data.max_value_y - self.data.min_value_y) / self.data.iterator_y)

        # begin loop_a
        for progress_meter_a, loop_step_a in enumerate(
                np.arange(self.data.min_value, self.data.max_value + self.data.iterator, self.data.iterator)):

            # assign loop_a value to desired property
            self.loop_assignment(loop_type=self.data.loop, loop_step=loop_step_a, loop_a_or_b="a")

            previous_highest_conc_loop_b = None
            previous_highest_conc_loop_b_dopants = {}

            # begin loop_b
            for progress_meter_b, loop_step_b in enumerate(
                    np.arange(self.data.min_value_y, self.data.max_value_y + self.data.iterator_y,
                              self.data.iterator_y)):

                # assign loop_b value to desired property
                self.loop_assignment(loop_type=self.data.loop_y, loop_step=loop_step_b, loop_a_or_b="b")

                print(f"..> Calculating defect concentrations for x: {progress_meter_a} of {number_of_iterations_a}"
                      f"  ||  y: {progress_meter_b} of {number_of_iterations_b}", end="\r", flush=True)

                # calculate entropy if requested
                if self.data.entropy_data and (self.data.loop == 1 or self.data.loop_y == 1):
                    self.entropy_vals = thermodynamics.calc_entropies(entropy_data=self.data.entropy_data,
                                                                      temperature=self.data.temperature)

                # initialise class for calculating chemical potentials
                chemical_potentials = ChemicalPotentials(data=self.data,
                                                         entropy_vals=self.entropy_vals
                                                         )

                # calculate chemical potentials depending on the method specified on the input
                chemical_potentials.call_chemical_pot_method()

                # add any dopants to the chem pots dictionary and determine how many require chemical potentials to be fitted
                if self.data.dopants:
                    chemical_potentials.add_dopants()

                # uncomment to view chemical potentials
                # print(chemical_potentials.chem_pots)

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

                    # for determining position of perfect stoich
                    if min_stoic_coords is None:
                        min_stoic = log_stoic
                        min_stoic_coords = [loop_step_a, loop_step_b]

                    elif log_stoic < min_stoic:
                        min_stoic = log_stoic
                        min_stoic_coords = [loop_step_a, loop_step_b]

                # calculate chemical potential sums of secondary phases
                if self.data.secondary_phases:
                    chemical_potentials.calc_secondary_phase_chem_pots()
                    # sort phases from the largest diff in (chem_pot - energy) to smallest
                    sorted_phases = sorted(chemical_potentials.phase_chem_pots.items(), key=lambda x: x[1]["difference"], reverse=True)

                    # only interested in the phase with the largest difference
                    if sorted_phases[0][1]["difference"] > 0:
                        stable_phase = True
                    else:
                        stable_phase = False

                else:
                    stable_phase = False

                # create an ordered list of defects from highest to lowest concentration
                ordered_defect_concs = sorted(defects.concentrations.items(), key=lambda x: x[1], reverse=True)

                # secondary phases overrule defect phases
                if stable_phase:
                    main_phase = sorted_phases[0][0]

                else:

                    # temporary variables to hold the 3 highest defect concentrations
                    highest_conc_defect = ordered_defect_concs[0][0]
                    second_highest_conc_defect = ordered_defect_concs[1][0]
                    third_highest_conc_defect = ordered_defect_concs[2][0]

                    # electron/holes are charged defects, therefore, require charge compensation
                    if highest_conc_defect == "electrons" or highest_conc_defect == "holes":

                        # just covers edge cases where neutral defects and electrons/holes are the highest conc, ignores electrons/holes
                        if second_highest_conc_defect in self.data.defects_data and self.data.defects_data[second_highest_conc_defect]["defect_charge"] == 0:
                            # "sorted" list for defect names ensures label is always joined in same order
                            main_defects = sorted([highest_conc_defect, second_highest_conc_defect, third_highest_conc_defect])
                        else:
                            main_defects = sorted([highest_conc_defect, second_highest_conc_defect])

                    # neutral defects have no charge compensating defects
                    elif self.data.defects_data[highest_conc_defect]["defect_charge"] == 0:
                        main_defects = [highest_conc_defect]

                    else:
                        main_defects = sorted([highest_conc_defect, second_highest_conc_defect])

                    # create phase label by joining defect names together
                    main_phase = '+'.join(main_defects)

                # add coords to relevant phases
                defect_phases = self.defect_phases_add_coords(phases_dict=defect_phases,
                                                              current_phase=main_phase,
                                                              loop_step_x=loop_step_a,
                                                              loop_step_y=loop_step_b,
                                                              prev_loop_step_y_phase=previous_highest_conc_loop_b
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

                        # loop through order defect concentrations list and determine the highest conc defect of this defect
                        for defect_name, defect_conc in ordered_defect_concs:

                            if defect_name == "electrons" or defect_name == "holes":
                                continue

                            else:
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
                                                                              prev_loop_step_y_phase=previous_dopant_phase
                                                                              )

                        # variables to hold the current phase as the highest conc phase for the next loop step
                        previous_highest_conc_loop_b_dopants[dopant] = main_dopant_phase


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

        # refit arrays so they are all same length, also shifts x-values so plotted polygons tessellate the diagram
        self.all_defect_phases = self.defect_phases_refit_arrays(phases_dict=defect_phases)

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

                # refit arrays so they are all same length, also shifts x-values so plotted polygons tessellate the diagram
                dopant_phases[dopant] = self.defect_phases_refit_arrays(phases_dict=dopant_phases[dopant])

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

        # write final output file
        write_defect_phases_output(conc_dataframe=concentrations_df, form_eng_dataframe=form_eng_df,
                                   chem_pot_dataframe=chem_pot_df, fermi_level_dataframe=fermi_level_df,
                                   secondary_phases_dataframe=phases_df,
                                   loop_type_a=self.data.loop, loop_type_b=self.data.loop_y,
                                   loop_a_vals=np.arange(self.data.min_value, self.data.max_value + self.data.iterator, self.data.iterator),
                                   loop_b_vals=np.arange(self.data.min_value_y, self.data.max_value_y + self.data.iterator_y, self.data.iterator_y),
                                   seedname=self.seedname
                                   )


    def brouwer(self):

        # check if defects are to be grouped together
        if "group" in self.data.tasks:
            self.grouped_defects = True

        # determine number of loop iterations there will be
        number_of_iterations = ((self.data.max_value - self.data.min_value) / self.data.iterator)

        stoic_vs_opp = {"stoic": [], "ox_pot": []}

        # begin loop
        for progress_meter, loop_step in enumerate(np.around(np.arange(self.data.min_value, self.data.max_value + self.data.iterator, self.data.iterator), decimals=5)):

            #loop_step = np.around(loop_step, decimals=5)

            # assign loop value being iterated over to the desired property
            self.loop_assignment(loop_type=self.data.loop, loop_step=loop_step, loop_a_or_b="a")

            print(f"..> Calculating defect concentrations: {progress_meter} of {number_of_iterations}", end="\r", flush=True)

            # if looping over temperature, ensure that entropy values are recalculated
            if self.data.loop == 1 and self.data.entropy_data:
                self.entropy_vals = thermodynamics.calc_entropies(entropy_data=self.data.entropy_data,
                                                                  temperature=self.data.temperature
                                                                  )

            # initialise class for calculting chemical potentials
            chemical_potentials = ChemicalPotentials(data=self.data,
                                                     entropy_vals=self.entropy_vals
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

                # calculate formation energies at valence band maximum then optimise fermi level to determine concentrations
                defects.calc_form_eng_vbm()
                defects.optimise_fermi_level()

            # warning message for defect concentrations > 1 pfu (indicates a -ve form eng)
            if max(list(defects.concentrations.values())) > 0 and not self.high_conc_check:
                print("\n<!> Very high concentrations predicted, exceeding 1 p.f.u.: "
                      "This will not be visible on default Brouwer diagram\n")

                self.high_conc_check = True

            # calculate the stoichiometry of a volatile system if indiciated
            if self.data.stoichiometry == 1 or self.data.stoichiometry == 2:
                log_stoic, stoic = defects.calc_stoichiometry()

                defects.concentrations["stoic"] = log_stoic

                # stoic_vs_opp[stoic] = math.log(10 ** loop_step) * (8.314 / 1000) * self.data.temperature
                stoic_vs_opp["stoic"].append(2 + stoic)
                stoic_vs_opp["ox_pot"].append(math.log(10 ** loop_step) * (8.314 / 1000) * self.data.temperature)

            # calculate chemical potential sums of secondary phases
            if self.data.secondary_phases:
                chemical_potentials.calc_secondary_phase_chem_pots()

            match self.grouped_defects:

                # if defects are to be grouped, determine defect groups and sum total concentration for each group
                case True:
                    grouped_defect_sums = {}
                    for defect_name, defect_conc in defects.concentrations.items():
                        # defect_group = defect_name for electrons, holes and stoich as they dont have a group assigned in the defects dict
                        defect_group = self.data.defects_data[defect_name]["group"] if defect_name != "electrons" and defect_name != "holes" and defect_name != "stoic" else defect_name

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
                case False:
                    for defect_name, defect_conc in defects.concentrations.items():
                        if defect_name not in self.all_defect_concentrations:
                            self.all_defect_concentrations[defect_name] = [defect_conc]
                        else:
                            self.all_defect_concentrations[defect_name].append(defect_conc)

            # add final calculated chemical potentials to the all_chemical_potentials,
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

        # create dataframe to plot brouwer results with, indexed using each loop step value
        brouwer_df = pd.DataFrame(self.all_defect_concentrations, index=np.arange(self.data.min_value, self.data.max_value + self.data.iterator, self.data.iterator))
        brouwer_df.index.name = "loop_step"
        brouwer_df.to_csv(f"brouwer_data.csv")

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

        # stoic_vs_opp_df = pd.DataFrame(stoic_vs_opp, index=np.arange(self.data.min_value, self.data.max_value + self.data.iterator, self.data.iterator))
        # stoic_vs_opp_df.index.name = "loop_step"
        # stoic_vs_opp_df.to_csv("stoic_vs_opp.csv")

        # create brouwer plot
        brouwer_plot(axes=(self.data.min_value, self.data.max_value, self.data.y_axis_min, self.data.y_axis_max),
                     defects_data=self.data.defects_data,
                     grouped_defects=self.grouped_defects,
                     loop_type=self.data.loop,
                     looping_dopants=self.looping_dopants,
                     art_dopant_chg=self.data.art_dopant_chg
                     )

        write_brouwer_output(conc_dataframe=brouwer_df, form_eng_dataframe=form_eng_df,
                             chem_pot_dataframe=chem_pot_df, fermi_level_dataframe=fermi_level_df,
                             secondary_phases_dataframe=phases_df,
                             loop_type=self.data.loop, seedname=self.seedname,
                             )
