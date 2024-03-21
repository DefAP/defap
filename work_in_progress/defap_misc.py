from bisect import bisect_left
import numpy as np


def format_constituents(constituents_lst, chemical_potential_method):

    """
    Function that formats the constituents into dictionaries that are more readable and can access
    variables with less loops. Formats based on the chemical potential method
    :param constituents_lst:
    :param chemical_potential_method:
    :return: all_constituents_dic
    """

    all_constituents_dic = {"volatile": None,
                            "compound": None,
                            "constituent_compounds": None
                            }
    total_rich_poor_fraction = 0

    # loop through list of constituents
    for constituent_index, constituent in enumerate(constituents_lst):
        individual_constituents_dic = {}

        if chemical_potential_method == "defined":

            # simple assign defined species with its dft energy
            individual_constituents_dic["elements"] = break_formula(constituent[0])
            individual_constituents_dic["dft_energy_pfu"] = float(constituent[1])
            all_constituents_dic[constituent[0]] = individual_constituents_dic

        elif chemical_potential_method == "rich-poor":

            # assign dft energy and rich-poor fraction for this specie to a dict
            individual_constituents_dic["elements"] = break_formula(constituent[0])
            individual_constituents_dic["dft_energy_pfu"] = float(constituent[1])
            individual_constituents_dic["fraction"] = float(constituent[2])

            # check total rich-poor fraction is not exceeded
            total_rich_poor_fraction += individual_constituents_dic["fraction"]
            if total_rich_poor_fraction > len(constituents_lst) - 1:
                raise Exception("ERROR <!> Sum of constituent rich-poor fractions too large for the number of constituents provided.\n"
                                f"Max sum of fractions: {len(constituents_lst) - 1}\n"
                                f"Total sum of fractions specified: {total_rich_poor_fraction}")

            # add specie to the dict for all constituents
            all_constituents_dic[constituent[0]] = individual_constituents_dic
            all_constituents_dic["total_fraction"] = total_rich_poor_fraction

        elif chemical_potential_method == "volatile" or chemical_potential_method == "volatile-reference":
            # first constituent is always the volatile specie
            if constituent_index == 0:
                individual_constituents_dic["volatile_element"] = constituent[0]
                individual_constituents_dic["log_pp"] = float(constituent[1])
                all_constituents_dic["volatile"] = individual_constituents_dic

            else:
                individual_constituents_dic["formula"] = constituent[0]
                individual_constituents_dic["elements"] = break_formula(constituent[0])
                individual_constituents_dic["compound_energy_pfu"] = float(constituent[1])
                individual_constituents_dic["metal_energy_pfu"] = float(constituent[2])
                individual_constituents_dic["std_formation_energy"] = float(constituent[3])
                all_constituents_dic["compound"] = individual_constituents_dic

        elif chemical_potential_method == "volatile-rich-poor":
            if constituent_index == 0:
                individual_constituents_dic["volatile_element"] = constituent[0]
                individual_constituents_dic["coefficient"] = float(constituent[1])
                individual_constituents_dic["log_pp"] = float(constituent[2])
                all_constituents_dic["volatile"] = individual_constituents_dic

            else:
                individual_constituents_dic["elements"] = break_formula(constituent[0])
                individual_constituents_dic["coefficient"] = float(constituent[1])
                individual_constituents_dic["compound_energy_pfu"] = float(constituent[2])
                individual_constituents_dic["metal_energy_pfu"] = float(constituent[3])
                individual_constituents_dic["std_formation_energy"] = float(constituent[4])
                individual_constituents_dic["fraction"] = float(constituent[5])

                total_rich_poor_fraction += individual_constituents_dic["fraction"]
                if total_rich_poor_fraction > len(constituents_lst) - 2: # -2 to account for the volatile in the list
                    raise Exception(
                        "ERROR <!> Sum of constituent rich-poor fractions too large for the number of constituents provided.\n"
                        f"Max sum of fractions: {len(constituents_lst) - 2}\n"
                        f"Total sum of fractions specified: {total_rich_poor_fraction}")

                if all_constituents_dic["constituent_compounds"] is None:
                    all_constituents_dic["constituent_compounds"] = {}
                    all_constituents_dic["constituent_compounds"][constituent[0]] = individual_constituents_dic
                else:
                    all_constituents_dic["constituent_compounds"][constituent[0]] = individual_constituents_dic

                all_constituents_dic["total_fraction"] = total_rich_poor_fraction



        else:
            raise Exception("ERROR <!> Invalid method for calculating chemical potentials.\n"
                            f"Your input: {chemical_potential_method}")

    return all_constituents_dic


def format_dopants(dopants_lst):

    """
    Function that formats the list of dopants provided
    :param dopants_lst:
    :return: all_dopants_dict
    """

    dopants_dict = {}
    dopants_to_fit = 0
    dopants_to_loop_conc = 0
    dopants_to_loop_press = 0


    for dopant_index, dopant in enumerate(dopants_lst):

        individual_dopant_dict = {"reference": dopant[1],
                                  "reference_elements": break_formula(dopant[1]),
                                  "chemical_potential": float(dopant[2]),
                                  "fitting_option": int(dopant[3]),
                                  }

        if int(dopant[3]) == 1 or int(dopant[3]) == 2: # fit concentration
            individual_dopant_dict["concentration_pfu"] = float(dopant[4])
            individual_dopant_dict["chem_pot_range"] = float(dopant[5])
            dopants_to_fit += 1

            if int(dopant[3]) == 2:
                dopants_to_loop_conc += 1

        elif int(dopant[3]) == 3 or int(dopant[3]) == 4: # calc chem pot from partial pressure
            individual_dopant_dict["log_PP_atm"] = float(dopant[4])

            if int(dopant[3]) == 4:
                dopants_to_loop_press += 1


        elif int(dopant[3]) == 0 or int(dopant[3]) == 5:
            pass


        else:
            raise Exception(f"ERROR<!> Invalid fitting option for {dopant[0]} dopant!")


        if (dopants_to_loop_conc + dopants_to_loop_press) > 2:
            raise ValueError("ERROR<!> Can't loop over more than 2 dopants at once!")

        dopants_dict[dopant[0]] = individual_dopant_dict

    return dopants_dict, dopants_to_fit


def format_secondary_phases(phases_lst):

    phases_dict = {}

    for phase in phases_lst:
        num_phases_vals = len(phase)

        phases_dict[phase[0]] = {"elements": break_formula(phase[0]),
                                 "energy": float(phase[1]),
                                 "add_temperature_contribution": False,
                                 "use_gibbs_energy": False,
                                 }

        if num_phases_vals > 2:

            if phase[2].lower() == "true" or phase[2].lower() == "t":
                phases_dict[phase[0]]["add_temperature_contribution"] = True

            elif phase[2].lower() == "gibbs_true":
                phases_dict[phase[0]]["use_gibbs_energy"] = True

    return phases_dict


def break_formula(formula):
    """
    Function that takes a chemical formula and breaks it down into a dictionary containing constituent elements
    and stoichiometries
    """

    # break formula into a dictionary of {"Element" : stoichiometry}
    constituent_formula_breakdown = {}

    # Split the constituent species by the hyphen
    split_formula = formula.split('-')

    # loop through species in the formula
    for species in split_formula:
        # split species into element and its stoichiometry
        split_stoichiometry = species.split("_")

        if len(split_stoichiometry) == 2:
            element, stoichiometry = split_stoichiometry[0], int(split_stoichiometry[1])
        else:
            # stoichiometry is 1 if no "_value" provided
            element, stoichiometry = split_stoichiometry[0], 1

        # add species and stoich to dictionary
        constituent_formula_breakdown[element] = stoichiometry

    return constituent_formula_breakdown


def check_input_errors(data_input):

    # check entropy data has been provided if needed
    if data_input.entropy_data is None and data_input.entropy == 1:
        print("\n>>> WARNING <<<\nInclusion of vibrational entropy specified in the input file (entropy : 1)\n"
              "However, no entropy data has been provided!\n"
              "Switching vibrational entropy off (entropy = 0)")

        # if not, switch off
        data_input.entropy = 0

    # check that functional units have been provided for entropy inclusion
    if data_input.entropy == 1 and data_input.entropy_units is None:
        raise ValueError("\n<!> ERROR\nInclusion of vibrational entropy specified in the input file (entropy : 1)\n"
                         "However, the number of functional units used to calculate entropy contribution has not been"
                         " specified! (entropy_units)\n"
                         "Please specify 'entropy_units : (int)' tag\n")

    # check gibbs data has been provided if needed
    if data_input.gibbs_data is None and data_input.include_gibbs == 1:
        print("\n>>> WARNING <<<\nInclusion of Gibbs temperature dependant energy contributions specified in the"
              " input file (gibbs : 1)\n"
              "However, no Gibbs energy data has been provided!\n"
              "Switching Gibbs energy contributions off (gibbs = 0)")

        # switch off it no data
        data_input.include_gibbs = 0

    # check dos data has been provided for fermi-dirac statistics
    if data_input.dos_data is None:
        if data_input.electron_method == "fermi-dirac" or data_input.hole_method == "fermi-dirac":
            raise ValueError("\n<!> ERROR\nFermi-Dirac method for calculating charge carrier concentrations requires "
                             "material density of states.\n"
                             "No density of state data has been provided!\n"
                             "Please provide a 'filename.dos' file\n")

    # check that defect data has been provided
    if data_input.defects_data is None:
        raise ValueError("\n<!> ERROR\n"
                         "Using the Defect Analysis Package with no defects provided!?!?! Madness!\n"
                         "Jokes aside... Have you provide a 'filename.defects' or 'filename_defects.csv' file?\n")


def take_closest(myList, myNumber):
    """
    For bisecting DOS energy list to determine index of VBM and CBM
    :param myList:
    :param myNumber:
    :return:
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]

    before = myList[pos - 1]
    after = myList[pos]

    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def refit_defect_phase_arrays(phases_dict, loop_min, loop_max, loop_iterator):
    # ensure all phase lists are the same length by appending the required num of np.nan
    # arrays need to be same length to create pandas dataframe

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

