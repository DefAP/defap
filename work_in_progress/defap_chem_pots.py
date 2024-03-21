from typing import Optional
from thermodynamics import calc_pressure_contribution
from thermodynamics import calc_temperature_contribution


class ChemicalPotentials:
    """
    Class for calculating chemical potentials
    """

    def __init__(self, data, entropy_vals):

        self.data = data
        self.entropy_vals = entropy_vals

        # ensures chemical potentials are returned in defap_tasks when chemical_potentials.chem_pots is called
        self.chem_pots = self.call_chemical_pot_method()

        self.phase_chem_pots = None

        self.fitting_dopants = []
        self.looping_dopants = []

    def call_chemical_pot_method(self):

        # auto call chemical potential function depending on method specified
        match self.data.chem_pot_method:

            case "defined":
                pass

            case "rich-poor":
                pass

            case "volatile":
                return self.calc_chemical_volatile()

            case "volatile-rich-poor":
                pass

    def calc_chemical_volatile(self):

        # energy of the host from input file
        host_energy = self.data.host_energy_pfu

        # unpack dictionary to get values of the compound provided
        compound_formula, compound_elements, compound_energy, compound_metal_energy, compound_std_form_eng = \
        self.data.constituents["compound"].values()

        # unpack elements to get element stoichiometries from compound provided
        # assume format of compound is: M_{a}-V_{b}
        metal_compound_stoic, volatile_compound_stoic = compound_elements.values()

        # calculate volatile standard chemical potential
        nu_volatile_std = (compound_energy - (
                    metal_compound_stoic * compound_metal_energy) - compound_std_form_eng) / volatile_compound_stoic

        # calculate temperature contribution to volatile chem pot
        temp_cont = 0.5 * calc_temperature_contribution(
            gas_species=self.data.constituents["volatile"]["volatile_element"],
            temperature=self.data.temperature,
            real_gas=self.data.real_gas
            )

        # calculate pressure contribution to volatile chem pot
        pressure_cont = calc_pressure_contribution(volatile_PP=self.data.constituents["volatile"]["log_pp"],
                                                   temperature=self.data.temperature
                                                   )

        # calculate volatile chemical potential at desired conditions
        nu_volatile = nu_volatile_std + temp_cont + pressure_cont

        # add entropy contribution to host energy if specified
        if self.entropy_vals:
            # host should be first key in the dict
            host_entropy_key = list(self.entropy_vals.keys())[0]

            if host_entropy_key == "host" or host_entropy_key == self.data.host["formula"]:
                host_entropy = self.entropy_vals[host_entropy_key]
                host_energy -= (host_entropy * self.data.temperature / self.data.entropy_units)
            else:
                raise Exception("ERROR! The host material is not the first defined compound in the entropy file."
                                f"\nFirst compound found: {host_entropy_key}")

        # get host elements and stoichiometries (incase a reference compound was used to calculate nu_volatile)
        # assume format of host is: M_{a}-V_{b}
        metal_host_element, volatile_element = self.data.host["elements"].keys()
        metal_host_stoic, volatile_host_stoic = self.data.host["elements"].values()

        # calculate chemical potential of metal in host
        nu_metal = (host_energy - (volatile_host_stoic * nu_volatile)) / metal_host_stoic

        chemical_potentials = {metal_host_element: nu_metal,
                               volatile_element: nu_volatile
                               }

        return chemical_potentials

    def calc_chemical_volatile_rich_poor(self):
        chemical_potentials = {}
        rich_poor_fraction_total = 0
        nu_volatile_std = 0

        # change energy of host if using gibbs function.
        # host should be first key
        if self.gibbs_energies:
            # assigns first key in dict as host_gibbs_key and leaves the rest in a list
            host_gibbs_key, *constituents_gibbs_keys = self.gibbs_energies.keys()

            if host_gibbs_key == "host" or host_gibbs_key == self.host["formula"]:
                host_energy = self.gibbs_energies[host_gibbs_key]
            else:
                raise Exception("ERROR! The host material is not the first defined compound in the gibbs energies file."
                                f"\nFirst compound found: {host_gibbs_key}")

        # loop through the constituent compounds that make up the host
        for compound in self.constituents["constituent_compounds"]:
            # unpack compound dictionary to get values
            compound_elements, compound_coefficient, compound_energy, compound_metal_energy, compound_std_form_eng, compound_rich_poor_frac = \
                self.constituents["constituent_compounds"][compound].values()

            # unpack elements to get element stoichiometries from compound provided
            # assume format of compound is: M_{a}-V_{b}
            metal_compound_stoic, volatile_compound_stoic = compound_elements.values()

            # accumulate the total rich-poor fraction
            rich_poor_fraction_total += compound_rich_poor_frac

            # calculate the contribution to the std volatile chem pot from this compound
            volatile_chem_pot_contribution = compound_rich_poor_frac * ((compound_energy - (
                    metal_compound_stoic * compound_metal_energy) - compound_std_form_eng) / volatile_compound_stoic)

            # accumulate std volatile chem pot
            nu_volatile_std += volatile_chem_pot_contribution

        # get final std volatile chem pot by dividing by total rich poor fraction
        nu_volatile_std /= rich_poor_fraction_total

        # calculate temperature contribution to the volatile chem pot
        temp_cont = 0.5 * calc_temperature_contribution(gas_species=self.constituents["volatile"]["volatile_element"],
                                                        temperature=self.temperature,
                                                        real_gas=self.real_gas
                                                        )

        # calculate pressure contribution to the volatile chem pot
        pressure_cont = calc_pressure_contribution(volatile_PP=self.constituents["volatile"]["log_pp"],
                                                   temperature=self.temperature
                                                   )

        # determine volatile chem pot at desired conditions
        nu_volatile = nu_volatile_std + temp_cont + pressure_cont

        # now loop over elements in the host and determine the final chemical potentials
        for host_element, element_stoichiometry in self.host["elements"].items():

            # only do non-volatile elements
            if host_element is not self.constituents["volatile"]["volatile_element"]:

                other_constituents_chem_pot_contribution = 0

                # loop over constituent compounds again to check if this element is in this compound
                for compound in self.constituents["constituent_compounds"]:
                    # unpack compound dictionary to get values
                    compound_elements, compound_coefficient, compound_energy, compound_metal_energy, compound_std_form_eng, compound_rich_poor_frac = \
                        self.constituents["constituent_compounds"][compound].values()

                    # change energies of constituents to gibbs energies at desired temperature if requested
                    if self.gibbs_energies:
                        compound_energy = self.gibbs_energies[compound]

                    if host_element not in compound_elements:
                        # sum up energies of constituents that DO NOT contain target host element
                        other_constituents_chem_pot_contribution += compound_coefficient * compound_energy
                    else:
                        # hold values for after the inner loop has finished
                        target_compound_energy = compound_energy
                        target_compound_coefficient = compound_coefficient
                        target_rich_poor_frac = compound_rich_poor_frac

                        # unpack elements to get element stoichiometries from compound provided
                        # assume format of compound is: M_{a}-V_{b}
                        target_metal_compound_stoic, target_volatile_compound_stoic = compound_elements.values()

                # calc chemical potential for element
                chemical_potential = target_rich_poor_frac * ((target_compound_energy - (
                            target_volatile_compound_stoic * nu_volatile)) / target_metal_compound_stoic) + (
                                                 1 - target_rich_poor_frac) * (((
                                                                                            host_energy - other_constituents_chem_pot_contribution - (
                                                                                                self.constituents[
                                                                                                    "volatile"][
                                                                                                    "coefficient"] * nu_volatile)) / target_compound_coefficient - (
                                                                                            target_volatile_compound_stoic * nu_volatile)) / target_metal_compound_stoic)

                # add to the dictionary
                chemical_potentials[host_element] = chemical_potential

            else:
                chemical_potentials[self.constituents["volatile"]["volatile_element"]] = nu_volatile

        return chemical_potentials, self.constituents["volatile"]["volatile_element"]

    def calc_secondary_phase_chem_pots(self):

        phase_chem_pots = {}

        for phase, phase_vals in self.data.secondary_phases.items():
            chem_pot_sum = 0

            for phase_element, element_stoic in phase_vals["elements"].items():
                chem_pot_sum += (self.chem_pots[phase_element] * element_stoic)

            if chem_pot_sum > phase_vals["energy"]:
                is_stable = True
            else:
                is_stable = False

            phase_chem_pots[phase] = {"energy": phase_vals["energy"],
                                      "chem_pot_sum": chem_pot_sum,
                                      "difference": chem_pot_sum - phase_vals["energy"],
                                      "is_stable": is_stable
                                      }

        self.phase_chem_pots = phase_chem_pots


    def add_dopants(self):

        # loop through dopants, add them to the chemical_potentials dict and determine if they need
        # chemical potentials fitting to a concentration
        for dopant, dopant_vals in self.data.dopants.items():

            reference = dopant_vals["reference"]
            ref_elements = dopant_vals["reference_elements"]
            dopant_chem_pot = dopant_vals["chemical_potential"]
            fitting_option = dopant_vals["fitting_option"]

            # ignore dopant
            if fitting_option == 0:
                for defect in list(self.data.defects_data.keys()):
                    if self.data.defects_data[defect]["added/removed"][dopant] != 0:
                        del self.data.defects_data[defect]

                    else:
                        del self.data.defects_data[defect]["added/removed"][dopant]

            # 1 = fixed concentration | 2 = loop over concentration
            # regardless, chemical potential need fitting
            elif fitting_option == 1 or fitting_option == 2:
                self.chem_pots[dopant] = dopant_chem_pot
                self.fitting_dopants.append(dopant)


            # 3 = fixed partial pressure | 4 - loop over partial pressure
            # no need to fit as chemical potential is calculated directly from partial pressure
            # only volatile elements available
            elif fitting_option == 3 or fitting_option == 4:

                valid_gas_dopants = ["H", "N", "O", "F", "Cl"]

                dopant_log_pp = dopant_vals["log_PP_atm"]

                # determine whether X or X2 has been provided as a dopant reference
                # e.g. Cl or Cl_2
                if reference in valid_gas_dopants:
                    volatile_element = reference
                    dopant_stoic = ref_elements[volatile_element]

                elif reference[:-2] in valid_gas_dopants:
                    volatile_element = reference[:-2]
                    dopant_stoic = ref_elements[volatile_element]

                else:
                    raise Exception(
                        f"\n<!> Error. Partial pressure dopant fitting option can only be used for diatomic gas molecules!\n")

                dopant_chem_pot /= dopant_stoic

                # calculate temperature and pressure contributions to the volatile dopant chemical potential
                temperature_contribution = 0.5 * calc_temperature_contribution(gas_species=volatile_element,
                                                                               temperature=self.data.temperature,
                                                                               real_gas=self.data.real_gas
                                                                               )

                pressure_contribution = calc_pressure_contribution(volatile_PP=dopant_log_pp,
                                                                   temperature=self.data.temperature
                                                                   )

                # calculate dopant chemical potential under desired conditions
                dopant_chem_pot += (temperature_contribution + pressure_contribution)

                self.chem_pots[dopant] = dopant_chem_pot

            elif fitting_option == 5:
                self.chem_pots[dopant] = dopant_chem_pot


        # for whether a dopant is being fitted or not
        if self.fitting_dopants == []:
            self.fitting_dopants = None

        elif len(self.fitting_dopants) == 1:
            self.fitting_dopants = self.fitting_dopants[0]


