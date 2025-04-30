from thermodynamics import calc_pressure_contribution
from thermodynamics import calc_temperature_contribution


class ChemicalPotentials:
    """
    Class for calculating chemical potentials
    """

    def __init__(self, data, entropy_vals, gibbs_vals):

        self.data = data
        self.data_copy = None
        self.entropy_vals = entropy_vals
        self.gibbs_vals = gibbs_vals

        # ensures chemical potentials are returned in defap_tasks when chemical_potentials.chem_pots is called
        self.chem_pots = self.call_chemical_pot_method()

        self.phase_chem_pots = None

        self.fitting_dopants = []
        self.looping_dopants = []

    def call_chemical_pot_method(self):

        # auto call chemical potential function depending on method specified
        if self.data.chem_pot_method == "defined":
            return self.calc_chemical_defined()

        elif self.data.chem_pot_method == "rich-poor":
            return self.calc_chemical_rich_poor()

        elif self.data.chem_pot_method == "volatile":
            return self.calc_chemical_volatile()

        elif self.data.chem_pot_method == "volatile-rich-poor":
            return self.calc_chemical_volatile_rich_poor()

        else:
            raise ValueError("\nERROR. Invalid chemical potential method!\n")


    def calc_chemical_defined(self):

        chemical_potentials = {}
        total_potential = 0

        # loop through host elements
        for host_element, stoic in self.data.host["elements"].items():

            for constituent_element in self.data.constituents["constituent_compounds"].keys():

                # assign chemical potential of this host element to what has been defined in the consitutuents tab
                if host_element == constituent_element:
                    chem_pot = self.data.constituents["constituent_compounds"][constituent_element]["dft_energy_pfu"]
                    chemical_potentials[host_element] = chem_pot

                    # sum total chemical potential
                    total_potential += stoic * chem_pot

        # check that host energy is not larger then chemical potential sum
        if abs(self.data.host_energy_pfu - total_potential) > 0.001:
            raise ValueError(
                f"<!> Error : The chemical potentials for the constituents do not add up to that for the host system\n"
                f"Host: {self.data.host_energy_pfu}\n"
                f"Sum of constituents: {total_potential}")

        return chemical_potentials


    def calc_chemical_rich_poor(self):
        chemical_potentials = {}

        # energy of the host from input file
        host_energy = self.data.host_energy_pfu

        # loop through elements in the host and get the respective stoichiometries
        for host_element, stoic in self.data.host["elements"].items():

            # total up rich chemical potentials of other constituents in the host
            other_chem_pot_total = 0
            # loop through the constituents
            for constituent_element, constituent_vals in self.data.constituents["constituent_compounds"].items():

                # get the rich chemical potential and rich poor fraction of the current constituent element in the host
                if host_element == constituent_element:
                    chem_pot_rich = constituent_vals["dft_energy_pfu"]
                    rp_fraction = constituent_vals["fraction"]

                else:
                    # sum the rich chemical potentials of the other elements in the host to determine the poor chemical potential limit of the current element
                    other_chem_pot_rich = constituent_vals["dft_energy_pfu"]
                    other_chem_pot_total += other_chem_pot_rich * self.data.host["elements"][constituent_element]

            # calculate poor chemical potential limit of current element
            chem_pot_poor = (host_energy - other_chem_pot_total) / stoic

            # determine final chemical potential by summing the fractions of the rich and poor chemical potentials
            final_chem_pot = rp_fraction*chem_pot_rich + (1 - rp_fraction)*chem_pot_poor

            chemical_potentials[host_element] = final_chem_pot

        return chemical_potentials

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
        nu_volatile_std = (compound_energy - (metal_compound_stoic * compound_metal_energy) - compound_std_form_eng) / volatile_compound_stoic

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

        # now calculate chemical potential of metal species in host

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

        # energy of the host from input file
        host_energy = self.data.host_energy_pfu

        # change energy of host if using gibbs function.
        # host should be first key
        if self.gibbs_vals:
            # assigns first key in dict as host_gibbs_key and leaves the rest in a list
            host_gibbs_key, *constituents_gibbs_keys = self.gibbs_vals.keys()

            if host_gibbs_key == "host" or host_gibbs_key == self.data.host["formula"]:
                host_energy = self.gibbs_vals[host_gibbs_key]
            else:
                raise Exception("ERROR! The host material is not the first defined compound in the gibbs energies file."
                                f"\nFirst compound found: {host_gibbs_key}")

        # loop through the constituent compounds that make up the host
        for compound in self.data.constituents["constituent_compounds"]:
            # unpack compound dictionary to get values
            compound_elements, compound_coefficient, compound_energy, compound_metal_energy, compound_std_form_eng, compound_rich_poor_frac = \
                self.data.constituents["constituent_compounds"][compound].values()

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
        temp_cont = 0.5 * calc_temperature_contribution(gas_species=self.data.constituents["volatile"]["volatile_element"],
                                                        temperature=self.data.temperature,
                                                        real_gas=self.data.real_gas
                                                        )

        # calculate pressure contribution to the volatile chem pot
        pressure_cont = calc_pressure_contribution(volatile_PP=self.data.constituents["volatile"]["log_pp"],
                                                   temperature=self.data.temperature
                                                   )

        # determine volatile chem pot at desired conditions
        nu_volatile = nu_volatile_std + temp_cont + pressure_cont

        # now loop over elements in the host and determine the final chemical potentials
        for host_element, element_stoichiometry in self.data.host["elements"].items():

            # only do non-volatile elements
            if host_element is not self.data.constituents["volatile"]["volatile_element"]:

                other_constituents_chem_pot_contribution = 0

                # loop over constituent compounds again to check if this element is in this compound
                for compound in self.data.constituents["constituent_compounds"]:
                    # unpack compound dictionary to get values
                    compound_elements, compound_coefficient, compound_energy, compound_metal_energy, compound_std_form_eng, compound_rich_poor_frac = \
                        self.data.constituents["constituent_compounds"][compound].values()

                    # change energies of constituents to gibbs energies at desired temperature if requested
                    if self.gibbs_vals:
                        compound_energy = self.gibbs_vals[compound]

                    if host_element not in compound_elements:
                        # sum up "rich" energies of constituents that DO NOT contain target host element
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
                chemical_potential = target_rich_poor_frac * ((target_compound_energy - (target_volatile_compound_stoic * nu_volatile)) / target_metal_compound_stoic) + (1 - target_rich_poor_frac) * (((host_energy - other_constituents_chem_pot_contribution - (self.data.constituents["volatile"]["coefficient"] * nu_volatile)) / target_compound_coefficient - (target_volatile_compound_stoic * nu_volatile)) / target_metal_compound_stoic)

                # add to the dictionary
                chemical_potentials[host_element] = chemical_potential

            else:
                chemical_potentials[self.data.constituents["volatile"]["volatile_element"]] = nu_volatile

        return chemical_potentials

    def calc_secondary_phase_chem_pots(self):

        phase_chem_pots = {}

        # loop through secondary phases provided
        for phase, phase_vals in self.data.secondary_phases.items():
            chem_pot_sum = 0

            # sum total chemical potentials of the consituents in the secondary phase
            for phase_element, element_stoic in phase_vals["elements"].items():
                chem_pot_sum += (self.chem_pots[phase_element] * element_stoic)


            # add temperature contribution to gaseous secondary phases using pyromat if requested
            if phase_vals["add_temperature_contribution"]:
                temp_cont = calc_temperature_contribution(
                    gas_species=phase,
                    temperature=self.data.temperature,
                    real_gas=3
                )

                phase_eng = phase_vals["energy"] + temp_cont

            elif phase_vals["use_gibbs_energy"]:
                phase_eng = self.gibbs_vals[phase]

            else:
                phase_eng = phase_vals["energy"]

            # secondary phase "stable" if sum of chemical potentials exceeds the chemical potential of the secondary phase
            if chem_pot_sum > phase_eng:
                is_stable = True
            else:
                is_stable = False

            phase_chem_pots[phase] = {"energy": phase_eng,
                                      "chem_pot_sum": chem_pot_sum,
                                      "difference": chem_pot_sum - phase_eng,
                                      "is_stable": is_stable
                                      }

        self.phase_chem_pots = phase_chem_pots

    def add_dopants(self):

        # loop through dopants, add them to the chemical_potentials dict and determine if they need
        # chemical potentials fitting to a concentration

        delete_dopants = None

        for dopant, dopant_vals in self.data.dopants.items():

            reference = dopant_vals["reference"]
            ref_elements = dopant_vals["reference_elements"]
            dopant_chem_pot = dopant_vals["chemical_potential"]
            fitting_option = dopant_vals["fitting_option"]

            # delete dopant
            if fitting_option == 0:

                if delete_dopants:
                    delete_dopants.append(dopant)
                else:
                    delete_dopants = [dopant]

                for defect in list(self.data.defects_data.keys()):

                    if self.data.defects_data[defect]["added/removed"][dopant] != 0:
                        del self.data.defects_data[defect]

                    else:
                        del self.data.defects_data[defect]["added/removed"][dopant]

            # 1 = fixed concentration | 2 = loop over concentration
            # regardless, chemical potential needs fitting
            elif fitting_option == 1 or fitting_option == 2:
                self.chem_pots[dopant] = dopant_chem_pot
                self.fitting_dopants.append(dopant)


            # 3 = fixed partial pressure | 4 - loop over partial pressure
            # no need to fit as chemical potential is calculated directly from partial pressure
            # only volatile elements available
            elif fitting_option == 3 or fitting_option == 4:

                valid_gas_dopants = ["H", "D", "T", "N", "O", "F", "Cl"]

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

            # dopant chem pot directly specified
            elif fitting_option == 5:
                self.chem_pots[dopant] = dopant_chem_pot

        # remove dopants from dopant dict if not being used
        # prevent errors on next loop step
        if delete_dopants:
            for dopant in delete_dopants:
                del self.data.dopants[dopant]

        # for whether a dopant is being fitted or not
        if self.fitting_dopants == []:
            self.fitting_dopants = None

        elif len(self.fitting_dopants) == 1:
            self.fitting_dopants = self.fitting_dopants[0]


