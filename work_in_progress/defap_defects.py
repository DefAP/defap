import decimal
import math
from scipy import optimize, interpolate
from decimal import *
decimal.getcontext().prec=50
from defap_misc import take_closest
import numpy as np


class Defects:
    def __init__(self, data, chemical_potentials, entropy_vals):
        self.data = data
        self.chemical_potentials = chemical_potentials
        self.entropy_vals = entropy_vals

        self.formation_energies_vbm = {}
        self.formation_energies = {}
        self.concentrations = {}
        self.fermi_level = None
        self.defect_chg_sum = 0
        self.temp = 0

        self.boltzmann = 0.000086173324

    def calc_form_eng_vbm(self):

        # calculate formation energies at valence band maximum
        for defect_name, defect_vals in self.data.defects_data.items():

            # pull vals from defects_dict for each defect
            defect_group, multiplicity, site_num, defect_charge, defect_energy, tab_correction, atoms_added_removed = defect_vals.values()

            # add chemical potential contributions of each element in the defect to the formation energy
            chem_pot_contributions = 0
            for element in self.chemical_potentials:
                chem_pot_contributions += atoms_added_removed[element] * self.chemical_potentials[element]


            # calculate defect form eng at valence band maximum
            defect_form_eng_vbm = defect_energy - self.data.host_energy_supercell + (defect_charge * self.data.e_vbm) + chem_pot_contributions

            # simple cubic point charge
            if self.data.coulombic_correction == 1:
                defect_form_eng_vbm += 14.39942 * 2.8383 * defect_charge**2 / (2*self.data.length * self.data.dielectric_constant)

            # anisotropic charge correction
            elif self.data.coulombic_correction == 2:
                defect_form_eng_vbm += (14.39942 * ((defect_charge ** 2 * self.data.screened_madelung) / 2))

            # add tabulated corrections in defects_dict to the formation energies
            if self.data.tab_correction:
                defect_form_eng_vbm += tab_correction

             # add entropy contributions to defect formation energies if provided
            if self.entropy_vals:
                host_name = list(self.entropy_vals.keys())[0]

                dS = self.entropy_vals[defect_name] - self.entropy_vals[host_name]

                defect_form_eng_vbm -= (self.data.temperature * dS)

            # add defect name and vbm form energy to the dict
            self.formation_energies_vbm[defect_name] = defect_form_eng_vbm


        return self.formation_energies_vbm

    def calc_form_eng_at_fermi(self, fermi_level):

        # add fermi level contribution to the formation energy
        for defect in self.formation_energies_vbm:
            self.formation_energies[defect] = self.formation_energies_vbm[defect] + (fermi_level * self.data.defects_data[defect]["defect_charge"])

        return self.formation_energies

    def calc_charge_carriers_conc(self, method, limits, energies, states, valband, condband, fermi_level, e_or_p):

        # calculate concentration of electrons or holes depending on the method specified
        if method == "off":
            # set to min concentration
            conc = 10e-200

        elif method == "boltzmann":
            if e_or_p == -1:
                conc = valband * math.exp(-((fermi_level) / (self.data.temperature * self.boltzmann)))
            elif e_or_p == 1:
                conc = condband * math.exp(-((self.data.bandgap - fermi_level) / (self.data.temperature * self.boltzmann)))
            else:
                raise ValueError("Invalid e_or_p value ???")

        elif method == "fermi-dirac":
            # divide DOS by number of formula units used to calculate DOS
            states_per_fu = states / self.data.fu_unit_cell

            # determine the VBM and CBM energy in the DOS file closest to the values in the limits provided
            closest_band_min = take_closest(energies, float(limits[0]))
            closest_band_max = take_closest(energies, float(limits[1]))

            # get the list index of the found VBM and CBM energies
            band_min_index = np.where(energies == closest_band_min)[0][0]
            band_max_index = np.where(energies == closest_band_max)[0][0]

            # numerically integrate the DOS between the limits provided to calculate electron/hole concentrations
            # e_or_p = 1 -> electrons | e_or_p = -1 -> holes
            if e_or_p == 1:
                conc = np.trapezoid(self.electron_intergrand(energies, states_per_fu,band_positions=[band_min_index, band_max_index],fermi_level=fermi_level),
                                energies[band_min_index:band_max_index],
                                dx=energies[1] - energies[0]
                                )
            else:
                conc = np.trapezoid(self.hole_intergrand(energies, states_per_fu, band_positions=[band_min_index, band_max_index],fermi_level=fermi_level),
                                energies[band_min_index:band_max_index],
                                dx=energies[1] - energies[0]
                                )

        elif method == "fixed":
            pass

        elif method == "effective_masses":
            vol_cell_cm3 = self.data.volume_unit_cell * 1e-30
            if e_or_p == 1:
                N_c = self.effective_masses(temperature=self.data.temperature, effect_mass=self.data.electron_effective_masses)
                conc = ((N_c * vol_cell_cm3) / self.data.fu_unit_cell) * math.exp(-((self.data.bandgap - fermi_level) / (self.data.temperature * self.boltzmann)))
            else:
                N_v = self.effective_masses(temperature=self.data.temperature, effect_mass=self.data.hole_effective_masses)
                conc = ((N_v * vol_cell_cm3) / self.data.fu_unit_cell) * math.exp(-((fermi_level) / (self.data.temperature * self.boltzmann)))

        return conc

    def electron_intergrand(self, energies, states, band_positions, fermi_level):
        # function to be integrated to calculate electron concentrations
        return states[band_positions[0]:band_positions[1]] * (1 / (1 + np.exp(
            (energies[band_positions[0]:band_positions[1]] - fermi_level) / (self.boltzmann * self.data.temperature))))

    def hole_intergrand(self, energies, states, band_positions, fermi_level):
        # function to be integrated to calculate hole concentrations
        return states[band_positions[0]:band_positions[1]] * (1 / (1 + np.exp(
            (fermi_level - energies[band_positions[0]:band_positions[1]]) / (self.boltzmann * self.data.temperature))))

    def effective_masses(self, temperature, effect_mass,):

        boltzmann_SI = 1.380649E-23
        planck_SI = 6.62607015E-34

        em = np.array(effect_mass)

        x, y = em[:,0], em[:,1]

        tck = interpolate.splrep(x, y)

        mass_eff = interpolate.splev(temperature, tck)

        return (2 * ((2* math.pi * mass_eff * 9.11e-31 * boltzmann_SI * temperature) / (planck_SI**2)) ** (3/2))


    def calc_defect_concentrations(self):

        # calculate defect concentrations and determine total ionic chg contribution of these defects
        concentrations = {}
        defect_chg_sum = 0

        for defect, defect_vals in self.data.defects_data.items():

            defect_multiplicity = defect_vals["multiplicity"]
            defect_charge = defect_vals["defect_charge"]

            if  self.data.defect_conc_method == "boltzmann":
                #print(defect, self.formation_energies[defect])
                concentration = defect_multiplicity * math.exp(-self.formation_energies[defect] / (self.boltzmann * self.data.temperature))

            elif self.data.defect_conc_method == "kasamatsu":
                pass

            # cant remember why we do this???
            # think python struggles with logs and precision when smaller than this
            if concentration < 1e-200:
                concentration = -200
            else:
                concentration = math.log10(concentration)

            # determine chg contribution of defects
            defect_chg_sum += (defect_charge * 10 ** concentration)

            concentrations[defect] = concentration

        return concentrations, defect_chg_sum

    def calc_concentrations(self, fermi_level):

        # calculate concentration of holes
        hole_conc = self.calc_charge_carriers_conc(method=self.data.hole_method,
                                                   limits=self.data.valenceband_limits,
                                                   energies=self.data.dos_data["energies"],
                                                   states=self.data.dos_data["states"],
                                                   valband=self.data.valenceband,
                                                   condband=self.data.conductionband,
                                                   fermi_level=fermi_level,
                                                   e_or_p=-1
                                                   )


        # concentration of electrons
        elec_conc = self.calc_charge_carriers_conc(method=self.data.electron_method,
                                                   limits=self.data.conductionband_limits,
                                                   energies=self.data.dos_data["energies"],
                                                   states=self.data.dos_data["states"],
                                                   valband=self.data.valenceband,
                                                   condband=self.data.conductionband,
                                                   fermi_level=fermi_level,
                                                   e_or_p=1
                                                   )


        # calculate formation energies at current fermi level
        self.calc_form_eng_at_fermi(fermi_level=fermi_level)

        # then calculate defect concentrations at this fermi level
        self.concentrations, self.defect_chg_sum = self.calc_defect_concentrations()

        # determine total charge
        # used to optimise fermi level
        total_charge = self.defect_chg_sum - elec_conc + hole_conc + (self.data.art_dopant_conc * self.data.art_dopant_chg)

        # add electron/holes to concentrations dict
        self.concentrations["electrons"] = math.log10(elec_conc)
        self.concentrations["holes"] = math.log10(hole_conc)

        return total_charge

    def optimise_fermi_level(self):

        # optimise the position of the fermi level by finding the point where total charge = 0
        # uses brentq scipy formula
        try:
            self.fermi_level = optimize.brentq(f=self.calc_concentrations, a=0, b=self.data.bandgap, maxiter=500)
        except ValueError:
            raise ValueError("\n<!> ERROR. Unable to determine the Fermi Level.\n"
                             "Your initial conditions likely result in the root Fermi level for charge neutrality to be"
                             f" outside the bandgap."
                             )

        return self.concentrations

    def calc_single_dopant_concentration(self, dopant_chemical_potential, dopant):

        # calculate the concentration of a single dopant at the provided chemical potential
        self.chemical_potentials[dopant] = dopant_chemical_potential
        target_dopant_conc = self.data.dopants[dopant]["concentration_pfu"]

        # calculate formation energies at vbm
        self.calc_form_eng_vbm()

        # now optimise fermi level with this dopant chemical potential
        # this also calculates defect concentrations
        self.optimise_fermi_level()


        # calculate total dopant concentration
        dopant_conc_sum = 0
        for defect, defect_vals in self.data.defects_data.items():
            # -1 in defects dict indicates that an element is added into the lattice
            if defect_vals["added/removed"][dopant] < 0:
                dopant_conc_sum += 10 ** (self.concentrations[defect])

        # calculate difference in calculated and desired dopant concentration
        target_conc_diff = dopant_conc_sum - target_dopant_conc


        return target_conc_diff

    def optimise_single_dopant(self, fitting_dopant):

        # optimise the chemical potential of a single dopant by fitting to a desired concentration

        # initial guess chemical potential of dopant
        dopant_chem_pot = self.data.dopants[fitting_dopant]["chemical_potential"]

        # +/- guessed bounds of chemical potential
        dopant_chem_pot_range = self.data.dopants[fitting_dopant]["chem_pot_range"]


        chem_pots_reattempts = np.arange(-50, 10, 1)
        attempt_num = 0
        while attempt_num < len(chem_pots_reattempts):

        # optimise dopant chemical potential by fitting to desired dopant chemical potential
            try:
                dopant_chem_pot = optimize.brentq(self.calc_single_dopant_concentration, dopant_chem_pot - dopant_chem_pot_range, dopant_chem_pot + dopant_chem_pot_range, args=(fitting_dopant))
                break
            except:

                dopant_chem_pot = chem_pots_reattempts[attempt_num]
                dopant_chem_pot_range = 1

                attempt_num += 1

        else:
            raise ValueError("\n<!> ERROR. Unable to determine dopant chemical potential.\n"
                             f"Your target dopant concentration may be unobtainable for the chemical potential range: {dopant_chem_pot - dopant_chem_pot_range} - {dopant_chem_pot + dopant_chem_pot_range} eV")

        self.chemical_potentials[fitting_dopant] = dopant_chem_pot


    def calc_multi_dopant_conc(self, dopant_chemical_potentials, fitting_dopants):

        # calculate the concentration of multiple fitted dopants

        # assign chemical potentials to the dopants into the chemical potentials dict
        target_dopant_conc = []
        for index, dopant_element in enumerate(fitting_dopants):
            self.chemical_potentials[dopant_element] = dopant_chemical_potentials[index]
            target_dopant_conc.append(self.data.dopants[dopant_element]["concentration_pfu"])

        # calculate formation energies at vbm
        self.calc_form_eng_vbm()

        # now optimise fermi level with these dopant chemical potentials
        # this also calculates defect concentrations
        self.optimise_fermi_level()

        # determine difference in calculated and desired dopant concentrations
        dopant_conc_diffs = []
        for index, dopant_element in enumerate(fitting_dopants):
            dopant_conc_sum = 0
            for defect, defect_vals in self.data.defects_data.items():
                if defect_vals["added/removed"][dopant_element] < 0:
                    dopant_conc_sum += 10 ** (self.concentrations[defect])

            # calculate absolute log difference in targe and desired concentration and add to list
            dopant_conc_diffs.append(np.abs(np.log(target_dopant_conc[index]) - np.log(dopant_conc_sum)))

        # convert list to numpy array
        target_dopant_concs_diff = np.array(dopant_conc_diffs)

        # calculate sum of the differences
        # this will be minimised using optimiser
        return np.sum(target_dopant_concs_diff)


    def optimise_multi_dopant(self):

        # optimise chemical potentials of multiple dopants

        # initial guesses
        x0 = []

        # chemical potential bounds of each dopant
        bnds = []

        # list of dopants to be fitted
        dopant_elements = []

        # loop through dopants, determine dopants to be fitted
        for dopant, dopant_vals in self.data.dopants.items():
            if dopant_vals["fitting_option"] == 1 or dopant_vals["fitting_option"] == 2:
                # add initial guess of dopant chemical potential to list
                x0.append(dopant_vals["chemical_potential"])

                # +/- guessed bounds of chemical potential
                element_bnds = (dopant_vals["chemical_potential"] - self.data.dopants[dopant]["chem_pot_range"],
                                dopant_vals["chemical_potential"] + self.data.dopants[dopant]["chem_pot_range"]
                                )

                bnds.append(element_bnds)
                dopant_elements.append(dopant)

        # the function itself is a constraint as it needs to sum of differences in target and calculated concentrations needs to be zero
        constraints = {"type": "eq", "fun": self.calc_multi_dopant_conc, "args": [dopant_elements]}

        attempt_num = 1
        while attempt_num < 100:

            # optimise chemical potentials using SLSQP
            sol = optimize.minimize(self.calc_multi_dopant_conc, np.array(x0), method="SLSQP", bounds=tuple(bnds),
                                    args=dopant_elements, constraints=(constraints), options={"disp":False})

            # check that individual dopants meet the target concentration
            # sometimes slsqp minimiser doesn't quite meet the criteria

            dopant_log_diffs = []
            for index, dopant in enumerate(dopant_elements):

                # calculate total dopant concetration
                dopant_conc_sum = 0
                for defect, defect_vals in self.data.defects_data.items():
                    if defect_vals["added/removed"][dopant] < 0:
                        dopant_conc_sum += 10 ** (self.concentrations[defect])

                # calc log difference between calculated and target concentration
                log_diff = np.log10(self.data.dopants[dopant]["concentration_pfu"]) - np.log10(dopant_conc_sum)

                # shift chemical potential bounds depending on if difference +ve or -ve
                if log_diff > 0:
                    bnds[index] = (self.chemical_potentials[dopant], bnds[index][1])
                else:
                    bnds[index] = (bnds[index][0], self.chemical_potentials[dopant])

                # re adjust initial guessed chem pot
                x0[index] = np.average(bnds[index])

                # add log difference between target and calculated dopant concentration to list
                dopant_log_diffs.append(abs(log_diff))

            # break loop if all dopant concentrations satisfactory
            if (np.array(dopant_log_diffs) < 5e-3).all():
                break

            attempt_num+=1

        else:
            raise ValueError(f"\n<!> ERROR. Unable to determine dopant chemical potential.\n{self.chemical_potentials}\n{dopant_log_diffs}")


    def calc_stoichiometry(self):

        # initial values for the volatile numerator / metal denominator
        # use decimal module to get increase precision for very small +-x
        volatile_numerator = Decimal(0)
        metal_denominator = Decimal(0)
        denominator_stoic_sum = Decimal(0)
        volatile_element = self.data.constituents["volatile"]["volatile_element"]

        # get perfect stoichiometric ratios of volatile and metal
        for host_element, element_stoic in self.data.host["elements"].items():
            if host_element == volatile_element:
                volatile_stoic = Decimal(element_stoic)
                volatile_numerator += volatile_stoic

            else:
                metal_denominator += Decimal(element_stoic)
                denominator_stoic_sum += Decimal(element_stoic)

        # loop through defects
        for defect_name, defect_vals in self.data.defects_data.items():

            # get concentration and which elements are added/removed in this defect
            defect_concentration = Decimal(self.concentrations[defect_name])
            defect_elements_added_removed = defect_vals["added/removed"]

            # +- stoic for numerator if defect is a volatile is added/removed
            for element, val in defect_elements_added_removed.items():
                if element == volatile_element:
                    #volatile_numerator += np.longdouble(-val * np.power(10, defect_concentration))
                    volatile_numerator += Decimal(-val * np.power(10, defect_concentration))

                # if not volatile and element is in the host, then consider it a metal
                elif element in self.data.host["elements"] and element is not volatile_element:
                    #metal_denominator += np.longdouble(-val * np.power(10, defect_concentration))
                    metal_denominator += Decimal(-val * np.power(10, defect_concentration))

                # for dopants to be consider metal elements
                elif self.data.stoichiometry == 2:
                    #metal_denominator += np.longdouble(-val * np.power(10, defect_concentration))
                    metal_denominator += Decimal(-val * np.power(10, defect_concentration))

        # calculate stoichiometry and log10(stoich) (for plotting)
        stoichiometry = ((volatile_numerator / (metal_denominator / denominator_stoic_sum)) - volatile_stoic)
        log_stoic = np.log10(np.abs(stoichiometry))

        return log_stoic, stoichiometry

