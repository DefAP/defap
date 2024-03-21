import os
import csv
import numpy as np
import defap_misc


class ReadInputData:

    def __init__(self, seedname: str):

        self.seedname = seedname

        # list of tasks
        self.tasks = None

        # loop properties for brouwer/defect phase task
        self.loop = None
        self.min_value = None
        self.max_value = None
        self.iterator = None

        # second loop variable for defect phase task
        self.loop_y = None
        self.min_value_y = None
        self.max_value_y = None
        self.iterator_y = None

        # plot as function of loop or stoichiometry | 1 = loop, 2 = stoich
        self.x_variable = 1

        # method to calculate stoichiometry | 0 = dont calc, 1 = consider defects leaving due to dopant
        #                                     2 = dopants considered as a metal species
        self.stoichiometry = 0

        # temperature of system
        self.temperature = 1000

        # properties of host species
        self.host = None
        self.host_energy_pfu = None
        self.host_energy_supercell = None

        # properties of bandstructure of system
        self.e_vbm = 0
        self.bandgap = 0
        self.charged_system = False

        # method to calculate chemical potentials
        self.chem_pot_method = None

        # constituents of host system
        self.constituents = None

        # dopants incorporated
        self.dopants = None
        self.dopants_to_fit = 0

        # any artificial dopant charges added and its concentration
        self.art_dopant_conc = 0
        self.art_dopant_chg = 0

        # dict for secondary phases
        self.secondary_phases = None

        # For calculating volatile chemical potentials | 0 = ideal gas, 1 = real gas (Shomate),
        #                                              | 2 = Johnson et al (Discontinued), 3 = pyromat library
        self.real_gas = 0

        # method for calculating electron/hole concentrations | boltzmann, fermi-dirac
        self.electron_method = "off"
        self.hole_method = "off"
        self.conductionband = None
        self.valenceband = None
        self.conductionband_limits = None
        self.valenceband_limits = None
        self.electron_fixed_conc = None
        self.hole_fixed_conc = None
        self.electron_effective_masses = None
        self.hole_effective_masses = None

        # functional units of unit cell
        self.fu_unit_cell = None

        # volume and length of cell
        self.volume_unit_cell = None
        self.length = None

        # method for calculting defect concentrations | boltzmann or kasamatsu
        self.defect_conc_method = "boltzmann"

        # concentration units | 0 = per formula unit, 1 = cm-3
        self.conc_units = 0

        # limits of y-axis on brouwer plot
        self.y_axis_min = -20
        self.y_axis_max = 0

        # convergence of multi-dopant fitting
        self.tolerance = 1e-10
        self.max_iteration = 100
        self.slsqp_dial = 10

        # convergence critera for total charge and dopant concentration
        self.dopant_convergence = 1e-3
        self.charge_convergence = 1e-10

        # brouwer colour scheme | 0 = default, 1 = user defined
        self.colour_scheme = 0

        # include entropy contributions to formation energies
        self.entropy = 0
        self.entropy_units = None

        # use gibbs temperature dependant energies for calculating chemical potentials
        self.include_gibbs = 0

        # whether to add charge corrections | 0 = dont add, 1 = cubic point charge, 2 = anisotropic point charge
        self.coulombic_correction = 0

        # add tabulated corrections in defects file
        self.tab_correction = False

        # values for calculating madelung constant for anisotropic point charge
        self.dielectric_constant = None
        self.screened_madelung = None
        self.dielectric_array = None
        self.lattice_array = None
        self.cutoff = 30
        self.gamma = 0.5

        # external file input data
        self.input_data = None
        self.defects_data = None
        self.dos_data = None
        self.entropy_data = None
        self.gibbs_data = None


    def read_input_file(self):

        """
        Function that reads .input file and appends variables to a dictionary
        :param seed:
        :return: self.input_data
        """

        # dictionary of variables with default values
        # only here so can be easily looped through to view all input variables
        self.input_data = {
            "tasks": None,
            "loop": None, "min_value": None, "max_value": None, "iterator": None,
            "loop_y": None, "min_value_y": None, "max_value_y": None, "iterator_y": None,
            "x_variable": 1,  # plot as function of loop
            "temperature": 1000,  # kelvin
            "host": None, "host_energy_pfu": None, "host_energy_supercell": None,
            "e_vbm": 0, "bandgap": 0,
            "chem_pot_method": None,
            "constituents": None,
            "dopants": None,
            "dopants_to_fit": 0,
            "art_dopant_conc": None, "art_dopant_chg": None,
            "secondary_phases": None,
            "real_gas": 0,  # ideal gas relations
            "electron_method": "off", "hole_method": "off",
            "conductionband": None, "valenceband": None,
            "conduction_band_limits": None, "valence_band_limits": None,
            "fu_unit_cell": None,
            "volume_unit_cell": None,
            "stoichiometry": 0,  # dont calculate
            "defect_conc_method": "boltzmann",
            "tab_correction": False,  # dont add
            "concentration_units": 0,  # per formula unit
            "y_axis_min": -15, "y_axis_max": 0,
            "tolerance": 1e-10, "max_iteration": 100, "SLSQP_dial": 10,  # for multi dopants
            "dopant_convergence": 1e-3,
            "charge_convergence": 1e-10,
            "colour_scheme": 0,  # defap default colours
            "entropy": 0, "entropy_units": None,
            "include_gibbs": 0,  # dont include
            "coulombic_correction": 0,  # no correction added
            "dielectric_constant": None,
            "screened_madelung": None,
            "length": None,
            "dielectric_array": None,
            "lattice_array": None,
            "cutoff": 30, "gamma": 0.5,  # for calculatinf madelung constant
        }

        input_file = f"{self.seedname}.input"

        if not os.path.isfile(input_file):
            raise Exception("ERROR<!> Input file not detected.\n"
                            f"Please provide a [name.input] file.")

        print(f">>> Reading in parameters from {input_file}\n")

        with open(input_file, "r") as f:
            # strip and split *.input file lines and append into a list. Skips empty lines
            lines = [s.strip().split() for s in f.readlines() if s.strip().split() != []]

            # loop through lines in .input file and add variables to dictionary. Certain cases require
            # additional processing lines to be added to the dict
            for line_index, line in enumerate(lines):

                variable = line[0].lower()

                match variable:

                    case "tasks":
                        self.input_data[variable] = [s.lower() for s in line[2:]]
                        self.tasks = [s.lower() for s in line[2:]]

                    case "loop":
                        self.input_data[variable] = int(line[2])
                        self.loop = int(line[2])

                    case "min_value":
                        self.input_data[variable] = float(line[2])
                        self.min_value = float(line[2])

                    case "max_value":
                        self.input_data[variable] = float(line[2])
                        self.max_value = float(line[2])

                    case "iterator":
                        self.input_data[variable] = float(line[2])
                        self.iterator = float(line[2])

                    case "loop_y":
                        self.input_data[variable] = float(line[2])
                        self.loop_y = float(line[2])

                    case "min_value_y":
                        self.input_data[variable] = float(line[2])
                        self.min_value_y = float(line[2])

                    case "max_value_y":
                        self.input_data[variable] = float(line[2])
                        self.max_value_y = float(line[2])

                    case "iterator_y":
                        self.input_data[variable] = float(line[2])
                        self.iterator_y = float(line[2])

                    case "x_variable":
                        self.input_data[variable] = int(line[2])
                        self.x_variable = int(line[2])

                    case "temperature":
                        self.input_data[variable] = float(line[2])
                        self.temperature = float(line[2])

                    case "host":
                        host_breakdown = defap_misc.break_formula(line[2])
                        self.input_data["host"] = {"formula": line[2], "elements": host_breakdown}
                        self.host = {"formula": line[2], "elements": host_breakdown}

                    case "host_energy_pfu":
                        self.input_data["host_energy_pfu"] = float(line[2])
                        self.host_energy_pfu = float(line[2])

                    case "host_energy_supercell":
                        self.input_data["host_energy_supercell"] = float(line[2])
                        self.host_energy_supercell = float(line[2])

                    case "e_vbm":
                        self.input_data[variable] = float(line[2])
                        self.e_vbm = float(line[2])

                    case "bandgap":
                        self.input_data[variable] = float(line[2])
                        self.bandgap = float(line[2])

                    case "chem_pot_method":
                        self.input_data[variable] = line[2]
                        self.chem_pot_method = line[2]

                    case "art_dopant_conc":
                        self.input_data[variable] = float(line[2])
                        self.art_dopant_conc = float(line[2])

                    case "art_dopant_chg":
                        self.input_data[variable] = int(line[2])
                        self.art_dopant_chg = int(line[2])

                    case "secondary_phases":
                        phases_lst = []
                        for i in range(int(line[2])):
                            phases_lst.append(lines[line_index + i + 1])

                        self.input_data["secondary_phases"] = phases_lst
                        self.secondary_phases = phases_lst

                    case "real_gas":
                        self.input_data[variable] = int(line[2])
                        self.real_gas = int(line[2])

                    case "electron_method":
                        self.input_data[variable] = line[2].lower()
                        self.electron_method = line[2].lower()

                    case "hole_method":
                        self.input_data[variable] = line[2].lower()
                        self.hole_method = line[2].lower()

                    case "conductionband":
                        self.input_data[variable] = float(line[2])
                        self.conductionband = float(line[2])

                    case "valenceband":
                        self.input_data[variable] = float(line[2])
                        self.valenceband = float(line[2])

                    case "fu_unit_cell":
                        self.input_data[variable] = int(line[2])
                        self.fu_unit_cell = int(line[2])

                    case "volume_unit_cell":
                        self.input_data[variable] = float(line[2])
                        self.volume_unit_cell = float(line[2])

                    case "stoichiometry":
                        self.input_data[variable] = int(line[2])
                        self.stoichiometry = int(line[2])

                    case "defect_conc_method":
                        self.input_data[variable] = line[2].lower()
                        self.defect_conc_method = line[2].lower()

                    case "concentration_units":
                        self.input_data[variable] = int(line[2])
                        self.conc_units = int(line[2])

                    case "y_axis_min":
                        self.input_data[variable] = float(line[2])
                        self.y_axis_min = float(line[2])

                    case "y_axis_max":
                        self.input_data[variable] = float(line[2])
                        self.y_axis_max = float(line[2])

                    case "tolerance":
                        self.input_data[variable] = float(line[2])
                        self.tolerance = float(line[2])

                    case "max_iteration":
                        self.input_data[variable] = float(line[2])
                        self.max_iteration = float(line[2])

                    case "slsqp_dial":
                        self.input_data[variable] = float(line[2])
                        self.slsqp_dial = float(line[2])

                    case "dopant_fit_convergence":
                        self.input_data[variable] = float(line[2])
                        self.dopant_convergence = float(line[2])

                    case "charge_convergence":
                        self.input_data[variable] = float(line[2])
                        self.charge_convergence = float(line[2])

                    case "colour_scheme":
                        self.input_data[variable] = int(line[2])
                        self.colour_scheme = int(line[2])

                    case "entropy":
                        self.input_data[variable] = int(line[2])
                        self.entropy = int(line[2])

                    case "entropy_units":
                        self.input_data[variable] = int(line[2])
                        self.entropy_units = int(line[2])

                    case "gibbs":
                        self.input_data["include_gibbs"] = int(line[2])
                        self.include_gibbs = int(line[2])

                    case "coulombic_correction":
                        self.input_data[variable] = int(line[2])
                        self.coulombic_correction = int(line[2])

                    case "tab_correction":
                        if line[2] == "1" or line[1].lower() == "t" or line[1].lower() == "true":
                            self.tab_correction = True
                            self.input_data[variable] = True
                        else:
                            self.tab_correction = False
                            self.input_data[variable] = False

                    case "dielectric_constant":
                        self.input_data[variable] = float(line[2])
                        self.dielectric_constant = float(line[2])

                    case "screened_madelung":
                        self.input_data[variable] = float(line[2])
                        self.screened_madelung = float(line[2])

                    case "length":
                        self.input_data[variable] = float(line[2])
                        self.length = float(line[2])

                    case "cutoff":
                        self.input_data[variable] = float(line[2])
                        self.cutoff = float(line[2])

                    case "gamma":
                        self.input_data[variable] = float(line[2])
                        self.gamma = float(line[2])

                    case "constituents":
                        constituents_lst = []
                        for i in range(int(line[2])):
                            constituents_lst.append(lines[line_index + i + 1])

                        self.input_data["constituents"] = constituents_lst

                    case "dopants":
                        dopants_lst = []
                        for i in range(int(line[2])):
                            dopants_lst.append(lines[line_index + i + 1])

                        self.input_data["dopants"] = dopants_lst

                    case "dopant_table":
                        dopants_lst = []
                        for i in range(int(line[2])):
                            dopants_lst.append(lines[line_index + i + 1])

                        self.input_data["dopants"] = dopants_lst

                    case "conduction_band_limits":
                        self.input_data[variable] = line[2:]
                        self.conductionband_limits = line[2:]

                    case "valence_band_limits":
                        self.input_data[variable] = line[2:]
                        self.valenceband_limits = line[2:]

                    case "lattice_array":
                        lattice_array_lst = []
                        for i in range(3):
                            lattice_array_lst.append(lines[line_index + i + 1])

                        self.input_data["lattice_array"] = lattice_array_lst

                    case "dielectric_array":
                        dielectric_array_lst = []
                        for i in range(3):
                            dielectric_array_lst.append(lines[line_index + i + 1])

                        self.input_data["dielectric_array"] = dielectric_array_lst

                    #case _:  # underscore used to define default case
                    #    if variable in list(self.input_data.keys()):#

                            # convert input values to float types if possible. If string, ensure all lower case
                            #try:
                            #    input_value = float(line[2])
                            #except:
                             #   input_value = line[2].lower()
                            #finally:
                             #   self.input_data[variable] = input_value

            # format constituents based on chem_pot_method
            self.input_data["constituents"] = defap_misc.format_constituents(self.input_data["constituents"],self.input_data["chem_pot_method"])
            self.constituents = self.input_data["constituents"]

            # format dopants if provided
            if self.input_data["dopants"]:
                self.input_data["dopants"], self.input_data["dopants_to_fit"] = defap_misc.format_dopants(self.input_data["dopants"])
                self.dopants, self.dopants_to_fit = self.input_data["dopants"], self.input_data["dopants_to_fit"]

            if self.secondary_phases:
                self.secondary_phases = defap_misc.format_secondary_phases(self.secondary_phases)
                self.input_data["secondary_phases"] = self.secondary_phases

            if self.bandgap > 0:
                self.charged_system = True

        return self.input_data


    def read_defects_file(self, host_compound_elements, dopants):
        self.defects_data = {}

        number_species = len(host_compound_elements)
        if dopants:
            number_species += len(dopants)

        # old .defects format
        if os.path.exists(f"./{self.seedname}.defects"):
            defectfile = f"{self.seedname}.defects"
            print("\n>>> Reading in parameters from ", defectfile)

            with open(defectfile, "r") as f:
                # strip and split lines. Skip empty lines
                lines = [s.strip().split() for s in f.readlines() if s.strip().split() != []]

                # loop through rows and and indicies
                for row_index, defect_row in enumerate(lines):

                    atoms_added_removed = {}

                    # check that the required number of values for each defect has been specified
                    if len(defect_row) != 7 + number_species:
                        raise Exception("ERROR! Insufficient number of values defined for the defect:\n"
                                        f"{defect_row[0]}\n"
                                        f"Number of values provided: {len(defect_row)}\n"
                                        f"Number of values needed: {7 + number_species}\n"
                                        f"Did you specify all the species added/removed for each defect?")

                    # assign defect values into dict
                    individual_defect_dict = {"group": defect_row[1],
                                              "multiplicity": float(defect_row[2]),
                                              "defect_site": int(defect_row[3]),
                                              "defect_charge": int(defect_row[4]),
                                              "defect_energy": float(defect_row[5]),
                                              "tab_correction": float(defect_row[6]),
                                              }

                    # loop through host elements and add whether an element is added/removed for each defect
                    for element_index, host_element in enumerate(host_compound_elements.keys()):
                        atoms_added_removed[host_element] = int(defect_row[7 + element_index])

                    # loop through dopants and add whether it is added/removed for each defect
                    if dopants:
                        for dopant_index, dopant in enumerate(dopants):
                            atoms_added_removed[dopant] = int(
                                defect_row[7 + len(host_compound_elements) + dopant_index])

                    individual_defect_dict["added/removed"] = atoms_added_removed

                    # check if defect name is unique to add to the dict
                    # if not, add a suffix to the name for the defect index
                    if defect_row[0] in self.defects_data:
                        defect_index = 2
                        new_defect_name = defect_row[0] + f'_{defect_index}'
                        while new_defect_name in self.defects_data:
                            defect_index += 1
                            new_defect_name = defect_row[0] + f'_{defect_index}'

                        defect_row[0] = new_defect_name

                    # add defect name and assign values to it in the dict
                    self.defects_data[defect_row[0]] = individual_defect_dict

            return self.defects_data

        elif os.path.exists(f"./{self.seedname}_defects.csv"):
            defectfile = f"{self.seedname}_defects.csv"
            print("\n>>> Reading in parameters from ", defectfile)

            with open(defectfile, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                for row_index, defect_row in enumerate(csv_reader):

                    atoms_added_removed = {}

                    if row_index == 0:
                        continue

                    elif len(defect_row) == 0:
                        continue

                    elif len(defect_row) != 7 + number_species:
                        raise Exception("ERROR! Insufficient number of values defined for the defect:\n"
                                        f"{defect_row[0]}\n"
                                        f"Number of values provided: {len(defect_row)}\n"
                                        f"Number of values needed: {7 + number_species}\n"
                                        f"Did you specify all the species added/removed for each defect?")

                    else:
                        individual_defect_dict = {"group": defect_row[1],
                                                  "multiplicity": float(defect_row[2]),
                                                  "defect_site": int(defect_row[3]),
                                                  "defect_charge": int(defect_row[4]),
                                                  "defect_energy": float(defect_row[5]),
                                                  "tab_correction": float(defect_row[6])
                                                  }

                        # loop through host elements and add whether an element is added/removed for each defect
                        for element_index, host_element in enumerate(host_compound_elements.keys()):
                            atoms_added_removed[host_element] = int(defect_row[7 + element_index])

                        # loop through dopants and add whether it is added/removed for each defect
                        if dopants:
                            for dopant_index, dopant in enumerate(dopants.keys()):
                                atoms_added_removed[dopant] = int(
                                    defect_row[7 + len(host_compound_elements) + dopant_index])

                        individual_defect_dict["added/removed"] = atoms_added_removed

                        # check if defect name is unique to add to the dict
                        # if not, add a suffix to the name for the defect index
                        if defect_row[0] in self.defects_data:
                            defect_index = 2
                            new_defect_name = defect_row[0] + f'_{defect_index}'
                            while new_defect_name in self.defects_data:
                                defect_index += 1
                                new_defect_name = defect_row[0] + f'_{defect_index}'

                            defect_row[0] = new_defect_name

                        # add defect name and assign values to it in the dict
                        self.defects_data[defect_row[0]] = individual_defect_dict

            return self.defects_data

        else:
            raise Exception("ERROR! No defects file is present.\nPlease provide a .defects or _defects.csv file")


    def read_dos_data(self):

        if os.path.exists(f"{self.seedname}.dos"):

            try:
                energy = np.loadtxt(f"{self.seedname}.dos", delimiter=' ', usecols=0)
                states = np.loadtxt(f"{self.seedname}.dos", delimiter=' ', usecols=1)

            except:
                try:
                    energy = np.loadtxt(f"{self.seedname}.dos", delimiter='\t', usecols=0)
                    states = np.loadtxt(f"{self.seedname}.dos", delimiter='\t', usecols=1)

                except:
                    raise Exception("\n<!> Error. Unable to read dos file.\n"
                                    "Ensure file format is in the format: 'energy' 'states'")

            self.dos_data = {"energies": energy, "states": states}

            return self.dos_data

        else:
            return None


    def read_entropy_data(self):

        heading_names = []

        # old .entropy txt file format
        if os.path.exists(f"./{self.seedname}.entropy") and self.entropy == 1:
            entropy_file = f"{self.seedname}.entropy"
            self.entropy_data = {}

            with open(entropy_file, "r") as f:
                # strip and split *.entropy file lines and append into a list. Skips empty lines
                lines = [s.strip().split() for s in f.readlines() if s.strip().split() != []]

                # loop through line rows
                for row_index, row in enumerate(lines):
                    # loop through values in the row
                    for value_index, row_value in enumerate(row):
                        # index 0 is a header row containing defect names
                        if row_index == 0:
                            # initialise dict with key being defect name and value being an empty list
                            self.entropy_data[row_value] = []

                            # store header names in separate list
                            heading_names.append(row_value)
                        else:
                            # add entropy values to list for this defect
                            self.entropy_data[heading_names[value_index]].append(float(row_value))

            return self.entropy_data

        # csv format
        elif os.path.exists(f"./{self.seedname}_entropy.csv") and self.entropy == 1:
            entropy_file = f"{self.seedname}_entropy.csv"
            self.entropy_data = {}

            with open(entropy_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                # as before, loop though rows and row values and store defect names and entropy values in the dict
                for row_index, row in enumerate(csv_reader):
                    for value_index, row_value in enumerate(row):
                        if row_index == 0:
                            self.entropy_data[row_value] = []
                            heading_names.append(row_value)
                        else:
                            self.entropy_data[heading_names[value_index]].append(float(row_value))

            return self.entropy_data

        else:
            return None
            #raise Exception("ERROR! No entropy file is present.\nPlease provide a .entropy or _entropy.csv file")


    def read_gibbs_data(self):

        self.gibbs_data = {}
        heading_names = []

        # old text .gibbs format
        if os.path.exists(f"./{self.seedname}.gibbs"):
            gibbs_file = f"{self.seedname}.gibbs"

            with open(gibbs_file, "r") as f:
                # strip and split *.gibbs file lines and append into a list. Skips empty lines
                lines = [s.strip().split() for s in f.readlines() if s.strip().split() != []]

                # loop through line rows
                for row_index, row in enumerate(lines):
                    # loop through row values
                    for value_index, row_value in enumerate(row):
                        # defect name / constituent names are index 0
                        if row_index == 0:
                            # initialise dict with key being defect/constituent name and value being an empty list
                            self.gibbs_data[row_value] = []
                            heading_names.append(row_value)
                        else:
                            # add gibbs energy temperature dependant values to list for this defect
                            self.gibbs_data[heading_names[value_index]].append(float(row_value))

            return self.gibbs_data

        # csv format
        elif os.path.exists(f"./{self.seedname}_gibbs.csv"):
            gibbs_file = f"{self.seedname}_gibbs.csv"
            with open(gibbs_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                # as before, loop though rows and row values and store names and gibbs energy values in the dict
                for row_index, row in enumerate(csv_reader):
                    for value_index, row_value in enumerate(row):
                        if row_index == 0:
                            self.gibbs_data[row_value] = []
                            heading_names.append(row_value)
                        else:
                            self.gibbs_data[heading_names[value_index]].append(float(row_value))

            return self.gibbs_data

        else:
            return None
            #raise Exception("ERROR! No Gibbs energy file is present.\nPlease provide a .gibbs or _gibbs.csv file")



    @staticmethod
    def print_input_data(input_val):
        """
        A method just to easily and quickly print the parameters from input files
        :param input_val:
        """

        if isinstance(input_val, dict):

            for key, val in input_val.items():
                print(key, val)

        elif isinstance(input_val, list) or isinstance(input_val, tuple):
            for val in input_val:
                print(val)

        else:
            print(input_val)

        print("\n")






