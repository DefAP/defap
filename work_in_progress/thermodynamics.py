import math
from defap_misc import break_formula
import pyromat as pm
from scipy import interpolate

def calc_pressure_contribution(volatile_PP: float,
                               temperature: float
                               ):
    std_pressure_atm = 1
    boltzmann = 0.000086173324

    # Change partial pressure from a log to atm
    partial_pressure_atm = 10 ** volatile_PP
    pressure_cont = (1 / 2) * boltzmann * temperature * math.log(partial_pressure_atm / std_pressure_atm)

    return pressure_cont


def calc_temperature_contribution(gas_species: str,
                                  temperature: float,
                                  real_gas: int
                                  ):

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

        elif gas_species == "D" or gas_species == "T":
            gas_species = "H_2"

        # break formula of gas molecule
        gas_array = break_formula(gas_species)

        # create string input for pyromat libray
        pyromat_input = "ig."
        for element, stoic in gas_array.items():
            if stoic == 1:
                str = element
            else:
                str = f"{element}{stoic}"

            pyromat_input += str

        # eg: gas_array = {'N': 1, 'O': 2}
        #     pyromat_input = "ig.NO2"
        pyromat_species_thermo_data = pm.get(pyromat_input)

        # G = H - TS
        pyromat_species_Gibbs = pyromat_species_thermo_data.h(T=temperature) - (
                    temperature * pyromat_species_thermo_data.s(T=temperature))

        pyromat_species_Gibbs_std = pyromat_species_thermo_data.h(T=std_temp) - (
                    std_temp * pyromat_species_thermo_data.s(T=std_temp))

        temp_cont = (pyromat_species_Gibbs - pyromat_species_Gibbs_std)

        return temp_cont[0]


def gas_thermo_values(volatile_species: str,
                      temperature: float,
                      real_gas: int
                      ):


    # units pre converted from J/mol --> eV

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

        if volatile_species == "H" or volatile_species == "D" or volatile_species == "T":
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

        elif volatile_species == "Br":
            if 332.5 <= temperature <= 3400:
                coefficients = {"aaa": 0.000399308,
                                "bbb": -2.04885E-05,
                                "ccc": 1.5817E-05,
                                "ddd": -2.05626E-06,
                                "eee": -1.92584E-06,
                                "fff": 0.000195639,
                                "ggg": 0.003021051,
                                "hhh": 0.000320361
                                }

            elif 3400 < temperature <= 6000:
                coefficients = {"aaa": 0.000362677,
                                "bbb": 9.58931E-05,
                                "ccc": -2.44762E-05,
                                "ddd": 1.59958E-06,
                                "eee": -0.000446456,
                                "fff": -7.73982E-05,
                                "ggg": 0.002835987,
                                "hhh": 0.000320361
                                }

            else:
                raise ValueError(f"<!> Cannot use real gas parameters at temperature of {temperature} K\n"
                                 f"The temperature range for bromine is between 332.5 and 6000 K.")

        elif volatile_species == "I":
            if 457.666 <= temperature <= 2000:
                coefficients = {"aaa": 0.000391746,
                                "bbb": 2.33666E-06,
                                "ccc": -9.458E-06,
                                "ddd": 1.07261E-05,
                                "eee": -8.68798E-07,
                                "fff": 0.000527218,
                                "ggg": 0.003170646,
                                "hhh": 0.000646951
                                }

            elif 2000 < temperature <= 6000:
                coefficients = {"aaa": 0.000795296,
                                "bbb": -4.19317E-05,
                                "ccc": -1.91547E-05,
                                "ddd": 2.27024E-06,
                                "eee": -0.000853954,
                                "fff": -0.00055834,
                                "ggg": 0.002914718,
                                "hhh": 0.000646951
                                }

            else:
                raise ValueError(f"<!> Cannot use real gas parameters at temperature of {temperature} K\n"
                                 f"The temperature range for iodine is between 457.666 and 6000 K.")


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


def calc_volatile_gibbs_free_energy(temperature: float,
                                    real_gas: int,
                                    aaa, bbb, ccc,
                                    ddd, eee, fff, ggg):

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

        return enthalpy - (temperature * entropy)

    elif real_gas == 2:

        Gibbs = (aaa * (temperature - (temperature * math.log(temperature / 1000))) -
                 (1 / 2) * bbb * (temperature ** 2) -
                 (1 / 6) * ccc * (temperature ** 3) -
                 (1 / 12) * ddd * (temperature ** 4) -
                 (eee / (2 * temperature)) +
                 fff -
                 ggg * temperature)

        return Gibbs


def calc_entropies(entropy_data, temperature):
    entropies = {}

    if not entropy_data:
        return None

    else:
        # separate temperature from the constituents in the dict keys
        temperature_key, *constituent_keys = entropy_data.keys()

        temperatures_lst = entropy_data[temperature_key]

        # check the desired temperature falls between the provided temperatures for the gibbs data
        if temperature < temperatures_lst[0] or temperature > temperatures_lst[-1]:
            raise Exception("\n<!> ERROR\nTemperature is outside of range of entropy data provided\n")

            # loop through the constituents
        for constituent in constituent_keys:
            # acquire the list of energies for this constituent
            constituent_entropies = entropy_data[constituent]

            # interpolate the entropy at the desired temperature
            tck = interpolate.splrep(temperatures_lst, constituent_entropies)
            entropy_at_temp = interpolate.splev(temperature, tck)

            # add constituent and interpolated entropy to the dictionary
            entropies[constituent] = float(entropy_at_temp)

        return entropies


def calc_gibbs_function(gibbs_data, temperature):

    gibbs_energies = {}

    if not gibbs_data:
        return None
    else:

        # separate temperature from the constituents in the dict keys
        temperature_key, *constituent_keys = gibbs_data.keys()

        temperatures_lst = gibbs_data[temperature_key]

        # check the desired temperature falls between the provided temperatures for the gibbs data
        if not (temperatures_lst[0] <= temperature <= temperatures_lst[-1]):
            raise Exception("\n<!> ERROR\nTemperature is outside of range of Gibbs data provided\n")


        for constituent in constituent_keys:
            constituent_gibbs_engergies = gibbs_data[constituent]

            # interpolate the entropy at the desired temperature
            tck = interpolate.splrep(temperatures_lst, constituent_gibbs_engergies)
            gibbs_at_temp = interpolate.splev(temperature, tck)

            # add constituent and interpolated entropy to the dictionary
            gibbs_energies[constituent] = float(gibbs_at_temp)

        return gibbs_energies

