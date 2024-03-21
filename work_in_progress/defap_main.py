from scipy.constants import physical_constants
import time
import sys
import numpy as np
import math
from scipy.integrate import quad
import defap_input
from defap_tasks import Tasks
import defap_misc
from defap_output import write_output_file

#kboltz = physical_constants["Boltzmann constant in eV/K"][0]
def main_wrapper(func):
    def wrapper():
        start_time = time.time()

        print("+-------------------------------------------+")
        print("|  ____        __   _    ____               |")
        print(r"| |  _ \  ___ / _| / \  |  _ \ _ __  _   _  |")
        print(r"| | | | |/ _ \ |_ / _ \ | |_) | '_ \| | | | |")
        print(r"| | |_| |  __/  _/ ___ \|  __/| |_) | |_| | |")
        print(r"| |____/ \___|_|/_/   \_\_| (_) .__/ \__, | |")
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


@main_wrapper
def main():
    seedname = sys.argv[1]

    # initialise class to read input data
    data = defap_input.ReadInputData(seedname=seedname)


    # read external files for input data
    input_data = data.read_input_file()
    data.read_defects_file(data.host["elements"], data.dopants)
    data.read_dos_data()
    data.read_entropy_data()
    data.read_gibbs_data()

    # uncomment to see all input parameters defap has read
    #data.print_input_data(data.input_data)
    #data.print_input_data(data.defects_data)
    #data.print_input_data(data.dos_data)
    #data.print_input_data(data.entropy_data)
    #data.print_input_data(data.gibbs_data)

    # check errors on input data
    defap_misc.check_input_errors(data_input=data)

    # initialise writing of the output file
    write_output_file(input_data=data,
                      seedname=seedname)


    # run brouwer tasks
    if "brouwer" in data.tasks:
        Tasks(data=data, seedname=seedname).brouwer()

    # run defect_phases task
    if "defect_phases" in data.tasks:
        Tasks(data=data, seedname=seedname).defect_phases()




if __name__ == "__main__":
    main()
