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

#####################################################
#                                                   #      
#           Defect Analysis Package 1.10            #
#                                                   #
#               by Samuel T. Murphy                 #
#               & William D. Neilson                #
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
# Please cite:                                      #
# S. T. Murphy and N. D. M. Hine, "Point defects    #
# and non-stoichiometry in Li2TiO3" Chemistry of    #
# Materials 26 (2014) 1629-1638.                    # 
#                                                   #
# If you encounter any problems with this script    #
# please contact: w.neilson@lancaster.ac.uk         #
#                                                   #
#####################################################
#                                                   #
# Last updated :  05/04/22                          #
#                                                   #
#####################################################

version = '1.10'

#Function to print header
def header():
    print( "+-------------------------------------------+")
    print( "|  ____        __   _    ____               |")
    print( "| |  _ \  ___ / _| / \  |  _ \ _ __  _   _  |")
    print( "| | | | |/ _ \ |_ / _ \ | |_) | '_ \| | | | |")
    print( "| | |_| |  __/  _/ ___ \|  __/| |_) | |_| | |")
    print( "| |____/ \___|_|/_/   \_\_| (_) .__/ \__, | |")
    print( "|                             |_|    |___/  |")
    print( "|                                           |")
    print( "|       Defect Analysis Package : 1.10      |")
    print( "|             mmmg.co.uk/defap              |")
    print( "+-------------------------------------------+")
    print( "|                                           |")
    print( "|             William D. Neilson            |")
    print( "|              Samuel T. Murphy             |")
    print( "|                                           |")
    print( "+-------------------------------------------+\n")
    
def inputs(seedname):

    filename = str(seedname)+".input"
    
    #Define variales and set some sensible defaults
    temperature = 1000          #Temperature in K
    def_statistics = 0          #Defect statistics 0 = Boltzmann, 1 = Kasamatsu
    tab_correction = 0          #Include a correction to the defect energy in the filename.defects file (0 = off, 1 = on)
    num_tasks = 0               #Number of tasks to perform
    host_energy = 0             #Energy of a formula unit of the host
    host_supercell = 0          #Energy of the host perfect host supercell
    chem_pot_method = 4         #Definition of the chemical potentials 0 = defined, 1 = rich-poor 2= volatile, 3=volatile-rich-poor. 
    use_coul_correction = 0     #Use a Coulombic correction 0 = none, 1 = Makov-Payne, 2 = screened Madelung
    length = 0                  #Supercell length for the MP correction
    dielectric = 1              #Dielectric constant for MP correction
    v_M = 1                     #Screened Madelung potential
    E_VBM = 0                   #Energy of the valence band maximum
    bandgap = 0                 #Bandgap for the host lattice
    condband = 0                #Effective conduction band integral
    valband = 0                 #Effective valence band integral
    electron_method = 0         #Method for calculating the electron concentration 0 = none, 1 = Boltzmann, 2 = Fermi-Dirac, 3 = Fixed
    hole_method = 0             #Method for calculating the hole concentration 0 = none, 1 = Boltzmann, 2 = Fermi-Dirac, 3 = Fixed
    fixed_e_conc = 0            #Fixed electron concentration
    fixed_p_conc = 0            #Fixed hole concentration
    art_dop_conc = 0            #Concentration of an artificial dopant
    art_dop_charge = 0          #Charge on the artifical dopant
    loop = 0                    #Property to loop over 0 = partial pressure, 1 = temperature, 2 = dopant conc, 3 = aritifical dopant conc
    min_value = -40             #Minimum value used in loop
    max_value = 0               #Maximum value using in loop
    iterator = 1                #Iterator between minimum and maximum
    gnuplot_version = 0         #Version of gnuplot 0 = v4, 1 = v5
    min_y_range = -20           #Minimum on the y axis for Brouwer plots
    max_y_range = 0             #Maximum on the y axis for Brouwer plots
    host_name =''               #Host name
    cond_band_min=0             #Conduction band minimum
    cond_band_max=2             #Conduction band maximum
    val_band_min=2              #Valence band minimum
    val_band_max=4              #Valence band maximum
    fu_uc=1                     #Number of functional units per unit cell. 
    electron_mass_eff=0         #DOS effective mass for electron
    hole_mass_eff=0             #DOS effective mass for hole
    unit_vol=0                  #Volume of unit cell (A^3) Used in DOS effective masses and for y axis unit conversion
    lines = 0                   # Not in use
    y_form_min=0                # Not in use
    y_form_max=10               # Not in use
    entropy_marker=0            #Use a vibrational entropy contribution (0 = off, 1 = on)
    entropy_units=1             #The number of functional units the entropy values that are entered in filename.entropy represen   
    scheme =0                   #Selection of colour scheme for plots produced by DefAP: 0: DefAP colour scheme (default).1: User customised scheme. Requires the input file, filename.plot
    stoichiometry=0             #Calculate and plot stoichiometry 1= on, 2= special option that considers dopants and calulates an O/M ratio. 
    x_variable =0               #Selection of x-axis in final defect concentration plots: 0: Plot as function of the property defined with loop (default). 1: Plot as a function of stoichiometry.
    y_variable =0                   #Selection of y-axis units in final defect concentration plots: 0: Per functional unit (default). 1: per cm^-3.
    real_gas = 0                #Calculate volatile chemical potetial with real gas parameters (1) 
    function_tol= 1e-10         #Sequential Least Squares Programming: Precision goal for the value of function in the stopping criterion.
    maxiter_dop= 100            #Maximum number of iterations to optimise dopant chemical potential(s) (SLSQP)
    charge_convergence = 0.0000000001       #The stopping criteria for the calculation of the Fermi level. Fermi level deemed satisfactory when the total charge does not exceed this value. 
    potential_convergence = 0.001           # Convergence criteria for dopant concentration: the difference between logarithmic target and calculated concentration. 
    
    #Data holds
    tasks = []
    constituents = []
    constituents_name_list = []
    dopref_name_list = []
    dopants = [0]
    stability =[0]
    dopant_fitting = 0
    host_array=[]               #Host array
    num_elements =0             #Number of elements in the host
    
    print(">>> Reading in parameters from ",filename)
   
    with open(filename) as file:
        for linenumber, line in enumerate(file):
            fields = line.strip().split()
            
            if len(fields) !=0:
                name=fields[0]
                            
                #Tasks
                if (name == "tasks"):
                    num_tasks = len(fields)-2
                    if (num_tasks == 0):
                        print("<!> There are no tasks to perform")
                        exit()
                    for i in np.arange(0, num_tasks, 1):
                        task = fields[2+i]
                        tasks.append(task)
                
                if (name == "loop"):
                    loop = float(fields[2])
                    #loop = 0 : loop over volatile partial pressure
                    #loop = 1 : loop over temperature
                    #loop = 2 : loop over dopant concentration
                    #loop = 3 : loop over artificial charge concentration
                    #loop = 4 : loop over dopant partial pressure
                            
                #Properties for loop
                if (name == "min_value"):
                    min_value = float(fields[2])
                if (name == "max_value"):
                    max_value = float(fields[2])
                if (name == "iterator"):
                    iterator = float(fields[2])

                #Host formula
                if (name == "Host"):
                    host_name = fields[2]
                    host_array = break_formula(host_name,0)
                    num_elements=host_array[0]

                #Calculate stoichiometry
                if (name == "Stoichiometry"):
                    stoichiometry = int(fields[2])
                    
                #Host Energy (eV)
                if (name == "Host_energy"):
                    host_energy = float(fields[2])
                if (name == "Host_supercell"):
                    host_supercell = float(fields[2])
                
                #Energy of the Valence Band Maximum (eV)
                if (name == "E_VBM"):
                    E_VBM = float(fields[2])
                
                #Bangap of the host material
                if (name == "Bandgap"):
                    bandgap = float(fields[2])
                
                #Effective conduction band integral
                if (name == "Conductionband"):
                    condband = float(fields[2])
                
                #Effective valence band integral
                if (name == "Valenceband"):
                    valband = float(fields[2])
                 
                #Electron calculation method
                if (name == "Electron_method"):
                    if (fields[2] == "Off"):
                        electron_method = 0
                    elif (fields[2] == "Boltzmann"):
                        electron_method = 1
                    elif (fields[2] == "Fermi-Dirac"):
                        electron_method = 2
                    elif (fields[2] == "Fixed"):
                        electron_method = 3
                        fixed_e_conc = fields[3]
                    elif (fields[2] == "Effective_masses"):
                        electron_method = 4
                        electron_mass_eff = eval(fields[3])                        
                    else:
                        print("<!> Error : Undefined method for calculating electron concentration")
                        exit()
                                        
                #Hole calculation method
                if (name == "Hole_method"):
                    if (fields[2] == "Off"):
                        hole_method = 0
                    elif (fields[2] == "Boltzmann"):
                        hole_method = 1
                    elif (fields[2] == "Fermi-Dirac"):
                        hole_method = 2
                    elif (fields[2] == "Fixed"):
                        hole_method = 3
                        fixed_p_conc = fields[3]
                    elif (fields[2] == "Effective_masses"):
                        hole_method = 4
                        hole_mass_eff = eval(fields[3])
                    else:
                        print("<!> Error : Undefined method for calculating hole concentration")
                        exit()
                    
                #Minimum and maximum for the valence and conduction bands
                if (name == "Valence_band_limits"):
                    val_band_min = float(fields[2])
                    val_band_max = float(fields[3])
                if (name == "Conduction_band_limits"):
                    cond_band_min = float(fields[2])
                    cond_band_max = float(fields[3])

                #Unit cell details
                if (name == "fu_unit_cell"):
                    fu_uc = float(fields[2])
                if (name == "Volume_unit_cell"):
                    unit_vol = float(fields[2])

                #Temperature
                if (name == "Temperature"):
                    temperature = float(fields[2])
                       
                #Chemical potential method
                if (name == "real_gas"):
                    real_gas = float(fields[2])
                if (name == "chem_pot_method"):
                    if fields[2].lower() == "Defined".lower():
                        chem_pot_method = 0
                    elif fields[2].lower() == "Rich-poor".lower():
                        chem_pot_method = 1
                    elif fields[2].lower() == "Volatile".lower():
                        chem_pot_method = 2
                    elif fields[2].lower() == "Volatile-Rich-Poor".lower():
                        chem_pot_method = 3
                    else:
                        print("<!> Error : Unknown chem_pot_method entered")
                        exit()

                #Convergence criteria
                if (name == "Charge_convergence"):
                     charge_convergence = float(fields[2])

                #Constituents
                if (name == "Constituents"):

                    defintion_total = 0
                
                    #Loop over list of constituents
                    for i in np.arange(1,host_array[0]+1 , 1):
                        
                        with open(filename) as file3:
                            for linenumber3, line3 in enumerate(file3):
                                fields3 = line3.strip().split()

                                if linenumber+i == linenumber3:
                                
                                    if (chem_pot_method == 0):                      #Use defined chemical potentials
                                    
                                        constituent_name = fields3[0]
                                        constituent_energy = float(fields3[1])

                                        constituents.append(constituent_name)
                                        constituents.append(constituent_energy)
                                        constituents_name_list.append(constituent_name)                                     
                            
                                    elif(chem_pot_method == 1):                    #Use rich-poor chemical potential method
                                
                                        constituent_name = fields3[0]
                                        constituent_energy = float(fields3[1])
                                        constituent_definition = float(fields3[3])

                                        constituents.append(constituent_name)
                                        constituents.append(constituent_energy)
                                        constituents.append(constituent_definition)
                                        constituents_name_list.append(constituent_name)
                                        
                                        if (constituent_definition > 1.0):
                                            print ("<!> Error : Constituent", constituent_name, "has greater definition than 1")
                                            exit()
                                
                                        defintion_total += constituent_definition
                                        if (defintion_total > (host_array[0]-1)):
                                            print("<!> Error : Total rich-poor balance greater than possible with this number of constituents")
                                            exit()

                                    if(chem_pot_method == 2):                  #Use volatile method with a binary system
                                        
                                        if i ==1:
                                            
                                            gaseous_species = fields3[0]                                
                                            partial_pressure = float(fields3[1])

                                            constituents.append(gaseous_species)                                   
                                            constituents.append(partial_pressure)
                                            
                                        else:
                                            
                                            constituent_name =fields3[0] 
                                            constituent_energy_DFT =float(fields3[1])
                                            constituent_metal_DFT =float(fields3[2])
                                            constituent_formation =float(fields3[3])

                                            constituents.append(constituent_name)
                                            constituents.append(constituent_energy_DFT)
                                            constituents.append(constituent_metal_DFT)
                                            constituents.append(constituent_formation)                                
                                            constituents_name_list.append(constituent_name)    

                                    if(chem_pot_method == 3):                  #Use rich-poor chemical potential and volatile method

                                        number_bin_oxides = host_array[0]-1
                                        
                                        if i ==1:
                                            
                                            gaseous_species = fields3[0]
                                            gaseous_stoichiometry = float(fields3[1])
                                            partial_pressure = float(fields3[2])

                                            constituents.append(gaseous_species)
                                            constituents.append(gaseous_stoichiometry)
                                            constituents.append(partial_pressure)
                                        else:
                                            constituent_name =fields3[0]
                                            constituent_stoich =float(fields3[1])
                                            constituent_energy_DFT =float(fields3[2])
                                            constituent_metal_DFT =float(fields3[3])
                                            constituent_formation =float(fields3[4])
                                            constituent_definition = float(fields3[5])

                                            constituents.append(constituent_name)
                                            constituents.append(constituent_stoich)
                                            constituents.append(constituent_energy_DFT)
                                            constituents.append(constituent_metal_DFT)
                                            constituents.append(constituent_formation)
                                            constituents.append(constituent_definition)
                                            constituents_name_list.append(constituent_name)
                                
                                            #print (constituent_name ,constituent_stoich ,constituent_energy_DFT ,constituent_metal_DFT ,constituent_formation ,constituent_definition)
                                            if (constituent_definition > 1.0):
                                                print("<!> Error : Constituent", constituent_name, "has greater definition than 1")
                                                exit()
                                    
                                            defintion_total += constituent_definition
                                            if (defintion_total > (host_array[0]-2)):
                                                print("<!> Error : Total rich-poor balance greater than possible with this number of constituents")
                                                exit()
                           
                #Dopants
                if (name == "Dopant_table"):
                    number_of_dopants = float(fields[2])
                    dopants[0] = number_of_dopants

                    #Loop over dopant table and fill dopants array
                    
                    for i in np.arange(1,number_of_dopants+1 , 1):
                        with open(filename) as file4:
                            for linenumber4, line4 in enumerate(file4):
                                fields4 = line4.strip().split()
                         
                                if linenumber+i == linenumber4:
                                    dopant_name = fields4[0]                                   
                                    #Break down details of the reference state
                                    reference_state = fields4[1]
                                    temp_array = break_formula(reference_state,1)
                                    dopants.append(dopant_name)
                                    
                                    reference_energy =float(fields4[2])
                                    dopants.append(reference_energy)
                                 
                                    fit_chempot =int(fields4[3])
                                    dopants.append(fit_chempot)

                                    if fit_chempot == 0:
                                        dopants.append(0)
                                        dopants.append(temp_array)
                                        dopants.append(0)

                                    if fit_chempot == 1 or fit_chempot == 2:
                                        dopant_fitting+=1    
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

                #Dopant optimise details
             
                if (name =="Tolerance"):
                    function_tol = float(fields[2])
                if (name =="max_iteration"):
                    maxiter_dop = float(fields[2])
                if (name =="Potential_convergence"):
                    potential_convergence = float(fields[2])
                    
                #Artificial dopants
                if (name =="Art_Dopant_Conc"):
                    art_dop_conc = float(fields[2])
                if (name =="Art_Dopant_Chg"):
                    art_dop_charge = float(fields[2])
                
                #Stability checks 
                if (name == "Stability_check"):
                    number_of_checks = float(fields[2])
                    stability[0]= number_of_checks

                    for i in np.arange(1,number_of_checks+1 , 1):
                        with open(filename) as file5:
                            for linenumber5, line5 in enumerate(file5):
                                fields5 = line5.strip().split()

                                if linenumber+i == linenumber5:
                                    constituent = fields5[0]
                                    reference_energy =float(fields5[1])

                                    stability.append(constituent)
                                    stability.append(reference_energy)

                                    #Break down details of the reference state
                                    temp_array = break_formula(constituent,1)
                                        
                                    stability.append(temp_array)
                                                           
                #Defect concentration method
                if (name == "Defect_conc_method"):
                    if (fields[2].lower() == "Boltzmann".lower()):
                        def_statistics = 0
            
                    elif (fields[2].lower() == "Kasamatsu".lower()):
                        def_statistics = 1;
            
                    else:
                        print ("<!> Error : Unknown defect statistics method entered")
                        exit()
          
                #Use correction schemes
                if (name =="Tab_correction"):
                    tab_correction = 1
                if (name == "Coulombic_correction"):
                    use_coul_correction = int(fields[2])
                if (name == "Dielectric_constant"):
                    dielectric = float(fields[2])
                if (name == "Length"):
                    length = float(fields[2])
                if (name == "Screened_Madelung"):
                    v_M = float(fields[2])

                #Formation plot preferences
                if name == "Formation_energy_limits":
                    y_form_min= float(fields[2])
                    y_form_max= float(fields[2])
                if name == "Lines":
                    lines = int(fields[2])

                #Entropy
                if name == "entropy":
                    entropy_marker = int(fields[2])
                if name == "entropy_units":
                    entropy_units = int(fields[2])
                
                #Plotting preferences 
                if (name == "x_variable"):
                    x_variable = int(fields[2])
                if (name == "y_axis"):
                    y_variable = int(fields[2])
                    if y_variable ==1:
                        max_y_range = 20
                if (name == "Gnuplot_version"):
                    gnuplot_version = fields[2]
                if (name == "min_y_range"):
                    min_y_range = fields[2]
                if (name == "max_y_range"):
                    max_y_range = fields[2]
                if name == "Scheme":
                    scheme = int(fields[2])

    #Some error messages

    if len(tasks) ==0:
        print("<!> There are no tasks to perform")
        exit()

    for i in tasks:
        if i not in ['brouwer','energy','form_plots','autodisplay','stability','madelung','bibliography','group']:
            print("<!> '",i,"' not an optional task")
            exit()

    if 'form_plots' in tasks:
        if 'energy' not in tasks:
            print("<!> 'The 'form_plots' task has no effect without the 'energy' task")
            exit()
            
    for i in tasks:
        if i in ['brouwer','energy','form_plots','autodisplay','stability','group']:
        
            if host_energy == 0:
                print("<!> Undefined 'Host_energy'")
                exit()
            if host_supercell == 0:
                print("<!> Undefined 'Host_supercell'")
                exit()
            break
        
    if(entropy_marker == 1 and def_statistics == 1):
        print("<!> Error: Entropy cannot be used with the Kasamatsu statistics")
        exit()
    if min_value >= max_value:
        print("<!> Error: Incompatible min_value and max_value")
        exit()

    if (unit_vol==0) and (y_variable ==1):
        print("<!> Error: Unit cell volume must be defined for the y axis to be in units of cm^-3")
        exit()

    if (unit_vol==0) and ((hole_method == 4)or(electron_method == 4)):
        print("<!> Error: Unit cell volume must be defined for the carrier concentration method selected")
        exit()
                
    #Output file construction

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")	
  
    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
        print('DefAP',version, file =f)
        print("Executed on", dt_string,"\n", file=f)
        print("-------------------------------------------------------","\n", file=f)
        for i in tasks:
            if i in ['brouwer','energy','form_plots','autodisplay','stability','group']:
                print(">>> Reading in parameters from ",filename,"\n",file=f)
                print("   Number of tasks :",num_tasks, file=f)
                for i in np.arange(1,num_tasks+1,1):
                    print("   Task",i,":",tasks[i-1],file=f)
                print("\n   Host :",host_name,file=f)
                print("   Number of elements in host :", num_elements,file=f)
                print("   DFT energy of host pfu:",host_energy,'eV',file=f)
                print("   DFT energy of host supercell:",host_supercell,'eV',file=f)
               
                print("   Energy of the valence band maximum:",E_VBM,'eV',file=f)
                print("   Stoichiometry method:",stoichiometry,file=f)

                print("\n>>> Electronic properties\n", file=f)
                print("   Bandgap of host material:",bandgap,'eV\n',file=f)
                
                if(electron_method == 0):
                    print("   Not calculating electron concentrations\n",file=f)         
                elif(electron_method == 1):
                    print("   Using Boltzmann statistics for electron concentrations",file=f)
                    print("   Effective conduction band integral:", condband,'eV pfu\n', file=f)          
                elif(electron_method == 2):
                    print("   Using Fermi-Dirac statistics for the electron concentration",file=f)
                    print("   Conduction_band_limits:",cond_band_min,'-',cond_band_max,'eV',file=f)
                    print("   Number of functional units per unit cell:", fu_uc,'\n', file=f)
                elif(electron_method == 3):
                    print("   Using fixed electron concentration of ",fixed_e_conc, "\n",file=f)
                elif(electron_method == 4):
                    print("   Using electron density of states effective masses",file=f)
                    print("   Electron density of states effective mass",electron_mass_eff,file=f)
                    print("   Number of functional units per unit cell:", fu_uc,file=f)
                    print("   Volume of the unit cell:", unit_vol,"Angstroms^3\n", file=f)
                  
                if(hole_method == 0):
                    print("   Not calculating hole concentrations\n",file=f)         
                elif(hole_method == 1):
                    print("   Using Boltzmann statistics for hole concentrations",file=f)
                    print("   Effective valence band integral:", valband,'eV pfu\n', file=f)          
                elif(hole_method == 2):
                    print("   Using Fermi-Dirac statistics for the hole concentration",file=f)
                    print("   Valence_band_limits:",val_band_min,'-',val_band_max,'eV',file=f)
                    print("   Number of functional units per unit cell:", fu_uc,'\n', file=f)
                elif(hole_method == 3):
                    print("   Using fixed hole concentration of ",fixed_p_conc, "\n",file=f)
                elif(hole_method == 4):
                    print("   Using hole density of states effective masses",file=f)
                    print("   Hole density of states effective mass",hole_mass_eff,file=f)
                    print("   Number of functional units per unit cell:", fu_uc,file=f)
                    print("   Volume of the unit cell:", unit_vol,"Angstroms^3\n", file=f)

                print(">>> Chemical potentials\n", file=f)
                if(chem_pot_method == 0):
                    print("   Chemical potentials defined\n",file=f)
                    print("   Defining chemical potentials of elements in host:",file=f)
                    print("   +---------+------------------------+",file=f)
                    print("   | Element | Chemical potential (eV)|",file=f)
                    print("   +---------+------------------------+",file=f)
                    for i in np.arange(0, len(constituents)/2, 1):
                        i = int(i)
                    print("   | %7s | %22f |" % (constituents[2*i], constituents[2*i+1]), file=f)
                    print("   +---------+------------------------+\n",file=f)
                if(chem_pot_method == 1):
                    print("   Rich-poor method selected\n",file=f)
                    print("   Defining chemical potentials of elements in host:",file=f)
                    print("   +---------+------------------------+-----------+",file=f)
                    print("   | Element | Chemical potential (eV)|  fraction |",file=f)
                    print("   +---------+------------------------+-----------+",file=f)
                    for i in np.arange(0, len(constituents)/3, 1):
                        i = int(i)
                    print("   | %7s | %22f | %8f |" % (constituents[3*i], constituents[2*i+1], constituents[3*i+2]), file=f)
                    print("   +---------+------------------------+-----------+\n",file=f)
                if(chem_pot_method == 2):
                    print("   Volatile method selected\n",file=f)
                    if(real_gas == 0):
                        print("   Using ideal gas specific heat", file=f)
                    elif(real_gas == 1) or (real_gas == 2):
                        print("   Using real gas relations for specific heat", file=f)
                    print("   Defining volatile species:",file=f)
                    print("   +-----------------+------------------+",file=f)
                    print("   | Gaseous species | Partial pressure |",file=f)
                    print("   +-----------------+------------------+",file=f)
                    print("   | %15s | %16f |" % (constituents[0], constituents[1]),file=f)
                    print("   +-----------------+------------------+\n",file=f)

                    print("   Defining properties of binary species:",file=f)
                    print("   +----------------+----------------+------------------------+-----------------------+",file=f)
                    print("   | Binary species | DFT energy (eV)| Cation DFT energy (eV) | Formation energy (eV) |",file=f)
                    print("   +----------------+----------------+------------------------+-----------------------+",file=f)
                    print("   | %14s | %14f | %22f | %21f |" % (constituents[2],constituents[3],constituents[4],constituents[5]),file=f)
                    print("   +----------------+----------------+------------------------+-----------------------+\n",file=f)

                if(chem_pot_method == 3):
                    print("   Volatile-Rich-Poor method selected\n",file=f)
                    if(real_gas == 0):
                        print("   Using ideal gas specific heat",file=f)
                    elif(real_gas == 1) or (real_gas == 2):
                        print("   Using real gas relations for specific heat",file=f)
               
                    print("   Defining volatile species:",file=f)
                    print("   +-----------------+---------------+------------------+",file=f)
                    print("   | Gaseous species | Stoichiometry | Partial pressure |",file=f)
                    print("   +-----------------+---------------+------------------+",file=f)
                    print("   | %15s | %13f | %16f |" % (constituents[0], constituents[1], constituents[2]),file=f)
                    print("   +-----------------+---------------+------------------+\n",file=f)

                    print("   Defining constituents of host:",file=f)
                    print("   +------------------+---------------+----------------+------------------------+--------------------------------------+----------+",file=f)
                    print("   | Constituent name | Stoichiometry | DFT energy (eV)| Cation DFT energy (eV) | Formation energy of constituent (eV) | fraction |",file=f)
                    print("   +------------------+---------------+----------------+------------------------+--------------------------------------+----------+",file=f)
                 
                    for i in np.arange(0, number_bin_oxides, 1):
                        i = int(i)
                        constituent_name = constituents[6*i+3]
                        constituent_stoich = float(constituents[6*i+4])
                        constituent_energy_DFT = float(constituents[6*i+5])
                        constituent_metal_DFT = float(constituents[6*i+6])
                        constituent_formation = float(constituents[6*i+7])
                        constituent_definition = float(constituents[6*i+8])
                            
                        print("   | %16s | %13f | %14f | %22f | %36f | %6f |" % (constituent_name,constituent_stoich,constituent_energy_DFT,constituent_metal_DFT,constituent_formation,constituent_definition),file=f)

                    print("   +------------------+---------------+----------------+------------------------+--------------------------------------+----------+\n",file=f)
               
                #Dopants
                print(">>> Dopants\n",file=f)
                print("   Number of dopants :", int(dopants[0]),file=f)
                if (dopants[0] != 0 ):
                    for i in np.arange(0, dopants[0], 1):
                         i = int(i)     
                         print("\n   Dopant",i+1,":",file=f)
                         fit_chempot =int(dopants[6*i+3])
                         if fit_chempot ==0:
                             print("   +----------------+------------------+-----------------------------+----------------+",file=f)
                             print("   | Dopant element | Dopant reference | DFT energy of reference (eV)| Fitting option |",file=f)
                             print("   +----------------+------------------+-----------------------------+----------------+",file=f)         
                             dopant_name = dopants[6*i+1]
                             reference_state =dopref_name_list[i]
                             reference_energy =float(dopants[6*i+2])
                             fit_chempot =int(dopants[6*i+3])
                             print("   | %14s | %16s | %27f | %14i |" % (dopant_name, reference_state,reference_energy,fit_chempot),file=f)
                             print("   +----------------+------------------+-----------------------------+----------------+\n",file=f)
                         elif fit_chempot == 1 or fit_chempot == 2:                            
                             print("   +----------------+------------------+-----------------------------+----------------+--------------------------+-------------------------------+",file=f)
                             print("   | Dopant element | Dopant reference | DFT energy of reference (eV)| Fitting option | Target concentration pfu | Chemical potential range (eV) |",file=f)
                             print("   +----------------+------------------+-----------------------------+----------------+--------------------------+-------------------------------+",file=f)         
                             dopant_name = dopants[6*i+1]
                             reference_state =dopref_name_list[i]
                             reference_energy =float(dopants[6*i+2])
                             target_conc = float(dopants[6*i+4])
                             dopant_range = float(dopants[6*i+6])
                             print("   | %14s | %16s | %27f | %14i | %24s | %29f |" % (dopant_name, reference_state,reference_energy,fit_chempot,"{:.10f}".format(target_conc),dopant_range),file=f)
                             print("   +----------------+------------------+-----------------------------+----------------+--------------------------+-------------------------------+\n",file=f)
                         elif fit_chempot == 3 or fit_chempot == 4:                    
                             print("   +----------------+------------------+-----------------------------+----------------+------------------+",file=f)
                             print("   | Dopant element | Dopant reference | DFT energy of reference (eV)| Fitting option | Partial pressure |",file=f)
                             print("   +----------------+------------------+-----------------------------+----------------+------------------+",file=f)         
                             dopant_name = dopants[6*i+1]
                             reference_state =dopref_name_list[i]
                             fit_chempot =int(dopants[6*i+3])
                             partial_pressure = float(dopants[6*i+6])
                             print("   | %14s | %16s | %27f | %14i | %16i |" % (dopant_name, reference_state,reference_energy,fit_chempot,partial_pressure),file=f)
                             print("   +----------------+------------------+-----------------------------+----------------+------------------+\n",file=f)             
               
                if loop != 3:
                    print("   Artificial dopant concentration:", art_dop_conc, file=f)
                    print("   Artificial dopant charge:", art_dop_charge, file=f)
               
                if dopant_fitting == 1 or dopant_fitting == 2:
                    print("\n   Fitting chemical potential of",dopant_fitting,"dopants", file=f)
                    if dopant_fitting ==1:
                        print("   Using Linear Bisection", file=f)
                    else:
                        print("   Using Seqential Least Squares Programming", file=f)
                        print("   Convergence criteria for logarithmic dopant concentration : ",potential_convergence, file=f)
                        print("   SLSQP precision goal : ",function_tol, file=f)
                        print("   SLSQP maximum iterations : ",maxiter_dop, file=f)
                else:
                    print("\n   No fitting of dopant chemical potentials selected", file=f)

                #Entropy
                print("\n>>> Entropy\n", file=f)
                if entropy_marker == 1:
                    print("   Entropy contribution ON", file=f)
                    print("   Number of functional units in supercell used to calculate entropy:", entropy_units, file=f)
                else:
                    print("   Entropy contribution OFF", file=f)
                
                print("\n>>> Defect methodology\n", file=f)
                if def_statistics ==0:
                    print( "   Defect concentration method : Boltzmann", file=f)
                elif def_statistics ==1:
                    print( "   Defect concentration method : Kasamatsu", file=f)
           
                #Use correction schemes
                if (tab_correction == 1) :
                    print( "   Tab correction ON, modifier will be read for each defect from column 7 of .defects file.", file=f)
                else:
                    print( "   Tab correction OFF", file=f)
               
                if (use_coul_correction ==1) and ('madelung' not in tasks):
                   
                    print("   Makov-Payne correction ON", file=f)
                    print("   Supercell length:",length,'Angstroms', file=f)
                    print("   Dielectric constant:",dielectric, file=f)
                    print("   Madelung constant: 2.8373", file=f)
                    
                elif use_coul_correction ==2 and ('madelung' not in tasks):
                    print( "   Screened Madelung correction ON", file=f)
                    print("   Screened Madelung potential:",v_M, file=f)

                elif ('madelung' in tasks):
                    use_coul_correction ==2
                    print( "   Screened Madelung correction ON", file=f)
                    print("   Screened Madelung potential to be calculated", file=f)

                else:
                    print( "   Makov-Payne and Screened Madelung corrections OFF", file=f)
                            
                if ('brouwer' in tasks):
                    print("\n>>> Instructions for: Task = brouwer", file=f)
                    print('\n   loop =',int(loop), file=f)
                    if(loop == 0):
                        print("   Looping over volatile partial pressure\n",file=f)
                        
                        print("   Temperature :",temperature , "K",file=f)
                        print( "   Volatile partial pressure range :",min_value, "-",max_value, "\n",file=f)
                    if(loop == 1):
                        print("   Looping over temperature\n",file=f)
                       
                        if(chem_pot_method == (2 or 3)):
                            print("   Volatile partial pressure :",partial_pressure,file=f)
                        print("   Temerature range :",min_value, "-",max_value,"K\n",file=f)
                    if(loop == 2):
                        print("   Looping over dopant concentration\n",file=f)
                        
                        print("   Temperature :",temperature,"K",file=f)
                        if(chem_pot_method == (2 or 3)):
                            print("   Volatile partial pressure :", partial_pressure,file=f)
                        print("   Target dopant concentration range :",min_value,"-",max_value, "pfu\n",file=f)
                    if(loop == 3):
                        print("   Looping over artificial dopant concentration\n",file=f)
                        
                        print("   Temperature :",temperature,"K",file=f)
                        if(chem_pot_method == (2 or 3)):
                            print("   Volatile partial pressure :", partial_pressure,file=f)
                        print("   Artificial dopant concentration range :",min_value,"-",max_value, "pfu\n",file=f)
                        print("   Artificial dopant charge:", art_dop_charge, file=f)

                    if(loop == 4):
                        print("   Looping over dopant partial pressure\n",file=f)
                        
                        print("   Temperature :",temperature , "K",file=f)
                        print( "   Dopant partial pressure range :",min_value, "-",max_value, "\n",file=f)

                    print("\n>>> Plotting preferences", file=f)
            
                    if y_variable ==0:
                        print('\n   Units of y axis set at \"concentration per functional unit\"', file=f)
                    if y_variable ==1:
                        print('\n   Units of y axis set at \"concentration per cm^-3\"', file=f)
                        print('   Conversion parameters:', file=f)
                        print('   Unit cell volume:',unit_vol,"Angstroms^3", file=f)
                        print('   Number of functional units in unit cell:',fu_uc, file=f)
                    print('   Minimum of y-axis set at',min_y_range, file=f)
                    print('   Maximum of y-axis set at',max_y_range, file=f)
                    if x_variable == 1:
                        print('   Plotting as a function of stoichiometery; default range -0.1 to +0.1', file=f)
                    if scheme == 0:
                        print('   Default coulour scheme will be used', file=f)
                    if scheme == 1:
                        print('   User defeined coulour scheme will be used from file.plot', file=f)
           
                if ('stability' in tasks):
                     print("\n>>> Instructions for: Task = stability", file=f)
                     print("\n   Checking the stability of",int(number_of_checks),"compounds",file=f)
                     print("\n   Compounds:",file=f)
                     print("   +------------------+-----------------+",file=f)
                     print("   |     Compound     | DFT energy (eV) | ",file=f)
                     print("   +------------------+-----------------+",file=f)
                     for i in np.arange(0, number_of_checks, 1):
                         i = int(i)
                         compound = stability[3*i+1]
                         compound_energy = float(stability[3*i+2])
                         print("   | %16s | %15f |" % (compound, compound_energy),file=f)
                     print("   +------------------+-----------------+",file=f)
                break
            
    print("..> Input file read successfully")
    
    return (host_array,dopants,tasks,constituents,constituents_name_list,temperature,def_statistics,tab_correction,host_energy,chem_pot_method,host_supercell,use_coul_correction,length,dielectric,v_M,E_VBM,bandgap,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,loop,min_value,max_value,iterator,gnuplot_version,min_y_range,max_y_range,host_name,val_band_min,val_band_max,cond_band_min,cond_band_max,y_form_min,y_form_max,lines, entropy_marker, entropy_units,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charge_convergence, potential_convergence,stability, scheme,stoichiometry,x_variable,real_gas,function_tol,maxiter_dop, y_variable)

#Subroutine for breaking down chemical formula
def break_formula(formula, index):
    
    temp_array=[]
    
    #Split the host definition on a hyphen
    splithost = formula.split('-')
    
    #Determine how many elements there are in the new array
    num_elements = len(splithost)
    if (index == 0):
        pass
    else:
        #print("Number of elements in subsystem",index,num_elements)   
        pass
    temp_array.append(num_elements)
    
    #Now loop over the number of elements in the formula
    for i in np.arange(0, num_elements, 1):
        splitelement = splithost[i].split('_')
        element = splitelement[0]
        if (len(splitelement)==2):
            stoich_number = splitelement[1]
        else:
            stoich_number = 1

        #Push details into temp_array
        temp_array.append(element)
        temp_array.append(stoich_number)
      
    return(temp_array)

def read_defects(seedname,elements,defects,dopants):

    defectfile = str(seedname)+".defects"
    print(">>> Reading in parameters from ",defectfile)
    total_species = dopants[0] + elements[0]
    charged_system = 0

    #Determine the  minimum number of columns required for each defect
    num_columns = 7 + total_species
    
    #Print header for defect summary table
    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
        print("\n>>> Summary of defects:", file=f)
        print( "   +------------+----------+--------------+------+--------+-------------+------------+","{0}".format('------+'*(int(total_species))),sep="",file=f)   
        
        element_print=''
        for i in np.arange(0, total_species, 1):
            if (i < elements[0]):
                element_print_i=(" n %2s |" % (elements[int(2*i+1)]))
            elif (i >= elements[0]):
                element_print_i=(" n %2s |" % dopants[int(6*(i-elements[0])+1)])
            element_print += element_print_i
        
        print( "   |   Defect   |   Group   | Multiplicity | Site | Charge | Energy /eV  | Correction |",element_print,sep="",file=f)
           
        print("   +------------+-----------+--------------+------+--------+-------------+------------+","{0}".format('------+'*(int(total_species))),sep="",file=f)   

        #Open file containing all the defect information
        file = open(defectfile)
        total_defects = 0
        
        for defect in file:
            
            fields = defect.strip().split()
            if len(fields) ==0:
                print("<!> Blank line detected in",defectfile)
                exit()

            #Prevent dopant defects inclusion if not requested in input file
            skip =0
            if dopants[0] == 0:
                if len(fields) > num_columns:
                    excess_columns = len(fields)-num_columns
                    for i in np.arange(0, excess_columns, 1):
                        i = int(i+1)
                        if fields[-i] != '0':
                            skip =1          
            if skip ==1:
                continue
            
            total_defects +=1
            defects.append(fields)
            if len(fields) < num_columns:
                print("<!> Error : Number of columns insufficient for", fields[0])
                exit()
            else:
                defect_name = fields[0]
                defect_group = fields[1]
                multiplicity = float(fields[2])
                site = int(fields[3])
                charge = float(fields[4])
                energy = float(fields[5])
                tabulated_correction = float(fields[6])
             
                #Quick check to see if overall system is charged
                if (charge != 0):
                    charged_system = 1;

                #Loop over elements and dopants in the host
                element_prints=''
                for i in np.arange(0, total_species, 1):
                    if (i < elements[0]):
                        element_prints_i=(" %4s |" % fields[int(7+i)])
                    elif (i >= elements[0]):
                        element_prints_i=(" %4s |" % fields[int(7+i)])
                    element_prints += element_prints_i
            
                print ("   | %10s | %9s | %12f | %4i | %6i | %11.3f | %10.3f |%2s " % (defect_name,defect_group,multiplicity,site,charge,energy,tabulated_correction,element_prints),file=f)
       
        print( "   +------------+-----------+--------------+------+--------+-------------+------------+","{0}".format('------+'*(int(total_species))),sep="",file=f)   
        print("   Number of defects :",total_defects,file=f)
        
        if (charged_system == 0):
             print ("   Treating system as charge neutral",file=f)
        
        if (charged_system == 1):
             print ("   Treating system as charged",file=f)

    print("..> Defect file read successfully (",total_defects,"defects )")

    return(defects,total_defects,total_species,charged_system)

def read_entropy(seedname):

    entropyfile = str(seedname)+".entropy"
    entropy_data = []
    file = open(entropyfile)
    for line in file:
        fields = line.strip().split()
        entropy_data.append(fields)
    
    return entropy_data

def entropy_check(entropy_data, defect_data, total_defects,constituents_name_list,chem_pot_method,seedname):

    entropyfile = str(seedname)+".entropy"
    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
        print("\n>>> Performing checks on", entropyfile,'\n',file=f)

        if chem_pot_method ==3:
            i = 0
            while i < len(constituents_name_list):
                constituent = constituents_name_list[i]
                entropy_constituent = entropy_data[0][i+2]
                if(entropy_constituent != constituent):
                    print("<!> ERROR: There is a mismatch in the name for the constituent", constituent, "and the entropy", entropy_constituent)
                    print("    Recommend you go back to ensure constituents occur in the same order in the .input and .entropy files and spellings are identical")
                    exit()
                i+=1
            print("   Constituents are listed in same order in .input and .entropy files",file=f)

            i = 0
            while i < total_defects:
                defect = defect_data[i][0]
                entropy_defect = entropy_data[0][i+len(constituents_name_list)+2]
                if(entropy_defect != defect):
                    print("<!> ERROR: There is a mismatch in the name for the defect formation energy", defect, "and the entropy", entropy_defect)
                    print("    Recommend you go back to ensure defects occur in the same order in the .dat and .entropy files and spellings are identical")
                    exit()
                i+=1
           
        else:
            i = 0
            while i < total_defects:
                defect = defect_data[i][0]
                entropy_defect = entropy_data[0][i+2]
                if(entropy_defect != defect):
                    print("<!> ERROR: There is a mismatch in the name for the defect formation energy", defect, "and the entropy", entropy_defect)
                    print("    Recommend you go back to ensure defects occur in the same order in the .dat and .entropy files and spellings are identical")
                    exit()
                i+=1
        print("   Defects are listed in same order in .defects and .entropy files\n",file=f)
            
def calc_entropy(entropy_data,temperature,total_defects,constituents_name_list,chem_pot_method,seedname,prnt):
    #Print table header to the output file
    
    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
        if prnt ==1:
            print("   Vibrational entropy at",temperature, "K",file=f)
            print("   +---------+---------------------+---------------+",file=f)
            print("   |  System |   Entropy /eV K^-1  |  ds /eV K^-1  |",file=f)
            print("   +---------+---------------------+---------------+",file=f)

        i = 0
        entropies =[]
        constituent_entropies =[]
        num_lines = len(entropy_data)

        if chem_pot_method == 3:
            length = total_defects+len(constituents_name_list)+1
            length2 = len(constituents_name_list)+1
        else:
            length = total_defects+1
            length2 = 1
            
        while i < length:
            #Get the system name
            system_name = entropy_data[0][i+1]
            x=[]
            y=[]
            j=0
            #extract the data as a function of temperature
            while j < num_lines-1:
                current_temp = float(entropy_data[j+1][0])
                if(j == 0):
                    min_temp = current_temp
                if(j == (num_lines-2)):
                    max_temp = current_temp
                current_entropy = float(entropy_data[j+1][i+1])
                x.append(current_temp)
                y.append(current_entropy)
                j+=1
                
            #Run a quick check to see whether temperature falls in the range of the data
            if(temperature < min_temp or temperature > max_temp):
                print("<!> WARNING Temperature is outside of range with entropy data provided\n")

            #Now use extrapolation to determine entropy of the perfect/defect system
            tck = interpolate.splrep(x, y)
            final_entropy = interpolate.splev(temperature, tck)
            if system_name == "host":
                perfect_entropy = final_entropy
            if i < length2:
                ds = ' '
                if prnt ==1:
                    print("   | %7s | %19f | %13s |" % (system_name, final_entropy, ds),file=f)
                
                constituent_entropies.append(final_entropy*1)
                   
            else:
                ds = final_entropy - perfect_entropy
                entropies.append(ds)
                if prnt ==1:
                    print("   | %7s | %19f | %13f |" % (system_name, final_entropy, ds),file=f)
            i+=1
        if prnt ==1:
            print("   +---------+---------------------+---------------+\n",file=f)
 
    return(entropies, constituent_entropies)

def madelung_input(seedname):

    filename = str(seedname)+".input"
    print(">>> Reading in parameters from ",filename)
    
    lattice = []	#Array to contain lattice parrallelpiped
    dielectric = []	#Array to contain the dielectric tensor
	
    #Initialise some variables
    gamma = 0.3
    real_space = 20
    total_charge = 1
    debug = 0
    num_atoms =1
    motif = [0,0,0,1]

    with open(filename) as file:
        for linenumber, line in enumerate(file):
            fields = line.strip().split()
            
            if len(fields) !=0:
                name=fields[0]

                #Gamma
                if (name == "Gamma"):
                    gamma = float(fields[2])

                #Cutoff
                if (name == "Cutoff"):
                    real_space = float(fields[2])

                #debug
                if (name == "debug"):
                    debug = float(fields[2])

                #Lattice 
                if (name == "Lattice"):
                    for i in np.arange(1,4,1):
                            with open(filename) as file2:
                                for linenumber2, line2 in enumerate(file2):
                                    fields2 = line2.strip().split()

                                    if linenumber+i == linenumber2:
                                        col1 = float(fields2[0])
                                        col2 = float(fields2[1])
                                        col3 = float(fields2[2])
                                        lattice.append(col1)
                                        lattice.append(col2)
                                        lattice.append(col3)

                #Dielectric
                if (name == "Dielectric"):
                    for i in np.arange(1,4,1):
                            with open(filename) as file4:
                                for linenumber4, line4 in enumerate(file4):
                                    fields4 = line4.strip().split()

                                    if linenumber+i == linenumber4:
                                        col1 = float(fields4[0])
                                        col2 = float(fields4[1])
                                        col3 = float(fields4[2])
                                        dielectric.append(col1)
                                        dielectric.append(col2)
                                        dielectric.append(col3)

                #Motif
                if (name == "Motif"):
                    motif=[]   #Array containing the motif
                    num_atoms = float(fields[1])
                    for i in np.arange(1,num_atoms+1,1):
                        with open(filename) as file3:
                                for linenumber3, line3 in enumerate(file3):
                                    fields3 = line3.strip().split()

                                    if linenumber+i == linenumber3:
                                        motif_x = float(fields3[0])
                                        motif_y = float(fields3[1])
                                        motif_z = float(fields3[2])
                                        charge = float(fields3[3])
                                        motif.append(motif_x)
                                        motif.append(motif_y)
                                        motif.append(motif_z)

                                        if (num_atoms == 1):
                                            charge = 1
                                            #print("Treating as a point charge, therefore, charge defined in motif is being ignored\n")
                
                                        else:
                                            print(motif_x,motif_y,motif_z,charge,"\n")

                                        motif.append(charge)

                                        #Calculate the total charge
                                        total_charge += charge
           

    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
          print("\n-------------------------------------------------------------------------------------------------------------------","\n", file=f)
          print(">>> Task = madelung","\n", file=f)
          print("   Real space lattice:", file=f)
          print("   %.6f  %.6f  %.6f" % (lattice[0], lattice[1], lattice[2]), file=f)
          print("   %.6f  %.6f  %.6f" % (lattice[3], lattice[4], lattice[5]), file=f)
          print("   %.6f  %.6f  %.6f" % (lattice[6], lattice[7], lattice[8]), file=f)
          print("\n   Dielectric tensor:", file=f)
          print("   %.6f  %.6f  %.6f" % (dielectric[0], dielectric[1], dielectric[2]), file=f)
          print("   %.6f  %.6f  %.6f" % (dielectric[3], dielectric[4], dielectric[5]), file=f)
          print("   %.6f  %.6f  %.6f" % (dielectric[6], dielectric[7], dielectric[8]), file=f)
          print("\n   gamma =",gamma, file=f)
          print("   Real space cutoff set to",real_space,"* longest lattice parameter", file=f)
          if (debug == 1):
              print("   Debugging settings enabled", file=f)       
          if (total_charge != 0):
              print("   System has an overall charge of",total_charge, file=f)
              print("   Applying charge neutralising background jellium", file=f)     
                                        
    return(dielectric,lattice,motif,gamma,real_space,num_atoms,total_charge,debug)

def calc_chemical_defined(host_array,constituents,chemical_potentials,host_energy,temperature, entropy_marker, constituent_entropies, entropy_units):
    
    #Define a limit for the discrepancy in chemcial potentials for numerical reasons
    error = 0.001
    
    total_potential = 0
    
    #Loop over host_array and match chemical potentials
    for i in np.arange(0, host_array[0], 1):
        i=int(i)

        current_element = host_array[2*i+1]
        stoichiometric_number = host_array[2*i+2]
        
        #Loop over constituents and match potential
        for j in np.arange(0, host_array[0], 1):
            j=int(j)

            if (constituents[2*j] == current_element):

                chemical_potential = float(constituents[2*j+1])

                chemical_potentials.append(current_element)
                chemical_potentials.append(chemical_potential)
        
                total_potential+=(stoichiometric_number*chemical_potential)
    
    #Compare the total chemical potential of the constituents with the host
    difference = math.sqrt((host_energy-total_potential)**2)
    if (difference == 0):
        pass   
    elif (difference <= error):
        print("<!> Warning : There is a small difference (<",error,") between the sum of the chemical potentials and that of the host")
    else:
        print("<!> Error : The chemical potentials for the constituents do not add up to that for the host system")
        exit()
        
    return (chemical_potentials)

def calc_chemical_rich_poor(host_array,constituents,chemical_potentials,host_energy,temperature, entropy_marker, constituent_entropies, entropy_units):

    #Loop over host_array and match chemical potentials
    for i in np.arange(0, host_array[0], 1):
        i=int(i)

        current_element = host_array[2*i+1]
        stoichiometric_number = float(host_array[2*i+2])
        
        running_total = 0
        
        #Loop over constituents to calculate the checmial potential
        for j in np.arange(0, host_array[0], 1):
            j=int(j)
      
            if (constituents[3*j] == current_element):

                rich_potential = float(constituents[3*j+1])
                xxx = float(constituents[3*j+2])
                #print("Rich potential",rich_potential,xxx)
            
            else:
            
                other = constituents[3*j]

                other_rich = float(constituents[3*j+1])
                
                #Find the stoichiometric number for the 'other' constituent
               
                for k in np.arange(0, host_array[0], 1):
                    k=int(k)

                    if (other == host_array[2*k+1]):

                        other_stoich = host_array[2*k+2]
                
                contribution = other_stoich*other_rich;
                running_total+=contribution
         
        chemical_potential = xxx*rich_potential + (1-xxx)*((host_energy_final - running_total)/stoichiometric_number)
        chemical_potentials.append(current_element)
        chemical_potentials.append(chemical_potential)

    return (chemical_potentials)

def calc_chemical_volatile(host_array,constituents,chemical_potentials,host_energy,temperature, entropy_marker, constituent_entropies, entropy_units,real_gas):

    #Some constants
    std_temp = 298.15
    std_pressue = 0.2
    boltzmann = 0.000086173324
  
    #Extract details for the volatile species
    volatile_species = constituents[0]
    partial_pressure = constituents[1]
    #print("Volatile species is", volatile_species,"with partial pressure of 10^{",partial_pressure,"} atm")
    
    nu_volatile_std = 0
    constituent_definition_total = 0
  
    #Calculate the chemical potential for volatile species under standard contitions  

    denominator = 0
    metal_stoich = 0
    
    formula = constituents[2];
    const_array = break_formula(formula,1)
   
    for j in np.arange(0, const_array[0], 1):
        sub_element = const_array[2*j+1]
        if (sub_element == volatile_species):

            denominator = float(const_array[2*j+2])
            
        else:
            metal_stoich = float(const_array[2*j+2])
     
    constituent_energy_DFT = float(constituents[3])
    constituent_metal_DFT = float(constituents[4])
    constituent_formation = float(constituents[5])

    nu_volatile_std = ((constituent_energy_DFT - (metal_stoich*constituent_metal_DFT) - constituent_formation)/denominator)

    temp_cont = temperature_cont(volatile_species,temperature,real_gas)
    
    #Change partial pressure from a log to atm
    partial_pressure_atm = 1/(10**-partial_pressure )
    pres_cont = (1/2)*boltzmann*temperature*math.log(partial_pressure_atm/std_pressue)
    
    #Calcate volatile element chemical potential under desired conditions
    nu_volatile = nu_volatile_std + temp_cont + pres_cont
  
    #print("At a temperature of",temperature,"K and oxygen partial pressure of",partial_pressure,"atm the chemical potentials for oxygen is:",nu_volatile)
  
    #Now loop over remaining elements in host and determine chemical potentials
    for i in np.arange(0, host_array[0]-1, 1):

        element = host_array[2*i+1];
        running_total = 0

        #Now loop over the constituents array to find the numbers to calculate the chemical potential
        for j in np.arange(0, host_array[0]-1, 1):

            formula = constituents[2]
            const_array = break_formula(formula,1)

            if (element in const_array):

                if(entropy_marker == 1):
                
                    modification = (constituent_entropies[j]*temperature)/entropy_units
                    const_energy = float(constituents[3])-modification #Entropy addition for each constituent constituent
                else:
                    const_energy = float(constituents[3])
                    
                #Loop over formula to extract elemental stoichiometric numbers
                for w in np.arange(0, const_array[0], 1):

                    el_name = const_array[2*w+1]
                    el_stoichiometry = float(const_array[2*w+2])
                    
                    if (el_name == element):

                        denominator2 = float(el_stoichiometry)
                    
                    elif (el_name == volatile_species):
                    
                        volatile_stoich = float(el_stoichiometry)
                             
        chemical_potential = ((const_energy-(volatile_stoich*nu_volatile))/denominator2) 
        chemical_potentials.append(element)
        chemical_potentials.append(chemical_potential)
        
    chemical_potentials.append(volatile_species)
    chemical_potentials.append(nu_volatile)  
    
    return (chemical_potentials)

def calc_chemical_volatile_rich_poor(host_array,constituents,chemical_potentials,host_energy,temperature, entropy_marker, constituent_entropies, entropy_units,real_gas):

    #Some constants
    std_temp = 298.15
    std_pressue = 0.2
    boltzmann = 0.000086173324

    #Extract details for the volatile species
    volatile_species = constituents[0]
  
    gaseous_stoichiometry = constituents[1]
    partial_pressure = constituents[2]
    #print("Volatile species is", volatile_species,"with partial pressure of 10^{",partial_pressure,"} atm")
    
    nu_volatile_std = 0
    constituent_definition_total = 0
    
    #Calculate the chemical potential for volatile species under standard contitions
    for i in np.arange(0, host_array[0]-1, 1):

        denominator = 0
        metal_stoich = 0
        
        formula = constituents[6*i+3];
        const_array = break_formula(formula,1)

        for j in np.arange(0, const_array[0], 1):
            sub_element = const_array[2*j+1]
            if (sub_element == volatile_species):

                denominator = float(const_array[2*j+2])
                
            else:
                metal_stoich = float(const_array[2*j+2])
         
        constituent_energy_DFT = float(constituents[6*i+5])
        constituent_metal_DFT = float(constituents[6*i+6])
        constituent_formation = float(constituents[6*i+7])
        constituent_definition = float(constituents[6*i+8])
        constituent_definition_total +=  constituent_definition

        contribution = constituent_definition * ((constituent_energy_DFT - (metal_stoich*constituent_metal_DFT) - constituent_formation)/denominator)
        
        nu_volatile_std+=contribution

    nu_volatile_std = nu_volatile_std/constituent_definition_total
  
    temp_cont = temperature_cont(volatile_species,temperature,real_gas)
        
    #Change partial pressure from a log to atm
    partial_pressure_atm = 1/(10**-partial_pressure )
    pres_cont = (1/2)*boltzmann*temperature*math.log(partial_pressure_atm/std_pressue)
    
    #Calcate volatile element chemical potential under desired conditions
    nu_volatile = nu_volatile_std + temp_cont + pres_cont
    
    #print("At a temperature of",temperature,"K and oxygen partial pressure of",partial_pressure,"atm the chemical potentials for oxygen is:",nu_volatile)

    #Modify the energy of the host to include vibrational if entropy contribution is used
    if(entropy_marker == 1):
        modification = (constituent_entropies[0]*temperature)/entropy_units
    else:
        modification = 0
    host_energy_final = host_energy - modification
  
    #Now loop over remaining elements in host and determine chemical potentials
    for i in np.arange(0, host_array[0]-1, 1):

        element = host_array[2*i+1];
        running_total = 0

        #Now loop over the constituents array to find the numbers to calculate the chemical potential
        for j in np.arange(0, host_array[0]-1, 1):


            formula = constituents[6*j+3]
            const_array = break_formula(formula,1)

            if (element in const_array):

                if(entropy_marker == 1):
                    modification = (constituent_entropies[j+1]*temperature)/entropy_units
                    const_energy = float(constituents[6*j+5])-modification #Entropy addition for each constituent                  
                else:
                    const_energy = float(constituents[6*j+5])
            
                const_stoich = float(constituents[6*j+4])
                const_defined = float(constituents[6*j+8])
                
                #Loop over formula to extract elemental stoichiometric numbers
                for w in np.arange(0, const_array[0], 1):

                    el_name = const_array[2*w+1]
                    el_stoichiometry = float(const_array[2*w+2])
                     
                    if (el_name == element):

                        denominator2 = float(el_stoichiometry)
                    
                    elif (el_name == volatile_species):
                    
                        volatile_stoich = float(el_stoichiometry)
                  
            else:
                if(entropy_marker == 1):
                    modification = (constituent_entropies[j+1]*temperature)/entropy_units
                    other_pot = float(constituents[6*j+5])-modification #Entropy addition for each constituent                  
                else:
                    other_pot = float(constituents[6*j+5])
               
                other_stoich = float(constituents[6*j+4])
                contribution = other_pot*other_stoich
                running_total += contribution
           
        chemical_potential = const_defined * ((const_energy-(volatile_stoich*nu_volatile))/denominator2) + (1-const_defined) * (((host_energy_final-running_total-(gaseous_stoichiometry*nu_volatile))/const_stoich-(volatile_stoich*nu_volatile))/denominator2);
        chemical_potentials.append(element)
        chemical_potentials.append(chemical_potential)
        
    chemical_potentials.append(volatile_species)
    chemical_potentials.append(nu_volatile)  

    return (chemical_potentials)

def temperature_cont(volatile_species,temperature,real_gas):

    std_temp = 298.15
    #All parmaters unless stated are taken from the NIST Chemistry WebBook. 
    if (volatile_species == "O"):
        entropy = 0.00212622
        heat_capacity = 0.000304546

        #legacy mode
        legacy =0
        if legacy ==1:
            entropy = 0.00212477008
            heat_capacity = 0.000302
                
        if(real_gas == 1):
            if temperature >= 100 and temperature <=700:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.000324659,-0.000209741,0.000599791,-0.00037839,-7.64321E-08,-9.22852E-05,0.002558046           
            elif temperature > 700 and temperature <=2000:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.000311288,9.09326E-05,-4.13373E-05,8.17093E-06,-7.68674E-06,-0.000117381,0.002447884
            elif temperature > 2000 and temperature <=6000:
                aaa,bbb,ccc,ddd,eee,fff,ggg =0.000216745,0.000111121,-2.09426E-05,1.51796E-06,9.58327E-05,5.53252E-05,0.002462936
            else:
                print("<!> Cannot use real gas parameters at temperature of",temperature,"K")
                exit()
            
    elif (volatile_species == "N"):
        entropy = 0.00198589	 
        heat_capacity = 0.00030187
        
        if(real_gas == 1):
            if temperature >= 100 and temperature <=500:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.000300447,1.92166E-05,-9.99967E-05,0.000172427,1.21271E-09,-8.98851E-05,0.002346829  
            elif temperature > 500 and temperature <=2000:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.00020218,0.000206131,-8.91245E-05,1.41979E-05,5.46863E-06,-5.11538E-05,0.00220144
            elif temperature > 2000 and temperature <=6000:
                aaa,bbb,ccc,ddd,eee,fff,ggg =0.000368155,1.16994E-05,-2.03262E-06,1.51973E-07,-4.72001E-05,-0.000196635,0.002331947
            else:
                print("<!> Cannot use real gas parameters at temperature of",temperature,"K")
                exit()

    elif (volatile_species == "H"):
        entropy = 0.00135436 
        heat_capacity = 0.000298891

        if(real_gas == 1):
            if temperature >= 100 and temperature <=1000:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.000342734,-0.000117783,0.000118502,-2.87411E-05,-1.64347E-06,-0.000103452,0.001790133
            elif temperature > 1000 and temperature <=2500:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.000192408,0.000127049,-2.96419E-05,2.78031E-06,2.0502E-05,-1.18933E-05,0.00161994
            elif temperature > 2500 and temperature <=6000:
                aaa,bbb,ccc,ddd,eee,fff,ggg =0.000449985,-4.44981E-05,1.31888E-05,-1.00413E-06,-0.000212835,-0.000399213,0.001679987
            else:
                print("<!> Cannot use real gas parameters at temperature of",temperature,"K")
                exit()
                
    elif (volatile_species == "F"):
        entropy = 0.00210186
        heat_capacity = 0.000324774

        if(real_gas == 1):
            if temperature >= 298 and temperature <=6000:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.000325931,8.72101E-05,-2.8803E-05,2.26067E-06,-2.18885E-06,-0.000108135,0.002459396
            else:
                print("<!> Cannot use real gas parameters at temperature of",temperature,"K")
                exit()

    elif (volatile_species == "Cl"):
        entropy = 0.00231205
        heat_capacity = 0.000351828

        if(real_gas == 1):
            if temperature >= 298 and temperature <=1000:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.000342572,0.000126759,-0.000125056,4.54543E-05,-1.65317E-06,-0.000112304,0.002684858
            elif temperature > 1000 and temperature <=3000:
                aaa,bbb,ccc,ddd,eee,fff,ggg = 0.000442354,-5.19246E-05,1.97416E-05,-1.71688E-06,-2.17509E-05,-0.00017921,0.002796914
            elif temperature > 3000 and temperature <=6000:
                aaa,bbb,ccc,ddd,eee,fff,ggg =-0.000441071,0.000432076,-7.38702E-05,4.01998E-06,0.001048366,0.00137611,0.002744529
            else:
                print("<!> Cannot use real gas parameters at temperature of",temperature,"K")
                exit()
    else:
        
        print("<!> Cannot calculate chemical potential for chosen volatile element")
        exit()
         
    #Calculate contributions from temperature depending on whether using a real or ideal gas
    if(real_gas == 0):
        temp_cont = -(1/2)*(entropy-heat_capacity)*(temperature-std_temp) + (1/2)*heat_capacity*temperature*math.log(temperature/std_temp) 

    elif(real_gas == 1):

        temperature_i  = temperature/1000
        std_temp_i = std_temp/1000
        
        H0=1000*aaa*std_temp_i + (1/2)*1000*bbb*std_temp_i**2 + (1/3)*1000*ccc*std_temp_i**3 + (1/4)*1000*ddd*std_temp_i**4 - (1000*eee)/(std_temp_i) + 1000*fff

        H=1000*aaa*temperature_i + (1/2)*1000*bbb*temperature_i**2 + (1/3)*1000*ccc*temperature_i**3 + (1/4)*1000*ddd*temperature_i**4 - (1000*eee)/(temperature_i) + 1000*fff 
        
        S0= aaa*math.log(std_temp_i) + bbb*std_temp_i + (1/2)*ccc*std_temp_i**2 + (1/3)*ddd*std_temp_i**3 - eee/(2*std_temp_i**2) + ggg
        
        S= aaa*math.log(temperature_i) + bbb*temperature_i + (1/2)*ccc*temperature_i**2 + (1/3)*ddd*temperature_i**3 - eee/(2*temperature_i**2) + ggg
       
        G0 = H0-std_temp*S0

        G = H-temperature*S

        temp_cont = (G-G0)/2
        
    elif(real_gas == 2): #Specific mode for oxygen and use of Johnston et al. parameters PHYSICAL REVIEW B 70, 085415

        aaa = 3.074E-4
        bbb = 6.36066E-8
        ccc = -1.22974E-11
        ddd = 9.927E-16
        eee = -2.2766
        fff = -1.022061
        ggg = 0.0024661578656

        G0 = aaa*(std_temp-std_temp*math.log(std_temp)) - (1/2)*bbb*std_temp**2 - (1/6)*ccc*std_temp**3 - (1/12)*ddd*std_temp**4 - eee/(2*std_temp) + fff - ggg*std_temp
            
        G = aaa*(temperature-temperature*math.log(temperature)) - (1/2)*bbb*temperature**2 - (1/6)*ccc*temperature**3 - (1/12)*ddd*temperature**4 - eee/(2*temperature) + fff - ggg*temperature
        
        temp_cont = (G-G0)/2
   
    return temp_cont

def stability_check(stability,chemical_potentials,indicator,b):

    stability_printout=[]
    #Loop over all constituents

    for i in np.arange(0,stability[0], 1):
        i = int(i)
        constituent = stability[3*i+1]
        formation = stability[3*i+2]
        constituent_breakdown = stability[3*i+3]
        contribution = 0
        stability_printout_i = []
        stability_printout_i.append(constituent)
        stability_printout_i.append(formation)
  
        #Loop over elements in each constituent
        for j in np.arange(0,constituent_breakdown[0], 1):
            j = int(j)
           
            element = constituent_breakdown[2*j+1]
            stoic = float(constituent_breakdown[2*j+2])                 

            #Search chemical potentials for matching element
            for k in np.arange(0,len(chemical_potentials)/2, 1):
                k = int(k)
                element_i = chemical_potentials[2*k]
                pot = chemical_potentials[2*k+1]      
            
                if element == element_i:

                    contribution += (pot * stoic)

        potential_diff = contribution-formation
        if potential_diff > 0:
            entry = "WARNING, unstable"
            indicator = 1
        else:
            entry = "Stable"
        stability_printout_i.append(contribution)
        stability_printout_i.append(potential_diff)
        stability_printout_i.append(entry)
        stability_printout.append(stability_printout_i)
   
    return stability_printout,indicator
    
def dopant_chemical(dopants,chemical_potentials,temperature,real_gas):

    #Some constants
    std_pressue = 0.2
    boltzmann = 0.000086173324

    number_dopants = dopants[0]
    opt_chem_pot = 0 
    #Loop over all dopants
   
    for i in np.arange(0,number_dopants, 1):
   
        running_pot_total = 0;
        
        target = dopants[int((6*i)+1)]
        reference_state_energy = float(dopants[int((6*i)+2)])
        potential_method = int(dopants[int((6*i)+3)])
        reference_breakdown = dopants[int((6*i)+5)]
        num_element_ref = reference_breakdown[0]

        #identify if optimise of dopant chemical potential is requested.
        if potential_method == 1 or potential_method ==  2:
            opt_chem_pot =1

        if potential_method == 3 or potential_method == 4:   
            partial_pressure = float(dopants[int((6*i)+6)])
            
        if potential_method != 3 or potential_method != 4:
            #Loop over elements in reference state
            for j in np.arange(0,num_element_ref, 1):

                element =reference_breakdown[int((2*j)+1)]
                
                if (element == target):

                    denominator = float(reference_breakdown[int((2*j)+2)])
                    
                else:

                    stoich_number = float(reference_breakdown[int((2*j)+2)])
                                      
                    #Find the chemical potential for element in chemical_potentials
                    elements_in_list = (len(chemical_potentials))/2
                    for w in np.arange(0,elements_in_list, 1):
                        ref_element = chemical_potentials[int(2*w)];

                        if (ref_element == element):

                            contribution = stoich_number*float(chemical_potentials[int(2*w+1)])
                            running_pot_total += contribution
                  
            final_chemical = (reference_state_energy - running_pot_total)/denominator

        if potential_method == 3 or potential_method == 4:
            #Loop over elements in reference state
            for j in np.arange(0,num_element_ref, 1):

                element =reference_breakdown[int((2*j)+1)]
                
                if (element == target):

                    denominator = float(reference_breakdown[int((2*j)+2)])
                    
                else:

                    stoich_number = float(reference_breakdown[int((2*j)+2)])
                                      
                    #Find the chemical potential for element in chemical_potentials
                    elements_in_list = (len(chemical_potentials))/2
                    for w in np.arange(0,elements_in_list, 1):
                        ref_element = chemical_potentials[int(2*w)];

                        if (ref_element == element):

                            contribution = stoich_number*float(chemical_potentials[int(2*w+1)])
                            running_pot_total += contribution
                  
            nu_volatile_std  = (reference_state_energy - running_pot_total)/denominator

            temp_cont = temperature_cont(target,temperature,real_gas)
            
            #Change partial pressure from a log to atm
            partial_pressure_atm = 1/(10**-partial_pressure )
            pres_cont = (1/2)*boltzmann*temperature*math.log(partial_pressure_atm/std_pressue)
            
            #Calcate volatile element chemical potential under desired conditions
            final_chemical = nu_volatile_std + temp_cont + pres_cont

        chemical_potentials.append(target)
        chemical_potentials.append(final_chemical)
   
    return chemical_potentials, opt_chem_pot

def calc_opt_chem_pot(b,loop,defects,dopants,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys, log_diff_conv,function_tol,maxiter_dop,environment,prog_meter,prog_bar,num_iter):         
      
    number_dopants = int(dopants[0])
    host_atoms = total_species- number_dopants

    #Determine number and postion of elements that are to be fitted 
    position=[]
    
    for i in np.arange(0,number_dopants, 1):

          fit_potential = float(dopants[int((6*i)+3)])
          if fit_potential != 0:
              position.append(i+host_atoms)
    
    #Determine number of fitted dopant(s) defects
    if len(defects[0]) ==7+total_species:
       
        signals_master=[]
        
        for j in np.arange(0,number_of_defects, 1):
            signals =[]
            signal_ii=0
            for k in np.arange(0,number_dopants, 1):
                signal_i = float(defects[int(j)][int(7+host_atoms+k)])
                if signal_i != 0:
                    signal = -1
                    signal_ii=-signal_i 
                    
                else:
                    signal = 0
                
                signals.append(signal)
                
            #Check to see whether this signal has been found before
            if (signals in signals_master):
                defects[int(j)].append(signals_master.index(signals)) #Defects with the same 'signal' summed later
                defects[int(j)].append(signals)                       #Used to retrieve correct dopant sum. 
                defects[int(j)].append(signal_ii) 
            else:
                signals_master.append(signals)
                defects[int(j)].append(signals_master.index(signals))
                defects[int(j)].append(signals)
                defects[int(j)].append(signal_ii) 

    dp_list = []
    for w in np.arange(0,len(position), 1):
            w=int(w)
            dp = chemical_potentials[int(2*(position[w]))]  
            dp_list.append(dp)
   
    #One dopant to optimise
    if len(position)==1:
        optimiser =1
       
        #Extract dopant to optimise, target conc and range
        nudp = chemical_potentials[int(2*(position[0])+1)]
        
        target_conc = float(dopants[int((6*(position[0]-host_atoms))+4)])
        dopant_range = float(dopants[int((6*(position[0]-host_atoms))+6)])
        
        #Create 'key' that corresponds to the dopant defects in .defect file. 
        key = number_dopants*[0]
        key[(int(position[0]-host_atoms))] = -1
    
        i = nudp-dopant_range
        j = nudp+dopant_range
        bnds=[(i,j)]
        conc_diff = 1
        log_conc_diff=1
        iteration = 1
       
        #Perform a check to ensure a root lies in the range i - j (also store values for the conc_diff at the initial i and j)
        fail =1
        while fail ==1:
            
            chemical_potentials[int(2*(position[0])+1)]=i
            x=[i]
            defects_form = defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,1)       
            (nu_e_final,concentrations,dopant_concentration_sums,fail) = calc_fermi_dopopt(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,x, bnds,dp_list,optimiser)
            if fail ==1:
                i +=0.05        
         
        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key)-1)]
        initial = dopant_concentration_sum - target_conc
        
        fail =1
        while fail ==1:
            
            chemical_potentials[int(2*(position[0])+1)]=j
            x=[j]
            defects_form = defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,1)       
            (nu_e_final,concentrations,dopant_concentration_sums,fail) = calc_fermi_dopopt(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,x, bnds,dp_list,optimiser)
            if fail ==1:
                j -= 0.05
           
        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key)-1)]
        final = dopant_concentration_sum - target_conc

        if j<=i:
            print("<!> No chemical potential in the specific range can give the requested defect concentraton!")
            exit()
        sign = initial*final;	
        if(sign > 0):
            print("<!> No chemical potential in the specific range can give the requested defect concentraton!")
            print("    I reccommend you increase Dopant_range from its current value of", dopant_range,"eV, if this fails you may need to revisit the chemical potential from which your dopant chemical potential is derived")
            exit()
               
        lower = initial			
        upper = final
        
        #Perform linear biesction search to find the chemical potential that gives the desired dopant concentration
        while(log_conc_diff > log_diff_conv):
            midpoint = (i+j)/2
            chemical_potentials[int(2*(position[0])+1)]=midpoint
            x=[midpoint]
            defects_form = defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,1)       
            (nu_e_final,concentrations,dopant_concentration_sums,fail) = calc_fermi_dopopt(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,x, bnds,dp_list,optimiser)
            dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key)-1)]
            conc_diff = dopant_concentration_sum-target_conc
            log_conc_diff=((math.log(target_conc)-math.log(dopant_concentration_sum))**2)**0.5
            if(lower*conc_diff < 0):
                j= midpoint
                upper = conc_diff
            if(upper*conc_diff < 0):
                i = midpoint
                lower = conc_diff
                 
            #print(iteration,i,j,dopant_concentration_sum, conc_diff)
            iteration+=1
        
        #Result:
        nudp_final = midpoint

    #Multiple dopant elements to optimise
    else:
        optimiser = 0
  
        #Extract dopant to optimise, target conc and range
    
        nudp_list=[]
        target_conc_list= []
        dopant_range_list= []
        key_list = []
        bnds =[]
        global iteration_slsqp
        iteration_slsqp=0
        for k in np.arange(0,len(position), 1):
            k=int(k)
           
            nudp = chemical_potentials[int(2*(position[k])+1)]
            target_conc = float(dopants[int((6*(position[k]-host_atoms))+4)])
            dopant_range = float(dopants[int((6*(position[k]-host_atoms))+6)])
            key = number_dopants*[0]
            key[(int(position[k]-host_atoms))] = -1
            lower = nudp -dopant_range
            upper =nudp +dopant_range
            bnd = (lower,upper)
            nudp_list.append(nudp)

            target_conc_list.append(target_conc)
            dopant_range_list.append(dopant_range)
            key_list.append(key)
            bnds.append(bnd)

        #Initial guess
        x0 = nudp_list

        #Impose constraints
        cons = ({'type':'ineq','fun':constraint,'args':[ target_conc_list, b,loop,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,position,key_list,bnds,dp_list,log_diff_conv,optimiser] })

        #Minimise function. (Minimising the difference between each dopant concentration and its target)
        sol = minimize(calc_opt_chem_pot_multidim, x0,args= (target_conc_list, b,loop,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,position,key_list,bnds,dp_list,optimiser,environment,prog_meter,prog_bar,num_iter), method = 'SLSQP',bounds =bnds,constraints= cons ,options={'ftol':function_tol,'disp':False,'maxiter':maxiter_dop})
        
        #Solution 
        xOpt= sol.x

        #Check the output of the optimiser that concentrations are correct. 
        for w in np.arange(0,len(position), 1):
            w=int(w)
            chemical_potentials[int(2*(position[w])+1)]=xOpt[w]
     
        defects_form = defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,1)       
        (nu_e_final,concentrations,dopant_concentration_sums,fail) = calc_fermi_dopopt(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,xOpt, bnds,dp_list,optimiser)

        minimise_array = len(position)*[0]  #This array will hold the difference between each dopants current concentration and its target concentration
    
        for j in np.arange(0,len(position), 1):
            j=int(j)
            key = key_list[j]
            dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key)-1)]
            target_conc = target_conc_list[j]
            #minimise_array[j] = (((target_conc)-(dopant_concentration_sum))**2)
            minimise_array[j] = ((math.log(target_conc)-math.log(dopant_concentration_sum))**2)**0.5 
            product = max(minimise_array)
         
            if product > (2*log_diff_conv):
                dopant_fail(xOpt, bnds,dp_list,2)

        if environment == 'energy':
            print("\n") #Improving printout

    #Return chemical potential array, with optimised dopant chemical potential now included.
        
    return chemical_potentials

def calc_opt_chem_pot_multidim(x, target_conc_list, b,loop,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,position,key_list, bnds,dp_list,optimiser,environment,prog_meter,prog_bar,num_iter):

    for i in np.arange(0,len(position), 1): #Update Chemical potentials with new trial
        i=int(i)
        chemical_potential = x[i]
        
        chemical_potentials[int(2*(position[i])+1)]=chemical_potential

    defects_form = defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,1)       
    (nu_e_final,concentrations,dopant_concentration_sums,fail) = calc_fermi_dopopt(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,x, bnds,dp_list,optimiser)

    minimise_array = len(position)*[0]  #This array will hold the difference between each dopants current concentration and its target concentration
    minimise_array2 = len(position)*[0]
    
    for j in np.arange(0,len(position), 1):
        j=int(j)
        key = key_list[j]
        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key)-1)]
        target_conc = target_conc_list[j]
        minimise_array[j] = (abs((target_conc)-(dopant_concentration_sum)))   #Will minimise the maximum value in this array, aiming for all to be zero.
        minimise_array2[j] = ((math.log(target_conc)-math.log(dopant_concentration_sum))**2)**0.5
        
    global iteration_slsqp
    iteration_slsqp+=1

    if environment == 'energy':
        print("..> SLSQP:", iteration_slsqp, ", max(log10([target])-log10([present])):",max(minimise_array2),'        ', end="\r", flush=True)

    else:
        print("..> Calculating defect concentrations for",environment,prog_meter, "of", num_iter," [{0}]   ".format('#' * (prog_bar) + ' ' * (25-prog_bar)),"SLSQP:", iteration_slsqp, ", max(log10([target])-log10([present])):",max(minimise_array2),'       ', end="\r", flush=True)

    return max(minimise_array)
    
def constraint(x, target_conc_list, b,loop,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,position,key_list, bnds,dp_list,log_diff_conv,optimiser):

    for i in np.arange(0,len(position), 1): #Update Chemical potentials with new trial
        i=int(i)
        chemical_potential = x[i]
        chemical_potentials[int(2*(position[i])+1)]=chemical_potential

    defects_form = defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,1)       
    (nu_e_final,concentrations,dopant_concentration_sums,fail) = calc_fermi_dopopt(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,number_dopants,x, bnds,dp_list,optimiser)

    minimise_array = len(position)*[0] #This array will hold the difference between each dopants current concentration and its target concentration

    for j in np.arange(0,len(position), 1):
        j=int(j)
        key = key_list[j]
        dopant_concentration_sum = dopant_concentration_sums[(dopant_concentration_sums.index(key)-1)]
        target_conc = target_conc_list[j]
        minimise_array[j] = ((math.log(target_conc)-math.log(dopant_concentration_sum))**2)**0.5
        product = sum(minimise_array)   #Will minimise the sum of the array, aiming for all to be zero.
 
    return (log_diff_conv- (number_dopants*sum(minimise_array)))

def calc_fermi_dopopt(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,dopants_opt,x, bnds,dp_list,optimiser):

    fail = 0 

    #Check that the point at which charge neutrality occurs falls in the bandgap

    #VBM   
    total_charge, concentrations,dopant_concentration_sum = calc_charge(defects_form,defects, number_of_defects, 0, bandgap,condband,valband,temperature,art_dop_conc,art_dop_charge,def_statistics,electron_method,fixed_e_conc,hole_method,fixed_p_conc,entropy_marker,entropies,seedname,cond_band_min,cond_band_max,val_band_min,val_band_max,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,dopants_opt)
    
    #This checks if a math error has occured (i.e. def form eng too low) 
    if dopant_concentration_sum  == 'flag' :
        fail =1
        return 1, concentrations,dopant_concentration_sum, fail

    if(total_charge < 0):
      
        if optimiser ==1:
            fail =1
            return 1, concentrations,dopant_concentration_sum, fail
        
        else:
            dopant_fail(x, bnds,dp_list,1)

    #CBM
    total_charge, concentrations, dopant_concentration_sum = calc_charge(defects_form,defects, number_of_defects, bandgap, bandgap,condband,valband,temperature,art_dop_conc,art_dop_charge,def_statistics,electron_method,fixed_e_conc,hole_method,fixed_p_conc,entropy_marker,entropies,seedname,cond_band_min,cond_band_max,val_band_min,val_band_max,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,dopants_opt)

    #This checks if a math error has occured (i.e. def form eng too low) 
    if dopant_concentration_sum  == 'flag' :
        fail =1
        return 1, concentrations,dopant_concentration_sum, fail
   
    if(total_charge > 0):
   
        if optimiser ==1:
            fail =1
            return 1, concentrations,dopant_concentration_sum, fail
           
        else:
            dopant_fail(x, bnds,dp_list,1)
   
    total_charge =1
    i = 0
    j = bandgap
    counter = 0
   
    while(total_charge > charge_convergence or total_charge < -charge_convergence):
        
        midpoint = (i+j)/2
        total_charge, concentrations, dopant_concentration_sum= calc_charge(defects_form,defects, number_of_defects,midpoint,bandgap,condband,valband,temperature,art_dop_conc,art_dop_charge,def_statistics,electron_method,fixed_e_conc,hole_method,fixed_p_conc,entropy_marker,entropies,seedname,cond_band_min,cond_band_max,val_band_min,val_band_max,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,dopants_opt)
        if(total_charge > 0):
            i = midpoint
            counter+=1
        elif(total_charge < 0):
            j = midpoint;
            counter+=1
    
        #print(midpoint,total_charge, charge_convergence)
        if (counter>100):
            if optimiser ==1:
                fail =1
                return 1, concentrations,dopant_concentration_sum, fail
            else:
                
                dopant_fail(x, bnds,dp_list,1)

    if charged_sys == 0:
        nu_e_final =0
    else:
        nu_e_final = midpoint

    return nu_e_final, concentrations,dopant_concentration_sum, fail

def dopant_fail(x, bnds,dp_list,fail):
    
    if fail ==1:
        print("\n<!> Optimisation of dopant chemical potential(s) terminated unsucessfully")
        print("    Could not calculate satisfactory Fermi level at following conditions:")
        print("   +--------+--------------------------------+-----------------+-----------------+")
        print("   | Dopant | Current chemical potential (eV)| Lower bound (eV)| Upper bound (eV)|")
        print("   +--------+--------------------------------+-----------------+-----------------+")
        for i in np.arange(0, len(x), 1):
            i = int(i)
            print("   | %6s | %30f | %15f | %15f |" % (dp_list[i], x[i],bnds[i][0],bnds[i][1]))

        print("   +--------+--------------------------------+-----------------+-----------------+")
        print("    Bounds or dopant reference energy should be altered to make these chemical potential(s) unattainable.")

    elif fail ==2:
        print("\n<!> Unable to calculate requested concentration of dopant(s)")
        print("    The following dopant chemical potential(s) represent the closest optimiser could achieve to request:")
        print("   +--------+--------------------------------+-----------------+-----------------+")
        print("   | Dopant | Current chemical potential (eV)| Lower bound (eV)| Upper bound (eV)|")
        print("   +--------+--------------------------------+-----------------+-----------------+")
        for i in np.arange(0, len(x), 1):
            i = int(i)
            print("   | %6s | %30f | %15f | %15f |" % (dp_list[i], x[i],bnds[i][0],bnds[i][1]))

        print("   +--------+--------------------------------+-----------------+-----------------+")
        print("    If printout shows that current chemical potential is at a boundary, consider increasing bounds or changing dopant reference energy.")
        print("    If not at boundary, recommend decreasing 'Tolerence' or increasing 'max_iteration'/'Potential_convergence'.")
        print("    Alternatively, no solution may be possible at the current conditions.")
  
    exit()
    
def defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,use_tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,opt_chem):
    
    defects_form= []
    
    #Define useful constants
    alpha = 2.8373
    electro_static_conv = 14.39942

    #loop over each defect
    for i in np.arange(0,number_of_defects, 1):
        defect_name = defects[int(i)][0]
        group = defects[int(i)][1]
        multiplicity = float(defects[int(i)][2])
        site = float(defects[int(i)][3])
        charge = float(defects[int(i)][4])   
        energy = float(defects[int(i)][5])
        correction = float(defects[int(i)][6])

        #Calculate defect formation energy
        def_form_energy = energy - host_supercell + charge*E_VBM

        #Add the chemical potentials to the defect formation energy
        for j in np.arange(0,total_species, 1):

            chem_pot_cont = float(defects[int(i)][int(7+j)]) * chemical_potentials[int(2*j+1)]
            def_form_energy += chem_pot_cont
            #print(defect_name, def_form_energy, chem_pot_cont)

        #Coulombic correction
        if (use_coul_correction == 1):

            mp_correction = electro_static_conv * ((charge**2 * alpha)/(2*length*dielectric))     #Cubic systems only           
            def_form_energy += mp_correction
            
        if (use_coul_correction == 2):

            mp_correction = electro_static_conv * ((charge**2 * v_M)/2)
            def_form_energy += mp_correction

        #Tabulated correction
        if (use_tab_correction == 1):

            def_form_energy += correction

        #Optimised chemical potential
        
        if opt_chem == 1:
            defects_form.append([defect_name,group,multiplicity,site,charge,def_form_energy,float(defects[int(i)][-3]),defects[int(i)][-2],float(defects[int(i)][-1])])
        else:
            defects_form.append([defect_name,group,multiplicity,site,charge,def_form_energy])
            
    return(defects_form)
    
def fermi_dirac(nu_e,seedname,temperature,elec_or_hole,minimum,maximum,fu_uc):
    
    #Open the $seedname.dos file
    dosfile = str(seedname)+".dos"
    dos_data = [[],[]]
    file = open(dosfile)
    for line in file:
        fields = line.strip().split()
        if len(fields) ==0:
            print("\n<!> Blank line detected in",dosfile)
            exit()
        x, y = fields
      
        dos_data[0].append(x)
        dos_data[1].append(y)

    #Determine the number of records in $seedname.dos
    num_records = len(dos_data[0])
    
    boltzmann = 0.000086173324
    
    #Determine the spacing
    step1 = float(dos_data[0][0])
    step2 = float(dos_data[0][1])
    dE = step2-step1
    
    running_total = 0
    
    #Now loop through file and calculate
    i =0
    while(i<num_records):
        energy = float(dos_data[0][i])
        states = (float(dos_data[1][i]))/fu_uc
        
        #Determine whether this is within the range defined
        if(energy >= minimum and energy <= maximum):
            #Calculate the contribution to the electron concentration
            contribution = 0
            if(elec_or_hole == 0):                  #Electrons
                contribution = states * (dE/(1 + math.exp(((energy-nu_e)/(boltzmann*temperature)))))
               
            if(elec_or_hole == 1):                  #Holes
                contribution = states * (dE/(1 + math.exp(((nu_e-energy)/(boltzmann*temperature)))))
        
            running_total += contribution
        i+=1
    
    return running_total

def eff_mass(temperature,mass_eff):

    if isinstance(mass_eff,float):
        pass
    else:
        x,y= [],[]
        for i in np.arange(0,len(mass_eff), 1):
            i = int(i)
            x.append(mass_eff[i][0])
            y.append(mass_eff[i][1])
                
        tck = interpolate.splrep(x, y)
        mass_eff = interpolate.splev(temperature, tck)    
   
    return(mass_eff)
    
def calc_charge(defects_form, defects,number_of_defects, nu_e, bandgap,condband,valband,temperature,art_dop_conc,art_dop_charge,def_statistics,electron_method,fixed_e_conc,hole_method,fixed_p_conc,entropy_marker,entropies,seedname,cond_band_min,cond_band_max,val_band_min,val_band_max,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,dopants_opt):

    #Some constants
    boltzmann = 0.000086173324
    
    #SI constants for working with electron/hole method 4. 
    boltzmann_SI = 1.380649E-23
    planck_SI = 6.62607015E-34

    concentrations =[]
    dopant_concentration_sum=[0]*2*(int(dopants_opt+1))
    
    #Calculate electron and hole contributions to the total charge
    #electrons
    if(electron_method == 0):           #Off
        electrons = 0
    elif(electron_method == 1):         #Boltzmann
        electrons = condband * math.exp(-((bandgap-nu_e)/(temperature*boltzmann)))
    elif(electron_method == 2):         #Fermi-Dirac
        electrons = fermi_dirac(nu_e,seedname,temperature,0,cond_band_min,cond_band_max,fu_uc)
    elif(electron_method == 3):         #Fixed concentration
        electrons = fixed_e_conc
    elif(electron_method == 4):         #Effective masses
       
        electron_mass_eff=eff_mass(temperature, electron_mass_eff)
        
        unit_vol_SI = unit_vol * 1E-30
        N_c = (2*((2*math.pi*electron_mass_eff*9.11E-31*boltzmann_SI*temperature)/(planck_SI**2))**(3/2))
        electrons = ((N_c*unit_vol_SI)/fu_uc) * math.exp(-((bandgap-nu_e)/(temperature*boltzmann)))
        
    #holes
    if(hole_method == 0):           #Off
        holes =0
    elif(hole_method == 1):         #Boltzmann
        holes = valband * math.exp(-((nu_e)/(temperature*boltzmann)))   
    elif(hole_method == 2):         #Fermi-Dirac
        holes = fermi_dirac(nu_e,seedname,temperature,1,val_band_min,val_band_max,fu_uc)
    elif(hole_method == 3):         #Fixed concentration
        holes = fixed_p_conc
    elif(hole_method == 4):         #Effective masses
    
        hole_mass_eff=eff_mass(temperature,hole_mass_eff)
        
        unit_vol_SI = unit_vol * 1E-30
        N_v = (2*((2*math.pi*hole_mass_eff*9.11E-31*boltzmann_SI*temperature)/(planck_SI**2))**(3/2))
        holes = ((N_v*unit_vol_SI)/fu_uc) * math.exp(-((nu_e)/(temperature*boltzmann)))
 
    total_charge = -1*electrons + holes 
    
    #Convert electron and hole concentrations to log values
    if(electron_method != 0):
        electrons = math.log(electrons)/math.log(10)
    if(hole_method != 0):
        holes = math.log(holes)/math.log(10)

    concentrations.append(electrons)
    concentrations.append(holes)
    
    #Add the contribution from an aritificial dopant
    total_charge += (art_dop_conc*art_dop_charge)
    
    #Loop over all defects and calculate concentration and contribution to the total charge
    for i in np.arange(0,number_of_defects, 1):
        #Read in details of the defect from the defects.dat file
        defect_name = defects_form[int(i)][0]
        multiplicity = float(defects_form[int(i)][2])
        site = int(defects_form[int(i)][3])
        charge = float(defects_form[int(i)][4])
        form_energy_vbm = float(defects_form[int(i)][5])

        #Calculate formation energy at $nu_e
        def_form_energy = form_energy_vbm + (charge*nu_e)
        	
        #Check to see whether the calculated defect formation energies are reasonable
        if(def_form_energy > 100 or def_form_energy < -100):
            print("<!> Error: Defect formation energy falls outside reasonable limits")
            print("    ",defect_name, charge, "has formation energy of", def_form_energy,"eV")
            print("    Check whether the host lattice has been defined correctly if so then you may need to revisit your DFT energies")
            exit()
      
        #Prevent math error:
        if (dopants_opt ==1) and ((-def_form_energy/(temperature*boltzmann)) > 705):
            return 'flag','flag','flag'            
            
        #Calculate the concentration and consequent contribution to total charge
        if(def_statistics== 0):           #Simple Boltzmann statistics
            
            concentration = multiplicity * math.exp(-def_form_energy/(temperature*boltzmann))
        
            if(entropy_marker == 1):
              
                concentration = concentration * math.exp(entropies[i]/boltzmann)

        if(def_statistics == 1):           #Kasamatsu statistics
            competing =0
            for j in np.arange(0,number_of_defects, 1):
                #Loop over all defects and determine if competing for the same site
             
                site2 = int(defects_form[int(j)][3])
                if (site == site2):
                    if (i == j):
                        pass
                        #This is the target defect and cannot compete with itself
                    else:
                        
                        defect_name2 = defects_form[int(j)][0]
                        multiplicity2 = float(defects_form[int(j)][2])
                        charge2 = float(defects_form[int(j)][4])
                        form_energy_vbm2 = float(defects_form[int(j)][5])
                       
                        def_form_energy2 = form_energy_vbm2 + (charge*nu_e)
                        
                        #Using this defect formation energy as to the sum in the denominator
                        competing += math.exp(-1*def_form_energy2/(temperature*boltzmann))
              
            concentration = multiplicity*(math.exp(-def_form_energy/(temperature*boltzmann)))/(1+competing)

        charge_contribution = concentration*charge
        total_charge += charge_contribution

        #Sum concentrations for use in dopant chemical potential optimisation
       
        if dopants_opt != 0:
            for k in np.arange(0,dopants_opt+1, 1):
                 marker = int(defects_form[int(i)][-3])
                 signal = defects_form[int(i)][-2]
                 multiply = defects_form[int(i)][-1]
                 if int(k) == marker:
                     dopant_concentration_sum[2*marker]=dopant_concentration_sum[2*marker]+(multiply*concentration)
                     dopant_concentration_sum[2*marker+1]=signal
        #print(defect_name,concentration)
        if concentration < 10e-200:
            concentration = 10e-200
        concentration = math.log(concentration)/math.log(10)
        concentrations.append(concentration)

    return total_charge, concentrations ,dopant_concentration_sum

def calc_fermi(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,dopants_opt):

    #Check that the point at which charge neutrality occurs falls in the bandgap
    total_charge, concentrations,dopant_concentration_sum = calc_charge(defects_form,defects, number_of_defects, 0, bandgap,condband,valband,temperature,art_dop_conc,art_dop_charge,def_statistics,electron_method,fixed_e_conc,hole_method,fixed_p_conc,entropy_marker,entropies,seedname,cond_band_min,cond_band_max,val_band_min,val_band_max,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,dopants_opt)
    
    if(total_charge < 0):
        if loop == 0:
            print( "<!> Error: Charge neutrality occurs outside of the band gap (nu_e < 0) at oxygen partial pressure of 10^(",b,") atm")
            exit()
        elif loop == 1:
            print( "<!> Error: Charge neutrality occurs outside of the band gap (nu_e < 0) at temperature of",b,"K")
            exit()
        elif loop == 2:
            print( "<!> Error: Charge neutrality occurs outside of the band gap (nu_e < 0) at dopant concentration of 10^(",b,") per f.u.")
            exit()
        elif loop == 3:
            print( "<!> Error: Charge neutrality occurs outside of the band gap (nu_e < 0) at artificial dopant concentration of 10^(",b,") per f.u.")
            exit()
            
    total_charge, concentrations, dopant_concentration_sum = calc_charge(defects_form,defects, number_of_defects, bandgap, bandgap,condband,valband,temperature,art_dop_conc,art_dop_charge,def_statistics,electron_method,fixed_e_conc,hole_method,fixed_p_conc,entropy_marker,entropies,seedname,cond_band_min,cond_band_max,val_band_min,val_band_max,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,dopants_opt)
    if(total_charge > 0):
        if loop == 0:
            print( "<!> Error: Charge neutrality occurs outside of the band gap (nu_e> Bandgap) at oxygen partial pressure of 10^(",b,") atm")
            exit()
        elif loop == 1:
            print( "<!> Error: Charge neutrality occurs outside of the band gap (nu_e> Bandgap) at temperature of",b,"K")
            exit()
        elif loop == 2:
            print( "<!> Error: Charge neutrality occurs outside of the band gap (nu_e> Bandgap) at dopant concentration of 10^(",b,") per f.u.")
            exit()
        elif loop == 3:
            print( "<!> Error: Charge neutrality occurs outside of the band gap (nu_e < 0) at artificial dopant concentration of 10^(",b,") per f.u.")
            exit()
    
    i = 0
    j = bandgap
    counter = 0
    while(total_charge > charge_convergence or total_charge < -charge_convergence):
        
        midpoint = (i+j)/2
        total_charge, concentrations, dopant_concentration_sum= calc_charge(defects_form,defects, number_of_defects,midpoint,bandgap,condband,valband,temperature,art_dop_conc,art_dop_charge,def_statistics,electron_method,fixed_e_conc,hole_method,fixed_p_conc,entropy_marker,entropies,seedname,cond_band_min,cond_band_max,val_band_min,val_band_max,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,dopants_opt)
        if(total_charge > 0):
            i = midpoint
            counter+=1
        if(total_charge < 0):
            j = midpoint
            counter+=1
    
        #print(midpoint,total_charge, charge_convergence)
        if (counter>100):
            print("<!> Could not deteremine the Fermi level that gives charge neutrality, recommened you examine your DFT energies")
            exit()

    if charged_sys == 0:
        nu_e_final =0
    else:
        nu_e_final = midpoint
    
    return nu_e_final, concentrations,dopant_concentration_sum

def group(final_concentrations,number_of_defects,defects,num_iter, stoichiometry):

    print("..> Summing defect concentrations according to group assigned")

    new_concs =[]
    final_group_list =[]
    
    #Loop over the number of records
    for i in np.arange(0,len(final_concentrations), 1):
        i = int(i)
        group_list=[]
        grouped_concs=[]

        #Extract the iterator condition and the electron and hole concentrations
        iterator = final_concentrations[i][0]
        fermi = final_concentrations[i][1] 
        electron = final_concentrations[i][2] 
        hole = final_concentrations[i][3]
        if stoichiometry !=0:
            stoic = final_concentrations[i][-1]
        
        #Now loop over the defects in each record
        for j in np.arange(0,number_of_defects, 1):
            j=int(j)
            #Extract the defect group
            group = defects[j][1]
            concentration = final_concentrations[i][j+4]    
            unlogged_conc = 10**concentration;
            
            #Check to see whether this group has been found before
            if (group in group_list):

                #Loop over group_list and determine the cell were this concentration should go
                group_list_length = len(group_list)
                for k in np.arange(0,group_list_length , 1):
                    k = int(k)
                    if (group == group_list[k]):
                        #Add the concentration to this grouping
                        grouped_concs[k] = grouped_concs[k]+unlogged_conc
                   
            else:
                group_list.append(group)
                group_list_length = len(group_list)
                grouped_concs.append(unlogged_conc)
            
            final_group_list = group_list

            size = len(grouped_concs)
        
        #Quick loop to relog everything
        group_list_length = len(group_list)
        new_concs.append([iterator,fermi,electron,hole])

        for w in np.arange(0,group_list_length , 1):
            w = int(w)
            log_conc = math.log(grouped_concs[w])/math.log(10)
            new_concs[-1].append(log_conc)
        if stoichiometry !=0:
              new_concs[-1].append(stoic)
    
    return (new_concs,final_group_list)

def stoich(concentrations, defects, host_array,number_of_defects,dopants, x_variable,stoichiometry):

    #Function that finds deviation in stoichiometry for volatile species
    #Volatile species must be the last element in the host.
    #Whether a defect contributes to hyper/hypo stoic is determined by input in .defects file. 

    #First, determine the metal and volatile elements' stoichiometry coefficients 
    stoic_sum=0
    for i in np.arange(0,host_array[0], 1):

        stoic = Decimal(host_array[2*i+2])
     
        if i==host_array[0]-1:
            volatile_stoic= stoic
        else:
            stoic_sum +=stoic

    #Now deterimine defect contributions.
    numerator=0
    denominator=0
    #loop over atoms in host
    for i in np.arange(0,host_array[0], 1):
        i = int(i)
        stoic = Decimal(host_array[2*i+2])
        contribution = 0
        #loop over defecs
        for j in np.arange(0,number_of_defects, 1):
            j = int(j)

            element_change = float(defects[j][7+i])
       
            if element_change != 0 :
                contribution += (10**float(concentrations[j+2]))*(-1*element_change)
                
        contribution = Decimal(contribution)     

        if i == host_array[0]-1:
            contribution +=stoic
            numerator += contribution
                 
        else:
            contribution = (contribution/volatile_stoic) + (stoic/stoic_sum)
            denominator += contribution        
                
    #Two options for dopants:
            #Stoichiometry = 1 calculates stoichiometry with original cations, considers the cation/volatile species leaving the system in a substitution, but not the dopant added.
            #Stoichiometry = 2 calculates a volatile to metal ratio, where any dopant added is treated as a metal.
            
    if stoichiometry ==2:
        #loop over dopant atoms
        if (dopants[0] > 0):      
            for i in np.arange(0,dopants[0], 1):
                i = int(i)
                contribution = 0
                #loop over defecs
                for j in np.arange(0,number_of_defects, 1):
                    j = int(j)

                    element_change = float(defects[j][7+host_array[0]+i])
               
                    if element_change != 0 :
                        contribution += (10**float(concentrations[j+2]))*(-1*element_change)

                contribution = Decimal(contribution)
                contribution = (contribution/volatile_stoic) 
                denominator += contribution

    final_stoic= -1*((numerator/denominator)-volatile_stoic)

    if  x_variable ==1: #Plotting as a function of stoichiometery

        new_stoichiometry = -final_stoic
        
        concentrations.insert(0,new_stoichiometry)
     
    else:
        #This function reflects the value of x so under hyperstoichiometry it is MO2+x and MO2-x for hypostoichiometry
        if(final_stoic<0):
           
            new_stoichiometry = -1*final_stoic
        else:
            new_stoichiometry = final_stoic

        log_stoichiometry = -200
        if (new_stoichiometry == 0):
            concentrations.append(log_stoichiometry)
        else:
            log_stoichiometry = math.log(new_stoichiometry)/math.log(10)
            concentrations.append(log_stoichiometry)
         
    return concentrations
    
def print_results(results,seedname):

    resultfile = str(seedname)+".res"
    print("..> Printing defect concentrations in", resultfile)
    with open(resultfile, 'w') as f:
        for i in results:
            print(*i, file=f)

def print_fermi(fermi,seedname):

    fermifile = str(seedname)+".fermi"
    print("..> Printing fermi energies in", fermifile)
    with open(fermifile, 'w') as f:
        for i in fermi:
            print(*i, file=f)

def print_formation(master_list):

    formationfile = str(seedname)+".formation"

    with open(formationfile, 'w') as f:
        for i in master_list:
            print(*i, file=f)
            
def graphical_inputs(seedname):
    
    filename = str(seedname)+".plot"

    file = open(filename)
    for line in file:
        fields = line.strip().split()
        if len(fields) >2:
            name = fields[0]
            if name == "concentration_colour":
                    conc_colour = fields[2:]
            if name == "formation_colour":
                    form_colour = fields[2:]
            if name == "electron_colour":
                    electron_colour = fields[2]
            if name == "hole_colour":
                    hole_colour = fields[2]
            
    return conc_colour,form_colour,electron_colour,hole_colour

def graphical_output(number_of_defects,min_value,max_value,final_concentrations,seedname,loop,gnuplot_version,min_y_range,host_name,defects,electron_method,hole_method,dopants,host_array,entry_marker,conc_colour,electron_colour,hole_colour,scheme, dopant_xvar, stoichiometry,x_variable,total_species,volatile_element,charged_sys, y_variable,max_y_range):
    
    print("..> Plotting defect concentrations")
 
    #Improve presentation of the host name
    host_name = host_name.replace("-", "")

    #colour choice
   
    if scheme ==0:
        
        colours =["#006e00","#b80058","#008cf9","#d163e6","#00bbad","#ff9287","#b24502","#878500","#00c6f8","#00a76c","#bdbdbd"]
        colour_electron ="#5954d6"
        colour_hole ="#ebac23"
    elif scheme ==1:
        colours = conc_colour
        colour_electron = electron_colour
        colour_hole=hole_colour

    colourx = "black"
    defect_colours = []
    defect_lines = []
    line_marker=0
    colour_marker= 0 

    #Assign colours and lines, for all defects. Each defect of same 'type' assigned same colour, with each given different line type. 
    if entry_marker ==0:
        key_master = []
        for i in np.arange(0, number_of_defects,1):
            i = int(i)
            key=''        
            for j in np.arange(0,total_species, 1):
            
                j = int(j)+7 
                key_i = defects[i][j]
                key+=key_i
            key_master.append(key)
       
        assigner = []
        j=0
        
        for i in np.arange(0, number_of_defects,1):
            i = int(i)
            key = key_master[i]
            if key in assigner:
                colour = assigner[assigner.index(key)+1]
                line = assigner[assigner.index(key)+2]+1
                assigner[assigner.index(key)+2]=line
           
                if line > 8:
                    line=1
                    if line_marker ==0:
                        print("<!> Unable to assign unique line dashes to all defects, due to a large number of defects of the same 'type'. Consider task = 'group'")
                        line_marker = 1 
                    
            else:
                if j > len(colours)-1:
                    colour = colourx
                    if colour_marker == 0:
                        print("<!> Colour list exceeded: some defects assigned colour black. Specify colour by increasing colours in 'concentration_colour' in", seedname,".plot file")
                        colour_marker = 1
                else:
                    colour = colours[j]
                line = 1
                j+=1
                assigner.append(key)
                assigner.append(colour)
                assigner.append(line)
            defect_colours.append(colour)
            defect_lines.append(line)

    #Assign colours and line types, if task = group.    
    if entry_marker ==1:
        i=1
        while i <9:
            j=0
            while j < len(colours):
                defect_colours.append(colours[j])
                defect_lines.append(i)
                j+=1
            i+=1
                   
    graphfile = str(seedname)+".p"
    outputfile = str(seedname)+".eps"
    resultfile = str(seedname)+".res"
    fermifile = str(seedname)+".fermi"
    fermiplotfile = str(seedname)+"_fermi.eps"
    
    with open(graphfile, 'w') as f:
        #Print header to file
        print("#GNUPLOT script for showing defect concentrations\n",file=f)
        print("set terminal postscript eps enhanced color font 'Helvetica,20'", file=f)
        print("set output \"",outputfile,"\"", sep="",file=f)
        print("set encoding iso_8859_1",file=f)
        
        if x_variable ==1:
            if host_array[-1] == 1:
                print("set xlabel \"x in ",host_name,"_{1+x}\"",sep="",file=f)
            else:
                print("set xlabel \"x in ",host_name,"_{+x}\"",sep="",file=f)
        else:     
            if(loop == 0):
                print("set xlabel 'log_{10}P_{",volatile_element,"_{2}} /atm'",sep="", file=f)
            elif(loop == 1):
                print("set xlabel 'Temperature /K'", file=f)
            elif(loop == 2):
                print("set xlabel 'log_{10}[",dopant_xvar,"] (per ",host_name,")'",sep="",file=f)
            elif(loop == 3):
                print("set xlabel 'log_{10}[artificial dopant conc] (per ",host_name,")'",sep="", file=f)
            elif(loop == 4):
                print("set xlabel 'log_{10}P_{",dopant_xvar,"_{2}} /atm'",sep="",file=f)
        if y_variable ==1:
            print("set ylabel 'log_{10}[D] (per cm^{-3})'\n",sep="", file=f)            
        else:
            print("set ylabel 'log_{10}[D] (per ",host_name,")'\n",sep="", file=f)               
        if x_variable ==1:
            print("set xrange [-0.1:0.1]",sep="", file=f)
        else:
            print("set xrange [",min_value,":",max_value,"]",sep="", file=f)
        print("set yrange [",min_y_range,":",max_y_range,"]\n",sep="", file=f)
        #Dashtype
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
            print("plot \"./",resultfile,"\" using 1:3 with lines lt 1 lw 2 lc rgb \"", colour_electron,"\" ti \"Electrons\",\\",sep="",file =f)

        if hole_method != 0 and electron_method !=0:
            print("\"./",resultfile,"\" using 1:4 with lines lt 1 lw 2 lc rgb \"", colour_hole,"\" ti \"Holes\",\\",sep="",file=f)
        if hole_method != 0 and electron_method ==0:
            print("plot \"./",resultfile,"\" using 1:4 with lines lt 1 lw 2 lc rgb \"", colour_hole,"\" ti \"Holes\",\\",sep="",file=f)

        #Plot concentration of every defect. each charge state assigned different 'dash type'
        if entry_marker ==0:
            i=0             
            while i < number_of_defects:
                defect = defects[i][0]
                group = defects[i][1]
                charge = int(defects[i][4])
                colour = defect_colours[i]
                line_type = defect_lines[i]
                if charged_sys ==1:
                    if i == 0 and electron_method == 0 and hole_method ==0:
                        print("plot \"./",resultfile,"\" using 1:",i+5," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect," ",charge,"\",\\",sep="",file=f)
                    else:
                        print("\"./",resultfile,"\" using 1:",i+5," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect," ",charge,"\",\\",sep="",file=f)
                else:
                    if i == 0 and electron_method == 0 and hole_method ==0:
                        print("plot \"./",resultfile,"\" using 1:",i+5," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect," \",\\",sep="",file=f)
                    else:
                        print("\"./",resultfile,"\" using 1:",i+5," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect," \",\\",sep="",file=f)
                i +=1
            if  stoichiometry != 0 and x_variable ==0:
                pm = r"\261"
                if host_array[-1] == 1:
                    print("\"./",resultfile,"\" using 1:",i+5," with lines lt 2 lw 2 lc rgb \"", colourx,"\" ti \"x in ",host_name,"_{1",pm,"x}\",\\",sep="",file=f)
                else:
                    print("\"./",resultfile,"\" using 1:",i+5," with lines lt 2 lw 2 lc rgb \"", colourx,"\" ti \"x in ",host_name,"_{",pm,"x}\",\\",sep="",file=f)
        
        #Plot sum of concentrations, based on group.
        elif entry_marker ==1:
            i=0   
            while i < len(defects):
                defect = defects[i]
                colour = defect_colours[i]
                line_type = defect_lines[i]
                if i == 0 and electron_method == 0 and hole_method ==0:
                    print("plot \"./",resultfile,"\" using 1:",i+5," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect,"\",\\",sep="",file=f)
                else:
                    print("\"./",resultfile,"\" using 1:",i+5," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect,"\",\\",sep="",file=f)           
                i +=1
       
            if  stoichiometry != 0 and x_variable ==0:
                pm = r"\261"
                if host_array[-1] == 1:
                    print("\"./",resultfile,"\" using 1:",i+5," with lines lt 2 lw 2 lc rgb \"", colourx,"\" ti \"x in ",host_name,"_{1",pm,"x}\",\\",sep="",file=f)
                else:
                    print("\"./",resultfile,"\" using 1:",i+5," with lines lt 2 lw 2 lc rgb \"", colourx,"\" ti \"x in ",host_name,"_{",pm,"x}\",\\",sep="",file=f)

        #Plot Fermi energy
        if charged_sys == 1: 

            print("\n\n#GNUPLOT script for showing Fermi energy\n",file=f)
            print("set output \"",fermiplotfile,"\"", sep="",file=f)
            if(loop == 0):
                print("set xlabel 'log_{10}P_{",volatile_element,"_{2}} /atm'",sep="", file=f)
            elif(loop == 1):
                print("set xlabel 'Temperature /K'", file=f)
            elif(loop == 2):
                print("set xlabel 'log_{10}[",dopant_xvar,"] (per ",host_name,")'",sep="",file=f)
            elif(loop == 3):
                print("set xlabel 'log_{10}[artificial_dopant_conc] (per ",host_name,")'",sep="", file=f)
            elif(loop == 4):
                print("set xlabel 'log_{10}P_{",dopant_xvar,"_{2}} /atm'",sep="",file=f)
            print("set autoscale y",sep="", file=f)
            print("set key off",sep="", file=f)
            print("set ylabel 'Fermi level (eV)'\n",sep="", file=f)
            print("plot \"./", fermifile,"\" using 1:2 with lines lt 1 lw 2 lc rgb \"#008cf9\" \\",sep="",file=f)

def form_energies(defects_form,number_of_defects,tasks,bandgap,seedname):

    defect_types=[]
    lowest_formation=[]
    formation=[]

    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
    
        #Print header for the formation energies
        print("\n-----------------------------------------------------------------------------------------","\n", file=f)
        print(">>> Formation energies\n",file=f)
        print("   +----------------+--------+----------------------+",file=f)
        print("   |     Defect     | Charge | Formation energy /eV |",file=f)
        print("   +----------------+--------+----------------------+",file=f)

        #Search through defects_form and print output
        for i in np.arange(0, number_of_defects, 1):
            i = int(i)
            defect_name = defects_form[i][0]
            defect_group = defects_form[i][1]
            charge = defects_form[i][4]
            form_energy = defects_form[i][5]            
            
            if ("form_plots" in tasks):

                if (defect_group in defect_types):
                    pass
                else:
                    defect_types.append(defect_group)
                
            print ("   | %14s | %6s | %20f |" % (defect_name, charge, form_energy),file=f)

        print("   +----------------+--------+----------------------+\n",file=f)

    print("..> Defect formation energies tabulated in", outputfile)  
     
    if ("form_plots" in tasks):
        
        #Find lowest formation energy for each class of defect across bandgap
        increment_fermi = 0.001
        i = 0
        while i <= bandgap:
            j=0    
            defect_form_list = [i]
            while j<len(defect_types):        
                group = defect_types[j]
                defect_group_form_list = []
                for w in np.arange(0, number_of_defects, 1):
                    w = int(w)
                    group_i=defects_form[w][1]    
                    charge = defects_form[w][4]
                    form_energy = defects_form[w][5]
                    if group == group_i:
                        defect_form =charge*i+form_energy
                        defect_group_form_list.append(defect_form) 
                
                defect_form_list.append(min(defect_group_form_list))
            
                j+=1
            lowest_formation.append(defect_form_list)
            i+=increment_fermi
        
        #Find formation energy for every defect across bandgap
        increment_fermi = 0.01
        i = 0
       
        while i <= bandgap:
            defect_form_list = [i]
            for w in np.arange(0, number_of_defects, 1):
                w = int(w)
                charge = defects_form[w][4]
                form_energy = defects_form[w][5]
                defect_form =charge*i+form_energy
                defect_form_list.append(defect_form)

            formation.append(defect_form_list)
            i+=increment_fermi

    formationfile = str(seedname)+".formation_grouped"
    formationfile2 = str(seedname)+".formation"

    with open(formationfile, 'w') as f:
        for i in  lowest_formation:
            print(*i, file=f)

    with open(formationfile2, 'w') as f:
        for i in formation:
            print(*i, file=f)

    return  defect_types
    
def formation_graphical_output(seedname, bandgap, defects, y_form_min, y_form_max,form_colour,scheme,number_of_defects,total_species, defect_types):


    graphfile = "formation_plot.p"
    outputfile1 = "formation_minimum.eps"
    outputfile2 = "formation.eps"
    resultfile1 = str(seedname)+".formation_grouped"
    resultfile2 = str(seedname)+".formation"

    if scheme ==0:
        colours =["#006e00","#b80058","#008cf9","#d163e6","#00bbad","#ff9287","#b24502","#878500","#00c6f8","#00a76c","#bdbdbd","#ebac23","#5954d6"]
        
    elif scheme ==1:
        colours = form_colour
    colourx = "black"

    defect_colours = []
    defect_lines = []
    defect_colours2 = []
    defect_lines2 = []
    line_marker=0
    colour_marker= 0 

    #Assign colours and lines, for all defects. Each defect of same 'type' assigned same colour, with each given different line type. 
  
    key_master = []
    for i in np.arange(0, number_of_defects,1):
        i = int(i)
        key=''        
        for j in np.arange(0,total_species, 1):
        
            j = int(j)+7 
            key_i = defects[i][j]
            key+=key_i
        key_master.append(key)
   
    assigner = []
    j=0
    
    for i in np.arange(0, number_of_defects,1):
        i = int(i)
        key = key_master[i]
        if key in assigner:
            colour = assigner[assigner.index(key)+1]
            line = assigner[assigner.index(key)+2]+1
            assigner[assigner.index(key)+2]=line
       
            if line > 8:
                line=1
                if line_marker ==0:
                    print("<!> Unable to assign unique line dashes to all defects, due to a large number of defects of the same 'type'. Consider task = 'group'")
                    line_marker = 1 
                
        else:
            if j > len(colours)-1:
                colour = colourx
                if colour_marker == 0:
                    print("<!> Colour list exceeded: some defects assigned colour black. Specify colour by increasing colours in 'formation_colour' in", seedname,".plot file")
                    colour_marker = 1
            else:
                colour = colours[j]
            line = 1
            j+=1
            assigner.append(key)
            assigner.append(colour)
            assigner.append(line)
        defect_colours.append(colour)
        defect_lines.append(line)

    #Assign colours and line types for grouped formation energies    
   
    i=1
    while i <9:
        j=0
        while j < len(colours):
            defect_colours2.append(colours[j])
            defect_lines2.append(i)
            j+=1
        i+=1

    with open(graphfile, 'w') as f:
        #Print header to file
        print("#GNUPLOT script for formation energies of defects\n",file=f)
        print("set terminal postscript eps enhanced color font 'Helvetica,20'", file=f)
        print("set xlabel 'Fermi level (eV)'", file=f)
        print("set ylabel 'Formation enegy (eV)'\n", file=f)
        print("set xrange [",0,":",bandgap,"]",sep="", file=f)
        #print("set yrange [",y_form_min,":",y_form_max,"]\n",sep="", file=f)
        #Dashtype
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
        #print("set key center below", file=f)
        #print("set key horizontal\n", file=f)

        #Print one line for each defect class
        print('..> Plotting minimum formation energy for each group')
        print("set output \"",outputfile1,"\"", sep="",file=f)
        i=2
        for group in defect_types:
            colour = defect_colours2[i-2]
            line_type = defect_lines2[i-2]
            if i==2:
                print("plot \"./",resultfile1,"\" using 1:",i," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",group,"\",\\",sep="",file=f)
            else:
                print("\"./",resultfile1,"\" using 1:",i," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",group,"\",\\",sep="",file=f)
            
            i+=1
        
        #Print every defect, assigning different dash type for each charge
    
        print('..> Plotting formation energy for every defect')
        print("\n set output \"",outputfile2,"\"", sep="",file=f)
  
        i=0
        while i < number_of_defects:
            defect = defects[i][0]
            charge = int(defects[i][4])
            colour = defect_colours[i]
            line_type = defect_lines[i]
            if i==0 :
                print("plot \"./",resultfile2,"\" using 1:",i+2," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect," ",charge,"\",\\",sep="",file=f)
            else:
                print("\"./",resultfile2,"\" using 1:",i+2," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect," ",charge,"\",\\",sep="",file=f)
            i +=1
            
        #Print individual groups on individual figures
      
        print('..> Plotting minimum formation energy for each group, with seperate figures for each group')
        
        i=2
        for group in defect_types:
            colour = defect_colours2[i-2]    
            outputfile = str(group)+"_min.eps"
            print("\n set output \"",outputfile,"\"", sep="",file=f)
            print("plot \"./",resultfile1,"\" using 1:",i," with lines lt 1 lw 2 lc rgb \"", colour,"\" ti \"",group,"\",\\",sep="",file=f)
                   
            i+=1
    
        #Print every defect on individual figures

        print('..> Plotting formation energy for every defect, with seperate figures for each group')
        group_position =[]  #A log of the group positions
        for group in defect_types:
            outputfile = str(group)+".eps"     
            print("\n set output \"",outputfile,"\"", sep="",file=f)
            i=1
            j=0
            group_position_i=[]
            while i < number_of_defects+1:
                group_i = defects[i-1][1]
                defect = defects[i-1][0]
                charge = int(defects[i-1][4])
                colour = defect_colours[i-1]
                line_type = defect_lines[i-1]
                if group == group_i:
                    group_position_i.append(i)
                    if j ==0:
                         print("plot \"./",resultfile2,"\" using 1:",i+1," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect," ",charge,"\",\\",sep="",file=f)
                    else:
                        print("\"./",resultfile2,"\" using 1:",i+1," with lines lt ",line_type," lw 2 lc rgb \"", colour,"\" ti \"",defect," ",charge,"\",\\",sep="",file=f)
                    j+=1                                            
                i+=1
            group_position.append(group_position_i)

def y_convert(final_concentrations,fu_uc, uc_volume,stoichiometry):

    #Nummber of Angstrom^3 in cm^3
    A3_2_cm3 = 1E24

    conversion = fu_uc * (1/uc_volume) * A3_2_cm3

    inc = 0  #Do not want to convert the final column if stoichiometry has been calculated
    if stoichiometry != 0:
        inc = 1 
        
    for i in np.arange(0,(len(final_concentrations)),1):      
        for j in np.arange(2,(len(final_concentrations[0])-inc),1):
            concentration = 10**final_concentrations[i][j]
            concentration = concentration *conversion          
            final_concentrations[i][j]= math.log(concentration)/math.log(10)

    return final_concentrations
     
def invert_matrix(input_mat, marker):       #Function for inverting a matrix
  
    #marker : Factor 0 for inverting dielectric and 1 for inverting lattice
    output_mat =[] 
    adjoint =[]
    
    if (marker == 0):
        factor = 1
    elif (marker == 1):
        factor = 2*math.pi

    #Calulate determinant of input matrix
    determinant = det(input_mat)
    
    #Calculate adjoint matrix
    adjoint.append(input_mat[4]*input_mat[8] - input_mat[7]*input_mat[5])
    adjoint.append(input_mat[3]*input_mat[8] - input_mat[6]*input_mat[5])
    adjoint.append(input_mat[3]*input_mat[7] - input_mat[6]*input_mat[4])
    adjoint.append(input_mat[1]*input_mat[8] - input_mat[7]*input_mat[2])
    adjoint.append(input_mat[0]*input_mat[8] - input_mat[6]*input_mat[2])
    adjoint.append(input_mat[0]*input_mat[7] - input_mat[6]*input_mat[1])
    adjoint.append(input_mat[1]*input_mat[5] - input_mat[4]*input_mat[2])
    adjoint.append(input_mat[0]*input_mat[5] - input_mat[3]*input_mat[2])
    adjoint.append(input_mat[0]*input_mat[4] - input_mat[3]*input_mat[1])
    
    #Calculate inverse
    output_mat.append((factor*adjoint[0])/determinant)
    output_mat.append( -(factor*adjoint[1])/determinant)
    output_mat.append( (factor*adjoint[2])/determinant)
    output_mat.append( -(factor*adjoint[3])/determinant)
    output_mat.append( (factor*adjoint[4])/determinant)
    output_mat.append( -(factor*adjoint[5])/determinant)
    output_mat.append( (factor*adjoint[6])/determinant)
    output_mat.append( -(factor*adjoint[7])/determinant)
    output_mat.append( (factor*adjoint[8])/determinant)
    
    return(output_mat)

#Subroutine for calculating the determinant of a matrix
def det(input_mat):

    #Calculate determinant
    determinant = input_mat[0]*(input_mat[4]*input_mat[8]-input_mat[7]*input_mat[5]) - input_mat[1]*(input_mat[3]*input_mat[8]-input_mat[6]*input_mat[5]) + input_mat[2]*(input_mat[3]*input_mat[7]-input_mat[6]*input_mat[4])
    
    return(determinant)

#Determine the longest lattice parameter and define limits for the real space
def limits_real(lattice,factor,seedname):

    real_limits = []
    
    #Calculate the cell lattice parameters
    latt_a = math.sqrt((lattice[0]**2 + lattice[1]**2 + lattice[2]**2))
    latt_b = math.sqrt((lattice[3]**2 + lattice[4]**2 + lattice[5]**2))
    latt_c = math.sqrt((lattice[6]**2 + lattice[7]**2 + lattice[8]**2))
       
    #Determine which of the lattice parameters is the largest
    if (latt_a >= latt_b) and (latt_a >= latt_c):
        longest = latt_a
    if (latt_b >= latt_a) and (latt_b >= latt_c):
        longest = latt_b
    if (latt_c >= latt_a) and (latt_c >= latt_b):
        longest = latt_c
    
    #Calculate real space cutoff
    r_c = factor * longest

    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
        print("   Supercell parameters %.6f  %.6f  %.6f" % (latt_a, latt_b, latt_c),file=f)
        print("   Longest lattice parameter =",longest,file=f)
        print("   Realspace cutoff =",r_c,file=f)
    
    #Estimate the number of boxes required in each direction to ensure r_c is contained (the tens are added to ensure the number of cells contains $r_c)
    a_range = r_c/latt_a + 10;
    b_range = r_c/latt_b + 10;
    c_range = r_c/latt_c + 10;
    a_range_final = round(a_range) 
    b_range_final = round(b_range) 
    c_range_final = round(c_range)
    
    #This defines the size of the supercell in which the real space section is performed, however only atoms within rc will be conunted
    real_limits.append(a_range_final)
    real_limits.append(b_range_final)
    real_limits.append(c_range_final)
    
    return(real_limits,r_c)

#Function to calculate the real and reciprocal space contributions
def real_recip(lattice,inv_dielectric,motif,real_limits,r_c,gamma,num_atoms,debug,determinant,recip_lattice,dielectric,volume,seedname):

    #lattice =		Lattice parallelpiped
    #inv_dielectric = Inverse of the dielectric tensor
    #motif = Motif
    #real_limits = Limits to the real space cell
    #r_c = Real space cutoff
    #gamma = gamma parameter
    #num_atoms = Number of defects in motif
    #debug = Debug flag
    #determinant = Determinant of the dielectric tensor
    #recip_lattice = Reciprocal lattice
    #dielectric = Dielectric tensor
    #volume = Volume of the supercell    
  
    #Calculate superlattice lattice parallelpiped
    superlattice=[]
    real_space=0
    recip_superlattice=[]
    reciprocal=0
    incell = 0
      
    #Calculate supercell parrallelpiped
    superlattice.append(real_limits[0] * lattice[0])
    superlattice.append(real_limits[0] * lattice[1])
    superlattice.append(real_limits[0] * lattice[2])
    superlattice.append(real_limits[1] * lattice[3])
    superlattice.append(real_limits[1] * lattice[4])
    superlattice.append(real_limits[1] * lattice[5])
    superlattice.append(real_limits[2] * lattice[6])
    superlattice.append(real_limits[2] * lattice[7])
    superlattice.append(real_limits[2] * lattice[8])
    
    #Calculate the reciprocal space parrallelpiped
    recip_superlattice.append(real_limits[0] * recip_lattice[0])
    recip_superlattice.append(real_limits[0] * recip_lattice[1])
    recip_superlattice.append(real_limits[0] * recip_lattice[2])
    recip_superlattice.append(real_limits[1] * recip_lattice[3])
    recip_superlattice.append(real_limits[1] * recip_lattice[4])
    recip_superlattice.append(real_limits[1] * recip_lattice[5])
    recip_superlattice.append(real_limits[2] * recip_lattice[6])
    recip_superlattice.append(real_limits[2] * recip_lattice[7])
    recip_superlattice.append(real_limits[2] * recip_lattice[8])
    
    #Print the real space superlattice
    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
    
        #Print the real space superlattice
        print("\n   Real space superlattice",file=f)
        print("   %.6f  %.6f  %.6f" % (superlattice[0], superlattice[1], superlattice[2]),file=f)
        print("   %.6f  %.6f  %.6f" % (superlattice[3], superlattice[4], superlattice[5]),file=f)
        print("   %.6f  %.6f  %.6f" % (superlattice[6], superlattice[7], superlattice[8]),file=f)
    
        #Print the real space superlattice
        print("\n   Reciprocal space superlattice",file=f)
        print("   %.6f  %.6f  %.6f" % (recip_superlattice[0], recip_superlattice[1], recip_superlattice[2]),file=f)
        print("   %.6f  %.6f  %.6f" % (recip_superlattice[3], recip_superlattice[4], recip_superlattice[5]),file=f)
        print("   %.6f  %.6f  %.6f" % (recip_superlattice[6], recip_superlattice[7], recip_superlattice[8]),file=f)
  
    ###########################
    # Real space contribution #
    ###########################
    print("..> Calcualting real space contribution")

    with open('REAL_SPACE', 'a') as f:
        #Loop over all atoms in the motif and calculate contributions
        for i in np.arange(0,num_atoms,1):
            i =int(i)

            #Convert fractional motif co-ordinates to cartesian
            motif_charge = motif[4*i+3]
            motif_cart_x = motif[4*i]*lattice[0] + motif[4*i+1]*lattice[3] + motif[4*i+2]*lattice[6]
            motif_cart_y = motif[4*i]*lattice[1] + motif[4*i+1]*lattice[4] + motif[4*i+2]*lattice[7]
            motif_cart_z = motif[4*i]*lattice[2] + motif[4*i+1]*lattice[5] + motif[4*i+2]*lattice[8]
            #printf ("Cartesian defect co-ordinates %.6f  %.6f  %.6f %.6f\n\n", $motif_cart_x, $motif_cart_y, $motif_cart_z, $motif_charge);
            #printf ("Fractional defect co-ordinates %.6f  %.6f  %.6f %.6f\n\n", $motif[4*$i], $motif[4*$i+1], $motif[4*$i+2], $motif_charge);
            
            #Convert fractional co-ordinates to reciprocal space
            motif_recip_x = motif[4*i]*recip_lattice[0] + motif[4*i+1]*recip_lattice[3] + motif[4*i+2]*recip_lattice[6]
            motif_recip_y = motif[4*i]*recip_lattice[1] + motif[4*i+1]*recip_lattice[4] + motif[4*i+2]*recip_lattice[7]
            motif_recip_z = motif[4*i]*recip_lattice[2] + motif[4*i+1]*recip_lattice[5] + motif[4*i+2]*recip_lattice[8]
            #printf ("Reciprocal space defect co-ordinates %.6f  %.6f  %.6f %.6f\n\n", $motif_recip_x, $motif_recip_y, $motif_recip_z, $motif_charge);
            
            #Loop over all other atoms in the motif
            for j in np.arange(0,num_atoms,1):
                j =int(j)

                incell_contribution = 0
                
                image_charge = motif[4*j+3]
                
                #Loop over all points in the superlattice
                for m in np.arange(-real_limits[0],real_limits[0],1):
                    m =int(m)

                    for n in np.arange(-real_limits[1],real_limits[1],1):
                        n =int(n)

                        for o in np.arange(-real_limits[2],real_limits[2],1):
                            o =int(o)
                    
                            real_contribution = 0
                            recip_contribution = 0
                        
                            #Calculate the defect's fractional position in the extended supercell
                            x_super = 1/(real_limits[0]) * m + motif[4*j+0]/(real_limits[0])
                            y_super = 1/(real_limits[1]) * n + motif[4*j+1]/(real_limits[1])
                            z_super = 1/(real_limits[2]) * o + motif[4*j+2]/(real_limits[2])
                            
                            #Convert these fractional co-ordinates to cartesian
                            x_cart = x_super*superlattice[0] + y_super*superlattice[3] + z_super*superlattice[6]
                            y_cart = x_super*superlattice[1] + y_super*superlattice[4] + z_super*superlattice[7]
                            z_cart = x_super*superlattice[2] + y_super*superlattice[5] + z_super*superlattice[8]
                            
                            #Test to see whether the new atom coordinate falls within r_c and then solve
                            seperation = math.sqrt((x_cart-motif_cart_x)**2 + (y_cart-motif_cart_y)**2 + (z_cart-motif_cart_z)**2)
                            
                            if ((i == j) and (m == 0) and (n == 0) and (o == 0)):   #Setting separation == 0 does not always work for numerical reasons
                        
                                #print("Found the central atom", m,n,o,  motif_cart_x, motif_cart_y, motif_cart_z,"\n")
                                incell += 0
                        
                            elif (seperation < r_c ):
                            
                                mod_x = (x_cart-motif_cart_x) * inv_dielectric[0] + (y_cart-motif_cart_y) * inv_dielectric[3] + (z_cart-motif_cart_z) * inv_dielectric[6]
                                mod_y = (x_cart-motif_cart_x) * inv_dielectric[1] + (y_cart-motif_cart_y) * inv_dielectric[4] + (z_cart-motif_cart_z) * inv_dielectric[7]
                                mod_z = (x_cart-motif_cart_x) * inv_dielectric[2] + (y_cart-motif_cart_y) * inv_dielectric[5] + (z_cart-motif_cart_z) * inv_dielectric[8]
                                
                                dot_prod = mod_x * (x_cart-motif_cart_x) + mod_y * (y_cart-motif_cart_y) + mod_z * (z_cart-motif_cart_z)
                                
                                
                                #This section calculates the Coulombic interactions inside the defect supercell
                                if (m == 0) and (n == 0) and (o == 0):

                                    incell_contribution = motif_charge * image_charge * (1/(math.sqrt(determinant))) * (1/(sqrt(dot_prod)))
                                    incell += incell_contribution
                                    #print("Calculating inner energy",motif_cart_x, motif_cart_y ,motif_cart_z ,x_cart ,y_cart ,z_cart ,incell_contribution ,incell)
                                
                                real_contribution = (motif_charge*image_charge)*(1/(math.sqrt(determinant)) * (special.erfc(gamma * math.sqrt(dot_prod)))/(math.sqrt(dot_prod)))
                                
                                if (debug == 1):
                                    print(x_cart, y_cart, z_cart, seperation, dot_prod, real_contribution, file =f)
                               
                                real_space += real_contribution
          
    
    #################################
    # Reciprocal space contribution #
    #################################
    print("..> Calcualting reciprocal space contribution")

    with open('RECIPROCAL_SPACE', 'a') as f:
        #Loop over all k-points
        recip_contribution = 0
        for s in np.arange(-real_limits[0],real_limits[0],1):
            s =int(s)

            for t in np.arange(-real_limits[1],real_limits[1],1):
                t =int(t)

                for u in np.arange(-real_limits[2],real_limits[2],1):
                    u =int(u)
                    
                    #Determine which k-point to calculate
                    x_recip_super = 1/(real_limits[0]) * s
                    y_recip_super = 1/(real_limits[1]) * t
                    z_recip_super = 1/(real_limits[2]) * u
                    
                    #Convert to reciprocal space
                    x_recip = x_recip_super*recip_superlattice[0] + y_recip_super*recip_superlattice[3] + z_recip_super*recip_superlattice[6]
                    y_recip = x_recip_super*recip_superlattice[1] + y_recip_super*recip_superlattice[4] + z_recip_super*recip_superlattice[7]
                    z_recip = x_recip_super*recip_superlattice[2] + y_recip_super*recip_superlattice[5] + z_recip_super*recip_superlattice[8]
                    
                    #my $recip_seperation = sqrt(($x_recip-$motif_recip_x)**2 + ($y_recip-$motif_recip_y)**2 + ($z_recip-$motif_recip_z)**2);
                    if (s == 0) and (t == 0) and (u == 0):
                         recip_contribution +=0
                         #print("Found image in reciprocal space", x_recip, y_recip,z_recip)
                    
                    else:

                        recip_mod_x = x_recip * dielectric[0] + y_recip * dielectric[3] + z_recip * dielectric[6]
                        recip_mod_y = x_recip * dielectric[1] + y_recip * dielectric[4] + z_recip * dielectric[7]
                        recip_mod_z = x_recip * dielectric[2] + y_recip * dielectric[5] + z_recip * dielectric[8]
                        recip_dot_prod = recip_mod_x * x_recip + recip_mod_y * y_recip + recip_mod_z * z_recip
                        
                        structure_factor = ((4*math.pi)/volume) * (1/recip_dot_prod) * (math.exp(-recip_dot_prod/(4*(gamma**2))))
                  
                        cos_cumulative = 0
                        sin_cumulative = 0
                        
                        #Loop over all atoms in the motif
                        for w in np.arange(0,num_atoms,1):
                            w =int(w)

                            #Convert fractional motif co-ordinates to cartesian
                            motif_charge = motif[4*w+3]
                            motif_cart_x = motif[4*w]*lattice[0] + motif[4*w+1]*lattice[3] + motif[4*w+2]*lattice[6]
                            motif_cart_y = motif[4*w]*lattice[1] + motif[4*w+1]*lattice[4] + motif[4*w+2]*lattice[7]
                            motif_cart_z = motif[4*w]*lattice[2] + motif[4*w+1]*lattice[5] + motif[4*w+2]*lattice[8]
                            
                            rdotG = motif_cart_x*x_recip + motif_cart_y*y_recip + motif_cart_z*z_recip

                            cos_term = motif_charge * math.cos(rdotG)
                            sin_term = motif_charge * math.sin(rdotG)
                            
                            cos_cumulative += cos_term
                            sin_cumulative += sin_term
                                     
                        recip_contribution = structure_factor * (cos_cumulative**2 + sin_cumulative**2)
                            #$recip_contribution = ($motif_charge*$image_charge)*(((4*pi)/$volume) * exp(-$rdotG) * ((exp(-$recip_dot_prod/(4*($gamma**2))))/$recip_dot_prod));
                            #print "$s $t $u $recip_cont\n";
                            #$current_atm_recip += $recip_contribution;
                        reciprocal += recip_contribution
                        if (debug == 1):
                            print(s, t, u, x_recip, y_recip, z_recip, recip_contribution, file =f)

    return(real_space,reciprocal,incell,1,1)

#Subroutine for calculating the self interaction term
def self_interaction(motif,gamma,determinant,num_atoms):
 
    summation = 0
    for k in np.arange(0,num_atoms,1):
        k =int(k)
    
        defect_charge = motif[4*k+3]
        #$summation += ($defect_charge**2) * (sqrt($gamma/(3.141592654*$determinant)));
        summation += (defect_charge**2)
    
    self_interaction = -((2*gamma)/math.sqrt(3.141592654*determinant)) * summation
    #my $self_interaction = -$summation;
    
    return(self_interaction)

#Subroutine for calculating the background contribution to the Madelung potential
def background(volume,gamma,total_charge):

    background_term = -3.141592654/(volume*gamma**2) * total_charge**2
    
    return(background_term)

#Function for printing the final results
def madelung_results(real_space,reciprocal,self_interaction,background_contribution,incell,num_atoms,seedname):

    #Unit conversion factor
    conversion = 14.39942

    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
        #Print the results based on the number of atoms
        if (num_atoms == 1):

            final_madelung = real_space + reciprocal + self_interaction + background_contribution - incell
            final_madelung_eV = (final_madelung * conversion)/2
            
            print("\n   --------------------------------------------------", file=f)
            print("   Results                      ", file=f)
            print("   --------------------------------------------------", file=f)
            print("   Real space contribution    =",real_space, file=f)
            print("   Reciprocal space component =",reciprocal, file=f)
            print("   Self interaction           =",self_interaction, file=f)
            print("   Neutralising background    =",background_contribution, file=f)
            print("   --------------------------------------------------", file=f)
            print("   Final Madelung potential   =",final_madelung, file=f)
            print("   --------------------------------------------------\n", file=f)
        
            #Print final point charge correction
            print("   Example corrections using the calculated Madelung potential:", file=f)
            print("   +--------+------------------+-----------------+", file=f)
            print("   | Charge | Point charge /eV | Lany-Zunger /eV |", file=f)
            print("   +--------+------------------+-----------------+", file=f)

            for chge_state in np.arange(1,7,1):
                chge_state =int(chge_state)
                
                makov_payne = 1/2 * final_madelung * chge_state**2 * conversion
                lany = 0.65*makov_payne
            
                print("   |   %i    |     %.10s   |    %.10s   |" % (chge_state,makov_payne, lany), file=f)
            
            print("   +--------+------------------+-----------------+\n", file=f)
        
        elif (num_atoms > 1):

            real_space_eV = (real_space * conversion)/2
            reciprocal_eV = (reciprocal * conversion)/2
            self_interaction_eV = (self_interaction * conversion)/2
            background_eV = (background_contribution * conversion)/2
            incell_eV = (incell * conversion)/2
            
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
    

def madelung(filename):

    ####################################################################################
    #Madelung potential for a periodic system with anisotropic dielectric properties. 
    ####################################################################################

    #Read in information from input file
    dielectric,lattice,motif,gamma,real_cutoff,num_atoms,total_charge,debug = madelung_input(seedname)

    #Calculate reciprocal lattice
    recip_lattice = invert_matrix(lattice,1)
            
    #Calculate inverse of the dielectric
    inv_dielectric = invert_matrix(dielectric,0)

    #Calculate volume
    volume = det(lattice)
    if (volume < 0):  #Check to make sure determinant (and hence volume) isn't negative
        volume = -volume

    #Calculate the determinant of the inverse dielectric
    determinant = det(dielectric)

    outputfile = str(seedname)+".output"
    with open(outputfile, 'a') as f:
          print("\n   Reciprocal space lattice:", file=f)
          print("   %.6f  %.6f  %.6f" % (recip_lattice[0], recip_lattice[1], recip_lattice[2]), file=f)
          print("   %.6f  %.6f  %.6f" % (recip_lattice[3], recip_lattice[4], recip_lattice[5]), file=f)
          print("   %.6f  %.6f  %.6f" % (recip_lattice[6], recip_lattice[7], recip_lattice[8]), file=f)
          print("\n   Inverse dielectric tensor:", file=f)
          print("   %.6f  %.6f  %.6f" % (inv_dielectric[0], inv_dielectric[1], inv_dielectric[2]), file=f)
          print("   %.6f  %.6f  %.6f" % (inv_dielectric[3], inv_dielectric[4], inv_dielectric[5]), file=f)
          print("   %.6f  %.6f  %.6f" % (inv_dielectric[6], inv_dielectric[7], inv_dielectric[8]), file=f)
          print("\n   Volume of the cell =",volume, "A^3", file=f)
          print("   Determinant of the dielectric tensor =",determinant,"\n", file=f)

    #Calculate limits for the real and reciprocal space sums
    real_limits,r_c = limits_real(lattice,real_cutoff,seedname)

    #Calculate real space term
    real_space,reciprocal,incell,real_duration,recip_duration = real_recip(lattice,inv_dielectric,motif,real_limits,r_c,gamma,num_atoms,debug,determinant,recip_lattice,dielectric,volume,seedname)

    #Calculate the self interaction term
    print("..> Calculating self interaction term")
    self_interaction_contribution = self_interaction(motif,gamma,determinant,num_atoms)

    #Calculate contribution to energy due to interaction with background potential
    print("..> Calculating background contribution")
    if (total_charge != 0):
        background_contribution = background(volume,gamma,total_charge)
        #print("Background contribution =",background_contribution,"eV")

    else:
        background_contribution = 0

    #Print results
    print("..> Printing final results and Madelung potential in", outputfile)

    v_M = madelung_results(real_space,reciprocal,self_interaction_contribution,background_contribution,incell,num_atoms,seedname)   

    return v_M

def bibliography(tasks,chem_pot_method,real_gas,entropy_marker):

    #Printing bibliograhy for processes used.
    print("..> Writing bibliography for processes used, in DefAP.bib")
    with open('DefAP.bib', 'w') as f:

        print("%DefAP Publication",file=f)
        print("@article{Murphy2014,",file=f)
        print("annote = {doi: 10.1021/cm4038473},",file=f)
        print("author = {Murphy, Samuel T and Hine, Nicholas D M},",file=f)
        print("doi = {10.1021/cm4038473},",file=f)
        print("issn = {0897-4756},",file=f)
        print("journal = {Chem. Mater.},",file=f)
        print("month = {feb},",file=f)
        print("pages = {1629--1638},",file=f)
        print("publisher = {American Chemical Society},",file=f)
        print("title = {{Point Defects and Non-stoichiometry in Li$_2$TiO$_3$}},",file=f)
        print("volume = {26},",file=f)
        print("year = {2014}",file=f)
        print("}",file=f)

        if ('brouwer' in tasks) or ('energy' in tasks):

            print("\n%Defect formation energy",file=f)
            print("@article{Zhang1991,",file=f)
            print("title = {Chemical potential dependence of defect formation energies in GaAs: Application to Ga self-diffusion},",file=f)
            print("author = {Zhang, S. B. and Northrup, John E.},",file=f)
            print("journal = {Phys. Rev. Lett.},",file=f)
            print("volume = {67},",file=f)
            print("issue = {17},",file=f)
            print("pages = {2339--2342},",file=f)
            print("year = {1991},",file=f)
            print("month = {Oct},",file=f)
            print("publisher = {American Physical Society},",file=f)
            print("doi = {10.1103/PhysRevLett.67.2339},",file=f)
            print("}",file=f)

        if (chem_pot_method == 2) or (chem_pot_method == 3):
        
            print("\n%Volatile chemical potential method.",file=f)
            print("@article{Finnis2005,",file=f)
            print("annote = {doi: 10.1146/annurev.matsci.35.101503.091652},",file=f)
            print("author = {Finnis, M W and Lozovoi, A Y and Alavi, A},",file=f)
            print("doi = {10.1146/annurev.matsci.35.101503.091652},",file=f)
            print("issn = {1531-7331},",file=f)
            print("journal = {Annu. Rev. Mater. Res.},",file=f)
            print("month = {jun},",file=f)
            print("pages = {167--207},",file=f)
            print("publisher = {Annual Reviews},",file=f)
            print("title = {{The Oxidation Of NiAl: What Can We Learn from Ab Initio Calculations?}},",file=f)
            print("volume = {35},",file=f)
            print("year = {2005}",file=f)
            print("}",file=f)

            if real_gas ==2:

                print("\n%Volatile chemical potential temperature dependence parameters.",file=f)
                print("@article{Johnston2004,",file=f)
                print("author = {Johnston, Karen and Castell, Martin R. and Paxton, Anthony T. and Finnis, Michael W.},",file=f)
                print("doi = {10.1103/PhysRevB.70.085415},",file=f)
                print("issn = {1098-0121},",file=f)
                print("journal = {Phys. Rev. B},",file=f)
                print("month = {aug},",file=f)
                print("pages = {085415},",file=f)
                print("publisher = {American Physical Society},",file=f)
                print("title = {{SrTiO$_3$ (001) (2 $\times$ 1) reconstructions: First-principles calculations of surface energy and atomic structure compared with scanning tunneling microscopy images}},",file=f)
                print("volume = {70},",file=f)
                print("year = {2004}",file=f)
                print("}",file=f)

            else:

                print("\n%Volatile chemical potential temperature dependence parameters.",file=f) 
                print("@book{NIST1,",file=f)
                print("address = {National Institute of Standards and Technology, Gaithersburg MD, 20899},",file=f)
                print("editor = {Linstrom, P.J and Mallard, W.J},",file=f)
                print("title = {{NIST Chemistry WebBook, NIST Standard Reference Database Number 69}},",file=f)
                print("url = {https://doi.org/10.18434/T4D303}",file=f)
                print("}",file=f)

        if(entropy_marker == 1):          

            print("\n%Vibrational entropy method.",file=f)
            print("@article{Soulie2018,",file=f)
            print("author = {Souli{\'{e}}, Aur{\'{e}}lien and Bruneval, Fabien and Marinica, Mihai-Cosmin and Murphy, Samuel and Crocombette, Jean-Paul},",file=f)
            print("doi = {10.1103/PhysRevMaterials.2.083607},",file=f)
            print("issn = {2475-9953},",file=f)
            print("journal = {Phys. Rev. Mater.},",file=f)
            print("month = {aug},",file=f)
            print("pages = {083607},",file=f)
            print("publisher = {American Physical Society},",file=f)
            print("title = {{Influence of vibrational entropy on the concentrations of oxygen interstitial clusters and uranium vacancies in nonstoichiometric UO$_2$}},",file=f)
            print("volume = {2},",file=f)
            print("year = {2018}",file=f)
            print("}",file=f)

            print("\n%Vibrational entropy method.",file=f) 
            print("@article{Cooper2018,",file=f)
            print("author = {Cooper, M. W.D. and Murphy, S. T. and Andersson, D. A.},",file=f)
            print("doi = {10.1016/j.jnucmat.2018.02.034},",file=f)
            print("issn = {00223115},",file=f)
            print("journal = {J. Nucl. Mater.},",file=f)
            print("pages = {251--260},",file=f)
            print("title = {{The defect chemistry of UO$_{2\pm x }$ from atomistic simulations}},",file=f)
            print("volume = {504},",file=f)
            print("year = {2018}",file=f)
            print("}",file=f)

        if ('madelung' in tasks):

            print("\n%Screened Madelung potential",file=f)
            print("@article{Murphy2013,",file=f)
            print("title = {Anisotropic charge screening and supercell size convergence of defect formation energies},",file=f)
            print("author = {Murphy, Samuel T. and Hine, Nicholas D. M.},",file=f)
            print("journal = {Phys. Rev. B},",file=f)
            print("volume = {87},",file=f)
            print("issue = {9},",file=f)
            print("pages = {094111},",file=f)
            print("numpages = {6},",file=f)
            print("year = {2013},",file=f)
            print("month = {Mar},",file=f)
            print("publisher = {American Physical Society},",file=f)
            print("doi = {10.1103/PhysRevB.87.094111},",file=f)
            print("}",file=f)
          
############################
#                          #
# This is the main program #
#                          #
############################

header()
if len(sys.argv) != 2:
    print ("No input file has been provided, remember to include")
    exit()
seedname = sys.argv[1]

outputfile = str(seedname)+".output"
if os.path.exists(outputfile):
    os.remove(outputfile)
if os.path.exists('REAL_SPACE'):
    os.remove('REAL_SPACE')
if os.path.exists('RECIPROCAL_SPACE'):
    os.remove('RECIPROCAL_SPACE')

#Create some arrays to store the data

details = []
defects=[]
final_concentrations = []
fermi = []
stoichiometry_list = []
#Defaults
indicator =0 
dopant_xvar='None'
volatile_element = ''
concentration_check =0

#Read in data
host_array,dopants,tasks,constituents,constituents_name_list,temperature,def_statistics,tab_correction,host_energy,chem_pot_method,host_supercell,use_coul_correction,length,dielectric,v_M,E_VBM,bandgap,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,loop,min_value,max_value,iterator,gnuplot_version,min_y_range,max_y_range,host_name,val_band_min,val_band_max,cond_band_min,cond_band_max,y_form_min,y_form_max,lines, entropy_marker, entropy_units,fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charge_convergence,potential_convergence,stability,scheme,stoichiometry,x_variable,real_gas,function_tol,maxiter_dop,y_variable =inputs(seedname)

#Read in data from seedname.defects
for i in tasks:
    if i in ['brouwer','energy','form_plots','autodisplay','stability','group']:
        defects,number_of_defects,total_species,charged_sys = read_defects(seedname, host_array, defects, dopants)
        break

#Determine whether entropy is being used
if(entropy_marker == 1):
    entropy_data = read_entropy(seedname)
    #Perform a check on input data
    entropy_check(entropy_data, defects, number_of_defects,constituents_name_list,chem_pot_method,seedname)
    if(loop ==1):
        entropies, constituent_entropies = calc_entropy(entropy_data,min_value,number_of_defects,constituents_name_list,chem_pot_method,seedname,1)
        entropies, constituent_entropies = calc_entropy(entropy_data,max_value,number_of_defects,constituents_name_list,chem_pot_method,seedname,1)  
    else:
        entropies, constituent_entropies = calc_entropy(entropy_data,temperature,number_of_defects,constituents_name_list,chem_pot_method,seedname,1) 
else:
    entropies, constituent_entropies = 0, 0

#Read in plotting customisation
if scheme ==1:
    conc_colour,form_colour,electron_colour,hole_colour = graphical_inputs(seedname)
else:
    conc_colour,form_colour,electron_colour,hole_colour=0,0,0,0

#Calculate madelung potential task
if ('madelung' in tasks):
    print("\n>>> Task:'madelung':")
    v_M = madelung(seedname)
elif chem_pot_method ==4:
    print("<!> Error : Unknown chem_pot_method")
    exit()
    
#Initialise progress meter
prog_meter = 1

#Formation energy task
if ('energy' in tasks):
    print("\n>>> Task:'energy':")

    #Calculate chemical potentials of host atoms
    chemical_potentials = []
    
    if (chem_pot_method == 0):
        chemical_potentials = calc_chemical_defined(host_array,constituents,chemical_potentials,host_energy,temperature, entropy_marker, constituent_entropies, entropy_units)
        pp='x'
    elif (chem_pot_method == 1):
        chemical_potentials = calc_chemical_rich_poor(host_array,constituents,chemical_potentials,host_energy,temperature, entropy_marker, constituent_entropies, entropy_units)
        pp='x'
    elif (chem_pot_method == 2):
        chemical_potentials = calc_chemical_volatile(host_array,constituents,chemical_potentials,host_energy,temperature,entropy_marker, constituent_entropies, entropy_units,real_gas)
        pp=constituents[1]
    elif (chem_pot_method == 3):
        chemical_potentials = calc_chemical_volatile_rich_poor(host_array,constituents,chemical_potentials,host_energy,temperature,entropy_marker, constituent_entropies, entropy_units,real_gas)
        pp= constituents[2]
        
    opt_chem_pot=0
    #Calculate the dopant chemical potentials
    if (dopants[0] > 0):
        chemical_potentials, opt_chem_pot = dopant_chemical(dopants,chemical_potentials,temperature,real_gas)
   
    nu_e = 1
    #Optimise the dopant checmial potentials, if requsted
    if opt_chem_pot ==1:
    
        chemical_potentials = calc_opt_chem_pot(pp,loop,defects,dopants,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,potential_convergence,function_tol,maxiter_dop,'energy',0,0,0)         
       
    #Calclate the defect formation energies
    defects_form = defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,0)
    
    #Print formation energies
    defect_types=form_energies(defects_form,number_of_defects,tasks,bandgap,seedname)

    #Perform stability check, if requested.
    if ('stability' in tasks):
        stability_printout,indicator= stability_check(stability,chemical_potentials,indicator,'-')
        outputfile = str(seedname)+".output"
        with open(outputfile, 'a') as f:
            print("\n   Stability check results", file=f)
            print("   +------------------+-----------------+-----------------------------+-----------------+-------------------+",file=f)
            print("   |     Compound     | DFT energy (eV) | Chemical potential sum (eV) | Difference (eV) |      Message      |",file=f)
            print("   +------------------+-----------------+-----------------------------+-----------------+-------------------+",file=f)
            for i in np.arange(0, stability[0], 1):
                i = int(i)
                compound = stability_printout[i][0]
                compound_energy = float(stability_printout[i][1])
                chem_pot_sum = float(stability_printout[i][2])
                diff  = float(stability_printout[i][3])
                message  = stability_printout[i][4]
                print("   | %16s | %15f | %27f | %15f | %17s |" % (compound, compound_energy, chem_pot_sum,diff,message),file=f)
            print("   +------------------+-----------------+-----------------------------+-----------------+-------------------+",file=f)
    
    if ('form_plots' in tasks):
        print("\n>>> Task:'form_plots':")
        
        #Print defect formation energies
        formation_graphical_output(seedname, bandgap, defects, y_form_min, y_form_max,form_colour,scheme,number_of_defects,total_species, defect_types)

        #Create formation energy figures, in new directroy

        #Due to number of plots, a new directory is made to store plots. 
        directory = "mkdir "+str(seedname)+"_formation_plots"
        mv_graphfile = "mv formation_plot.p "+str(seedname)+".formation "+str(seedname)+".formation_grouped "+str(seedname)+"_formation_plots"
        directory_i = str(seedname)+"_formation_plots"
        form_graphfile = "gnuplot formation_plot.p"

        if os.path.exists(directory_i):
            shutil.rmtree(directory_i)
       
        os.system(directory)
        os.system(mv_graphfile)
        os.chdir(directory_i)
        os.system(form_graphfile)
        os.chdir('../')

        print("..> Successfully plotted formation energies. Plots found in", directory_i,)
    
#Brouwer diagram task
if ('brouwer' in tasks):
    print("\n>>> Task:'brouwer':")

    #Calculate the number of iterations in the loop
    num_iter = ((max_value-min_value)/iterator)+1
    #print("Number of iterations in the loop =",num_iter)
   
    #Loop over the requested range
    for b in np.arange(min_value, max_value+(iterator/2), iterator):
    
        if (loop == 0):     #Volatile partial pressure
            volatile_element = constituents[0]
            if (chem_pot_method == 2):
                constituents[1] = b
            if (chem_pot_method == 3):
                constituents[2] = b
            
            environment = "partial pressure"
        if (loop == 1):     #Temperature
            temperature = b
            if(entropy_marker == 1):
                entropies, constituent_entropies = calc_entropy(entropy_data,b,number_of_defects,constituents_name_list,chem_pot_method,seedname,0) 
            environment = "temperature"
           
        if (loop == 2):     #Dopant concentration
            number_dopants = int(dopants[0])
            fit_counter = 0 
            for i in np.arange(0,number_dopants, 1):          
                fit_potential = float(dopants[int((6*i)+3)])
                if fit_potential == 2: 
                    dopants[int((6*i)+4)]= (10**b)
                    dopant_xvar=dopants[int((6*i)+1)]
                    fit_counter+=1
            if fit_counter != 1:
                print("<!> No dopant (or too many) selected as independent variable. Review input file")
                exit()
            environment = "dopant concentration"
           
        if (loop == 3):     #Artificial charge
            art_dop_conc= (10**b)
            environment = "artificial dopant concentration"

        if (loop == 4):     #Dopant partial pressure
            number_dopants = int(dopants[0])
            fit_counter = 0 
            for i in np.arange(0,number_dopants, 1):          
                fit_potential = float(dopants[int((6*i)+3)])
                if fit_potential == 4:
                    dopants[int((6*i)+6)]= b
                    dopant_xvar=dopants[int((6*i)+1)]
                    fit_counter+=1
            if fit_counter != 1:
                print("<!> No dopant (or too many) selected as independent variable. Review input file")
                exit()
            environment = "dopant partial pressure"
             
        if ( x_variable == 1):     #Stoichiometry
            stoichiometry = 1          
     
        prog_bar = round((prog_meter/num_iter)*25)
        print("..> Calculating defect concentrations for",environment,prog_meter, "of", num_iter," [{0}]   ".format('#' * (prog_bar) + ' ' * (25-prog_bar)), end="\r", flush=True)
    
        #Calculate chemical potentials of host atoms
       
        chemical_potentials = [] 
        if (chem_pot_method == 0):
            chemical_potentials = calc_chemical_defined(host_array,constituents,chemical_potentials,host_energy,temperature, entropy_marker, constituent_entropies, entropy_units)
        elif (chem_pot_method == 1):
            chemical_potentials = calc_chemical_rich_poor(host_array,constituents,chemical_potentials,host_energy,temperature, entropy_marker, constituent_entropies, entropy_units)
        elif (chem_pot_method == 2):
            chemical_potentials = calc_chemical_volatile(host_array,constituents,chemical_potentials,host_energy,temperature,entropy_marker, constituent_entropies, entropy_units,real_gas)
            pp= constituents[1]
        elif (chem_pot_method == 3):
            chemical_potentials = calc_chemical_volatile_rich_poor(host_array,constituents,chemical_potentials,host_energy,temperature,entropy_marker, constituent_entropies, entropy_units,real_gas)
            pp= constituents[2]
            
        opt_chem_pot=0
        #Calculate the dopant checmical potentials
        if (dopants[0] > 0):
            
            chemical_potentials, opt_chem_pot = dopant_chemical(dopants,chemical_potentials,temperature,real_gas)
          
        nu_e = 1
        #Optimise the dopant chemical potentials, if requsted
        if opt_chem_pot ==1:
            
            chemical_potentials = calc_opt_chem_pot(b,loop,defects,dopants,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,potential_convergence,function_tol,maxiter_dop,environment,prog_meter,prog_bar,num_iter)         
       
        #Perform stability check, if requested.
        if ('stability' in tasks):
            stability_printout,indicator= stability_check(stability,chemical_potentials,indicator,b) 
     
        #Calclate the defect formation energies
        defects_form = defect_energies(defects,chemical_potentials,number_of_defects,host_supercell,tab_correction,E_VBM,total_species,use_coul_correction,length,dielectric,v_M,0)

        #Calculate final Fermi level and concentrations of defects. 
        (nu_e_final,concentrations,dopant_concentration_sum) = calc_fermi(b,loop,defects,defects_form,number_of_defects,bandgap,temperature,def_statistics,nu_e,condband,valband,electron_method,hole_method,fixed_e_conc,fixed_p_conc,art_dop_conc,art_dop_charge,charge_convergence,val_band_min,val_band_max,cond_band_min,cond_band_max,seedname,entropies, fu_uc,electron_mass_eff,hole_mass_eff,unit_vol,charged_sys,0)
        fermi.append([b,nu_e_final])

        #Perform check to determine if a very high conctration has been calcualted
        if concentration_check ==0:
            max_concentration = max(concentrations)
            if max_concentration > 0:
                print("<!> Very high concentrations predicted, exceeding 1 p.f.u.: This will not be visable on default Brouwer diagram.")
                concentration_check =1
 
        #Calculate the stoichiometry, if requested
      
        if (stoichiometry == 1 or stoichiometry ==2):
            
            concentrations = stoich(concentrations, defects, host_array,number_of_defects,dopants, x_variable, stoichiometry)
       
        if x_variable ==1:     
            concentrations.insert(1,nu_e_final)    
            concentrations.append(b)
            stoichiometry_list.append(concentrations[0])
        else:
            concentrations.insert(0,nu_e_final)    
            concentrations.insert(0,b)
            
        final_concentrations.append(concentrations)

        #Output file printing
        
        with open(outputfile, 'a') as f:
            print("\n-------------------------------------------------------------------------------------------------------------------","\n", file=f)
            print(">>> Task = brouwer, condition",prog_meter, "of", num_iter,"\n", file=f)
            if(loop == 0):
                print("   Volatile partial pressure : 10^(",b,") atm",file=f)
                print("   Temperature :",temperature , "K",file=f)
            if(loop == 1):
                if(chem_pot_method == (2 or 3)):
                    print("   Volatile partial pressure : 10^(",pp,") atm",file=f)
                print("   Temperature :",temperature , "K",file=f)
            if(loop == 2):
                print("   Temperature :",temperature,"K",file=f)
                if(chem_pot_method == (2 or 3)):
                    print("   Volatile partial pressure : 10^(",pp,") atm",file=f)
                print("   Dopant concentration : 10^(",b,") per f.u.",file=f)
            if(loop == 3):
                print("   Temperature :",temperature,"K",file=f)
                if(chem_pot_method == (2 or 3)):
                    print("   Volatile partial pressure : 10^(",pp,") atm",file=f)
                print("   Artificial dopant concentration : 10^(",b,") per f.u.",file=f)
                print("   Artificial dopant charge:", art_dop_charge, file=f)

            print("\n   Calculated chemical potentials:","\n", file=f)
            for i in np.arange(0,len(chemical_potentials)/2 , 1):
                i=int(i)
                print("   ",chemical_potentials[2*i],":",chemical_potentials[2*i+1],"eV",file=f)
            print("\n   Fermi level:",nu_e_final,"eV", file=f)
            print("\n   Concentrations:", file=f)
            print("   +----------------+--------+----------------------------------------+", file=f)
            print("   |     Defect     | Charge | log_{10}[Concentration] (per f.u.) /eV |", file=f)
            print("   +----------------+--------+----------------------------------------+", file=f)
            print ("   | %14s | %6s | %38f |" % ('Electrons', '-1', concentrations[2]), file=f)
            print ("   | %14s | %6s | %38f |" % ('Holes', '1', concentrations[3]), file=f)

            #Search through defects_form and print output
            for i in np.arange(0, number_of_defects, 1):
                i = int(i)
                defect_name = defects_form[i][0]
                charge = defects_form[i][4]
                concentration = concentrations[i+4]                
            
                print ("   | %14s | %6s | %38f |" % (defect_name, charge, concentration), file=f)

            print("   +----------------+--------+----------------------------------------+", file=f)

            if ('stability' in tasks):
                print("\n   Stability check results", file=f)
                print("   +------------------+-----------------+-----------------------------+-----------------+-------------------+",file=f)
                print("   |     Compound     | DFT energy (eV) | Chemical potential sum (eV) | Difference (eV) |      Message      |",file=f)
                print("   +------------------+-----------------+-----------------------------+-----------------+-------------------+",file=f)
                for i in np.arange(0, stability[0], 1):
                    i = int(i)
                    compound = stability_printout[i][0]
                    compound_energy = float(stability_printout[i][1])
                    chem_pot_sum = float(stability_printout[i][2])
                    diff  = float(stability_printout[i][3])
                    message  = stability_printout[i][4]
                    print("   | %16s | %15f | %27f | %15f | %17s |" % (compound, compound_energy, chem_pot_sum,diff,message),file=f)
                print("   +------------------+-----------------+-----------------------------+-----------------+-------------------+",file=f)
            prog_meter+=1
    print("\n..> Loop successfully executed")

    #print the seedname.fermi file
    if charged_sys == 1:
        print_fermi(fermi,seedname)    
    
    #Obtain new x range, if plotting as function as stoichiometry 
    if x_variable ==1:
        min_value = min(stoichiometry_list)
        max_value = max(stoichiometry_list)
          
    #Group defect concentrations

    if ('group' in tasks):     
        print("\n>>> Task:'group':")

        (final_grouped_concs,group_list) = group(final_concentrations,number_of_defects,defects,num_iter, stoichiometry)

        #Convert concentrations to cm^-3, if requested
        if y_variable == 1:
            final_grouped_concs = y_convert(final_grouped_concs,fu_uc, unit_vol,stoichiometry)

        #Print the seedname.res file
        print_results(final_grouped_concs,seedname)
       
        #Generate Brouwer diagram
        graphical_output(number_of_defects,min_value,max_value,final_concentrations,seedname,loop,gnuplot_version,min_y_range,host_name,group_list,electron_method,hole_method,dopants,host_array,1,conc_colour,electron_colour,hole_colour,scheme, dopant_xvar,stoichiometry,x_variable,total_species,volatile_element,charged_sys, y_variable,max_y_range )
        
    else:
        #Convert concentrations to cm^-3, if requested
        if y_variable == 1:
            final_concentrations = y_convert(final_concentrations,fu_uc, unit_vol,stoichiometry)

        #Print the seedname.res file
        print_results(final_concentrations,seedname)

        #Generate Brouwer diagram
        graphical_output(number_of_defects,min_value,max_value,final_concentrations,seedname,loop,gnuplot_version,min_y_range,host_name,defects,electron_method,hole_method,dopants,host_array,0,conc_colour,electron_colour,hole_colour,scheme, dopant_xvar, stoichiometry,x_variable,total_species,volatile_element,charged_sys, y_variable,max_y_range )

#print stability readout, if requested
if ('stability' in tasks):
    print("\n>>> Task:'stability':")
    if indicator == 1:
        print("<!> WARNING: Stability check for supplied compounds has found unstable compounds. See"," ",seedname,".output for details",sep="" )
    else:
        print("..> Stability check for supplied compounds complete. No unstable compounds found, see"," ",seedname,".output for details",sep="" )

if ('bibliography' in tasks):
    print("\n>>> Task:'bibliography':")
    bibliography(tasks,chem_pot_method,real_gas,entropy_marker)
    
#Launch gnuplot
        
if ('brouwer' in tasks):
    outputfile = str(seedname)+".eps"
    graphfile ="gnuplot "+str(seedname)+".p"
    if os.path.exists(outputfile):
        os.remove(outputfile)
    print("\n..> gnuplot messages:")
    os.system(graphfile)

#Plot and visulise Brouwer diagram. 
if('autodisplay' in tasks):

    osys = platform.system()
    
    if ('brouwer' in tasks):
        print("\n>>> Task: 'autodisplay':")
        print("..> Displaying defect concentration figure")
        
        if osys == 'Linux':
            command = "gv "+outputfile
            os.system(command)
        elif osys == 'Darwin':
            command = "open "+outputfile
            os.system(command)
        else:
            print("<!> Unable to open",outputfile,"on this system")

print("\n>>> Tasks complete")   
