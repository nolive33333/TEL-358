
import os
import csv
import pyaedt
from pyaedt import general_methods
import matplotlib.pyplot as plt 
import sys
import io
def draw_strucutre(ltota,wpatch,f1,f2):
        
    ###############################################################################
    # Launch Ansys Electronics Desktop (AEDT)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    non_graphical = True
    new_thread = True
    n=32
    m=32
    ###############################################################################
    # Save the project and results in the local folder
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    project_folder = os.path.join(os.getcwd(), "HFSS")
    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    project_name = os.path.join(project_folder, general_methods.generate_unique_name("wgf", n=2))


    # Instantiate the HFSS application
    hfss = pyaedt.Hfss(projectname=project_name + '.aedt',
                    designname="Waveguide",
                    specified_version="2023.1",
                    non_graphical=non_graphical,
                    new_desktop_session=False,
                    close_on_exit=False)

    # hfss.settings.enable_debug_methods_argument_logger = False  # Only for debugging.

    var_mapping = dict()  # Used by parse_expr to parse expressions.
    ###############################################################################
    # Initialize design parameters in HFSS.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    hfss.modeler.model_units = "mm"  # Set to mm

    hfss.materials.add_material("cond")
    hfss.materials["cond"].conductivity=str(6e7)
    hfss.materials.add_material("noncond")
    hfss.materials["noncond"].conductivity=str(0)
    hfss['ltota']=str(ltota)+"mm"
    hfss['wpatch']=str(wpatch)+"mm"
    hfss['h']=str(0.508)+"mm"
    hfss['e']=str(17)+"um"
    hfss.modeler.create_box(["-ltota/3", "-ltota", "0"], ["ltota*1.85", "ltota*2", "h"],
                            name="substrat", matname="Rogers RO4350 (tm)")
   
    l1=hfss.modeler.create_rectangle("XY",["wpatch/2", "-ltota", "h"], ["wpatch", "ltota-wpatch*"+str(int(n/2))],
                            name="microstrip")

    l2=hfss.modeler.create_rectangle("XY",[str(m)+"*wpatch-wpatch/2", "-ltota", "h"], ["wpatch", "ltota-wpatch*"+str(int(n/2))],
                            name="microstrip1")

    l3=hfss.modeler.create_rectangle("XY",[str(n)+"*wpatch-wpatch/2", "ltota", "h"], ["wpatch", "-ltota+wpatch*"+str(int(n/2))],
                            name="microstrip2")

    l4=hfss.modeler.create_rectangle("XY",["wpatch/2", "ltota", "h"], ["wpatch", "-ltota+wpatch*"+str(int(n/2))],
                            name="microstrip3")
    vac=hfss.modeler.create_box(["-ltota/3", "-ltota", "-h"], ["ltota*1.85", "ltota*2", "9*h"],
                            name="vac", matname="Vacuum")
    hfss.assign_perfecte_to_sheets(l1)
    hfss.assign_perfecte_to_sheets(l2)
    hfss.assign_perfecte_to_sheets(l3)
    hfss.assign_perfecte_to_sheets(l4)
    # hfss.assign_perfecte_to_sheets(gnd)
    # hfss.materials.add_material("mat")
    vac.display_wireframe = True
    hfss.assign_radiation_boundary_to_objects(vac)

    gnd=hfss.modeler.get_faceid_from_position([0, 0, 0], obj_name="substrat")
    hfss.assign_perfecte_to_sheets(gnd)
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()


    patch=[]
    for j in range(n):
        for i in range(m):
            # hfss.materials.add_material("mat"+str(j)+"_"+str(i))
            patch.append(hfss.modeler.create_box([str(j)+"*wpatch+wpatch/2",'-wpatch*'+str(int(m/2))+'+'+str(i)+'*wpatch' , 'h'], ['wpatch', 'wpatch',  'e'],
                                                    name="P"+str(j)+"_"+str(i), matname="copper"))
    sys.stdout = save_stdout
    
   ###############################################################################
# Assign wave ports to the end faces of the waveguid

    ports = []
    #P1
    p1=hfss.modeler.create_rectangle("XZ",[ "-8*wpatch/2","-ltota", "0"],["4*h","9*wpatch"])
    ports.append(hfss.wave_port(p1, name="P1", renormalize=False))
    #P2
    p2=hfss.modeler.create_rectangle("XZ",["-8*wpatch/2","ltota",  "0"],["4*h","9*wpatch"])
    ports.append(hfss.wave_port(p2, name="P2", renormalize=False))

    p3=hfss.modeler.create_rectangle("XZ",[str(m)+"*wpatch-8*wpatch/2", "-ltota", "0"],["4*h","9*wpatch"])
    ports.append(hfss.wave_port(p3, name="P3", renormalize=False))

    
    p4=hfss.modeler.create_rectangle("XZ",[str(n)+"*wpatch-8*wpatch/2", "ltota", "0"],["4*h","9*wpatch"])
    ports.append(hfss.wave_port(p4, name="P4", renormalize=False))    
    ###############################################################################
# create setup

    setup = hfss.create_setup("Setup1", setuptype="HFSSDriven",
                            Frequency=str((f1+f2)/2)+"Hz",
                            save_fields=False,
                            MaximumPasses=20,
                            MinimumPasses=5)
    sweep=setup.create_frequency_sweep(
        unit="GHz",
        sweepname="Sweep1",
        freqstart=f1/1e9,
        freqstop=f2/1e9,
        sweep_type="Discrete",
        num_of_freq_points=101, 
        save_fields=False,
    )
    setup.props["MaxDeltaS"] =0.02
    
    #################################################################################
#  Solve the project with one task

    # setup.analyze(num_tasks=1)
    return hfss, patch, setup




def mise_a_jour_cond(hfss,tab,patch,k):

 
    for i in range(len(tab)):
        if tab[i]==1:
            # hfss.materials["mat"+str(j)+"_"+str(i)].conductivity=str(6e7)
            patch[i].color=(255,255,0)
            patch[i].material_name='cond'
        else:
            patch[i].color=(0,0,0)
            patch[i].material_name='noncond'
            # hfss.materials["mat"+str(j)+"_"+str(i)].conductivity=str(0)
    
    with open("patch"+str(k)+".csv", 'w+',newline ='') as csvfile:   
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(tab)

def run_and_get_data(hfss,setup,k):
#   Solve the project with one task

    setup.analyze(num_tasks=1,num_cores=16)

    x = hfss.post.get_solution_data(["mag(S(P1,P1))","mag(S(P2,P2))","mag(S(P3,P3))","mag(S(P4,P4))","mag(S(P2,P1))","mag(S(P3,P1))","mag(S(P4,P1))","mag(S(P3,P2))","mag(S(P4,P2))","mag(S(P4,P3))"])
    # x.plot(curves=["dB(S(P2,P1))"])# math_formula="re", xlabel="Freq", ylabel="L and Q")
    x.export_data_to_csv("data"+str(k)+".csv", delimiter=";")
    
    

def createSetup(hfss,f1,f2):
    
    setup = hfss.create_setup("Setup1", setuptype="HFSSDriven",
                            Frequency=str((f1+f2)/2/1e9)+"GHz",
                            save_fields=False,
                            MaximumPasses=15,
                            MinimumPasses=5)
    sweep=setup.create_frequency_sweep(
        unit="GHz",
        sweepname="Sweep1",
        freqstart=f1/1e9,
        freqstop=f2/1e9,
        sweep_type="Discrete",
        num_of_freq_points=101,
        save_fields=False, 
    )
    setup.props["MaxDeltaS"] =0.02
    return setup