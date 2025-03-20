import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from math import inf
from ctypes import *
import pickle
import traceback
import socket

# Ensure a suitable Matplotlib backend
matplotlib.use('TkAgg')  # Adjust backend as needed

# Import necessary modules (assumes these are available in your environment)
from PSGManager import PSGManager
from PPCL_Bare_Bones import LaserControl
from PolCTRLManager import PolCTRLManager
from NIDAQ_USBv4 import NIDAQ_USB
from OSWManager import OSWManager
from GenericUser_v2 import GenericUser
from PSOManager_v2_1 import PSOManager
from TimeTaggerFunctions import TimeTaggerManager
from RigolManager import RigolDG4162Manager

from PSO_Running_v24 import *

def HOM_time_scan(Alice,Bob,Charlie):
    Alice.PSG.polSET(Alice.PSG.H)
    Bob.PSG.polSET(Bob.PSG.H)

    delays = np.arange(20000, 60000, 500)
    counts_data = np.zeros([4, len(delays)])
    coincidences_data = []
    # Define channel pairs for coincidences
    channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)] # Example: Coincidences between Ch1 and Ch4
    #channel_pairs = [(1,3), ]
    ch1_params = Charlie.Rigol1.set_channel_params(channel="1", amplitude=1.5, burst_delay=delays[0],burst_cycles=800 ,trigger_source="EXT")
    Charlie.Rigol1.configure_channel(**ch1_params)
    Charlie.Rigol1.control_channel(1, "ON")
    
    ch2_params = Charlie.Rigol2.set_channel_params(channel="2", amplitude=1.5, burst_delay=0, burst_cycles=800, trigger_source="EXT")
    # Rigol.control_channel(2, "OFF")Charlie.
    Charlie.Rigol2.configure_channel(**ch2_params)
    Charlie.Rigol2.control_channel(2, "ON")


    for i, delay in enumerate(delays):
        burst_delay = delay * 1E-9

        #Rigol.configure_channel(**ch1_params)
        #Rigol.control_channel(1, "ON")
        Charlie.Rigol1.dev.write(f"SOUR1:BURS:TDEL {burst_delay}")
        
        # Get coincidences for Ch1 and Ch4
        coincidence_hist = Charlie.TimeTaggerManager.getCoincidences(channel_pairs,measurement_time=1, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
        HOM_coincidence_hist=coincidence_hist[1]
        print(f'Delay:{delay*1E-3}us, Coin counts {np.sum(HOM_coincidence_hist)}')

        # coincidence_hist2 = TTU.getCoincidences( [(1, 4),], bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
        # print(coincidence_hist2)
        coincidences_data.append(np.sum(HOM_coincidence_hist))
        HOM_vis = ( np.mean(np.sort(coincidences_data)[-5:]) - min(coincidences_data) ) / np.mean(np.sort(coincidences_data)[-5:])
        time.sleep(0.01)
    print(f'HOM visibility:{HOM_vis}')
    # # Plot coincidences histogram for Ch1 and Ch4
    
    # ax_HOM.clear()
    # ax_HOM.set_xlabel("Delay (us)")
    # ax_HOM.set_ylabel("Coin Counts ")
    # ax_HOM.set_title("HOM")
    # ax_HOM.plot(delays/1000, coincidences_data)

    # plt.ion()
    # plt.show()
    
    print(np.argmin(coincidences_data))
    print(f'Best delay: {delays[np.argmin(coincidences_data)]*1E-3} us ')
    Charlie.Rigol1.dev.write(f"SOUR1:BURS:TDEL {delays[np.argmin(coincidences_data)]*1E-9}")

    return np.sum(coincidences_data)

def BSM_psi_plus(Alice, Bob, Charlie,measurement_time=1):

    Alice.PSG.polSET(Alice.PSG.D)
    Bob.PSG.polSET(Bob.PSG.D)

    Charlie.Rigol1.control_channel(1, "ON") 
    Charlie.Rigol2.control_channel(2, "ON")

    channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    #single_counts = Charlie.TimeTaggerManager.getChannelCountRate([1, 2, 3, 4], printout=True)
    coincidence_hist = Charlie.TimeTaggerManager.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100, measurement_time=measurement_time) # binwidth in ps (check this)
    # print(coincidence_hist)
    for j in range(len(coincidence_hist)):
        print(f"Channel pair: {channel_pairs[j]}, Total coincidences: {np.sum(coincidence_hist[j])}")
    coincidence_his_sum=np.sum(coincidence_hist,axis=1)
    print(coincidence_his_sum)



    psi_p=np.sum(coincidence_hist[0])+np.sum(coincidence_hist[5])
    psi_m=np.sum(coincidence_hist[2])+np.sum(coincidence_hist[3])

    vis=(psi_p-psi_m)/psi_p
    print(f'Visibility:{vis}')

    # ax_BSM.clear()
    # ax_BSM.set_xlabel("Channel pair")
    # ax_BSM.set_ylabel("Coin Counts rate (1/s)")
    # ax_BSM.set_title("BSM")
    # ax_BSM.bar([1,2,3,4,5,6],coincidence_his_sum , color=['r','b','b','b','b','r'], alpha=0.7)
    # ax_BSM.set_xticks([1, 2, 3, 4, 5, 6])
    # ax_BSM.set_xticklabels(['(1,2)', '(1,3)', '(1,4)', '(2,3)', '(2,4)', '(3,4)'])
    

    plt.ion()
    plt.show()
    plt.pause(0.01)

    return psi_p, psi_m

def BSM_psi_plus_fast(Alice, Bob, Charlie,measurement_time=1):


    channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    #single_counts = Charlie.TimeTaggerManager.getChannelCountRate([1, 2, 3, 4], printout=True)
    coincidence_hist = Charlie.TimeTaggerManager.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100, measurement_time=measurement_time) # binwidth in ps (check this)
    # print(coincidence_hist)

    psi_p=np.sum(coincidence_hist[0])+np.sum(coincidence_hist[5])
    psi_m=np.sum(coincidence_hist[2])+np.sum(coincidence_hist[3])

    vis=(psi_p-psi_m)/psi_p
    print(f'Visibility:{vis}')

    return psi_p, psi_m

def BSM_psi_minus(Alice, Bob, Charlie, measurement_time=1):

    Alice.PSG.polSET(Alice.PSG.D)
    Bob.PSG.polSET(Bob.PSG.A)

    Charlie.Rigol1.control_channel(1, "ON") 
    Charlie.Rigol2.control_channel(2, "ON")

    channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    #single_counts = Charlie.TimeTaggerManager.getChannelCountRate([1, 2, 3, 4], printout=True)
    coincidence_hist = Charlie.TimeTaggerManager.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100, measurement_time=measurement_time) # binwidth in ps (check this)
    # print(coincidence_hist)
    for j in range(len(coincidence_hist)):
        print(f"Channel pair: {channel_pairs[j]}, Total coincidences: {np.sum(coincidence_hist[j])}")
    coincidence_his_sum=np.sum(coincidence_hist,axis=1)
    print(coincidence_his_sum)



    psi_p=np.sum(coincidence_hist[0])+np.sum(coincidence_hist[5])
    psi_m=np.sum(coincidence_hist[2])+np.sum(coincidence_hist[3])

    vis=(psi_m-psi_p)/psi_m
    print(f'Visibility:{vis}')


    # ax_BSM.clear()
    # ax_BSM.set_xlabel("Channel pair")
    # ax_BSM.set_ylabel("Coin Counts rate (1/s)")
    # ax_BSM.set_title("BSM")
    # ax_BSM.bar([1,2,3,4,5,6],coincidence_his_sum , color=['b','b','r','r','b','b'], alpha=0.7)
    # ax_BSM.set_xticks([1, 2, 3, 4, 5, 6])
    # ax_BSM.set_xticklabels(['(1,2)', '(1,3)', '(1,4)', '(2,3)', '(2,4)', '(3,4)'])
    

    # plt.ion()
    # plt.show()
    # plt.pause(0.01)

    return psi_p, psi_m

def BSM_psi_minus_fast(Alice, Bob, Charlie, measurement_time=1):


    channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    #single_counts = Charlie.TimeTaggerManager.getChannelCountRate([1, 2, 3, 4], printout=True)
    coincidence_hist = Charlie.TimeTaggerManager.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100, measurement_time=measurement_time) # binwidth in ps (check this)

    psi_p=np.sum(coincidence_hist[0])+np.sum(coincidence_hist[5])
    psi_m=np.sum(coincidence_hist[2])+np.sum(coincidence_hist[3])

    vis=(psi_m-psi_p)/psi_m
    print(f'Visibility:{vis}')

    return psi_p, psi_m

def init_device():
    # TODO Add conditions checking Bob visibilities during steps 1 and 2 and Alice checking that Bob's visibilities are reasonable

    num_runs = 1 # it can take around 10-25 minutes to run
    duration = 2*60 * 60 # amount of time to run the stabilization algorithm for
    tracking_interval=30
    # Define Alice and Bob visibilities dictionaries to save to .pkl files later

    # Define np.array of duration per run
    timing = np.zeros((num_runs))

    # Create log file with unique name
    current_date = datetime.now().strftime("%Y-%m-%d")
    path = "C:/Users/QUANT-NET Admin/Documents/Python codes/QUANTNET/INQNET_DLPolarizationStabilization_Main/"
    runner_idx = 1
    os.makedirs(path + "PSO Data", exist_ok=True)
    while os.path.exists(path + f"PSO Data/{current_date}_Polarization_Stabilization_PSO_log_{runner_idx}.txt"):
        runner_idx += 1
    log_filename = path + f"PSO Data/{current_date}_Polarization_Stabilization_PSO_log_{runner_idx}.txt"

    # Important device resources
    Rigol_resources = ("USB0::0x1AB1::0x0641::DG4E251600982::INSTR", "TCPIP0::10.0.0.201::INSTR" )

    log_file = open(log_filename, 'w')
    message = f"Particle Swarm Optimization Polarization Stabilization Test {runner_idx}, {current_date} step-by-step log" # message should not be larger than 116 characters
    # notes = f"(Bouncing from the bounds with particle velocity damping = {DAMPING})"
    # notes = "(Clipping at the boundaries)"
    notes = "(Clipping at the boundaries and resetting a particle's velocity and voltage to a random value"
    notes2 = "if it hits the boundary)"
    notes3 = "Using LBNL laser and time tagger with live-plotting"
    log_event(log_file, "[" + "=" * 118 + "]")
    format_title(log_file, message)
    format_title(log_file, notes)
    format_title(log_file, notes2)
    format_title(log_file, notes3)
    log_event(log_file, "[" + "=" * 118 + "]\n")

    # Initialize devices
    log_event(log_file, "##### Initializing Alice's devices #####")
    #Alice = GenericUser(name= "Alice", laser_port="COM19", PSG_port="COM12", PolCTRL_port1="COM20") # LBNL
    Alice = GenericUser(name= "Alice", PSG_port="COM12", PolCTRL_port1="COM20") # LBNL
    #Alice.laser.connect_laser()
    # Alice.laser.turn_on()
    Alice.PSG.connect()
    Alice.Pol_CTRL1.connect()
    log_event(log_file, "##### Alice's devices are initialized #####")
    
    log_event(log_file, "##### Initializing Bob's devices ##### \n")
    
    Bob = GenericUser(name= "Bob", PSG_port="COM9") # LBNL
    
    Bob.PSG.connect()
    
    log_event(log_file, "##### Bob's devices are initialized #####\n")

    log_event(log_file, "##### Initializing Charlie's devices #####\n")
    Charlie = GenericUser(name= "Charlie", PolCTRL_port1="COM22", PolCTRL_port2="COM21",
                          NIDAQ_USB_id="Dev1", time_tagger_filename="TimeTaggerConfig.yaml",
                          Rigol_resources=Rigol_resources) # LBNL
    # Rigol channel params
    for Rigol in Charlie.Rigols:
        Rigol.ch1_params = Rigol.set_channel_params(channel="1", burst_delay=47E-6,burst_cycles=800, trigger_source="EXT")
        Rigol.ch2_params = Rigol.set_channel_params(channel="2", burst_delay=0 ,   burst_cycles=800, trigger_source="EXT")
        Rigol.connect()
    # COM 8 V_2π = 0.887 V, COM 11 V_2π = 0.94 V, COM 12 V_2π = 0.752
    Charlie.Pol_CTRL1.connect()
    Charlie.Pol_CTRL2.connect()
    
    Charlie.meas_device = Charlie.TimeTaggerManager
    log_event(log_file, "##### Charlie's devices are initialized #####\n")

    # Define important values
    dim = 3

    pso_params = PSOParams(Alice, Bob, Charlie,
                           Measure_CostFunction, log_event, log_file, Charlie.meas_device)
    
    ### Default OSW connetion

    OSW_operate(OSW=Charlie.Rigols, Alice_Switch_status=1, Bob_Switch_status=0,
                meas_device=Charlie.meas_device,
                channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                Alice=Alice, Bob=Bob)
    raw_Alice_power = MeasureFunction(Charlie.meas_device, [1,2,3,4])
    print(f"Alice raw power: {raw_Alice_power}")
    Charlie.meas_device.Alice_power = np.sum(raw_Alice_power)
    print(f"Alice power: {Charlie.meas_device.Alice_power}")

    # Charlie.OSW.StatusSet(Charlie.OSW.Alice_Switch, 0)
    # Charlie.OSW.StatusSet(Charlie.OSW.Bob_Switch, 1)
    _, Bob_power = OSW_operate(OSW=Charlie.Rigols, Alice_Switch_status=0, Bob_Switch_status=1, meas_device=Charlie.meas_device, channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                Alice=Alice, Bob=Bob)
    raw_Bob_power = MeasureFunction(Charlie.meas_device, [1,2,3,4])
    print(f"Bob power after switch: {Bob_power}")
    print(f"Bob raw power: {raw_Bob_power}")
    Charlie.meas_device.Bob_power = np.sum(raw_Bob_power)
    print(f"Bob power: {Charlie.meas_device.Bob_power}")

    while True:
        OSW_operate(OSW=Charlie.Rigols, Alice_Switch_status=0, Bob_Switch_status=0, meas_device=Charlie.meas_device, channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                    Alice=Alice, Bob=Bob)
        raw_zero_power = MeasureFunction(Charlie.meas_device, [1,2,3,4])
        print(f"Raw zero power: {raw_zero_power}")
        Charlie.meas_device.zero_power = raw_zero_power
        log_event(log_file, f"measure zero power: {Charlie.meas_device.zero_power}")
        break # Remove after adding condition below
    
    BSM_tracking_data=[]

    try:
        tic_total = time.perf_counter()
        run_plural = "s" if num_runs != 1 else ""


        ps = PolarizationStabilization(pso_params, duration,tracking_interval)
        #ps.initial_stabilization()

        min_delay = 45000
        ch1_params = Charlie.Rigol1.set_channel_params(channel="1",amplitude=1, burst_delay=min_delay*1E-9, burst_cycles=800, trigger_source="EXT")
        Charlie.Rigol1.configure_channel(**ch1_params)
        Charlie.Rigol1.control_channel(1, "ON")

        ch2_params = Charlie.Rigol2.set_channel_params(channel="2",amplitude=1, burst_delay=0, burst_cycles=800, trigger_source="EXT")
        Charlie.Rigol2.configure_channel(**ch2_params)
        Charlie.Rigol2.control_channel(2, "ON")
        #time.sleep(5)
        return 0
    
    except KeyboardInterrupt:
        log_event(log_file, "Polarization stabilization interrupted by user.")
        return 1
    except Exception as e:
        log_event(log_file, f"An error occurred: {e}")
        traceback.print_exc()  # Print full traceback
        return 1

def stabilization():
    try:
        ps.initial_stabilization()
        # ps.check_H_stability()
        # ps.check_D_stability()        
        
        # ##BSM
        # while True:
        #     print('='*120)
        #     command = input("Enter command or 'exit' to quit: ")

        #     command_list=[command]
        #     #command_list=['pol_init','many_psi+','many_psi-','check_H','many_psi+','many_psi+','many_psi+','check_D','many_psi-','exit']

        #     for command in command_list:
        #         if command.lower() == "exit":
        #             print("Exiting...")
                    
        #             break  # Exit the loop

                
        #         if command == "psi+":
        #             BSM_psi_plus(Alice,Bob,Charlie)

        #         elif command == "psi-":
        #             BSM_psi_minus(Alice,Bob,Charlie)

        #         elif command == "HOM":
        #             HOM_time_scan(Alice,Bob,Charlie)

        #         elif command == "only_check":
        #             ps.only_check()
                    
        #         elif command == "check_H":
        #             ps.check_H_stability()

        #         elif command == "check_D":
        #             ps.check_D_stability()

        #         elif command == "pol_init":
        #             ps.initial_stabilization()

        #         elif command == 'many_psi+':
        #             for _ in range(4):
        #                 BSM_psi_plus(Alice,Bob,Charlie)
        #                 for _ in range(30):
        #                     psi_p,psi_m=BSM_psi_plus_fast(Alice,Bob,Charlie)
        #                     tocPS = time.perf_counter()
        #                     time_elapsed = tocPS - ps.ticPS  
        #                     BSM_tracking_data.append([time_elapsed,psi_p,psi_m])                        
        #                 ps.only_check()

        #         elif command == 'many_psi-':
                    
        #             for _ in range(4):
        #                 BSM_psi_minus(Alice,Bob,Charlie)
        #                 for _ in range(30):
        #                     psi_p,psi_m=BSM_psi_minus_fast(Alice,Bob,Charlie)
        #                     tocPS = time.perf_counter()
        #                     time_elapsed = tocPS - ps.ticPS  
        #                     BSM_tracking_data.append([time_elapsed,psi_p,psi_m])
        #                 ps.only_check()

        #         elif command == 'test':
        #             for _ in range(10*60*2):
        #                 time.sleep(30)
        #                 ps.only_check()

        #         else:
        #             print("Invalid input.")
        #             continue  # Ask for input again
        #     if command.lower() == "exit":
        #         print("Exiting...")                  
        #         break  # Exit the loop

    except KeyboardInterrupt:
        log_event(log_file, "Polarization stabilization interrupted by user.")
    except Exception as e:
        log_event(log_file, f"An error occurred: {e}")
        traceback.print_exc()  # Print full traceback
    
def cleanup():
    toc_total = time.perf_counter()
    delta_t_total = toc_total - tic_total
    delta_t_total_hr = int(delta_t_total / 3600)
    delta_t_total_min = int((delta_t_total % 3600) / 60)
    delta_t_total_sec = delta_t_total % 60
    log_event(log_file, f"\nTotal time elapsed for all {num_runs} Run{run_plural} is {delta_t_total_hr} hours, {delta_t_total_min} minutes and {delta_t_total_sec} seconds")

    log_event(log_file, "end")
    # Alice.laser.turn_off()
    Alice.disconnect_all()
    log_event(log_file, "Alice's laser is off")
    # Bob.laser.turn_off()
    Bob.disconnect_all()
    log_event(log_file, "Bob's laser is off")
    Charlie.Rigol1.control_channel(1, "OFF")
    Charlie.Rigol2.control_channel(2, "OFF")
    Charlie.disconnect_all()
    Charlie.OSW.disconnect()
    log_event(log_file, "")

    # ps.save_tracking_data(path + f"PSO Data/{current_date}_Polarization_Stabilization_TrackingData_{runner_idx}.npz") # Only saves last run
    # ps.save_plot(path + f"PSO Data/{current_date}_Polarization_Stabilization_TrackingData_{runner_idx}.png") # Only saves last run
    # np.savez(path + f"PSO Data/{current_date}_BSM_TrackingData_{runner_idx}.npz", BSM_tracking_data=BSM_tracking_data)
    
    log_event(log_file, 'Data saved!')
    log_event(log_file, path + f"PSO Data/{current_date}_Polarization_Stabilization_TrackingData_{runner_idx}.png")
    log_file.close()

if __name__ == "__main__":
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((socket.gethostname(), 5006))
    serversocket.listen(5)

    while True:
        (clientsocket, address) = serversocket.accept()
        msg = clientsocket.recv(1024)
        if msg == b"init":
            try:
                # result = init_device()
                result = 0
            except Exception:
                result = "Not OK"
        elif msg == b"calib":
            # result = stabilization()
            result = 0
        elif msg == b"cleanup":
            # result = cleanup()
            result = 0
        clientsocket.send(str(result).encode('utf8'))
