import pyvisa
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml

from PSGManager import PSGManager
from PPCL_Bare_Bones import LaserControl
from PolCTRLManager import PolCTRLManager
from NIDAQ_USBv4 import NIDAQ_USB
from OSWManager import OSWManager
from GenericUser_v2 import GenericUser
from PSOManager_v2 import PSOManager
from TimeTaggerFunctions import TimeTaggerManager
from RigolManager import RigolDG4162Manager

def log_event(log_file, message, log=True):
    """Log the event message to the file and print it."""
    print(message)
    if log==True:
        log_file.write(message + "\n")


class HOM_Delay_Measurement:
    def __init__(self, users, delays, log_event, log_file,
                 burst_cycles=800, trigger_source="EXT"):
        for key, user in users.items():
            setattr(self, key, user)
        
        self.delays = delays
        self.channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        self.log_event = log_event
        self.log_file = log_file
        self.burst_cycles = burst_cycles
        self.trigger_source = trigger_source

        self.Rigol = self.Charlie.Rigol
        self.TTU = self.Charlie.TimeTaggerManager

        self.coincidences_dict = {}
        for channel_pair in channel_pairs:
            self.coincidences_dict[f"{channel_pair}"] = np.zeros(len(delays))
        
        self.coincidences_array = np.zeros([len(channel_pairs), len(delays)])

        self.min_delay = 1j
    
    def config_Rigol(self):
            ch1_params = self.Rigol.set_channel_params(
                channel="1",
                burst_delay=self.delays[0],
                burst_cycles=self.burst_cycles,
                trigger_source=self.trigger_source
            )
            # self.Rigol.control_channel(1, "OFF") # Purely for debugging purposes
            self.Rigol.configure_channel(**ch1_params)
            self.Rigol.control_channel(1, "ON")

            ch2_params = self.Rigol.set_channel_params(
                channel="2",
                burst_delay=0,
                burst_cycles=self.burst_cycles,
                trigger_source=self.trigger_source
            )
            # self.Rigol.control_channel(2, "OFF")
            self.Rigol.configure_channel(**ch2_params)
            self.Rigol.control_channel(2, "ON")

            # return ch1_params, ch2_params
    
    def measure(self, auto_config=True):
        try:
            if auto_config:
                self.config_Rigol()
            for i, delay in enumerate(self.delays):
                burst_delay = delay * 1E-9
                self.log_event(self.log_file, f"Difference in delay between Channel 1 and Channel 2: {delay*1E-3} µs")

                #Rigol.configure_channel(**ch1_params)
                #Rigol.control_channel(1, "ON")
                self.Rigol.dev.write(f"SOUR1:BURS:TDEL {burst_delay}")
                
                # Get coincidences for all channel pairs
                coincidence_hist = self.TTU.getCoincidences(self.channel_pairs, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
                for j in range(len(coincidence_hist)):
                    self.log_event(self.log_file, f"Channel pair: {self.channel_pairs[j]}, Total coincidences: {np.sum(coincidence_hist[j])}")

                self.coincidences_array[:, i] = np.sum(coincidence_hist, axis=1)
                
                time.sleep(0.1)
                
            self.min_delay = self.delays[np.argmin(coincidences_array)]
        except KeyboardInterrupt:
            self.log_event(self.log_file, "HOM measurement interrupted by user")
        except Exception as e:
            self.log_event(self.log_file, str(e))
        finally:
            for cp_idx in range(len(self.channel_pairs)):
                channel_pair = self.channel_pairs[cp_idx]
                self.coincidences_dict[f"{channel_pair}"] = self.coincidences_array[cp_idx, :]
        
        return self.min_delay, self.coincidences_array, self.coincidences_dict

    def plot_coincidences(self, channel_pair):
        """Plot coincidences histogram for the given channel pair"""
        plt.figure()
        plt.title(f"Coincidences Histogram: Channels {channel_pair}")
        plt.plot(self.delays/1000, self.coincidences_dict[f"{channel_pair}"])
        plt.xlabel("Time (µs)")
        plt.ylabel("Counts")
    
    def save_coincidences_dict(self, filename):
        with open(filename, 'w') as file:
            yaml.dump(self.coincidences_dict, file, default_flow_style=False)

class BSM_Delay_Measurement:
    def __init__(self, users, log_event, log_file, min_delay=55000,
                 burst_cycles=800, trigger_source="EXT"): # Maybe adapt it later to calculate min_delay by itself or from a file?
        for key, user in users.items():
            setattr(self, key, user)
        
        self.min_delay = min_delay
        self.log_event = log_event
        self.log_file = log_file
        self.channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        self.burst_cycles = burst_cycles
        self.trigger_source = trigger_source

        self.Rigol = self.Charlie.Rigol
        self.TTU = self.Charlie.TimTaggerManager

        self.error_psi_minus = np.inf
        self.error_psi_plus = np.inf
    
    def config_Rigol(self, ch12_on_off=["ON", "OFF"]):
        ch1_params = self.Rigol.set_channel_params(
            channel="1",
            burst_delay=self.min_delay,
            burst_cycles=self.burst_cycles,
            trigger_source=self.trigger_source
        )
        self.Rigol.control_channel(1, ch12_on_off[0])

        ch2_params = self.Rigol.set_channel_params(
            channel="2",
            burst_delay=0,
            burst_cycles=self.burst_cycles,
            trigger_source=self.trigger_source
        )
        self.Rigol.control_channel(2, ch12_on_off[1])
    
    def coincidences_for_2_polarizations(self, pol_Alice, pol_Bob):
        self.Alice.PSG.polSET(getattr(self.Alice.PSG, pol_Alice))
        self.Bob.PSG.polSET(getattr(self.Alice.PSG, pol_Bob))

        single_counts = self.TTU.getChannelCountRate([1, 2, 3, 4], printout=True)
        coincidence_hist = self.TTU.getCoincidences(self.channel_pairs, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
        # print(coincidence_hist)
        coincidences = np.sum(coincidence_hist, axis=1)
        for j in range(len(coincidence_hist)):
            self.log_event(self.log_file, f"Channel pair: {self.channel_pairs[j]}, Total coincidences: {np.sum(coincidence_hist[j])}")

        return coincidences

    def measure(self, vis_type="H&V", auto_config=True):
        if auto_config:
            self.config_Rigol()
        psi_minus_ch_pairs = [(1,4),(2,3)]
        psi_plus_ch_pairs = [(1,2),(3,4)]
        if vis_type == "D&A":
            coincidences_DD = self.coincidences_for_2_polarizations(pol_Alice="D", pol_Bob="D")
            N_DD_minus = coincidences_DD[channel_pairs.index((1,4))] + coincidences_DD[channel_pairs.index((2,3))]
            N_DD_plus = coincidences_DD[channel_pairs.index((1,2))] + coincidences_DD[channel_pairs.index((3,4))]

            coincidences_DA = self.coincidences_for_2_polarizations(pol_Alice="D", pol_Bob="A")
            N_DA_minus = coincidences_DA[channel_pairs.index((1,4))] + coincidences_DA[channel_pairs.index((2,3))]
            N_DA_plus = coincidences_DA[channel_pairs.index((1,2))] + coincidences_DA[channel_pairs.index((3,4))]

            self.error_psi_minus = N_DD_minus / (N_DD_minus + N_DA_minus)
            self.error_psi_plus = N_DD_plus / (N_DD_plus + N_DA_plus)
            
            self.log_event(self.log_file, f"BSM Visibility Psi-: {self.error_psi_minus}")
            self.log_event(self.log_file, f"BSM Visibility Psi+: {self.error_psi_plus}")
        
        elif vis_type == "H&V":
            coincidences_HH = self.coincidences_for_2_polarizations(pol_Alice="H", pol_Bob="H")
            N_HH_minus = coincidences_HH[channel_pairs.index((1,4))] + coincidences_HH[channel_pairs.index((2,3))]
            N_HH_plus = coincidences_HH[channel_pairs.index((1,2))] + coincidences_HH[channel_pairs.index((3,4))]

            coincidences_HV = self.coincidences_for_2_polarizations(pol_Alice="H", pol_Bob="V")
            N_HV_minus = coincidences_HV[channel_pairs.index((1,4))] + coincidences_HV[channel_pairs.index((2,3))]
            N_HV_plus = coincidences_HV[channel_pairs.index((1,2))] + coincidences_HV[channel_pairs.index((3,4))]

            self.error_psi_minus = N_HH_minus / (N_HH_minus + N_HV_minus)
            self.error_psi_plus = N_HH_plus / (N_HH_plus + N_HV_plus)
            
            self.log_event(self.log_file, f"BSM Visibility Psi-: {self.error_psi_minus}")
            self.log_event(self.log_file, f"BSM Visibility Psi+: {self.error_psi_plus}")
        
        return self.error_psi_minus, self.error_psi_plus

if __name__ == "__main__":
    # Create log file with unique name
    current_date = datetime.now().strftime("%Y-%m-%d")
    path = "C:/Users/QUANT-NET Admin/Documents/Python codes/QUANTNET/INQNET_DLPolarizationStabilization_Main/"
    runner_idx = 1
    os.makedirs(path + "PSO Data", exist_ok=True)
    while os.path.exists(path + f"PSO Data/{current_date}_Polarization_Stabilization_PSO_log_{runner_idx}.txt"):
        runner_idx += 1
    log_filename = path + f"PSO Data/{current_date}_Polarization_Stabilization_PSO_log_{runner_idx}.txt"

    # Important device resources
    # Rigol_resource = "USB0::0x1AB1::0x0641::DG4D152500730::INSTR"
    Rigol_resource = "USB0::0x1AB1::0x0641::DG4E251600982::INSTR"

    log_file = open(log_filename, 'w')
    message = f"Particle Swarm Optimization Polarization Stabilization Test {runner_idx}, {current_date} step-by-step log" # message should not be larger than 116 characters
    # notes = f"(Bouncing from the bounds with particle velocity damping = {DAMPING})"
    # notes = "(Clipping at the boundaries)"
    notes = "(Clipping at the boundaries and resetting a particle's velocity and voltage to a random value"
    notes2 = "if it hits the boundary)"
    notes3 = "Using LBNL laser and time tagger"
    log_event(log_file, "[" + "=" * 118 + "]")
    format_title(log_file, message)
    format_title(log_file, notes)
    format_title(log_file, notes2)
    format_title(log_file, notes3)
    log_event(log_file, "[" + "=" * 118 + "]\n")

    # Initialize devices
    log_event(log_file, "##### Initializing Alice's devices #####")
    Alice = GenericUser(name= "Alice", laser_port="COM19", PSG_port="COM12", PolCTRL_port1="COM20") # LBNL
    Alice.laser.connect_laser()
    # Alice.laser.turn_on()
    Alice.PSG.connect()
    Alice.Pol_CTRL1.connect()
    log_event(log_file, "##### Alice's devices are initialized #####")
    
    log_event(log_file, "##### Initializing Bob's devices ##### \n")
    # Bob = GenericUser(name= "Bob", laser_port="COM10", PSG_port="COM9", PolCTRL_port1="COM12") # COM4 is LUNA, COM21 is LADAQ+INQNET PolCTRL
    Bob = GenericUser(name= "Bob", laser_port="COM10", PSG_port="COM9") # LBNL
    Bob.laser.connect_laser()
    Bob.PSG.connect()
    # Bob.laser.turn_on()
    log_event(log_file, "##### Bob's devices are initialized #####\n")

    log_event(log_file, "##### Initializing Charlie's devices #####\n")
    Charlie = GenericUser(name= "Charlie", PolCTRL_port1="COM22", PolCTRL_port2="COM21",
                          NIDAQ_USB_id="Dev1", time_tagger_filename="TimeTaggerConfig.yaml",
                          Rigol_resource=Rigol_resource) # LBNL
    Charlie.Pol_CTRL1.connect()
    Charlie.Pol_CTRL2.connect()
    Charlie.Rigol.connect()
    Charlie.meas_device = Charlie.TimeTaggerManager
    log_event(log_file, "##### Charlie's devices are initialized #####\n")

    delays = np.arange(35000, 70000, 1000)
    
    users = [Alice, Bob, Charlie]
    try:
        tic_total = time.perf_counter()
        HOM = HOM_Delay_Measurement(users, delays, log_event, log_file,
                 channel_pairs=[(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)],
                 burst_cycles=800, trigger_source="EXT")
        
        min_delay, HOM_coincidences_array, HOM_coincidences_dict = HOM.measure()

        min_delay = 55000 # checked in previous measurements

        BSM = BSM_Delay_Measurement(users, log_event, log_file, min_delay=min_delay,
                 channel_pairs=[(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)],
                 burst_cycles=800, trigger_source="EXT")
        
        error_psi_minus_DA, error_psi_plus_DA = BSM.measure(vis_type="D&A")
        error_psi_minus_HV, error_psi_plus_HV = BSM.measure(vis_type="H&V")
    
    except KeyboardInterrupt:
        log_event(log_file, "Polarization stabilization interrupted by user.")
    except Exception as e:
        log_event(log_file, f"An error occurred: {e}")
        traceback.print_exc()  # Print full traceback
    finally:
        toc_total = time.perf_counter()
        delta_t_total = toc_total - tic_total
        delta_t_total_hr = int(delta_t_total / 3600)
        delta_t_total_min = int((delta_t_total % 3600) / 60)
        delta_t_total_sec = delta_t_total % 60
        log_event(log_file, f"\nTotal time elapsed for HOM and BSM is {delta_t_total_hr} hours, {delta_t_total_min} minutes and {delta_t_total_sec} seconds")

        log_event(log_file, "end")
        # Alice.laser.turn_off()
        Alice.disconnect_all()
        log_event(log_file, "Alice's laser is off")
        # Bob.laser.turn_off()
        Bob.disconnect_all()
        log_event(log_file, "Bob's laser is off")
        Charlie.disconnect_all()
        Charlie.Rigol.disconnect()
        log_event(log_file, "")
        log_file.close()