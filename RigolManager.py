import pyvisa
import time
import numpy as np
import matplotlib.pyplot as plt
from TimeTaggerFunctions import TimeTaggerManager
from PSGManager import PSGManager
from PSGManagerBob import PSGManagerBob
class RigolDG4162Manager:
    def __init__(self, resource):
        self.resource = resource
        self.rm = None
        self.dev = None

    def connect(self):
        """Connect to the Rigol DG4162 function generator."""
        self.rm = pyvisa.ResourceManager()
        self.dev = self.rm.open_resource(self.resource)
        print(f"Connected to {self.resource}")

    def disconnect(self):
        """Disconnect from the Rigol DG4162 function generator."""
        if self.dev:
            self.dev.close()
            print("Connection closed.")
        self.dev = None

    def set_channel_params(self, channel, frequency=80E6, amplitude=1.5, offset=0, phase=0, burst_cycles=800, burst_period=100E-6, burst_delay=33.4E-6, trigger_source="EXT"):
        """
        Set parameters for a Rigol DG4162 channel.
        """
        return {
            "channel": channel,
            "frequency": frequency,
            "amplitude": amplitude,
            "offset": offset,
            "phase": phase,
            "burst_cycles": burst_cycles,
            "burst_period": burst_period,
            "burst_delay": burst_delay,
            "trigger_source": trigger_source
        }

    def configure_channel(self, channel, frequency, amplitude, offset, phase, burst_cycles, burst_period, burst_delay, trigger_source):
        """
        Configure a Rigol DG4162 channel with specified waveform and burst settings.
        """
        self.dev.write(f"SOUR{channel}:FUNC SIN")
        self.dev.write(f"SOUR{channel}:FREQ {frequency}")
        self.dev.write(f"SOUR{channel}:VOLT {amplitude}")
        self.dev.write(f"SOUR{channel}:VOLT:OFFS {offset}")
        self.dev.write(f"SOUR{channel}:PHAS {phase}")
        self.dev.write(f"SOUR{channel}:BURS:STAT ON")     
        self.dev.write(f"SOUR{channel}:BURS:MODE TRIG")
        self.dev.write(f"SOUR{channel}:BURS:NCYC {burst_cycles}")
        self.dev.write(f"SOUR{channel}:BURS:INT:PER {burst_period}")
        self.dev.write(f"SOUR{channel}:BURS:TDEL {burst_delay}")
        self.dev.write(f"SOUR{channel}:BURS:TRIG:SOUR {trigger_source}")
        #self.dev.write(f"SOUR{channel}:BURS:STAT OFF") 
        self.dev.write(f"OUTP{channel} OFF")
        print(f"CH{channel} configured: {frequency/1E6}MHz Sine, Burst Mode, {burst_cycles} Cycles, Delay {burst_delay*1E6}µs, Trigger Source: {trigger_source}")

    def control_channel(self, channel, state):
        """
        Turn a specific channel ON or OFF.
        """
        self.dev.write(f"OUTP{channel} {state}")
        print(f"CH{channel} turned {state}.")

if __name__ == "__main__":
    # Rigol = RigolDG4162Manager("USB0::0x1AB1::0x0641::DG4D152500730::INSTR")

    Rigol1 = RigolDG4162Manager("USB0::0x1AB1::0x0641::DG4E251600982::INSTR") # Use this for LBNL Rigol
    #Rigol = RigolDG4162Manager("TCPIP0::10.0.0.201::INSTR") # UCB Rigol
    Rigol1.connect()

    ch1_params = Rigol1.set_channel_params(channel="1",frequency=80.0e6 ,amplitude=1, burst_delay=45000*1e-9,burst_cycles=800 ,trigger_source="EXT")
    Rigol1.configure_channel(**ch1_params)
    Rigol1.control_channel(1, "ON")

    ch12_params = Rigol1.set_channel_params(channel="2",frequency=80e6, amplitude=1, burst_delay=000*1e-9,burst_cycles=800 ,trigger_source="EXT")
    Rigol1.configure_channel(**ch12_params)
    Rigol1.control_channel(2, "OFF")

    Rigol2 = RigolDG4162Manager("TCPIP0::10.0.0.201::INSTR") # UCB Rigol
    Rigol2.connect()

    ch2_params = Rigol2.set_channel_params(channel="2",frequency=80e6 ,amplitude=1, burst_delay=0,burst_cycles=800 ,trigger_source="EXT")
    Rigol2.configure_channel(**ch2_params)
    Rigol2.control_channel(2, "ON")
    Rigol2.disconnect()



    Alice_PSG = PSGManager("COM12")
    Bob_PSG = PSGManager("COM9")
    Bob_PSG2 = PSGManagerBob()

    Alice_PSG.connect()
    Bob_PSG.connect()
    Bob_PSG2.connect()
    Alice_PSG.polSET(Alice_PSG.H)
    Bob_PSG.polSET(Bob_PSG.H)
    Bob_PSG2.polSET(Bob_PSG.H)

    # # HOM_time_scan
    # delays = np.arange(30000, 60000, 500)
    # counts_data = np.zeros([4, len(delays)])
    # coincidences_data = []

    # # Define channel pairs for coincidences
    # channel_pairs = [(1, 3),]  # Example: Coincidences between Ch1 and Ch4

    # TTU = TimeTaggerManager(filename="TimeTaggerConfig.yaml")

    # for i, delay in enumerate(delays):
    #     burst_delay = delay * 1E-9

    #     #Rigol.configure_channel(**ch1_params)
    #     #Rigol.control_channel(1, "ON")
    #     Rigol1.dev.write(f"SOUR1:BURS:TDEL {burst_delay}")
        
    #     # counts = TTU.getChannelCounts(Chlist=TTU.Chlist, measurement_time=0.2, printout=False)[:,0]
    #     # counts_data[:, i] = counts
        
    #     # Get coincidences for Ch1 and Ch4
    #     coincidence_hist = TTU.getCoincidences(channel_pairs,measurement_time=1, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
    #     print(f'Delay:{delay*1E-3}ns, Coin counts {np.sum(coincidence_hist)}')

    #     # coincidence_hist2 = TTU.getCoincidences( [(1, 4),], bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
    #     # print(coincidence_hist2)
    #     coincidences_data.append(np.sum(coincidence_hist))
        
    #     time.sleep(0.01)

    # # Plot coincidences histogram for Ch1 and Ch4
    # plt.figure()
    # plt.title("Coincidences Histogram: Ch1 & Ch3")
    # plt.plot(delays/1000, coincidences_data)
    # plt.xlabel("Time (µs)")
    # plt.ylabel("Counts")
    # plt.show()

    # coincidences_data = np.array(coincidences_data)
    # min_delay = delays[np.argmin(coincidences_data)]

    # # BSM
    # # min_delay = 55000
    # # ch1_params = Rigol.set_channel_params(channel="1",amplitude=1, burst_delay=min_delay*1E-9, trigger_source="EXT")
    # # Rigol.configure_channel(**ch1_params)
    # # Rigol.control_channel(1, "ON")

    # # ch2_params = Rigol.set_channel_params(channel="2",amplitude=1, burst_delay=0, trigger_source="EXT")
    # # Rigol.configure_channel(**ch2_params)
    # # Rigol.control_channel(2, "ON")

    # # # channel_pairs = [(1,4), (2,3), (1,3), (2,4)]
    # # # pol_pairs = [(Alice_PSG.H, Bob_PSG.V), (Alice_PSG.V, Bob_PSG.H)]

    # # # for i in range(2):
    # # #     pol_pair_i = pol_pairs[i]
    # # #     print(pol_pair_i[0])
    # # #     print(pol_pair_i[1])
    # # #     print(channel_pairs[i])

    # # #     Alice_PSG.polSET(pol_pair_i[0])
    # # #     Bob_PSG.polSET(pol_pair_i[1])
        
    # # #     coincidence_hist = TTU.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
    # # #     print(coincidence_hist)
    # # #     print(f"Channel pair : {np.sum(coincidence_hist[0:2])}")
    # # #     print(np.sum(coincidence_hist[2:4]))
        
    # # #     time.sleep(0.1)
    
    # # # 1: H, 2: V, 3: H, 4: V;

    # # Alice_PSG.polSET(Alice_PSG.H)
    # # Bob_PSG.polSET(Bob_PSG.V)
    # # # channel_pairs = [(1,4), (2,3), (1,2), (3,4)]
    # # channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    # # single_counts = TTU.getChannelCountRate([1, 2, 3, 4], printout=True)
    # # coincidence_hist = TTU.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
    # # # print(coincidence_hist)
    # # for j in range(len(coincidence_hist)):
    # #     print(f"Channel pair: {channel_pairs[j]}, Total coincidences: {np.sum(coincidence_hist[j])}")
    
    

