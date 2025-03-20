import pyvisa
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from TimeTaggerFunctions import TimeTaggerManager
from PSGManager import PSGManager

class RigolDG4162Manager:
    def __init__(self, resource, BUFFER_SIZE=4096, remote_host=None, remote_port=None, ssh_user=None):
        self.resource = resource
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.ssh_user = ssh_user
        self.rm = None
        self.dev = None
        self.ssh_tunnel = None
        self.sock = None
        self.BUFFER_SIZE = BUFFER_SIZE

    def start_ssh_tunnel(self):
        if self.remote_host and self.remote_port:
            local_port = self.remote_port
            ssh_command = [
                "ssh",
                "-L",
                f"{local_port}:localhost:{local_port}",
                "-N",
                "-f",
                f"{self.ssh_user}"#@{self.remote_host}"
            ]
            try:
                self.ssh_tunnel = subprocess.Popen(ssh_command)
                print(f"SSH tunnel started on localhost:{local_port} → {self.remote_host}:{local_port}")
                time.sleep(1)
            except Exception as e:
                print(f"Failed to start SSH tunnel: {e}")

    def stop_ssh_tunnel(self):
        if self.ssh_tunnel:
            self.ssh_tunnel.terminate()
            self.ssh_tunnel.wait()
            print("SSH tunnel closed.")

    def connect(self):
        self.rm = pyvisa.ResourceManager()
        if self.remote_host and self.remote_port:
            self.start_ssh_tunnel()
            visa_address = f"TCPIP::localhost::{self.remote_port}::SOCKET"
        else:
            visa_address = self.resource
        try:
            self.dev = self.rm.open_resource(visa_address)
            print(f"Connected to {visa_address}")
        except Exception as e:
            print(f"Connection failed: {e}")
            if self.remote_host:
                self.stop_ssh_tunnel()
    
    def _send_command_and_receive_response(self, command):
        if self.remote_host and self.remote_port:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.sock:
                self.sock.connect(("localhost", self.remote_port))
                self.sock.sendall(command.encode() + b"\n")
                response = self.sock.recv(self.BUFFER_SIZE).decode().strip()
                return response
        else:
            self.dev.write(command)
            return f"Executed: {command}"

    def disconnect(self):
        if self.dev:
            self.dev.close()
            print("Instrument connection closed.")
        if self.remote_host:
            self.stop_ssh_tunnel()
        self.dev = None

    def set_channel_params(self, channel, frequency=80E6, amplitude=2, offset=0, phase=0, burst_cycles=800, burst_period=100E-6, burst_delay=33.4E-6, trigger_source="EXT"):
        params = {
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
        # response = self._send_command_and_receive_response(f"SET_PARAMS {params}")
        # print(response)
        return params

    def configure_channel(self, channel, frequency, amplitude, offset, phase, burst_cycles, burst_period, burst_delay, trigger_source):
        commands = [
            f"SOUR{channel}:FUNC SIN",
            f"SOUR{channel}:FREQ {frequency}",
            f"SOUR{channel}:VOLT {amplitude}",
            f"SOUR{channel}:VOLT:OFFS {offset}",
            f"SOUR{channel}:PHAS {phase}",
            f"SOUR{channel}:BURS:STAT ON",
            f"SOUR{channel}:BURS:MODE TRIG",
            f"SOUR{channel}:BURS:NCYC {burst_cycles}",
            f"SOUR{channel}:BURS:INT:PER {burst_period}",
            f"SOUR{channel}:BURS:TDEL {burst_delay}",
            f"SOUR{channel}:BURS:TRIG:SOUR {trigger_source}",
            f"OUTP{channel} OFF"
        ]
        responses = [self._send_command_and_receive_response(cmd) for cmd in commands]
        print("\n".join(responses))
        # return f"CH{channel} configured successfully."
        return responses

    def control_channel(self, channel, state):
        command = f"OUTP{channel} {state}"
        response = self._send_command_and_receive_response(command)
        print(f"response = {response}")
        return response

if __name__ == "__main__":
    Rigol = RigolDG4162Manager("USB0::0x1AB1::0x0641::DG4D152500730::INSTR", remote_host="10.0.0.5", remote_port=33595, ssh_user="UCB5")
    # Rigol = RigolDG4162Manager("USB0::0x1AB1::0x0641::DG4E251600982::INSTR", remote_host="10.0.0.5", remote_port=33595, ssh_user="UCB5")
    Rigol.connect()
    # TTU = TimeTaggerManager(filename="TimeTaggerConfig.yaml")
    # Alice_PSG = PSGManager("COM12")
    # Bob_PSG = PSGManager("COM9")

    # Alice_PSG.connect()
    # Alice_PSG.polSET(Alice_PSG.H)
    # Bob_PSG.connect()
    # Bob_PSG.polSET(Bob_PSG.H)

    # delays = np.arange(45000, 70000, 1000)
    # counts_data = np.zeros([4, len(delays)])
    # coincidences_data = []

    # # # HOM
    # # # Define channel pairs for coincidences
    # # channel_pairs = [(1, 3),]  # Example: Coincidences between Ch1 and Ch4

    # # ch1_params = Rigol.set_channel_params(channel="1", burst_delay=delays[0], trigger_source="EXT")
    # # Rigol.control_channel(1, "ON")

    # # ch2_params = Rigol.set_channel_params(channel="2", burst_delay=0, trigger_source="EXT")
    # # # Rigol.control_channel(2, "OFF")
    # # Rigol.configure_channel(**ch2_params)
    # # Rigol.control_channel(2, "ON")
    # # for i, delay in enumerate(delays):
    # #     burst_delay = delay * 1E-9
    # #     print(delay*1E-3)

    # #     #Rigol.configure_channel(**ch1_params)
    # #     #Rigol.control_channel(1, "ON")
    # #     Rigol.dev.write(f"SOUR1:BURS:TDEL {burst_delay}")
        
    # #     counts = TTU.getChannelCounts(Chlist=TTU.Chlist, measurement_time=1, printout=False)[:,0]
    # #     counts_data[:, i] = counts
        
    # #     # Get coincidences for Ch1 and Ch4
    # #     coincidence_hist = TTU.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
    # #     print(coincidence_hist)

    # #     coincidence_hist2 = TTU.getCoincidences( [(1, 4),], bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
    # #     print(coincidence_hist2)
    # #     coincidences_data.append(np.sum(coincidence_hist))
        
    # #     time.sleep(0.1)

    # # # Plot coincidences histogram for Ch1 and Ch4
    # # plt.figure()
    # # plt.title("Coincidences Histogram: Ch1 & Ch3")
    # # plt.plot(delays/1000, coincidences_data)
    # # plt.xlabel("Time (µs)")
    # # plt.ylabel("Counts")
    # # plt.show()

    # # coincidences_data = np.array(coincidences_data)
    # # min_delay = delays[np.argmin(coincidences_data)]

    # # BSM
    # min_delay = 55000
    # ch1_params = Rigol.set_channel_params(channel="1", burst_delay=min_delay, trigger_source="EXT")
    # Rigol.control_channel(1, "ON")

    # ch2_params = Rigol.set_channel_params(channel="2", burst_delay=0, trigger_source="EXT")
    # Rigol.control_channel(2, "OFF")

    # # channel_pairs = [(1,4), (2,3), (1,3), (2,4)]
    # # pol_pairs = [(Alice_PSG.H, Bob_PSG.V), (Alice_PSG.V, Bob_PSG.H)]

    # # for i in range(2):
    # #     pol_pair_i = pol_pairs[i]
    # #     print(pol_pair_i[0])
    # #     print(pol_pair_i[1])
    # #     print(channel_pairs[i])

    # #     Alice_PSG.polSET(pol_pair_i[0])
    # #     Bob_PSG.polSET(pol_pair_i[1])
        
    # #     coincidence_hist = TTU.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
    # #     print(coincidence_hist)
    # #     print(f"Channel pair : {np.sum(coincidence_hist[0:2])}")
    # #     print(np.sum(coincidence_hist[2:4]))
        
    # #     time.sleep(0.1)
    
    # # 1: H, 2: V, 3: H, 4: V;

    # Alice_PSG.polSET(Alice_PSG.D)
    # Bob_PSG.polSET(Bob_PSG.D)
    # # channel_pairs = [(1,4), (2,3), (1,2), (3,4)]
    # channel_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    # single_counts = TTU.getChannelCountRate([1, 2, 3, 4], printout=True)
    # coincidence_hist = TTU.getCoincidences(channel_pairs, bindwidth=1e6, n_bins=100) # binwidth in ps (check this)
    # # print(coincidence_hist)
    # for j in range(len(coincidence_hist)):
    #     print(f"Channel pair: {channel_pairs[j]}, Total coincidences: {np.sum(coincidence_hist[j])}")

    # ch2_params = Rigol.set_channel_params(channel="2", burst_delay=0, trigger_source="EXT")
    # Rigol.configure_channel(**ch2_params)
    # print(Rigol.control_channel(2, "ON"))
    Rigol.disconnect()
