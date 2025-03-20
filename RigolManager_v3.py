import pyvisa
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
import subprocess

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
                f"{self.ssh_user}@{self.remote_host}"
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
        print(f"Attempting to connect to VISA address: {visa_address}")
        try:
            self.dev = self.rm.open_resource(visa_address)
            print(f"Connected to {visa_address}")
            # Send a simple command to check the response
            response = self.dev.query("*IDN?")
            print(f"Instrument response: {response}")
        except Exception as e:
            print(f"Connection failed: {e}")
            self.dev = None
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
        print(response)
        return response

if __name__ == "__main__":
    # List available VISA resources for debugging
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    print("Available VISA resources:")
    for resource in resources:
        print(resource)

    Rigol = RigolDG4162Manager("TCPIP0::10.0.0.201::INSTR")
    # Rigol = RigolDG4162Manager("USB0::0x1AB1::0x0641::DG4E251600982::INSTR", remote_host="10.0.0.5", remote_port=33595, ssh_user="UCB5")
    # Rigol = RigolDG4162Manager("ASRL/dev/ttyACM0::INSTR", remote_host="10.0.0.5", remote_port=33595, ssh_user="UCB5")
    Rigol.connect()
    if Rigol.dev is None:
        print("Failed to connect to the instrument.")
    else:
        # Your additional code here
        Rigol.disconnect()