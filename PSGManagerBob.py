import socket
import time

UDP_REMOTE_PORT = 55000
UDP_LOCAL_PORT = 62689
IP_QNODE = '10.7.0.26'  # Connected via SSH to Luna Pol control board

class PSGManagerBob:
    def __init__(self, ip_address=IP_QNODE, remote_port=UDP_REMOTE_PORT, local_port=UDP_LOCAL_PORT):
        self.ip_address = ip_address
        self.remote_port = remote_port
        self.local_port = local_port
        self.sock = None
        self.H = "S 7\n"
        self.V = "S 5\n"
        self.D = "S 4\n"
        self.A = "S 6\n"
        self.L = "S 2\n"
        self.R = "S 13\n"

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('', self.local_port))
            print("PSG connected via UDP.")
        except Exception as e:
            print(f"Error connecting PSG: {e}")
            self.sock = None
        return self.sock

    def send_command(self, command):
        if self.sock is None:
            print("PSG is not connected. Attempting to reconnect...")
            self.connect()
        
        if self.sock:
            try:
                self.sock.sendto(command.encode(), (self.ip_address, self.remote_port))
                self.read_msg()
                time.sleep(2)
            except Exception as e:
                print(f"Error sending command: {e}")
    
    def read_msg(self):
        try:
            msg, _ = self.sock.recvfrom(1024)
            msg = msg.decode()
            print(f"Received response: {msg}")
        except Exception as e:
            print(f"Error receiving message: {e}")
    
    def polSET(self, command):
        if isinstance(command, str) and command in [self.H, self.V, self.D, self.A, self.L, self.R]:
            print(f"Setting polarization command: {command.strip()}")
            self.send_command(command)
        else:
            print("Invalid polarization input.")
    
    def disconnect(self):
        if self.sock:
            print("PSG disconnected.")
            self.sock.close()
            self.sock = None
        else:
            print("No active connection to disconnect.")

if __name__ == "__main__":
    PSG = PSGManagerBob()
    PSG.connect()
    
    PSG.polSET(PSG.H)
    PSG.read_msg()
    time.sleep(2)
    PSG.polSET(PSG.D)
    PSG.read_msg()
    
    PSG.disconnect()
