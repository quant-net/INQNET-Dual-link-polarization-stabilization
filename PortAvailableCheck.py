import socket

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False

port = 1155#62689
if is_port_available(port):
    print(f"Port {port} is available.")
else:
    print(f"Port {port} is in use.")