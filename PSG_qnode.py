import serial
import time
import socket
import re
import math

polH = "S 4\n"
polD = "S 5\n"
polV = "S 6\n"
polA = "S 7\n"
polLCP = "S 2\n"
polRCP = "S 13\n"
UDP_IP = ""
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0',5005))

ip_BSM = '192.168.0.169'

UDP_remote_port = 55180
UDP_local_port = 5005

def is_device_connected(port_name):
    try:
        ser = serial.Serial(port_name)
        ser.close()
        return True
    except serial.SerialException:
        return False
    
def connect():
    #dev = serial.Serial(port='COM4', baudrate=9600, timeout = 0.1)
    dev = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout = 0.1)
    print(dev.name)
    msg_device = "####### PSG is connected ############# "
    print(msg_device)
    return dev, msg_device


def polSET(dev, ip, PSGpol):
    try: 
        dev_connected_flag = is_device_connected('/dev/ttyACM0')
       
        if (~dev_connected_flag):
            msg_devflag = "device connected flag:1, PSG is already connected"
            print(msg_devflag)
        else:
            msg_devflag = "device connected flag:0, PSG was not connected. Connecting now"
            print(msg_devflag)
            dev, msg_device=connect()
        
        try:
            dev.write(PSGpol.encode())
            msg_write = "Written: Ok"
            print(msg_write)
        except Exception as e:
            print(e)
            msg_write = "Write ISSUE"
            print(msg_write)
        
        time.sleep(1)
        try: 
            b=dev.read(20)
            msg_read = "Read: Ok"           
            print(msg_read)
        except Exception as e:
            print(e)
            msg_read="Read Issue"
            print("")     
        polset_error= "Polset error None"
        print(polset_error)
    except:
        polset_error = "Polarization: Could NOT set" 
        print(polset_error)
    
    return msg_devflag, msg_write, msg_read, polset_error

def disconnect(dev):
    try:
        msg_disconnect = "####### PSG is disconnected ############# "
        print(msg_disconnect)
        dev.close
    except:
        msg_disconnect = "PSG: Could NOT disconnect"
    
    return msg_disconnect

dev_PSG, msg_device = connect()
#msg_devflag, msg_write, msg_read, polset_error= polSET(dev_PSG, 0, polRCP)


try:
    while True:
        try:
            print("Waiting for data")
            data, addr = sock.recvfrom(1024)
            print(data)
            #data1 = polD
            #data2=(re.findall(r'\d+', data1))
            #print("data2:", data2)
            #polStr = "S "+data2[0]+"\n"
            msg_devflag, msg_write, msg_read, polset_error = polSET(dev_PSG, 0,  data.decode())
            complete_msg = msg_devflag+"\n"+msg_write+"\n"+msg_read+"\n"+polset_error
            print(complete_msg)
            sock.sendto(complete_msg.encode(),(ip_BSM, UDP_remote_port))
            #msg_disconnect = disconnect(dev_PSG)
        except Exception as e:
            print(e)
            break        
except KeyboardInterrupt:
    pass




