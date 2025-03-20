import time
import serial
from NIDAQ_USBv4 import NIDAQ_USB
import numpy as np
import matplotlib.pyplot as plt
# from ThorlabsPMFunctions import PowerMeter

class OSWManager:
    def __init__(self, com_port,baudrate=9600, timeout=0.5):
        self.com_port = com_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.device = None

    def is_device_connected(self):
        try:
            with serial.Serial(self.com_port, timeout=1) as ser:
                return False
        except serial.SerialException:
            return True
        
    def connect(self):
        if not self.is_device_connected():
            #print("OSW not connected. Connecting now...")
            self.device = serial.Serial(port=self.com_port, baudrate=self.baudrate, timeout=self.timeout)
            print("OSW connected")
            time.sleep(1)
        else:
            print(" already connected.")
        return self.device

    def StatusSet(self, channel, status, sleep_time=0.5):
        #print(f'Voltages from inside V set: {voltages}')
        try:
            if self.device == None or not self.is_device_connected():
                print("OSW not connected")
                self.device = self.connect()
            elif self.device != None:
                command = f"MEMS22 {channel} {status}\n"
                start_time = time.time() 
                self._send_command(command, start_time)
            time.sleep(sleep_time) #Needs 0.1s max for stabilization

        except Exception as e:
            print(e)
            print("Could not set the OSW")
     
    def _send_command(self, command, start_time):
        self.device.write(bytes(command, 'utf-8'))
        while True:
            data = self.device.readline()
            dataStr = data.decode().strip()
            if dataStr == "+ok":
                # print(f"Data received: {command.strip()} and running time: {time.time() - start_time:.2f} seconds")
                break
            else:
                print(f"Received data: {dataStr}")
                print(f"Data not received at {self.com_port} and running time: {time.time() - start_time:.2f} seconds")
                self.connect()
                break


    def disconnect(self):
        if self.device != None and self.is_device_connected():
            print("####### OSW is disconnected ############# ")
            self.device.close()
            self.device = None
        else:
            print("No OSW to disconnect.")


if __name__ == "__main__":
    # Replace with the correct COM port for your device
    COM_PORT = "COM9"  # Adjust based on your hardware setup
    OSW = OSWManager(com_port=COM_PORT)

    daq = NIDAQ_USB(devID="Dev1", ai_samples=100, ai_rate=25000, ai_channels=4)
    daq.setup_ai_task()

    Alice_Switch = 0
    Bob_Switch = 1
    Switch2 = 2
    Switch3 = 3

    OSW.connect()
    OSW.StatusSet(Alice_Switch, 0)
    OSW.StatusSet(Bob_Switch, 1)
    # PM = PowerMeter(resource_name_pyvisa='USB0::0x1313::0x8078::P0023583::INSTR', wavelength=1536, use_pyvisa=True)
    # PM.measure_power()

    try: 
        for i in range(1):
            status = 1
            OSW.StatusSet(Alice_Switch, 1, sleep_time=1)
            # OSW.StatusSet(Bob_Switch, 1, sleep_time=1)
            # OSW.StatusSet(Switch3, status, sleep_time=1)
            # print(f"Changing Alice's switch status to {(i+1)%2}, power measured:{PM.measure_power()}")

            # OSW.StatusSet(Bob_Switch, i%2, sleep_time=1)
            # print(f"Changing Bob's switch status to {i%2}, power measured:{PM.measure_power()}")
            # print(f" power measured:{PM.measure_power()}")

            # time.sleep(5)
        #     measurement = np.mean(daq.ai_measurement(10), axis=1)
        #     print((measurement[1:5]))
        # # for i in range(5):
        #     OSW.StatusSet(0, i%2)
        #     time.sleep(5)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Step 4: Disconnect the OSW
        OSW.disconnect()
        pass 
