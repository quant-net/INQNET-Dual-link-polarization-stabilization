import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from NIDAQ_USBv4 import NIDAQ_USB

class PolCTRLManager: # remember to install the polctrl_firmware that has mystrtrod function
    def __init__(self, com_port, baudrate=9600, timeout=0.5):
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
            #print("Polarization Controller not connected. Connecting now...")
            self.device = serial.Serial(port=self.com_port, baudrate=self.baudrate, timeout=self.timeout)
            print("Polarization Controller connected")
            time.sleep(1)
        else:
            print("Polarization Controller already connected.")
        return self.device
    

    def Vset(self, voltages, sleep_time=0.015):
        #print(f'Voltages from inside V set: {voltages}')
        try:
            if self.device is None or not self.is_device_connected():
                print("Pol CTRL not connected")
                self.device = self.connect()
            commands = [f"Vset {i+1} {volt}\n" for i, volt in enumerate(voltages)]

            start_time = time.time()
            for command in commands:
                tic = time.time()
                self._send_command(command, start_time)
                # print(f"Time taken to set voltage:{time.time()-tic}")
            time.sleep(sleep_time) #Needs 0.1s max for stabilization

        except Exception as e:
            print(e)
            print("Could not set the polarization controller voltage")
     
    def _send_command(self, command, start_time):
        self.device.write(bytes(command, 'utf-8'))
        if self.device != None:
            data = self.device.readline()
            dataStr = data.decode().strip()
            if dataStr == "+ok":
                # print(f"Data received: {command.strip()} and running time: {time.time() - start_time:.2f} seconds")
                # PSODevice.log_event(message=f'Voltage recieved: {command.strip()}')
                # time.sleep(0.01)
                pass
            else:
                print(f"Received data: {dataStr}")
                print(f"Data not received at {self.com_port} and running time: {time.time() - start_time:.2f} seconds")
                self.connect()
    def disconnect(self):
        print("Unknown command to disconnect Polarization Controller")


if __name__ == "__main__":
    polCTRL = PolCTRLManager("COM21") #EPC1 COM22;EPC2 COM21; EPC A COM20
    polCTRL.connect()
    polCTRL.Vset(np.array([1, 2, 2]))
    #polCTRL.Vset(np.array([1.04648364,0.27236673,0.18366485]))
    # daq = NIDAQ_USB(devID="Dev1", ai_samples=100, ai_rate=25000, ai_channels=4)
    # daq.setup_ai_task()

    # voltage_index = 1  # Set the index of the voltage to vary (0, 1, or 2)
    # voltages = np.arange(0.1, 1.902, 0.002)  # Voltage range from 0.1V to 1.9V
    # # voltages = np.array([1,1,1])
    # data_values = np.zeros((5, len(voltages)))
    # data_std = np.zeros((5, len(voltages)))

    # for i, voltage in enumerate(voltages):
    #     voltage_array = np.array([0.1, 0.1, 0.1])  # Default voltages
    #     voltage_array[voltage_index] = voltage  # Modify only the selected index
    #     polCTRL.Vset(voltage_array)
    #     measurement = daq.ai_measurement(100)
    #     data_values[:, i] = np.mean(measurement, axis=1)
    #     data_std[:, i] = np.std(measurement, axis=1)
    # print(data_values)
    # colors = ["blue", "red", "green", "navy"]
    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # fig.suptitle(f"Voltage Sweep for Index {voltage_index}")

    # for i, (ax, color) in enumerate(zip(axs.flat, colors), start=1):
    #     ax.errorbar(voltages, data_values[i, :], yerr=data_std[i, :], fmt='o', label=f"Channel {i}", color=color)
    #     ax.set_xlabel("Voltage (V)")
    #     ax.set_ylabel("Mean Voltage Reading (V)")
    #     ax.set_title(f"Channel {i}")
    #     ax.legend()
    #     ax.grid()

    # plt.tight_layout()
    # plt.show()
    # daq.close_all_tasks()
    polCTRL.disconnect()
               