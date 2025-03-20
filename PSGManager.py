import time
import serial
from NIDAQ_USBv4 import NIDAQ_USB
import numpy as np
import matplotlib.pyplot as plt

class PSGManager:
    def __init__(self, com_port,baudrate=9600, timeout=0.5):
        self.com_port = com_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.dev = None
        self.H = "S 7\n"
        self.V = "S 5\n"
        self.D = "S 4\n"
        self.A = "S 6\n"
        self.L = "S 2\n"
        self.R = "S 13\n"

    def is_device_connected(self):
        try:
            with serial.Serial(self.com_port, timeout=1) as ser:
                return False
        except serial.SerialException:
            return True
        
    def connect(self):
        if not self.is_device_connected():
            #print("PSG not connected. Connecting now...")
            self.dev = serial.Serial(port=self.com_port, baudrate=self.baudrate, timeout=self.timeout)
            print("PSG connected")
            time.sleep(1)
        else:
            print("PSG already connected.")
        return self.dev

    def polSET(self, PSGpol):
        if not self.is_device_connected():
            self.connect()
        
        if self.is_device_connected():  # Ensure self.dev is still connected
            try:
                self.dev.write(PSGpol.encode())
                # time.sleep(0.1)
                #print(f'PSG UPDATED')
            except serial.SerialException as e:
                print(f"Error sending command to PSG: {e}")
        else:
            print("Failed to connect to PSG. Cannot send command.")

    
    # def polSET(self, PSGpol):
    #     if self.dev is None:
    #         try: 
    #             self.connect()
    #         except:
    #             print("Failed to connect to PSG. Cannot send command.")
    #     else: 
    #         try:
    #             self.dev.write(PSGpol.encode())
    #             # time.sleep(0.001)
    #             #print(f'PSG UPDATED')
    #         except serial.SerialException as e:
    #             print(f"Error sending command to PSG: {e}")

    def disconnect(self):
        if self.dev != None and self.is_device_connected():
            print("####### PSG is disconnected ############# ")
            self.dev.close()
            self.dev = None
        else:
            print("No PSG to disconnect.")


if __name__ == "__main__":
    # Replace with the correct COM port for your device
    #COM_PORT = "COM9"  # (Bob) Adjust based on your hardware setup
    #COM_PORT = "COM18" 
    COM_PORT = "COM12"  # (Alice) Adjust based on your hardware setup
    PSG = PSGManager(com_port=COM_PORT)

    try:
        # # Initialize NIDAQ_USB
        # duration_sec = 10  # Data acquisition time for each measurement
        # ai_rate = 25000
        # ai_samples = int(duration_sec * ai_rate)
        # daq = NIDAQ_USB(devID="Dev1", ai_samples=ai_samples, ai_rate=ai_rate, ai_channels=4)
        
        # Step 1: Connect to the PSG
        PSG.connect()
        print(PSG.dev)

        # Step 2: Measure timing for a polSET operation
        start_time = time.time()
        PSG.polSET(PSG.H)  # Example: Setting polarization to Horizontal (H)
        end_time = time.time()

        # Step 3: Print the time taken for the operation
        print(f"Time taken for polSET operation: {end_time - start_time:.6f} seconds")


        # # Step 4: Data acquisition for the first voltage
        # daq.setup_ai_task()
        # data_values1 = daq.ai_measurement(ai_samples)


        # # Step 5: Measure timing for a polSET operation
        # start_time = time.time()
        # PSG.polSET(PSG.H)  # Example: Setting polarization to Horizontal (H)
        # end_time = time.time()

        # # Step 6: Print the time taken for the operation
        # print(f"Time taken for polSET operation: {end_time - start_time:.6f} seconds")


        # # Step 7: Data acquisition for the first voltage
        # daq.setup_ai_task()
        # data_values2 = daq.ai_measurement(ai_samples)

        # data_values = np.concatenate((data_values1, data_values2), axis=1)


        # time_values = np.linspace(0, 2*duration_sec, data_values.shape[1])

        # # Plot and save
        # colors = ["blue", "red", "green", "navy"]
        # plt.figure(figsize=(12, 8))

        # for i, color in enumerate(colors, start=1):
        #     y_data = data_values[i, :]

        #     try:
        #         # Fit the data to the exponential model
        #         # popt, _ = curve_fit(exponential_fit, time_values, y_data, p0=(1, 1, 1))
        #         # y_fit = exponential_fit(time_values, *popt)

        #         # Plot the data and the fit
        #         plt.subplot(2, 2, i)
        #         plt.plot(time_values, y_data, label=f"Channel {i} Data", color=color)
        #         # plt.plot(time_values, y_fit, linestyle="--",
        #         #         label=f"Fit: {popt[0]:.2f} * exp(-{popt[1]:.2f} * x) + {popt[2]:.2f}", color=color)
        #         plt.xlabel("Time (s)")
        #         plt.ylabel("Voltage (V)")
        #         plt.title(f"Channel {i}")
        #         plt.legend()
        #         plt.grid()

        #     except Exception as e:
                # print(f"Error fitting data for Channel {i}: {e}")

        # plt.tight_layout()
        # plt.show()


    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Step 4: Disconnect the PSG
        PSG.disconnect()
        pass 
