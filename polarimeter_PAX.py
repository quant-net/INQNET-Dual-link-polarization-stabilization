import os
import os.path
import time
import ctypes
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt

from PSGManager import PSGManager
from NIDAQ_USBv4 import NIDAQ_USB

lib = cdll.LoadLibrary("C:\\Program Files\\IVI Foundation\\VISA\\Win64\\Bin\\TLPAX_64.dll")
# Detect and initialize PAX1000 device
instrumentHandle = c_ulong()
IDQuery = True
resetDevice = False
resource = c_char_p(b"")
deviceCount = c_int()
scanID = c_int()

class PAXManager:
    def __init__(self):
        pass
    def connect(self):
        try:
            polarimeter_connection_start = time.time()

            # Check how many PAX1000 are connected
            lib.TLPAX_findRsrc(instrumentHandle, byref(deviceCount))
            if deviceCount.value < 1 :
                print("No PAX1000 device found.")
                exit()
            else:
                print(deviceCount.value, "PAX1000 device(s) found.")
                print("")

            # Connect to the first available PAX1000
            lib.TLPAX_getRsrcName(instrumentHandle, 0, resource)
            if (0 == lib.TLPAX_init(resource.value, IDQuery, resetDevice, byref(instrumentHandle))):
                print("Connection to first PAX1000 initialized.")
            else:
                print("Error with initialization.")
                exit()
            print("")

            # Short break to make sure the device is correctly initialized
            time.sleep(2)

            lib.TLPAX_setMeasurementMode(instrumentHandle, 9)
            lib.TLPAX_setWavelength(instrumentHandle, c_double(1550e-9))
            lib.TLPAX_setBasicScanRate(instrumentHandle, c_double(300))
            time.sleep(2)

            lib.TLPAX_getLatestScan(instrumentHandle, byref(scanID))
            polarimeter_connection_end = time.time()
            polarimeter_connection_time = polarimeter_connection_end - polarimeter_connection_start;
        except KeyboardInterrupt:
            self.disconnect()
        except Exception:
            self.disconnect()

        return polarimeter_connection_time

    def measure_SOP(self):
        s1 = c_double()
        s2 = c_double()
        s3 = c_double()
        try:
            lib.TLPAX_getStokesNormalized(instrumentHandle, scanID.value, byref(s1), byref(s2), byref(s3))
        except KeyboardInterrupt:
            lib.TLPAX_releaseScan(instrumentHandle, scanID)
            time.sleep(0.5)
            lib.TLPAX_close(instrumentHandle)
        except Exception:
            lib.TLPAX_releaseScan(instrumentHandle, scanID)
            time.sleep(0.5)
            lib.TLPAX_close(instrumentHandle)

        # Do something here and when you key board interrupt
        # The except block will capture the keyboard interrupt and exit
        SOP = np.array([s1.value, s2.value, s3.value])
        lib.TLPAX_releaseScan(instrumentHandle, scanID)
        return SOP

    def disconnect(self):
        lib.TLPAX_releaseScan(instrumentHandle, scanID)
        time.sleep(0.5)
        lib.TLPAX_close(instrumentHandle)
        print("PAX disconnected")

if __name__ == "__main__":
    sleep_time = 1
    numReps = 100
    ai_samples = 1000
    ai_data = np.zeros(ai_samples)

    PSG = PSGManager(com_port="COM9")
    device1 = NIDAQ_USB(devID="Dev1", ai_samples=1000, ai_rate=25000, ai_channels=4)
    pax = PAXManager()

    print(f"device:{device1}")

    try:
        PSG.connect()
        PSG.polSET(PSG.H)
        
        for i in range(2):
            # Synchronous Measurement

            tic1 = time.time()
            print(pax.connect())
            print(pax.measure_SOP())
            pax.disconnect()   
            print(f"Time taken for measurement: {time.time()-tic1}")
            device1.setup_ai_task()
            ai_data_sync = device1.ai_measurement(1000)
            print(f"Synchronous Data:\n{ai_data_sync}")
            device1.close_task(device1.ai_task)
        # tic = time.perf_counter()
        # for i in range(numReps):
        #     print(f"Iteration {i+1}/{numReps}")
        #     device.setup_ai_task()
        #     ai_data_sync = device.ai_measurement(1000)
        #     print(f"Synchronous Data:\n{ai_data_sync}")
        #     device.close_task(device.ai_task)

        # toc = time.perf_counter()
        # print(f"Duration: {toc - tic}")

    except Exception as e:
        print(e)
    finally:
        pax.disconnect()
        PSG.disconnect()
    
    # try:
    #     print(pax.connect())
    #     PSG.connect()
    #     for i in range(numReps):
    #         print(f"Iteration {i+1}/{numReps}")
    #         PSG.polSET(PSG.H)
    #         time.sleep(sleep_time)
    #         H_data_i = pax.measure_SOP()
    #         H_data[:, i] = H_data_i
    #         print(f"PSG H: {H_data_i}")

    #         PSG.polSET(PSG.V)
    #         time.sleep(sleep_time)
    #         V_data_i = pax.measure_SOP()
    #         V_data[:, i] = V_data_i
    #         print(f"PSG V: {V_data_i}")

    #         PSG.polSET(PSG.D)
    #         time.sleep(sleep_time)
    #         D_data_i = pax.measure_SOP()
    #         D_data[:, i] = D_data_i
    #         print(f"PSG D: {D_data_i}")

    #         PSG.polSET(PSG.A)
    #         time.sleep(sleep_time)
    #         A_data_i = pax.measure_SOP()
    #         A_data[:, i] = A_data_i
    #         print(f"PSG A: {A_data_i}")
    # except Exception as e:
    #     print(e)
    # finally:
    #     pax.disconnect()
    #     PSG.disconnect()