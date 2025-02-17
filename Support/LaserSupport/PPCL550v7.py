import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#print(f'PPCLv4 {parent_dir}')
sys.path.append(parent_dir)

import serial
import numpy as np
import math
import time
import LaserSupport.ITLA_v3 as ITLA
# import yaml

# TODO Create a laser class
class PPCL550:
    """PPCL550 Laser Control Class"""
    def __init__(self, com_port, baud_rate, min_freq=191.50, max_freq=196.25, min_pow=600, max_pow=1700) -> None:
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.freq = 0#freq
        self.power = 0#power
        self.sercon = 0 # serial laser connection
        self.NOP = 0
        self.MIN_FREQ = min_freq#191.50#MIN_FREQ
        self.MAX_FREQ = max_freq#196.25#MAX_FREQ
        self.MIN_POW = min_pow#600#MIN_POW
        self.MAX_POW = max_pow#1000#MAX_POW
        self.LASER_ON = [0x81,0x32,0x00,0x08]
        self.LASER_OFF = [0x01,0x32,0x00,0x00]
        self.Cjump_enable_flag = False
        self.GridSize = 10
        self.Cjump_enable_flag = False
        self.CPower = 700
        self.CPOW_THRESH = 0.05
        self.CJUMP_MIN = 1528.5
        self.CJUMP_MAX = 1550

    def __str__(self) -> str:
        return ("PPCL550 laser:"
                + "\n---------------"
                + f"\nCOM Port: {self.com_port}"
                + f"\nBaud Rate: {self.baud_rate}"
                + f"\nFrequency: {self.freq} THz"
                + f"\nPower: {self.power} * 0.01 dBm")
    
    def check_settings(self):
        if (self.freq > self.MAX_FREQ or self.freq < self.MIN_FREQ):
            raise ValueError("Laser frequency value is too high!")
        if (self.power > self.MAX_POW or self.power < self.MIN_POW):
            raise ValueError("Laser power value is too high!")
        print("Settings are good")
        return(self.freq,self.power)
    
    # Connect laser and turn it ON or OFF
    def connect_laser(self):
        self.sercon = ITLA.ITLAConnect(self.com_port,baudrate=self.baud_rate)
        return self.sercon
    
    def NOP_register(self):
        self.NOP = ITLA.ITLA(self.sercon, 0, 0, 0)
        return self.NOP
        
    class NOPException(Exception):
        def __init__(self, message):
            super().__init__(message)
        
        def __str__(self):
            return self.args[0]
    
    def is_NOP_correct(self, throw=False):
        if self.NOP == 0x10:
            return True
        elif throw:
            raise self.NOPException(f"NOP register failed to return 0x10, instead returned {hex(self.NOP)}")
        else:
            return False
    
    def laser_on(self):
        return ITLA.ITLA(self.sercon, self.LASER_ON[1],
                         (self.LASER_ON[2]*256 + self.LASER_ON[3]), 1)
    
    def laser_off(self):
        return ITLA.ITLA(self.sercon, self.LASER_OFF[1],
                         (self.LASER_OFF[2]*256 + self.LASER_OFF[3]), 1)
    
    def disconnect_laser(self):
        self.sercon.close()
    
    # Read and Write Methods
    def read_freq(self):
        return ITLA.ITLA(self.sercon, 0x35, self.freq, 0) + ITLA.ITLA(self.sercon, 0x36, self.freq, 0)/10000
    
    def write_freq(self, f):
        if (f > self.MAX_FREQ or f < self.MIN_FREQ):
            raise ValueError("Laser frequency value is too high!")
        fTHz = int(f)
        f100MHz = np.round((f - fTHz) * 10000, 0)
        responseTHz = ITLA.ITLA(self.sercon, 0x35, fTHz, 1)
        response100MHz = ITLA.ITLA(self.sercon, 0x36, f100MHz, 1)
        return responseTHz + response100MHz/10000
    
    def read_power(self):
        return ITLA.ITLA(self.sercon, 0x31, self.power, 0)
    
    def write_power(self, p):
        if (p > self.MAX_POW or p < self.MIN_POW):
            raise ValueError("Laser power value is too high!")
        return ITLA.ITLA(self.sercon, 0x31, p, 1)

    # Whisper mode enable    
    def Whispermode_enable(self):
        """Enable Whisper mode, retrying up to 10 times if it is not already enabled."""
        max_attempts = 10
        attempt = 0

        while attempt < max_attempts:
            WhispermodeResponse = ITLA.ITLA(self.sercon, 0x90, 0, 0)  # Check status

            if WhispermodeResponse == 2:
                # print(f"Whisper mode enabled.{WhispermodeResponse}")
                return WhispermodeResponse
            else:
                print(f"Whisper mode is not enabled. Attempting to enable it. Attempt {attempt + 1} of {max_attempts}. Whispermode status: {WhispermodeResponse}")
                setWhispermode = ITLA.ITLA(self.sercon, 0x90, 2, 1)  # Try enabling
                time.sleep(0.5)
                attempt += 1

        # If after 10 attempts, Whisper mode is not enabled
        print("Failed to enable Whisper mode after 10 attempts.")
        return -1

    # Dither mode enable
    def DitherMode_enable(self):
        DitherModeResponse = ITLA.ITLA(self.sercon, 0x90, 0, 1)
        if DitherModeResponse == 0:
            print("Dither mode enabled")
        else:
            print("Issue in enabling dither mode")
        return DitherModeResponse
    
    # # Clean jump function
    
    # def SetGrid(self, Gridsize):
    #     while True:
    #         grid_value = 0.1 * ITLA.ITLA(self.sercon, 0x34, Gridsize*10, 1)
    #         grid_value = 0.1 *  ITLA.ITLA(self.sercon, 0x34, 0, 0)
    #         print(f"Grid settings in GHz = {grid_value}")
    #         # print(f"grid_value = {grid_value*0.1} GHz")
    #         if grid_value != 0:
    #             self.GridSize = grid_value
    #             print(f"Grid size: {self.GridSize} GHz")
    #             break
    #         else: 
    #             print(f"Error in setting the grid size. Reg response {grid_value}. Retrying grid settings")
    #             self.SetGrid(Gridsize)
                

    # def CjumpEnable(self):
    #     self.Cjump_enable_flag = ITLA.ITLA(self.sercon, 0xD0, 1, 1)
    #     print(f"jump enabled: {self.Cjump_enable_flag}")
    #     if self.Cjump_enable_flag!=1:
    #         print("Retrying Cjump enable")
    #         self.CjumpEnable() #Retrying to enable
    #     # return self.Cjump_enable_flag
        
    # def Cjump(self, setpoint=None, wavelength=None, return_full=False):
    #     """Perform a clean jump based on setpoint or wavelength input, with retry logic and return status."""
        
    #     # Check if both setpoint and wavelength are not provided
    #     if setpoint is None and wavelength is None:
    #         raise ValueError("Either 'setpoint' or 'wavelength' must be provided.")
        
    #     # Check if Whisper mode is enabled before proceeding
    #     whisper_mode_status = self.Whispermode_enable()  # Assuming it returns 0 when enabled
    #     if whisper_mode_status != 2:
    #         raise RuntimeError("Whisper mode is not enabled. Enable Whisper mode before performing a clean jump.")
        
    #     # Handle the wavelength case
    #     frequency = None
    #     if wavelength is not None:
    #         # Check if the wavelength is within the allowed range
    #         if wavelength < self.CJUMP_MIN or wavelength > self.CJUMP_MAX:
    #             raise ValueError(f"Wavelength {wavelength} nm is out of range. It must be between {self.CJUMP_MIN} nm and {self.CJUMP_MAX} nm.")
            
    #         # Convert the wavelength to frequency
    #         frequency = wavelength_to_frequency(wavelength)
            
    #         # Find the setpoint for the given frequency from LaserCalibrationStandard.yaml
    #         filename = 'LaserCalibrationStandard.yaml'
    #         setpoint = get_setpoint_for_frequency(filename, frequency)
            
    #         # Print the setpoint, wavelength, and frequency
    #         print(f"Wavelength: {wavelength} nm, Frequency: {frequency} THz, Setpoint: {setpoint}")
        
    #     # Handle the setpoint case (if setpoint is directly provided)
    #     if setpoint is not None and wavelength is None:
    #         # Retrieve frequency from the setpoint in LaserCalibrationStandard.yaml
    #         filename = 'LaserCalibrationStandard.yaml'
    #         frequency = get_frequency_for_setpoint(filename, setpoint)
            
    #         # Print the setpoint and frequency
    #         print(f"Setpoint: {setpoint}, Frequency: {frequency} THz")

    #     # Start the clean jump process with a max of 20 attempts
    #     max_attempts = 20
    #     attempt = 0
    #     Cjumpstatus = False

    #     while attempt < max_attempts and not Cjumpstatus:
    #         start_time = time.time()
    #         self.SetCjump = ITLA.ITLA(self.sercon, 0xD0, setpoint, 1)
    #         # time.sleep(1)

    #         # Enable clean jump
    #         self.CjumpEnable()
    #         Cjumpstatus = self.CjumpRunningCheck()
            
    #         if Cjumpstatus:
    #             # print(f"Clean jump for setpoint {setpoint} completed in {time.time() - start_time} seconds")
    #             pass
    #         else:
    #             print(f"Clean jump failed on attempt {attempt + 1}. Retrying...")
    #             attempt += 1
        
    #     # If it fails after max_attempts
    #     if not Cjumpstatus:
    #         print(f"Clean jump failed after {max_attempts} attempts.")
        
    #     # Return only Cjumpstatus if return_full is False, else return all parameters
    #     if return_full:
    #         return setpoint, wavelength, frequency, Cjumpstatus
    #     else:
    #         return Cjumpstatus
    

    # def CjumpRunningCheck(self):
    #     Cjumpstatus = False
    #     StuckFlag = 0
    #     while True:
    #         self.CjumpinMHz = ITLA.ITLA(self.sercon, 0xD1, 0, 0)
    #         if self.CjumpinMHz==0:
    #             Cjumpstatus=True
    #             break
    #         elif 0.1*self.CjumpinMHz==self.GridSize or 0.1*self.CjumpinMHz==100:
    #             StuckFlag+=1
    #             print(f"Stuck {StuckFlag}", end="\r")
    #             time.sleep(0.5)
    #             if StuckFlag==20:
    #                 break
    #         else:
    #             print(f"jump in MHz: {self.CjumpinMHz}", end="\r")
    #     print(f"jump in MHz: {self.CjumpinMHz}")
    #     return Cjumpstatus

    # def CjumpCalibrationEnable(self, Calibration_channels):
    #     self.CjumpCaliEnable = ITLA.ITLA(self.sercon, 0xD2, Calibration_channels, 1)

#     def CjumpCalibrationRunningCheck(self):
#         CalibrationStatus =False

#         start_time = time.perf_counter()
#         Estimated_time_for_calibration = Calibration_channels*40
        
#         while True:
#             CalibrationRunningCheck = ITLA.ITLA(self.sercon, 0xD2, 0, 0)
#             if CalibrationRunningCheck == 0:
#                 print("Calibration is done ...")
#                 CalibrationStatus = True
#                 break
#             else: 
#                 print(f"Calibration is running. Status: {CalibrationRunningCheck}. Calibration running time: {int(time.perf_counter()-start_time)}s, Estimated time of completion: {int(Estimated_time_for_calibration - (time.perf_counter()-start_time))}s", end="\r")
#             # time.sleep(1)

#         return CalibrationStatus 
    

# def initialize_yaml(filename):
#     data = {'calibration': []}
#     # Create setpoints from 0 to 499, with all frequencies set to 0
#     for i in range(500):
#         data['calibration'].append({'setpoint': i, 'frequency': 0})

#     # Write the initialized data to the file
#     with open(filename, 'w') as file:
#         yaml.dump(data, file, sort_keys=False)

# def update_calibration_in_yaml(filename, calibration_frequencies):
#     # Check if the file exists; if not, initialize it
#     if not os.path.exists(filename):
#         print(f"{filename} does not exist. Initializing it with setpoints from 0 to 499...")
#         initialize_yaml(filename)

#     # Load existing data from the YAML file
#     with open(filename, 'r') as file:
#         data = yaml.safe_load(file)

#     # Update the calibration data
#     for i, freq in enumerate(calibration_frequencies):
#         # Check if the setpoint exists and update its frequency
#         if i < len(data['calibration']):
#             data['calibration'][i]['frequency'] = float(np.round(freq, 3))
#         else:
#             raise IndexError(f"Setpoint {i} is out of range. The YAML file only supports setpoints from 0 to 499.")

#     # Write the updated data back to the YAML file
#     with open(filename, 'w') as file:
#         yaml.dump(data, file, sort_keys=False)


# def get_frequency_for_setpoint(filename, setpoint):
#     """Read the frequency for the given setpoint from the YAML file."""
#     with open(filename, 'r') as file:
#         data = yaml.safe_load(file)
    
#     for entry in data['calibration']:
#         if entry['setpoint'] == setpoint:
#             return entry['frequency']
    
#     raise ValueError(f"Setpoint {setpoint} not found in {filename}")

def wavelength_to_frequency(wavelength_nm):
    """Convert wavelength in nm to frequency in THz."""
    C = 299792458 # m/s
    wavelength_m = wavelength_nm * 1e-9  # Convert nm to meters
    frequency_thz = C / (wavelength_m * 1e12)  # Convert to THz
    return np.round(frequency_thz, 3)


# def get_setpoint_for_frequency(filename, target_frequency):
#     """Find the setpoint for a given target frequency from the YAML file."""
#     with open(filename, 'r') as file:
#         data = yaml.safe_load(file)

#     # Find the closest setpoint for the given frequency
#     closest_setpoint = None
#     min_diff = float('inf')
    
#     for entry in data['calibration']:
#         frequency = entry['frequency']
#         diff = abs(frequency - target_frequency)
        
#         if diff < min_diff:
#             min_diff = diff
#             closest_setpoint = entry['setpoint']

#     if closest_setpoint is not None:
#         return closest_setpoint
#     else:
#         raise ValueError(f"No suitable setpoint found for frequency {target_frequency} THz")