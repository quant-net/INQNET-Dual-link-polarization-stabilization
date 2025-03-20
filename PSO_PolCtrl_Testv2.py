# Laser
# PSO
# ThorlabsPM
# PolCtrl

import os
import numpy as np
from ctypes import *
import matplotlib.pyplot as plt
import time
import traceback
import datetime
import yaml
import pickle

from PSGManager import PSGManager
from PPCL_Bare_Bones import LaserControl
from PolCTRLManager import PolCTRLManager
from NIDAQ_USBv4 import NIDAQ_USB
# from OSWManager import OSWManager
from GenericUser_v1 import GenericUser
from PSOManager_v2 import PSOManager
# from TimeTaggerFunctions import TimeTaggerManager

"""
Fiber setup diagram
                            ┌─D THPM1 (to be minimized)
Laser ─ PSG ─ PolCtrl ─ PBS ┤
                            └─D THPM2 (to be maximized)
"""

class PSOParams:
    def __init__(self, Alice, Bob, Charlie, Measure_CostFunction, log_event, log_file, meas_device):
        self.num_particles = 20
        self.max_iter = 20
        self.threshold_cost1 = 0.01
        self.threshold_cost2 = 0.01
        self.threshold_cost_Alice = 0.04
        self.channels12 = [1, 2]
        self.channels34 = [3, 4]
        self.Measure_CostFunction = Measure_CostFunction  # Assigning function
        self.log_event = log_event  # Assigning function
        self.log_file = log_file
        self.meas_device = meas_device
        self.Alice = Alice
        self.Bob = Bob
        self.Charlie = Charlie
        self.visTolPercent = 0.015


def OSW_operate(OSW, Alice_Switch_status, Bob_Switch_status, meas_device, channels, log_file, initCheck = False,
                Alice=None, Bob=None):
    success = False
    while True:
        if Alice_Switch_status==0:
            Alice.laser.turn_off()
        elif Alice_Switch_status==1: 
            Alice.laser.turn_on(10)
        
        if Bob_Switch_status==0:
            Bob.laser.turn_off()
        elif Bob_Switch_status==1:
            Bob.laser.turn_on(10)
            
        # OSW.StatusSet(OSW.Alice_Switch, Alice_Switch_status)
        # OSW.StatusSet(OSW.Bob_Switch, Bob_Switch_status)
        power_check = np.sum(MeasureFunction(meas_device, channels))
        # print(f"Measured power: {power_check}")
        if initCheck:
            break
        else:
            if Alice_Switch_status == 0 and Bob_Switch_status == 0:
                reference_power = meas_device.zero_power
            elif Alice_Switch_status == 0 and Bob_Switch_status == 1:
                reference_power = meas_device.Bob_power
            elif Alice_Switch_status == 1 and Bob_Switch_status == 0:
                reference_power = meas_device.Alice_power
            else:
                reference_power = meas_device.Alice_power + meas_device.Bob_power
            
            if abs(power_check / reference_power - 1) < 0.2:
                    success = True
                    break
    return success, power_check

def log_event(log_file, message, log=True):
    """Log the event message to the file and print it."""
    print(message)
    if log==True:
        log_file.write(message + "\n")

def Measure_CostFunction(user, meas_device, channels):
    if user.name == "Alice":
        measurement1 = MeasureFunction(meas_device, channels=[1, 2])
        visibilityH = VisibilityCal(measurement1)
        user.PSG.polSET(user.PSG.D)
        measurement2 = MeasureFunction(meas_device, channels=[3, 4])
        visibilityD = VisibilityCal(measurement2)
        measurement = measurement1+measurement2
        cost = visibilityH+visibilityD
        user.PSG.polSET(user.PSG.H)
    elif user.name == "Bob":
        measurement = MeasureFunction(meas_device, channels)
        cost = VisibilityCal(measurement)
    return measurement, cost

# Synchronous Analog Input Measurement Function
def MeasureFunctionSync(meas_device, channels): 
    # Each measurement takes around 0.062 s using setup and close, around 0.059 s without
    tic = time.perf_counter()
    meas_device.setup_ai_task()
    data = meas_device.ai_measurement(samples=meas_device.ai_samples)
    mean_data = np.mean(data, axis=1)
    meas_device.close_all_tasks()
    toc = time.perf_counter()
    # log_event(log_file, f"Time for measurement was {toc - tic} seconds")
    # log_event(log_file, f"mea:{mean_data}")
    # Adjust channels from 1-based to 0-based and return only the requested channels
    return mean_data[channels]

# Asynchronous Analog Input Measurement Function
def MeasureFunctionAsync(meas_device, channels): 
    # Each measurement takes around 0.062 s using setup and close, around 0.059 s without
    # tic = time.perf_counter()
    meas_device.setup_ai_task()
    ai_duration = np.round(meas_device.ai_samples/meas_device.ai_rate, 2)
    data = meas_device.ai_measurement_async(
        duration_sec=ai_duration,
        buffer_size=1000, # this is never used
        callback_interval=meas_device.ai_samples # number of samples per measurement (?)
        )
    mean_data = np.mean(data, axis=1)
    meas_device.close_all_tasks()
    # toc = time.perf_counter()
    # log_event(log_file, f"Time for measurement was {toc - tic} seconds")
    # log_event(log_file, f"mea:{mean_data}")
    # Adjust channels from 1-based to 0-based and return only the requested channels
    return mean_data[channels]

# NIDAQ Measurement Function with either Sync or Async Modes
def MeasureFunction(meas_device, channels, sync=True):
    if sync:
        channel_data = MeasureFunctionSync(meas_device, channels)
    else:
        channel_data = MeasureFunctionAsync(meas_device, channels)
    
    return channel_data

def VisibilityCal(channels):
    return channels[1]/channels[0]

def calculate_average_visibility(device, channels, log_file, controller=None, voltage=None, num_measurements=10):
    # Set the voltage on the controller
    if controller is not None and voltage is not None:
        controller.Vset(voltage)

    # Perform measurements and calculate visibility
    total_visibility = 0
    for i in range(num_measurements):
        measure = MeasureFunction(device, channels)
        visibility = VisibilityCal(measure)
        total_visibility += visibility
        # log_event(log_file, f"Measurement {i+1}: Visibility = {visibility}")

    # Calculate the average visibility
    average_visibility = total_visibility / num_measurements
    # log_event(log_file, f"Average Visibility: {average_visibility}")

    return average_visibility

def calculate_user_visibilities(user, device, channels, log_file, controller=None, voltage=None):
    polarization_states = ["H", "V", "D", "A", "L", "R"]
    visibilities = {}

    for state in polarization_states:
        getattr(user.PSG, f"polSET")(getattr(user.PSG, state))
        avg_visibility = calculate_average_visibility(device, channels, log_file, controller, voltage)
        visibilities[state] = avg_visibility
        log_event(log_file, f"{user.name} visibility {state}: {avg_visibility}\n")

    return visibilities

def format_title(log_file, message):
    message_length = len(message)
    space_remaining = 119-message_length
    spacing_left = " " * (space_remaining//2)
    spacing_right = " " * (space_remaining - 1 - space_remaining//2)
    formatted_title = (f"|{spacing_left}{message}{spacing_right}|")
    log_event(log_file, formatted_title)


