import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from math import inf
from ctypes import *
import pickle
import traceback

# Ensure a suitable Matplotlib backend
matplotlib.use('TkAgg')  # Adjust backend as needed

# Import necessary modules (assumes these are available in your environment)
from PSGManager import PSGManager
from PPCL_Bare_Bones import LaserControl
from PolCTRLManager import PolCTRLManager
from NIDAQ_USBv4 import NIDAQ_USB
from OSWManager import OSWManager
from GenericUser_v1 import GenericUser
from PSOManager_v2 import PSOManager

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

# Define polarization stabilization experimental classes
class Step:
    def __init__(self, name, pso_params):
        self.name = name
        self.success = False
        self.pso_params = pso_params
        self.failed_badly = True
        
        if isinstance(pso_params, dict): # Checks if pso_params is a dictionary
            for key, value in pso_params.items():
                setattr(self, key, value)
        elif isinstance(pso_params, PSOParams): # Checks if pso_params is a PSOParams object
            self.Alice = pso_params.Alice
            self.Bob = pso_params.Bob
            self.Charlie = pso_params.Charlie

            self.num_particles = pso_params.num_particles
            self.max_iter = pso_params.max_iter
            self.threshold_cost1 = pso_params.threshold_cost1
            self.threshold_cost2 = pso_params.threshold_cost2
            self.threshold_cost_Alice = pso_params.threshold_cost_Alice
            self.channels12 = [1, 2]
            self.channels34 = [3, 4]
            self.Measure_CostFunction = pso_params.Measure_CostFunction
            self.log_file = pso_params.log_file
            self.log_event = pso_params.log_event
            self.meas_device = pso_params.meas_device
            self.visTolPercent = pso_params.visTolPercent # percentage away from threshold_cost that checks how badly stabilization fails
        else:
            raise ValueError(f"pso_params was passed as {type(pso_params)}, accepted types are dict and PSOParams only")
        
        self.best_voltage = None
        self.visibility = None
        self.pso = None

        self.H1_visibility = None
        self.D2_visibility = None
        self.H2_visibility = None
    
    def setup_PSOManager(self, threshold_cost):
        """Sets up a PSOManager object"""
        self.pso = PSOManager(pso_params, threshold_cost)
    
    def run(self):
        """Runs the step and updates success status"""
        raise NotImplementedError
    
    def check(self):
        """Checks if the step is still valid and updates success status"""
        raise NotImplementedError
    
    def reset(self):
        """Reset success state for retry"""
        self.success = False
        self.failed_badly = True

class Bob_H1_Stabilization(Step):
    def check(self):
        """Checks the current visibility or polarization stability for Bob H1"""
        self.log_event(self.log_file, "Setting H polarization from Bob's PSG")
        self.Bob.PSG.polSET(self.Bob.PSG.H)

        _ = OSW_operate(
            self.Charlie.OSW,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4],
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )
        
        Bob_H1_avg_visibility = calculate_average_visibility(
            device=self.Charlie.NIDAQ_USB,
            channels=self.channels12,
            log_file=self.log_file,
            controller=self.Charlie.Pol_CTRL1,
            voltage=self.best_voltage
        )
        
        # Bob_visibilities_dict[f"Run {i}, Step 1: (Avg) Bob H"] = Bob_H_avg_visibility
        self.log_event(self.log_file, f"Bob H1 visibility: {Bob_H1_avg_visibility}")
        
        if Bob_H1_avg_visibility<=self.threshold_cost1:
            self.success = True
        else:
            self.success = False
            if Bob_H1_avg_visibility <= (1 + self.visTolPercent) * self.threshold_cost1:
                self.failed_badly = False
            else:
                self.failed_badly = True
        self.log_event(self.log_file, f"{self.name} visibility: {'Valid' if self.success else 'Failed'}")

        self.visibility = Bob_H1_avg_visibility
    
    def run(self):
        self.reset()
        ticBH = time.perf_counter()
        # log_file = self.log_file # not sure if these are ok to use
        # log_event = self.log_event
        pso = PSOManager(self.pso_params, self.threshold_cost1)
        # setup_PSOManager(self.threshold_cost1)

        self.log_event(self.log_file, f"Running {self.name}...")
        self.log_event(self.log_file, "Setting H polarization from Bob's PSG")
        # self.Bob.PSG.polSET(self.Bob.PSG.H)
        
        _ = OSW_operate(
            self.Charlie.OSW,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4],
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )

        for _ in range(5):
            # Optimize Charlie Pol_CTRL1 to find best PSO voltage and cost
            self.best_voltage, self.best_cost, self.success = pso.optimize_polarization(
                user=self.Bob,
                pol=self.Bob.PSG.H,
                channels=self.channels12,
                user_ctrl=self.Charlie.Pol_CTRL1
            )

            if self.success:
                self.failed_badly = False
                break
        
        tocBH = time.perf_counter()
        self.log_event(self.log_file, f"Time taken for Bob H1 Stabilization: {tocBH - ticBH} seconds")
        # time.sleep(5)

        self.log_event(self.log_file, f"{self.name} Completed Successfully")

class Bob_D2_Stabilization(Step):
    def check(self):
        """Checks the current visibility or polarization stability for Bob D2"""
        self.log_event(self.log_file, "Setting D polarization from Bob's PSG")
        self.Bob.PSG.polSET(self.Bob.PSG.D)

        _ = OSW_operate(
            self.Charlie.OSW,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4], 
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )

        Bob_D2_avg_visibility = calculate_average_visibility(
            device=self.Charlie.NIDAQ_USB,
            channels=self.channels34,
            log_file=self.log_file,
            controller=self.Charlie.Pol_CTRL2,
            voltage=self.best_voltage
        )
        
        self.log_event(self.log_file, f"Bob D2 visibility: {Bob_D2_avg_visibility}")
        
        if Bob_D2_avg_visibility<=self.threshold_cost2:
            self.success = True
        else:
            self.success = False
            if Bob_D2_avg_visibility <= (1 + self.visTolPercent) * self.threshold_cost2:
                self.failed_badly = False
            else:
                self.failed_badly = True
        self.log_event(self.log_file, f"{self.name} visibility: {'Valid' if self.success else 'Failed'}")
        
        self.visibility = Bob_D2_avg_visibility
    
    def run(self):
        self.reset()
        ticBD = time.perf_counter()
        pso = PSOManager(self.pso_params, self.threshold_cost2)

        self.log_event(self.log_file, f"Running {self.name}...")
        self.log_event(self.log_file, "Setting D polarization from Bob's PSG")
        
        _ = OSW_operate(
            self.Charlie.OSW,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4], 
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )

        for _ in range(5):
            # Optimize Charlie Pol_CTRL2 to find best PSO voltage and cost
            self.best_voltage, self.best_cost, self.success = pso.optimize_polarization(
                user=self.Bob,
                pol=self.Bob.PSG.D,
                channels=self.channels34,
                user_ctrl=self.Charlie.Pol_CTRL2
            )

            if self.success:
                self.failed_badly = False
                break
        
        tocBD = time.perf_counter()
        self.log_event(self.log_file, f"Time taken for Bob D2 Stabilization: {tocBD - ticBD} seconds")
        # time.sleep(5)

        self.log_event(self.log_file, f"{self.name} Completed Successfully")

class Alice_H1D2_Stabilization(Step):
    def check_H1D2(self):
        """Checks the current visibility or polarization stability for Alice H1 and D2"""
        self.log_event(self.log_file, "Checking Alice's current visibilities...")
        self.log_event(self.log_file, "Setting H polarization from Alice's PSG")
        self.Alice.PSG.polSET(self.Alice.PSG.H)
        
        _ = OSW_operate(
            self.Charlie.OSW,
            Alice_Switch_status=1,
            Bob_Switch_status=0,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4],
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )
        
        self.H1_visibility = calculate_average_visibility(
            device=self.meas_device,
            channels=self.channels12,
            log_file=self.log_file,
            # It is actually important to add these last two arguments since we are constantly checking
            # if each step is still vaild
            controller=self.Alice.Pol_CTRL1,
            voltage=self.best_voltage
        )

        self.log_event(self.log_file, f"Setting D polarization from Alice's PSG: best_voltage = {self.best_voltage}")
        self.Alice.PSG.polSET(self.Alice.PSG.D) # TODO Need to be able to check Alice H2 as well

        self.D2_visibility = calculate_average_visibility(
            device=self.meas_device,
            channels=self.channels34,
            log_file=self.log_file,
            controller=self.Alice.Pol_CTRL1,
            voltage=self.best_voltage
        )

        self.Alice_all_visibilities = calculate_user_visibilities(
            user=self.Alice,
            device=self.meas_device,
            channels=self.channels12,
            log_file=self.log_file
        )
        
        # Bob_visibilities_dict[f"Run {i}, Step 1: (Avg) Bob H"] = Bob_H_avg_visibility
        self.log_event(self.log_file, 
                       f"Alice H1 visibility: {self.H1_visibility}, Alice D2 visibility: {self.D2_visibility}")
        
        if ((self.H1_visibility<=self.threshold_cost_Alice)):
            self.H1_success = True
            self.H1_failed_badly = False
        else:
            self.H1_success = False
            if self.H1_visibility <= (1 + self.visTolPercent) * self.threshold_cost_Alice:
                self.H1_failed_badly = False
            else:
                self.H1_failed_badly = True

        if ((self.D2_visibility<=self.threshold_cost_Alice)):
            self.D2_success = True
            self.D2_failed_badly = False
        else:
            self.D2_success = False
            if self.D2_visibility <= (1 + self.visTolPercent) * self.threshold_cost_Alice:
                self.D2_failed_badly = False
            else:
                self.D2_failed_badly = True
        
        self.success = self.H1_success and self.D2_success
        self.failed_badly = self.H1_failed_badly or self.D2_failed_badly
        self.log_event(self.log_file, f"{self.name} visibility: {'Valid' if self.success else 'Failed'}")

        self.D2_visibility = self.D2_visibility

    def check_H1H2(self):
        """Checks the current visibility or polarization stability for Alice H1 and H2"""
        self.log_event(self.log_file, "Checking Alice's current visibilities...")
        self.log_event(self.log_file, "Setting H polarization from Alice's PSG")
        self.Alice.PSG.polSET(self.Alice.PSG.H)
        
        _ = OSW_operate(
            self.Charlie.OSW,
            Alice_Switch_status=1,
            Bob_Switch_status=0,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4],
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )
        self.H1_visibility = calculate_average_visibility(
            device=self.meas_device,
            channels=self.channels12,
            log_file=self.log_file,
            # It is actually important to add these last two arguments since we are constantly checking
            # if each step is still vaild
            controller=self.Alice.Pol_CTRL1,
            voltage=self.best_voltage
        )

        # self.log_event(self.log_file, "Setting D polarization from Alice's PSG")
        # self.Alice.PSG.polSET(self.Alice.PSG.D) # TODO Need to be able to check Alice H2 as well

        self.H2_visibility = calculate_average_visibility(
            device=self.meas_device,
            channels=self.channels34,
            log_file=self.log_file,
            controller=self.Alice.Pol_CTRL1,
            voltage=self.best_voltage
        )

        self.Alice_all_visibilities = calculate_user_visibilities(
            user=self.Alice,
            device=self.meas_device,
            channels=self.channels12,
            log_file=self.log_file
        )
        
        
        # Bob_visibilities_dict[f"Run {i}, Step 1: (Avg) Bob H"] = Bob_H_avg_visibility
        self.log_event(self.log_file, 
                       f"Alice H1 visibility: {self.H1_visibility}, Alice H2 visibility: {self.H2_visibility}")
        
        if ((self.H1_visibility<=self.threshold_cost_Alice)):
            self.H1_success = True
            self.H1_failed_badly = False
        else:
            self.H1_success = False
            if self.H1_visibility <= (1 + self.visTolPercent) * self.threshold_cost_Alice:
                self.H1_failed_badly = False
            else:
                self.H1_failed_badly = True

        if ((self.H2_visibility<=self.threshold_cost_Alice)):
            self.H2_success = True
            self.H2_failed_badly = False
        else:
            self.H2_success = False
            if self.H2_visibility <= (1 + self.visTolPercent) * self.threshold_cost_Alice:
                self.H2_failed_badly = False
            else:
                self.H2_failed_badly = True
        
        self.success = self.H1_success and self.H2_success
        self.failed_badly = self.H1_failed_badly or self.H2_failed_badly
        self.log_event(self.log_file, f"{self.name} visibility: {'Valid' if self.success else 'Failed'}")

    def run(self, step1, step2):
        self.reset() # Automatically sets self.success to False and self.failed_badly to True
        MAX_TRIES = 10
        num_tries = 0
        ticAH = time.perf_counter()
        pso = PSOManager(self.pso_params, self.threshold_cost_Alice)

        self.log_event(self.log_file, f"Running {self.name}...")
        self.log_event(self.log_file, "Setting H polarization from Alice's PSG")

        # log_event(log_file, f"Running {self.name}... Checking Step 1 and Step 2 validity")

        all_succeeded = self.success and step1.success and step2.success
        while num_tries < MAX_TRIES and not all_succeeded:
            num_tries += 1
            _ = OSW_operate(
                self.Charlie.OSW,
                Alice_Switch_status=1,
                Bob_Switch_status=0,
                meas_device=self.meas_device,
                channels=[1, 2, 3, 4],
                log_file=self.log_file,
                Alice=self.Alice,
                Bob = self.Bob
            )

            # Optimize Charlie Pol_CTRL1 to find best PSO voltage and cost
            self.best_voltage, self.best_cost, self.success = pso.optimize_polarization(
                user=self.Alice,
                pol=self.Alice.PSG.H,
                channels=self.channels12,
                user_ctrl=self.Alice.Pol_CTRL1
            )

            self.check_H1D2() # Problem here
            self.log_event(self.log_file,
                       f"Alice H1 visibility: {self.H1_visibility}, Alice D2 visibility: {self.D2_visibility}")
            
            for key in self.Alice_all_visibilities.keys():
                self.log_event(self.log_file,
                        f"Alice {key} visibility: {self.Alice_all_visibilities[key]}")

            # Revalidate Step 1 and Step 2
            step1.check()
            step2.check()

            # if step1_valid and step2_valid:
            #     break # If both are valid, Step 3 is successful

            # Retry only failed steps
            if not step1.success:
                log_event(log_file, "Step 1 visibilites are above threshold, Redoing Step 1")
                step1.run()
            if not step2.success:
                log_event(log_file, "Step 2 visibilites are above threshold, Redoing Step 2")
                step2.run()
            
            all_succeeded = self.success and step1.success and step2.success # TODO fix this to only deal with Alice

        if num_tries==MAX_TRIES:
            raise TimeoutError(f"Alice H1 and D2 failed to stabilize within {num_tries} tries")
        
        if self.success:
            self.failed_badly = False
        
        log_event(log_file, f"{self.name} Completed Successfully")
        tocAH = time.perf_counter()
        self.log_event(self.log_file, f"Time taken for Alice H1 and D2 Stabilization: {tocAH - ticAH} seconds")

class Bob_H2_Stabilization(Step):
    def check(self):
        """Checks the current visibility or polarization stability for Bob H2"""
        self.Bob.PSG.polSET(self.Bob.PSG.H)

        _ = OSW_operate(
            self.Charlie.OSW,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4],
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )
        
        Bob_H2_avg_visibility = calculate_average_visibility(
            device=self.meas_device,
            channels=self.channels34,
            log_file=self.log_file,
            controller=self.Charlie.Pol_CTRL2,
            voltage=self.best_voltage
        )
        
        self.log_event(self.log_file, f"Bob H2 visibility: {Bob_H2_avg_visibility}")
        
        if Bob_H2_avg_visibility<=self.threshold_cost2:
            self.success = True
        else:
            self.success = False
            if Bob_H2_avg_visibility <= (1 + self.visTolPercent) * self.threshold_cost2:
                self.failed_badly = False
            else:
                self.failed_badly = True
        self.log_event(self.log_file, f"{self.name} visibility: {'Valid' if self.success else 'Failed'}")

        self.visibility = Bob_H2_avg_visibility
    
    def run(self):
        self.reset()
        ticBH = time.perf_counter()
        pso = PSOManager(self.pso_params, self.threshold_cost2)

        self.log_event(self.log_file, f"Running {self.name}...")
        self.log_event(self.log_file, "Setting H polarization from Bob's PSG")
        # self.Bob.PSG.polSET(self.Bob.PSG.H)
        
        _ = OSW_operate(
            self.Charlie.OSW,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4],
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )

        for _ in range(5):
            # Optimize Charlie Pol_CTRL1 to find best PSO voltage and cost
            self.best_voltage, self.best_cost, self.success = pso.optimize_polarization(
                user=self.Bob,
                pol=self.Bob.PSG.H,
                channels=self.channels34,
                user_ctrl=self.Charlie.Pol_CTRL2
            )

            if self.success:
                self.failed_badly = False
                break
        
        tocBH = time.perf_counter()
        self.log_event(self.log_file, f"Time taken for Bob H2 Stabilization: {tocBH - ticBH} seconds")
        # time.sleep(5)

        self.log_event(self.log_file, f"{self.name} Completed Successfully")

class PolarizationStabilization:
    def __init__(self, pso_params, duration, tracking_interval=30, visTolPercent=0.015):

        self.step1 = Bob_H1_Stabilization("Step 1", pso_params)
        self.step2 = Bob_D2_Stabilization("Step 2", pso_params)
        self.step3 = Alice_H1D2_Stabilization("Step 3", pso_params)
        self.step4 = Bob_H2_Stabilization("Step 4", pso_params)

        self.pso_params = pso_params
        self.duration = duration
        self.log_file = self.pso_params.log_file
        self.log_event = self.pso_params.log_event

        self.tracking_check_interval = tracking_interval
        self.visTolPercent = visTolPercent
    
    def initial_stabilization(self):
        # NOTE Step.run() does not check whether self.failed_badly is True or False when self.success = False,
        # only Step.check() does this
        self.log_event(self.log_file, "Running Initial Stabilization...")
        self.step1.run()
        self.step2.run()
        self.step3.run(self.step1, self.step2)
        self.step4.run()
        self.log_event(self.log_file, "Initial Stabilization Completed")

    def stabilization_tracking(self):
        ticPS = time.perf_counter()
        self.initial_stabilization()
        self.log_event(self.log_file, "Running Stabilization Tracking...")
        tocPS = time.perf_counter()
        time_elapsed = tocPS - ticPS
        while time_elapsed < self.duration:
            # Check if steps 1, 3, and 4 are still valid
            self.step1.check() # Bob H1
            self.step3.check_H1H2() # Alice H1 and H2
            self.step4.check() # Bob H2
            
            # Ensure Steps 1 & 2 complete successfully
            if self.step1.success and self.step3.success and self.step4.success:
                self.log_event(self.log_file, "All visibilities are below their respective thresholds"
                                + "\nStabilization is good for now!")
                
            elif ((not self.step3.H1_success and not self.step3.H2_success)
                  and (self.step1.success and self.step4.success)): # Alice failed and Bob did not
                # if not self.step3.H1_failed_badly and not self.step3.H2_failed_badly: # Alice H1 and H2 are still within visTolPercent of threshold_cost
                #     self.step3.run(self.step1, self.step2)
                # else:
                self.initial_stabilization()
            elif ((self.step3.H1_success and self.step3.H2_success)
                  and (not self.step1.success and not self.step4.success)): # Bob failed and Alice did not
                # if not self.step1.failed_badly and not self.step4.failed_badly: # Bob H1 and H2 are still within visTolPercent of threshold_cost
                #     self.step1.run()
                #     self.step4.run()
                # else: # If Bob H1 or H2 failed badly
                self.initial_stabilization()

            elif ((not self.step3.H1_success and not self.step1.success)
                    and (self.step3.H2_success and self.step4.success)): # H1 failed for both, H2 is fine for both
                self.step1.run()
            elif ((self.step3.H1_success and self.step1.success)
                    and (not self.step3.H2_success and not self.step4.success)): # H2 failed for both, H1 is fine for both
                self.step4.run()

            else:
                self.initial_stabilization()

            time.sleep(self.tracking_check_interval)
            tocPS = time.perf_counter()
            time_elapsed = tocPS - ticPS
            self.log_event(self.log_file, f"Current time elapsed: {time_elapsed} seconds\n")
        self.log_event(self.log_file, "Stabilization Tracking Completed\n")
        # delta_t = tocPS - ticPS
        # delta_t_hr = delta_t // 3600
        # delta_t_min = (delta_t % 3600) // 60
        # delta_t_sec = delta_t % 60
        # self.log_event(self.log_file, f"Total time elapsed for full stabilization tracking loop is {delta_t_hr} hours, {delta_t_min} minutes, and {delta_t_sec} seconds")

if __name__ == "__main__":
    # TODO Add conditions checking Bob visibilities during steps 1 and 2 and Alice checking that Bob's visibilities are reasonable

    num_runs = 1 # it can take around 10-25 minutes to run
    duration = 15 * 60 # amount of time to run the stabilization algorithm for

    # Define Alice and Bob visibilities dictionaries to save to .pkl files later
    Alice_visibilities_dict = {}
    Bob_visibilities_dict = {}
    # Define np.array of duration per run
    timing = np.zeros((num_runs))

    # Create log file with unique name
    current_date = datetime.now().strftime("%Y-%m-%d")
    path = "C:/Users/CPTLab/Documents/Claire_Ellison/Cleaning_Code-1/Components/"
    runner_idx = 1
    os.makedirs("Data", exist_ok=True)
    while os.path.exists(path + f"PSO Data/{current_date}_Polarization_Stabilization_PSO_log_{runner_idx}.txt"):
        runner_idx += 1
    log_filename = path + f"PSO Data/{current_date}_Polarization_Stabilization_PSO_log_{runner_idx}.txt"

    log_file = open(log_filename, 'w')
    message = f"Particle Swarm Optimization Polarization Stabilization Test {runner_idx}, {current_date} step-by-step log" # message should not be larger than 116 characters
    # notes = f"(Bouncing from the bounds with particle velocity damping = {DAMPING})"
    # notes = "(Clipping at the boundaries)"
    notes = "(Clipping at the boundaries and resetting a particle's velocity and voltage to a random value"
    notes2 = "if it hits the boundary)"
    log_event(log_file, "[" + "=" * 118 + "]")
    format_title(log_file, message)
    format_title(log_file, notes)
    format_title(log_file, notes2)
    log_event(log_file, "[" + "=" * 118 + "]\n")

    # Initialize devices
    log_event(log_file, "##### Initializing Alice's devices #####")
    Alice = GenericUser(name= "Alice", laser_port="COM14", PSG_port="COM6", PolCTRL_port1="COM12") # COM4 is LUNA, COM21 is LADAQ+INQNET PolCTRL
    Alice.laser.connect_laser()
    Alice.laser.turn_on()
    Alice.PSG.connect()
    Alice.Pol_CTRL1.connect()
    log_event(log_file, "##### Alice's devices are initialized #####")
    
    log_event(log_file, "##### Initializing Bob's devices ##### \n")
    # Bob = GenericUser(name= "Bob", laser_port="COM10", PSG_port="COM6", PolCTRL_port1="COM12") # COM4 is LUNA, COM21 is LADAQ+INQNET PolCTRL
    Bob = GenericUser(name= "Bob", laser_port="COM3", PSG_port="COM7")
    Bob.laser.connect_laser()
    Bob.PSG.connect()
    Bob.laser.turn_on()
    log_event(log_file, "##### Bob's devices are initialized #####\n")

    log_event(log_file, "##### Initializing Charlie's devices #####\n")
    Charlie = GenericUser(name= "Charlie", PolCTRL_port1="COM8", PolCTRL_port2="COM11", NIDAQ_USB_id="Dev1")#, OSW_port="COM9")
    # Charlie = GenericUser(name= "Charlie", PolCTRL_port1="COM11", PolCTRL_port2="COM8", NIDAQ_USB_id="Dev1")#, OSW_port="COM9")
    # COM 8 V_2π = 0.887 V, COM 11 V_2π = 0.94 V, COM 12 V_2π = 0.752
    Charlie.Pol_CTRL1.connect()
    Charlie.Pol_CTRL2.connect()
    # Charlie.OSW.connect()
    # Charlie.NIDAQ_USB.setup_ai_task()
    log_event(log_file, "##### Charlie's devices are initialized #####\n")

    # Define important values
    dim = 3
    # bounds = [(0.1, 1.1)] * dim # deprecated in current PSOManager and PSOManager_v2 code
    # Alice_bounds = [(0.01, 2.49)] * dim
    # num_particles = 20
    # max_iter = 20
    threshold_cost1 = 0.005
    threshold_cost2 = 0.01
    threshold_cost_Alice = 0.04
    # DAMPING = 0.01
    channels21 = [2,1]
    channels43 = [4,3]

    # pso_params = {
    #     'dim': dim,
    #     'num_particles': 20,
    #     'max_iter': 20,
    #     'CostFunction': VisibilityCal,
    #     'MeasureFunction': MeasureFunction,
    #     'log_event': log_event,
    #     'log_file': log_file,
    #     'Alice': Alice,
    #     'Bob': Bob,
    #     'Charlie': Charlie,
    #     'meas_device': Charlie.NIDAQ_USB,
    #     'threshold_cost1': threshold_cost1,
    #     'threshold_cost2': threshold_cost2,
    #     'threshold_cost_Alice': threshold_cost_Alice,
    # }

    pso_params = PSOParams(Alice, Bob, Charlie,
                           Measure_CostFunction, log_event, log_file, Charlie.NIDAQ_USB)

    ### Default OSW connetion
    Charlie.OSW.Alice_Switch = 0
    Charlie.OSW.Bob_Switch = 1

    # Charlie.OSW.StatusSet(Charlie.OSW.Alice_Switch, 1)
    # Charlie.OSW.StatusSet(Charlie.OSW.Bob_Switch, 0)
    OSW_operate(OSW=None, Alice_Switch_status=1, Bob_Switch_status=0, meas_device=Charlie.NIDAQ_USB, channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                Alice=Alice, Bob=Bob)
    raw_Alice_power = MeasureFunction(Charlie.NIDAQ_USB, [1,2,3,4])
    print(f"Alice raw power: {raw_Alice_power}")
    Charlie.NIDAQ_USB.Alice_power = np.sum(raw_Alice_power)
    print(f"Alice power: {Charlie.NIDAQ_USB.Alice_power}")

    # Charlie.OSW.StatusSet(Charlie.OSW.Alice_Switch, 0)
    # Charlie.OSW.StatusSet(Charlie.OSW.Bob_Switch, 1)
    _, Bob_power = OSW_operate(OSW=None, Alice_Switch_status=0, Bob_Switch_status=1, meas_device=Charlie.NIDAQ_USB, channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                Alice=Alice, Bob=Bob)
    raw_Bob_power = MeasureFunction(Charlie.NIDAQ_USB, [1,2,3,4])
    print(f"Bob power after switch: {Bob_power}")
    print(f"Bob raw power: {raw_Bob_power}")
    Charlie.NIDAQ_USB.Bob_power = np.sum(raw_Bob_power)
    print(f"Bob power: {Charlie.NIDAQ_USB.Bob_power}")

    while True:
        OSW_operate(OSW=None, Alice_Switch_status=0, Bob_Switch_status=0, meas_device=Charlie.NIDAQ_USB, channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                    Alice=Alice, Bob=Bob)
        raw_zero_power = MeasureFunction(Charlie.NIDAQ_USB, [1,2,3,4])
        print(f"Raw zero power: {raw_zero_power}")
        Charlie.NIDAQ_USB.zero_power = raw_zero_power
        log_event(log_file, f"measure zero power: {Charlie.NIDAQ_USB.zero_power}") 
        if np.mean(Charlie.NIDAQ_USB.zero_power)<0.01:
            break

    try:
        tic_total = time.perf_counter()
        run_plural = "s" if num_runs != 1 else ""
        for i in range(num_runs):
            log_event(log_file, "=" * 120)
            log_event(log_file, f"Run # {i+1}/{num_runs}")
            log_event(log_file, "=" * 120)

            tic_stablization = time.perf_counter()

            ps = PolarizationStabilization(pso_params, duration)
            ps.stabilization_tracking()

            Charlie.OSW.StatusSet(Charlie.OSW.Bob_Switch, 0)
            toc_stablization = time.perf_counter()
            delta_t = toc_stablization - tic_stablization
            delta_t_hr = int(delta_t / 3600)
            delta_t_min = int((delta_t % 3600) / 60)
            delta_t_sec = delta_t % 60
            log_event(log_file, f"Total time elapsed for Run # {i+1}/{num_runs} is {delta_t_hr} hours, {delta_t_min} minutes and {delta_t_sec} seconds")
            timing[i] = delta_t

    except KeyboardInterrupt:
        log_event(log_file, "Polarization stabilization interrupted by user.")
    except Exception as e:
        log_event(log_file, f"An error occurred: {e}")
        traceback.print_exc()  # Print full traceback
    finally:
        toc_total = time.perf_counter()
        delta_t_total = toc_total - tic_total
        delta_t_total_hr = int(delta_t_total / 3600)
        delta_t_total_min = int((delta_t_total % 3600) / 60)
        delta_t_total_sec = delta_t_total % 60
        log_event(log_file, f"\nTotal time elapsed for all {num_runs} Runs is {delta_t_total_hr} hours, {delta_t_total_min} minutes and {delta_t_total_sec} seconds")

        log_event(log_file, "end")
        Alice.laser.turn_off()
        Alice.disconnect_all()
        log_event(log_file, "Alice's laser is off")
        Bob.laser.turn_off()
        Bob.disconnect_all()
        log_event(log_file, "Bob's laser is off")
        Charlie.disconnect_all()
        Charlie.OSW.disconnect()
        log_event(log_file, "")
        log_file.close()

        with open(path + f"PSO Data/{current_date}_Polarization_Stabilization_AliceVisibilities_{runner_idx}.pkl", 'wb') as f_a:
            pickle.dump(Alice_visibilities_dict, f_a)
        with open(path + f"PSO Data/{current_date}_Polarization_Stabilization_BobVisibilities_{runner_idx}.pkl", 'wb') as f_b:
            pickle.dump(Bob_visibilities_dict, f_b)
