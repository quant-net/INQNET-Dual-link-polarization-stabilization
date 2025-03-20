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
from GenericUser_v2 import GenericUser
from PSOManager_v2 import PSOManager
# from PSOManager_v3 import PSOManager
from TimeTaggerFunctions import TimeTaggerManager
from RigolManager import RigolDG4162Manager

class PSOParams:
    def __init__(self, Alice, Bob, Charlie, Measure_CostFunction, log_event, log_file, meas_device):
        self.num_particles = 20
        self.max_iter = 20
        self.threshold_cost1 = 0.02 # 0.015
        self.threshold_cost2 = 0.03 #0.025
        self.threshold_cost_Alice = 0.045
        self.channels12 = [1, 2]
        self.channels34 = [3, 4]
        self.Measure_CostFunction = Measure_CostFunction  # Assigning function
        self.log_event = log_event  # Assigning function
        self.log_file = log_file
        self.meas_device = meas_device
        self.Alice = Alice
        self.Bob = Bob
        self.Charlie = Charlie
        self.visTol = 0.015

def OSW_operate(OSW, Alice_Switch_status, Bob_Switch_status, meas_device, channels, log_file, initCheck = False,
                Alice=None, Bob=None):
    success = False
    # OSW.control_channel(1, "OFF")
    # OSW.control_channel(2, "OFF")
    while True:
        if Alice_Switch_status==0:
            OSW.control_channel(1, "OFF")
        elif Alice_Switch_status==1: 
            OSW.configure_channel(**OSW.ch1_params)
            OSW.control_channel(1, "ON")
        
        if Bob_Switch_status==0:
            OSW.control_channel(2, "OFF")
        elif Bob_Switch_status==1:
            OSW.configure_channel(**OSW.ch2_params)
            OSW.control_channel(2, "ON")

        # # swap Alice and Bob
        # if Bob_Switch_status==0:
        #     OSW.control_channel(1, "OFF")
        # elif Bob_Switch_status==1: 
        #     OSW.configure_channel(**OSW.ch1_params)
        #     OSW.control_channel(1, "ON")
        
        # if Alice_Switch_status==0:
        #     OSW.control_channel(2, "OFF")
        # elif Alice_Switch_status==1:
        #     OSW.configure_channel(**OSW.ch2_params)
        #     OSW.control_channel(2, "ON")

        time.sleep(1)
            
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
            
            if abs(power_check / reference_power - 1) < 0.9: # TODO Most likely need to change this tolerance here
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

# TimeTagger Measure Function
def MeasureFunction(meas_device, channels, measurement_time=0.01):
    return meas_device.getChannelCountRate(channels, measurement_time=measurement_time)

# # Synchronous Analog Input Measurement Function
# def MeasureFunctionSync(meas_device, channels): 
#     # Each measurement takes around 0.062 s using setup and close, around 0.059 s without
#     tic = time.perf_counter()
#     meas_device.setup_ai_task()
#     data = meas_device.ai_measurement(samples=meas_device.ai_samples)
#     mean_data = np.mean(data, axis=1)
#     meas_device.close_all_tasks()
#     toc = time.perf_counter()
#     # log_event(log_file, f"Time for measurement was {toc - tic} seconds")
#     # log_event(log_file, f"mea:{mean_data}")
#     # Adjust channels from 1-based to 0-based and return only the requested channels
#     return mean_data[channels]

# # Asynchronous Analog Input Measurement Function
# def MeasureFunctionAsync(meas_device, channels): 
#     # Each measurement takes around 0.062 s using setup and close, around 0.059 s without
#     # tic = time.perf_counter()
#     meas_device.setup_ai_task()
#     ai_duration = np.round(meas_device.ai_samples/meas_device.ai_rate, 2)
#     data = meas_device.ai_measurement_async(
#         duration_sec=ai_duration,
#         buffer_size=1000, # this is never used
#         callback_interval=meas_device.ai_samples # number of samples per measurement (?)
#         )
#     mean_data = np.mean(data, axis=1)
#     meas_device.close_all_tasks()
#     # toc = time.perf_counter()
#     # log_event(log_file, f"Time for measurement was {toc - tic} seconds")
#     # log_event(log_file, f"mea:{mean_data}")
#     # Adjust channels from 1-based to 0-based and return only the requested channels
#     return mean_data[channels]

# # NIDAQ Measurement Function with either Sync or Async Modes
# def MeasureFunction(meas_device, channels, sync=True):
#     if sync:
#         channel_data = MeasureFunctionSync(meas_device, channels)
#     else:
#         channel_data = MeasureFunctionAsync(meas_device, channels)
    
#     return channel_data

def VisibilityCal(channels): # TODO Maybe add a way to normalize with respect to Alice/Bob zero power?
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
            self.visTol = pso_params.visTol # percentage away from threshold_cost that checks how badly stabilization fails
        else:
            raise ValueError(f"pso_params was passed as {type(pso_params)}, accepted types are dict and PSOParams only")
        
        self.sleep_time = 0.001
        self.MAX_TRIES = 5

        self.best_voltage = None
        self.visibility = -1
        self.pso = None

        self.H1_visibility = -1
        self.D2_visibility = -1
        self.H2_visibility = -1
    
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
            self.Charlie.Rigol,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4],
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )
        
        Bob_H1_avg_visibility = calculate_average_visibility(
            device=self.meas_device,
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
            if Bob_H1_avg_visibility <= (1 + self.visTol) * self.threshold_cost1:
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
            self.Charlie.Rigol,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4], # TODO Need to check channels for TimeTaggerManager (is 6 given as 2 or 6?)
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )
        
        #self.check()
        num_tries = 0
        while num_tries <= self.MAX_TRIES and not self.success:
            # Optimize Charlie Pol_CTRL1 to find best PSO voltage and cost
            self.best_voltage, self.visibility, self.success = pso.optimize_polarization(
                user=self.Bob,
                pol=self.Bob.PSG.H,
                channels=self.channels12,
                user_ctrl=self.Charlie.Pol_CTRL1
            )

            if self.success:
                self.failed_badly = False
            
            num_tries += 1

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
            self.Charlie.Rigol,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4], 
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )

        Bob_D2_avg_visibility = calculate_average_visibility(
            device=self.meas_device,
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
            if Bob_D2_avg_visibility <= (1 + self.visTol) * self.threshold_cost2:
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
            self.Charlie.Rigol,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4], 
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )
            
        #self.check()
        num_tries = 0
        while num_tries <= self.MAX_TRIES and not self.success:
            # Optimize Charlie Pol_CTRL2 to find best PSO voltage and cost
            self.best_voltage, self.visibility, self.success = pso.optimize_polarization(
                user=self.Bob,
                pol=self.Bob.PSG.D,
                channels=self.channels34,
                user_ctrl=self.Charlie.Pol_CTRL2
            )

            if self.success:
                self.failed_badly = False
            
            num_tries += 1
        
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
            self.Charlie.Rigol,
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
            if self.H1_visibility <= (1 + self.visTol) * self.threshold_cost_Alice:
                self.H1_failed_badly = False
            else:
                self.H1_failed_badly = True

        if ((self.D2_visibility<=self.threshold_cost_Alice)):
            self.D2_success = True
            self.D2_failed_badly = False
        else:
            self.D2_success = False
            if self.D2_visibility <= (1 + self.visTol) * self.threshold_cost_Alice:
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
            self.Charlie.Rigol,
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
            if self.H1_visibility <= (1 + self.visTol) * self.threshold_cost_Alice:
                self.H1_failed_badly = False
            else:
                self.H1_failed_badly = True

        if ((self.H2_visibility<=self.threshold_cost_Alice)):
            self.H2_success = True
            self.H2_failed_badly = False
        else:
            self.H2_success = False
            if self.H2_visibility <= (1 + self.visTol) * self.threshold_cost_Alice:
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
                self.Charlie.Rigol,
                Alice_Switch_status=1,
                Bob_Switch_status=0,
                meas_device=self.meas_device,
                channels=[1, 2, 3, 4],
                log_file=self.log_file,
                Alice=self.Alice,
                Bob = self.Bob
            )

            # Revalidate Step 1 and Step 2
            step1.check()
            step2.check()

            # Retry only failed steps
            if not step1.success:
                log_event(log_file, "Step 1 visibilites are above threshold, Redoing Step 1")
                step1.run()
            if not step2.success:
                log_event(log_file, "Step 2 visibilites are above threshold, Redoing Step 2")
                step2.run()
            
            self.check_H1D2()
            # Optimize Charlie Pol_CTRL1 to find best PSO voltage and cost
            if step1.success and step2.success and (not self.H1_success or not self.D2_success):
                self.best_voltage, self.best_cost, self.success = pso.optimize_polarization(
                    user=self.Alice,
                    pol=self.Alice.PSG.H,
                    channels=self.channels12,
                    user_ctrl=self.Alice.Pol_CTRL1
                )

            self.check_H1D2()
            self.log_event(self.log_file,
                       f"Alice H1 visibility: {self.H1_visibility}, Alice D2 visibility: {self.D2_visibility}")
            
            for key in self.Alice_all_visibilities.keys():
                self.log_event(self.log_file,
                        f"Alice {key} visibility: {self.Alice_all_visibilities[key]}")

            # if step1_valid and step2_valid:
            #     break # If both are valid, Step 3 is successful

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
            self.Charlie.Rigol,
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
            if Bob_H2_avg_visibility <= (1 + self.visTol) * self.threshold_cost2:
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
            self.Charlie.Rigol,
            Alice_Switch_status=0,
            Bob_Switch_status=1,
            meas_device=self.meas_device,
            channels=[1, 2, 3, 4],
            log_file=self.log_file,
            Alice=self.Alice,
            Bob = self.Bob
        )
    
        #self.check()
        num_tries = 0
        while num_tries <= self.MAX_TRIES and not self.success:
            # Optimize Charlie Pol_CTRL1 to find best PSO voltage and cost
            self.best_voltage, self.visibility, self.success = pso.optimize_polarization(
                user=self.Bob,
                pol=self.Bob.PSG.H,
                channels=self.channels34,
                user_ctrl=self.Charlie.Pol_CTRL2
            )

            if self.success:
                self.failed_badly = False
            
            num_tries += 1
        
        tocBH = time.perf_counter()
        self.log_event(self.log_file, f"Time taken for Bob H2 Stabilization: {tocBH - ticBH} seconds")
        # time.sleep(5)

        self.log_event(self.log_file, f"{self.name} Completed Successfully")

class PolarizationStabilization:
    def __init__(self, pso_params, duration, tracking_interval=30, visTol=0.015):

        self.step1 = Bob_H1_Stabilization("Step 1", pso_params)
        self.step2 = Bob_D2_Stabilization("Step 2", pso_params)
        self.step3 = Alice_H1D2_Stabilization("Step 3", pso_params)
        self.step4 = Bob_H2_Stabilization("Step 4", pso_params)

        self.pso_params = pso_params
        self.duration = duration
        self.log_file = self.pso_params.log_file
        self.log_event = self.pso_params.log_event

        self.tracking_check_interval = tracking_interval
        self.visTol = visTol

        self.SLEEP_TIME_H = 1*60
        self.SLEEP_TIME_D = 5*60

        self.tracking_data = -np.ones([7, int(2 * duration/tracking_interval)])
        self.cleaned_tracking_data = None
    
    def initial_stabilization(self):
        # NOTE Step.run() does not check whether self.failed_badly is True or False when self.success = False,
        # only Step.check() does this
        self.log_event(self.log_file, "Running Initial Stabilization...")
        self.step1.run()
        self.step2.run()
        self.step3.run(self.step1, self.step2)
        self.step4.run()
        self.log_event(self.log_file, "Initial Stabilization Completed")

    def stabilization_tracking(self, init=True):
        ticPS = time.perf_counter()
        if init:
            self.initial_stabilization()
        self.log_event(self.log_file, "Running Stabilization Tracking...")
        tocPS = time.perf_counter()
        time_elapsed = tocPS - ticPS
        tic_H_tracking = time.perf_counter()
        tic_D_tracking = time.perf_counter()
        tracking_iter = 0
        
        while time_elapsed < self.duration:
            self.log_event(self.log_file, f"Stabilization Tracking Loop Iteration {tracking_iter}")
            # Wait for a specificed period of time
            time.sleep(self.tracking_check_interval)

            # Check H stability
            if (time.perf_counter() - tic_H_tracking) >= self.SLEEP_TIME_H: # Add this as a separate method
                self.log_event(self.log_file, "Tracking H polarization stabilization...")

                self.step1.check() # Bob H1
                while not self.step1.success:
                    self.log_event(self.log_file, "Step 1: Failed, rerunning Step 1...")
                    self.step1.run()
                self.log_event(self.log_file, "Step 1: Valid")
            
                self.step4.check() # Bob H2
                while not self.step4.success:
                    self.log_event(self.log_file, "Step 4: Failed, rerunning Step 4...")
                    self.step4.run()
                self.log_event(self.log_file, "Step 4: Valid")

                self.step3.check_H1H2() # Alice H1 and H2
                while not self.step3.H1_success or not self.step3.H2_success:
                    self.log_event(self.log_file, "Step 3: Failed, rerunning Step 3...")
                    # self.step1.check()
                    # self.step2.check()
                    self.step3.run(self.step1, self.step2)
                self.log_event(self.log_file, "Step 3: Valid")
                
                # Recheck if all are still valid
                self.log_event(self.log_file, "Rechecking H polarization steps...")
                # self.step1.check()
                # self.step4.check()
                # self.step3.check()
                if (not self.step1.success or not self.step3.H1_success
                    or not self.step3.H2_success or not self.step4.success):
                    self.log_event(self.log_file, "One or more steps failed after checks, rerunning initial stabilization")
                    self.initial_stabilization()
                else:
                    self.log_event(self.log_file, "All visibilities are below their respective thresholds"
                                    + "\nStabilization is good for now!")
                
                tic_H_tracking = time.perf_counter()
            
            # Check D stability
            if (time.perf_counter() - tic_D_tracking)>= self.SLEEP_TIME_D:
                self.log_event(self.log_file, "Tracking D polarization stabilization...")

                self.step1.check() # Bob H1
                while not self.step1.success:
                    self.log_event(self.log_file, "Step 1: Failed, rerunning Step 1...")
                    self.step1.run()
                self.log_event(self.log_file, "Step 1: Valid")

                self.step2.check() # Bob D2
                while not self.step2.success:
                    self.log_event(self.log_file, "Step 2: Failed, rerunning Step 2...")
                    self.step2.run()
                self.log_event(self.log_file, "Step 2: Valid")

                self.step3.check_H1D2() # Alice H1 and D2
                while not self.step3.success:
                    self.log_event(self.log_file, "Step 3: Failed, rerunning Step 3...")
                    # self.step1.check()
                    # self.step2.check()
                    self.step3.run(self.step1, self.step2)
                self.log_event(self.log_file, "Step 3: Valid")

                self.step4.check() # Bob H2
                while not self.step4.success:
                    self.log_event(self.log_file, "Step 4: Failed, rerunning Step 4...")
                    self.step4.run()
                self.log_event(self.log_file, "Step 4: Valid")
                
                # Recheck if all are still valid
                self.log_event(self.log_file, "Rechecking D polarization steps...")
                # self.step1.check()
                # self.step2.check()
                # self.step4.check()
                # self.step3.check()
                if (not self.step1.success or not self.step3.H1_success
                    or not self.step3.D2_success or not self.step4.success
                    or not self.step2.success):
                    self.log_event(self.log_file, "One or more steps failed after checks, rerunning initial stabilization")
                    self.initial_stabilization()
                else:
                    self.log_event(self.log_file, "All visibilities are below their respective thresholds"
                                    + "\nStabilization is good for now!")
                    
                tic_D_tracking = time.perf_counter()
                
            # time.time(), Alice H1 visibility,

            # Update time elapsed
            tocPS = time.perf_counter()
            time_elapsed = tocPS - ticPS

            self.log_event(self.log_file, f"Current time elapsed: {time_elapsed} seconds\n")
            self.log_event(self.log_file, f"Listing visibilities:")
            self.log_event(self.log_file,
                           f"Time: {time_elapsed:<20}\nBob   H1 {self.step1.visibility:<20}\nBob   D2 {self.step2.visibility:<20}\nBob   H2 {self.step4.visibility:<20}\nAlice H1 {self.step3.H1_visibility:<20}\nAlice D2 {self.step3.D2_visibility:<20}\nAlice H2 {self.step3.H2_visibility:<20}\n")
            
            self.tracking_data[:, tracking_iter] = np.array(
                [time_elapsed,
                 self.step1.visibility,
                 self.step2.visibility,
                 self.step4.visibility,
                 self.step3.H1_visibility,
                 self.step3.D2_visibility,
                 self.step3.H2_visibility]
            )
            tracking_iter += 1

        self.log_event(self.log_file, "Stabilization Tracking Completed\n")
        # delta_t = tocPS - ticPS
        # delta_t_hr = delta_t // 3600
        # delta_t_min = (delta_t % 3600) // 60
        # delta_t_sec = delta_t % 60
        # self.log_event(self.log_file, f"Total time elapsed for full stabilization tracking loop is {delta_t_hr} hours, {delta_t_min} minutes, and {delta_t_sec} seconds")

    def save_tracking_data(self, filename):
        # Find columns where all values are -1
        mask = ~(self.tracking_data == -1).all(axis=0)

        # Keep only the columns that are not completely -1
        self.cleaned_tracking_data = self.tracking_data[:, mask]

        if self.cleaned_tracking_data.size > 0:
            self.log_event(self.log_file, f"{self.cleaned_tracking_data}")
            np.savez(filename, raw_tracking_data=self.tracking_data, cleaned_tracking_data=self.cleaned_tracking_data)
    
    def load_tracking_data(self, filename):
        loaded_data = np.load(filename)
        self.tracking_data = loaded_data["raw_tracking_data"]
        self.cleaned_tracking_data = loaded_data["cleaned_tracking_data"]

if __name__ == "__main__":
    # TODO Add conditions checking Bob visibilities during steps 1 and 2 and Alice checking that Bob's visibilities are reasonable

    num_runs = 1 # it can take around 10-25 minutes to run
    duration = 1 * 60 # amount of time to run the stabilization algorithm for

    # Define Alice and Bob visibilities dictionaries to save to .pkl files later
    Alice_visibilities_dict = {}
    Bob_visibilities_dict = {}
    # Define np.array of duration per run
    timing = np.zeros((num_runs))

    # Create log file with unique name
    current_date = datetime.now().strftime("%Y-%m-%d")
    path = "C:/Users/QUANT-NET Admin/Documents/Python codes/QUANTNET/INQNET_DLPolarizationStabilization_Main/"
    runner_idx = 1
    os.makedirs(path + "PSO Data", exist_ok=True)
    while os.path.exists(path + f"PSO Data/{current_date}_Polarization_Stabilization_PSO_log_{runner_idx}.txt"):
        runner_idx += 1
    log_filename = path + f"PSO Data/{current_date}_Polarization_Stabilization_PSO_log_{runner_idx}.txt"

    # Important device resources
    # Rigol_resource = "USB0::0x1AB1::0x0641::DG4D152500730::INSTR"
    Rigol_resource = "USB0::0x1AB1::0x0641::DG4E251600982::INSTR"

    log_file = open(log_filename, 'w')
    message = f"Particle Swarm Optimization Polarization Stabilization Test {runner_idx}, {current_date} step-by-step log" # message should not be larger than 116 characters
    # notes = f"(Bouncing from the bounds with particle velocity damping = {DAMPING})"
    # notes = "(Clipping at the boundaries)"
    notes = "(Clipping at the boundaries and resetting a particle's velocity and voltage to a random value"
    notes2 = "if it hits the boundary)"
    notes3 = "Using LBNL laser and time tagger"
    log_event(log_file, "[" + "=" * 118 + "]")
    format_title(log_file, message)
    format_title(log_file, notes)
    format_title(log_file, notes2)
    format_title(log_file, notes3)
    log_event(log_file, "[" + "=" * 118 + "]\n")

    # Initialize devices
    log_event(log_file, "##### Initializing Alice's devices #####")
    Alice = GenericUser(name= "Alice", laser_port="COM19", PSG_port="COM12", PolCTRL_port1="COM20") # LBNL
    #Alice = GenericUser(name= "Alice", laser_port="COM19", PSG_port="COM9", PolCTRL_port1="COM20") # LBNL
    Alice.laser.connect_laser()
    # Alice.laser.turn_on()
    Alice.PSG.connect()
    Alice.Pol_CTRL1.connect()
    log_event(log_file, "##### Alice's devices are initialized #####")
    
    log_event(log_file, "##### Initializing Bob's devices ##### \n")
    # Bob = GenericUser(name= "Bob", laser_port="COM10", PSG_port="COM9", PolCTRL_port1="COM12") # COM4 is LUNA, COM21 is LADAQ+INQNET PolCTRL
    Bob = GenericUser(name= "Bob", laser_port="COM10", PSG_port="COM9") # LBNL
    #Bob = GenericUser(name= "Bob", laser_port="COM10", PSG_port="COM12") # LBNL
    Bob.laser.connect_laser()
    Bob.PSG.connect()
    # Bob.laser.turn_on()
    log_event(log_file, "##### Bob's devices are initialized #####\n")

    log_event(log_file, "##### Initializing Charlie's devices #####\n")
    Charlie = GenericUser(name= "Charlie", PolCTRL_port1="COM22", PolCTRL_port2="COM21",
                          NIDAQ_USB_id="Dev1", time_tagger_filename="TimeTaggerConfig.yaml",
                          Rigol_resource=Rigol_resource) # LBNL
    # Rigol channel params
    Charlie.Rigol.ch1_params = Charlie.Rigol.set_channel_params(channel="1", burst_delay=47E-6,burst_cycles=7000, trigger_source="EXT")
    Charlie.Rigol.ch2_params = Charlie.Rigol.set_channel_params(channel="2", burst_delay=0 ,   burst_cycles=7000, trigger_source="EXT")
    # Charlie = GenericUser(name= "Charlie", PolCTRL_port1=None, PolCTRL_port2=None, NIDAQ_USB_id="Dev1")#, OSW_port="COM9")
    # Charlie = GenericUser(name= "Charlie", PolCTRL_port1="COM11", PolCTRL_port2="COM8", NIDAQ_USB_id="Dev1")#, OSW_port="COM9")
    # COM 8 V_2π = 0.887 V, COM 11 V_2π = 0.94 V, COM 12 V_2π = 0.752
    Charlie.Pol_CTRL1.connect()
    Charlie.Pol_CTRL2.connect()
    Charlie.Rigol.connect()
    Charlie.meas_device = Charlie.TimeTaggerManager
    # Charlie.OSW.connect()
    # Charlie.NIDAQ_USB.setup_ai_task()
    log_event(log_file, "##### Charlie's devices are initialized #####\n")

    # Define important values
    dim = 3

    pso_params = PSOParams(Alice, Bob, Charlie,
                           Measure_CostFunction, log_event, log_file, Charlie.meas_device)
    
    ### Default OSW connetion
    Charlie.Rigol.Alice_Switch = 0
    Charlie.Rigol.Bob_Switch = 1

    # Charlie.OSW.StatusSet(Charlie.OSW.Alice_Switch, 1)
    # Charlie.OSW.StatusSet(Charlie.OSW.Bob_Switch, 0)
    OSW_operate(OSW=Charlie.Rigol, Alice_Switch_status=1, Bob_Switch_status=0,
                meas_device=Charlie.meas_device,
                channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                Alice=Alice, Bob=Bob)
    raw_Alice_power = MeasureFunction(Charlie.meas_device, [1,2,3,4])
    print(f"Alice raw power: {raw_Alice_power}")
    Charlie.meas_device.Alice_power = np.sum(raw_Alice_power)
    print(f"Alice power: {Charlie.meas_device.Alice_power}")

    # Charlie.OSW.StatusSet(Charlie.OSW.Alice_Switch, 0)
    # Charlie.OSW.StatusSet(Charlie.OSW.Bob_Switch, 1)
    _, Bob_power = OSW_operate(OSW=Charlie.Rigol, Alice_Switch_status=0, Bob_Switch_status=1, meas_device=Charlie.meas_device, channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                Alice=Alice, Bob=Bob)
    raw_Bob_power = MeasureFunction(Charlie.meas_device, [1,2,3,4])
    print(f"Bob power after switch: {Bob_power}")
    print(f"Bob raw power: {raw_Bob_power}")
    Charlie.meas_device.Bob_power = np.sum(raw_Bob_power)
    print(f"Bob power: {Charlie.meas_device.Bob_power}")

    while True:
        OSW_operate(OSW=Charlie.Rigol, Alice_Switch_status=0, Bob_Switch_status=0, meas_device=Charlie.meas_device, channels=[1, 2, 3, 4], log_file=log_file, initCheck = True,
                    Alice=Alice, Bob=Bob)
        raw_zero_power = MeasureFunction(Charlie.meas_device, [1,2,3,4])
        print(f"Raw zero power: {raw_zero_power}")
        Charlie.meas_device.zero_power = raw_zero_power
        log_event(log_file, f"measure zero power: {Charlie.meas_device.zero_power}")
        break # Remove after adding condition below
        # if np.mean(Charlie.meas_device.zero_power)<0.01:
        #     break

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

            # Charlie.OSW.StatusSet(Charlie.OSW.Bob_Switch, 0)
            toc_stablization = time.perf_counter()
            delta_t = toc_stablization - tic_stablization
            delta_t_hr = int(delta_t / 3600)
            delta_t_min = int((delta_t % 3600) / 60)
            delta_t_sec = delta_t % 60
            log_event(log_file, f"Total time elapsed for Run # {i+1}/{num_runs} is {delta_t_hr} hours, {delta_t_min} minutes and {delta_t_sec} seconds")
            timing[i] = delta_t
            
            # tracking_data_all_runs[f"[Raw Tracking Data] Run #{num_runs:<3}"] = ps.tracking_data

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
        log_event(log_file, f"\nTotal time elapsed for all {num_runs} Run{run_plural} is {delta_t_total_hr} hours, {delta_t_total_min} minutes and {delta_t_total_sec} seconds")

        log_event(log_file, "end")
        # Alice.laser.turn_off()
        Alice.disconnect_all()
        log_event(log_file, "Alice's laser is off")
        # Bob.laser.turn_off()
        Bob.disconnect_all()
        log_event(log_file, "Bob's laser is off")
        Charlie.disconnect_all()
        Charlie.OSW.disconnect()
        log_event(log_file, "")
        log_file.close()

        ps.save_tracking_data(path + f"PSO Data/{current_date}_Polarization_Stabilization_TrackingData_{runner_idx}.npz") # Only saves last run

        with open(path + f"PSO Data/{current_date}_Polarization_Stabilization_AliceVisibilities_{runner_idx}.pkl", 'wb') as f_a:
            pickle.dump(Alice_visibilities_dict, f_a)
        with open(path + f"PSO Data/{current_date}_Polarization_Stabilization_BobVisibilities_{runner_idx}.pkl", 'wb') as f_b:
            pickle.dump(Bob_visibilities_dict, f_b)