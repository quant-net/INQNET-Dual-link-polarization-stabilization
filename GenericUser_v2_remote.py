import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from math import inf
from ctypes import *
import pickle

# Ensure a suitable Matplotlib backend
matplotlib.use('TkAgg')  # Adjust backend as needed

# Import necessary modules (assumes these are available in your environment)
from PSGManager import PSGManager
from PSGManagerBob import PSGManagerBob
from PPCL_Bare_Bones import LaserControl
from PolCTRLManager import PolCTRLManager
from NIDAQ_USBv4 import NIDAQ_USB
from OSWManager import OSWManager
from TimeTaggerFunctions import TimeTaggerManager
from RigolManager import RigolDG4162Manager

class GenericUser:
    def __init__(self, name = None, laser_port=None, wavelength=1550, power=6,
                 PSG_port=None, PolCTRL_port1=None,
                 PolCTRL_port2=None, NIDAQ_USB_id = None, OSW_port = None, time_tagger_filename=None,
                 Rigol_resources=(None,None)):
        self.name = name
        self.laser = LaserControl(laser_port, wavelength, power)
        if self.name == "Bob":
            self.PSG = PSGManagerBob()
            # self.PSG = PSGManager(PSG_port)
        else:
            self.PSG = PSGManager(PSG_port)

        self.Pol_CTRL1 = PolCTRLManager(PolCTRL_port1)
        self.Pol_CTRL2 = PolCTRLManager(PolCTRL_port2)
        self.NIDAQ_USB = NIDAQ_USB(devID=NIDAQ_USB_id)
        self.OSW = OSWManager(OSW_port)
        self.TimeTaggerManager = TimeTaggerManager(time_tagger_filename)

        self.Rigols = ( RigolDG4162Manager(Rigol_resources[0]), RigolDG4162Manager(Rigol_resources[1]) ) 
        self.Rigol1, self.Rigol2=self.Rigols
        self.device_list = [
            self.laser,
            self.PSG,
            self.Rigol1,
            self.Rigol2
        ] # please do not touch this list, it will be useful for adding other devices

    def add_device(self, device, name):
        setattr(self, name, device)
        # Add to self.device_list if the device connection can be disconnected
        if hasattr(device, "disconnect") and callable(getattr(device, "disconnect")):
            self.device_list.append(device)

    def disconnect_all(self):
        if self.laser.dev != None:
            self.laser.turn_off()
            self.laser.disconnect()
        for device in self.device_list:
            if device.dev != None:
                device.disconnect()
        # self.laser.turn_off()
        # self.laser.disconnect()
        # self.PSG.disconnect()
        # self.Pol_CTRL1.disconnect()
        # self.Pol_CTRL2.disconnect()