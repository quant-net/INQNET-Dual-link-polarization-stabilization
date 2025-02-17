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
from PPCL_Bare_Bones import LaserControl
from PolCTRLManager import PolCTRLManager
from NIDAQ_USBv4 import NIDAQ_USB
from OSWManager import OSWManager

class GenericUser:
    def __init__(self, name = None, laser_port=None, wavelength=1550, power=6,
                 PSG_port=None,
                 PolCTRL_port1=None,
                 PolCTRL_port2=None, NIDAQ_USB_id = None, OSW_port = None):
        self.name = name
        self.laser = LaserControl(laser_port, wavelength, power)
        self.PSG = PSGManager(PSG_port)
        self.Pol_CTRL1 = PolCTRLManager(PolCTRL_port1)
        self.Pol_CTRL2 = PolCTRLManager(PolCTRL_port2)
        self.NIDAQ_USB = NIDAQ_USB(devID=NIDAQ_USB_id)
        self.OSW = OSWManager(OSW_port)
        self.device_list = [
            self.laser,
            self.PSG,
        ] # please do not touch this list, it will be useful for adding other devices

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