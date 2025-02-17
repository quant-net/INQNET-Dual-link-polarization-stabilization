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

# TODO Create a laser class
class PPCL550:
    """PPCL550 Laser Control Class"""
    def __init__(self, com_port, baud_rate, min_freq=191.50, max_freq=196.25, min_pow=600, max_pow=1000) -> None:
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.freq = 0#freq
        self.power = 0#power
        self.sercon = 0 # serial laser connection
        self.MIN_FREQ = min_freq#191.50#MIN_FREQ
        self.MAX_FREQ = max_freq#196.25#MAX_FREQ
        self.MIN_POW = min_pow#600#MIN_POW
        self.MAX_POW = max_pow#1000#MAX_POW
        self.LASER_ON = [0x81,0x32,0x00,0x08]
        self.LASER_OFF = [0x01,0x32,0x00,0x00]
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
        if self.com_port is None:
            self.sercon= None
        else:
            # NOP_register(self)
            # if 
            self.sercon = ITLA.ITLAConnect(self.com_port,baudrate=self.baud_rate)
        return self.sercon
    
    def NOP_register(self):
        return ITLA.ITLA(self.sercon, 0, 0, 0)
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

if __name__ == "__main__":
    # sleep_time = 0.001
    freq1 = 196.00
    freq2 = 191.560
    power1 = 700
    # laser1 = PPCL550('COM4', 9600)
    laser1 = PPCL550('COM3', 9600)
    print(f"Connecting laser...")
    print(laser1.connect_laser())
    # time.sleep(sleep_time)
    print(f"NOP Register:")
    print(laser1.NOP_register())
    # time.sleep(sleep_time)
    # print(laser1.write_freq(193.560))
    print(f"Setting Frequency to {freq1}...")
    print(laser1.write_freq(freq1))
    # time.sleep(sleep_time)
    print(f"Setting Power to {power1}...")
    print(laser1.write_power(power1))
    # time.sleep(sleep_time)
    print(f"Turning laser ON...")
    print(laser1.laser_on())
    time.sleep(30)
    # time.sleep(sleep_time)
    print(f"Reading Frequency and Power:")
    print(laser1.read_freq())
    # time.sleep(sleep_time)
    print(laser1.read_power())
    # time.sleep(sleep_time)
    print(laser1.laser_off())

    print(f"Setting Frequency to {freq2}...")
    print(laser1.write_freq(freq2))
    # time.sleep(sleep_time)
    print(f"Setting Power to {power1}...")
    print(laser1.write_power(power1))
    # time.sleep(sleep_time)
    print(laser1.laser_on())
    time.sleep(30)
    # time.sleep(sleep_time)
    print(f"Reading Frequency and Power:")
    print(laser1.read_freq())
    # time.sleep(sleep_time)
    print(laser1.read_power())
    # time.sleep(sleep_time)
    print(f"Turning laser OFF...")
    print(laser1.laser_off())
    print(f"Disconnecting laser...")
    laser1.disconnect_laser()