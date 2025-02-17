# Polarization-stabilization-dual-link
Developed by Iraj, Micheal and Claire.
For assistance: pumesh@caltech.edu, mbregar@caltech.edu


GenericUser_V1.py: Creates Generic Users (Alice, Bob or Charlie) with different devices like Polarization Controller, PSG, Laser and NIDAQ USB.

NIDAQ_USBv4.py: Communicates to NI-USB 6003 data acquisition python code

OSWManager.py: Communicates to drive Agiltron optical switches. OSWManager >> Teensy >> FETs >> Agiltron Switches. 

PPCL_Bare_Bones.py: Communicates to PPCL590/PPCL690 lasers. Additional files needed: Support/LaserSupport/ITLA_v3 and PPCL550v4 (or PPCL550v7)

PSGManager.py: Communicates to Luna's Polarization state generator. This file should be in BSM node (or central node, Charlie).

PSGManagerBob.py: Communicates to Luna's Polarization state generator at a quantum node (or Bob).

PSOManager.py: Particle Swarm Optimization algorithm to optimize outputs based on input and output. 

PolCTRLManager.py: Communicates to Polarization controller. PolCTRLManager >> Teensy >> DAC >> Amplifiers >> Luna polarization controller  

PSO_running_v18.py: Python file for Dual link polarization stabilization. Utilizes OSW.

PSO_running_v18c.py: Python file for Dual link polarization stabilization. Instead of OSW lasers at Alice and Bob are turned on and off.






 


