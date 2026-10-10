import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load energy and wave speed data for a=0.1
energy_data = np.load('/Users/coalminer/Desktop/流体模拟/HW1/wave_speed/a0.1_vortex_right.npz')
wave_data = np.load('/Users/coalminer/Desktop/流体模拟/HW1/wave_speed/a0.1_wavepacket.npz')

# Check what keys are available
print("Energy data keys:", energy_data.files)
print("Wave data keys:", wave_data.files)
