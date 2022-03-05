import sys; sys.path.append('../../'); # To import CubeNet
import CubeNet

import numpy as np
import matplotlib.pyplot as plt

data_dict = np.load('data.npz');
data = data_dict['waveform'];
print('Shape of data (TraceID * Components * Sampling pts) :',data.shape);

# Load Prepared station file
sta = data_dict['sta']; # ATTENTION: unit is km
print('Shape of station array (TraceID * X/Y) :',sta.shape);

# Using CubeNet to predict seismic phases
picker = CubeNet.picker;
picker.RegCube(sta)
_,pick_result,fs = picker.pick(data);