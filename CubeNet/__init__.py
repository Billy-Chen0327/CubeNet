import os
import torch
import CubeNet.net
import CubeNet.Picker
import CubeNet.config

network = CubeNet.net.UNet(); network.load_state_dict(torch.load(os.path.join(os.path.split(CubeNet.__file__)[0],'Para.pt')));
picker = CubeNet.Picker.IrrPicker(arr_info=CubeNet.config.arr_info,
                                  para_path = os.path.join(os.path.split(CubeNet.__file__)[0],'Para.pt'),
                                  net = network,
                                  predict_batch = CubeNet.config.basic_info['batch_size'],
                                  device = CubeNet.config.basic_info['device'])
