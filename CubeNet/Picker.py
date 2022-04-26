import torch
import numpy as np
from scipy import signal

class IrrPicker:
    def __init__(self,arr_info,para_path,net,predict_batch,device):
        self.batch = predict_batch;
        self.device = device;
        self.net = net;
        self.net.load_state_dict(torch.load(para_path));
        self.net.to(device); self.net.eval();
        self.Cube_Reged = None;
        self.Hdist_runningdata = None;
        self.arr_info = arr_info;
        
    def standardization(self,data):
        if max(abs(data)) == min(abs(data)):
            return np.zeros(data.shape);
        else:
            return (data-np.mean(data))/np.std(data);
        
    def vector_norm(self,input_vec):
        _range = np.max(input_vec) - np.min(input_vec);
        if _range != 0:
            return (input_vec - np.min(input_vec)) / _range;
        if _range == 0:
            return np.zeros(input_vec.shape);
        
    def Cal_cubegrid(self,rcv):
    
        rcv_new = rcv.copy();
        rcv_new[:,0] = self.vector_norm(rcv_new[:,0]); rcv_new[:,1] = self.vector_norm(rcv_new[:,1]);
        
        cube_vec = np.linspace(0,1,8);
        cube_grid = np.zeros([8,8],dtype=int);
        
        for i in range(len(rcv)):
            x = np.argmin(abs(cube_vec-rcv_new[i,0]));
            y = np.argmin(abs(cube_vec-rcv_new[i,1]));
            cube_grid[x,y] += 1;
        
        return cube_grid;
    
    def Cal_BestRcv(self,rcv,deg=None):
        rcv_new = rcv.copy();
        center_x = np.average(rcv[:,0]); center_y = np.average(rcv[:,1]);
        rcv_new[:,0] = rcv_new[:,0] - center_x;
        rcv_new[:,1] = rcv_new[:,1] - center_y;
        
        if deg == None:
            non_zero_num = 0;
            for i in np.arange(0,90.1,5):
                rcv_rotate = np.zeros(rcv_new.shape);
                for k in range(2,rcv_new.shape[-1]):
                    rcv_rotate[:,k] = rcv_new[:,k];
                rcv_rotate[:,0] = np.cos(i/180*np.pi)*rcv_new[:,0] - np.sin(i/180*np.pi)*rcv_new[:,1];
                rcv_rotate[:,1] = np.cos(i/180*np.pi)*rcv_new[:,1] + np.sin(i/180*np.pi)*rcv_new[:,0];
                cube_grid = self.Cal_cubegrid(rcv_rotate); 
                if non_zero_num < len(np.nonzero(cube_grid)[0]):
                    non_zero_num = len(np.nonzero(cube_grid)[0]);
                    deg = i;
            if non_zero_num == 0:
                deg = 0;
        i = deg;
        rcv_rotate = np.zeros(rcv_new.shape);
        for k in range(2,rcv_new.shape[-1]):
            rcv_rotate[:,k] = rcv_new[:,k];
        rcv_rotate[:,0] = np.cos(i/180*np.pi)*rcv_new[:,0] - np.sin(i/180*np.pi)*rcv_new[:,1];
        rcv_rotate[:,1] = np.cos(i/180*np.pi)*rcv_new[:,1] + np.sin(i/180*np.pi)*rcv_new[:,0];
        rcv_rotate[:,0] -= np.min(rcv_rotate[:,0]);
        rcv_rotate[:,1] -= np.min(rcv_rotate[:,1]);
        return rcv_rotate;
    
    def get_CubeData(self,Data,Cube_now):
        Cube_Data = np.zeros([len(Cube_now),3,8,8,Data.shape[-1]]);
        for i in range(len(Cube_now)):
            Cube_mat = Cube_now[i];
            for ii in range(Cube_mat.shape[0]):
                for iii in range(Cube_mat.shape[1]):
                    if Cube_mat[ii,iii] != -1:
                        Cube_Data[i,:,ii,iii,:] = Data[Cube_mat[ii,iii],:,:];
        return Cube_Data;
    
    def refresh_result(self,result,result_flag,net_output,Cube_now):
        for i in range(len(Cube_now)):
            Cube_mat = Cube_now[i];
            for ii in range(Cube_mat.shape[0]):
                for iii in range(Cube_mat.shape[1]):
                    if Cube_mat[ii,iii] != -1:
                        tem_max = np.max(net_output[i,0,ii,iii,:]);
                        if tem_max > result_flag[Cube_mat[ii,iii]]:
                            result_flag[Cube_mat[ii,iii]] = tem_max;
                            result[Cube_mat[ii,iii],:,:] = net_output[i,:,ii,iii,:];
        return result,result_flag;
    
    def ResampleData(self,Data,data_fs,New_fs):
        new_seq_len = int(Data.shape[-1]/data_fs*New_fs);
        Data_new = np.zeros([Data.shape[0],Data.shape[1],new_seq_len]);
        for i in range(Data.shape[0]):
            for ii in range(Data.shape[1]):
                Data_new[i,ii,:] = signal.resample(Data[i,ii,:],new_seq_len);
        return Data_new;
    
    def ban_ZeroTrace(self,Data,Result):
        for i in range(Data.shape[0]):
            X_cond = max(abs(Data[i,0,:])) == 0;
            Y_cond = max(abs(Data[i,1,:])) == 0;
            Z_cond = max(abs(Data[i,2,:])) == 0;
            if np.all((X_cond,Y_cond,Z_cond)):
                Result[i,:,:] = np.zeros(Result[i,:,:].shape);
        return Result
    
    def RegCube(self,rcv):
        rcv_new = self.Cal_BestRcv(rcv);
        self.Hdist_runningdata = np.median(( ( rcv_new[:,0]-np.average(rcv_new[:,0]) )**2 + (rcv_new[:,1] - np.average(rcv_new[:,1]))**2 )**.5);
        rcv_new[:,0] = self.vector_norm(rcv_new[:,0]);
        rcv_new[:,1] = self.vector_norm(rcv_new[:,1]);
        cube_vec = np.linspace(0,1,8);
        cube_grid = np.zeros([8,8],dtype=int);
        
        for i in range(cube_grid.shape[0]):
            for ii in range(cube_grid.shape[1]):
                locals()['Index_'+str(i)+'_'+str(ii)] = [];
                
        for i in range(len(rcv_new)):
            x = np.argmin(abs(cube_vec-rcv_new[i,0]));
            y = np.argmin(abs(cube_vec-rcv_new[i,1]));
            cube_grid[x,y] += 1;
            locals()['Index_'+str(x)+'_'+str(y)].append(i);
        judge_grid = np.array(cube_grid,dtype=bool);
            
        whole_judge_mat = [];
        for iter_num in range(np.max(cube_grid)):
            judge_mat = -np.ones([8,8],dtype=int);
            for i in range(cube_grid.shape[0]):
                for ii in range(cube_grid.shape[1]):
                    if len(locals()['Index_'+str(i)+'_'+str(ii)]) == 0:
                        continue;
                    if len(locals()['Index_'+str(i)+'_'+str(ii)]) > iter_num:
                        judge_mat[i,ii] = locals()['Index_'+str(i)+'_'+str(ii)][iter_num];
                    else:
                        judge_mat[i,ii] = locals()['Index_'+str(i)+'_'+str(ii)][0];
            whole_judge_mat.append(judge_mat);
        self.Cube_Reged = whole_judge_mat;
        
    def pick(self,Data,std=True,Resample_Mode=False,data_fs=None,avg_adjust=False):
        if self.Cube_Reged == None:
            Error_info = "Cube has not been regularized";
            raise TypeError(Error_info);
        if std == True:
            for i in range(Data.shape[0]):
                for ii in range(Data.shape[1]):
                    Data[i,ii,:] = self.standardization(Data[i,ii,:]);
        if Resample_Mode == True:
            dataset_info = self.arr_info;
            dist_training = (dataset_info['vir_evt_depth']**2 + dataset_info['median_app']**2)**.5;
            if self.Hdist_runningdata == None:
                Error_info = "Can't find the median travel path, please regularize your cube first";
                raise ValueError(Error_info);
            else:
                if data_fs == None:
                    Error_info = "Please input the sampling rate of your data! ";
                    raise ValueError(Error_info);
                else:                    
                    deltaPS_training = dist_training/dataset_info['vir_vs'] - dist_training/dataset_info['vir_vp'];
                    delta_pts_training = (deltaPS_training*dataset_info['model_vaild_fs'][0],deltaPS_training*dataset_info['model_vaild_fs'][1]);
                    
                    dist_running = (self.Hdist_runningdata**2 + dataset_info['vir_evt_depth']**2)**.5;
                    deltaPS_running = dist_running/dataset_info['vir_vs'] - dist_running/dataset_info['vir_vp'];
                    delta_pts_running = deltaPS_running*data_fs;
                    
                    if delta_pts_running >= delta_pts_training[0] and delta_pts_running <= delta_pts_training[1]:
                        New_fs = data_fs;
                    else:
                        if avg_adjust == False:
                            if delta_pts_running < delta_pts_training[0]:
                                New_fs = data_fs*delta_pts_training[0]/delta_pts_running;
                                New_fs = round(New_fs/10)*10;
                            else:
                                New_fs = data_fs*delta_pts_training[1]/delta_pts_running;
                                New_fs = round(New_fs/10)*10;
                        else:
                            New_fs = data_fs*np.average(delta_pts_training)/delta_pts_running;
                            New_fs = round(New_fs/10)*10;
            if New_fs != data_fs:
                print('Sampling rate has been adjusted to '+ str(int(New_fs)) +' Hz!');
                Data = self.ResampleData(Data,data_fs,New_fs);
            else:
                print('Sampling rate is suitable! No need for change!');
        else:
            New_fs = data_fs;
        
        result = np.zeros(Data.shape); result_flag = np.zeros(Data.shape[0]);
        for i in range(0,len(self.Cube_Reged),self.batch):
            Cube_now = self.Cube_Reged[i:min(i+self.batch,len(self.Cube_Reged))];
            Cube_data = self.get_CubeData(Data,Cube_now);
            Cube_data = torch.Tensor(Cube_data);
            Cube_data = Cube_data.to(self.device);
            net_output = self.net(Cube_data);
            net_output = net_output.cpu().detach().numpy();
            result,result_flag = self.refresh_result(result,result_flag,net_output,Cube_now);
        result = self.ban_ZeroTrace(Data,result);
        
        return Data,result,New_fs;
