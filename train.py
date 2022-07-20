import os
import torch
import numpy as np

def Data_Gen(data_path,batch_size):
    
    Training_Data_List = os.listdir(data_path);
    num_examples = len(Training_Data_List);
    indices = list(range(num_examples));
    np.random.shuffle(indices);
    
    for i in range(0,num_examples,batch_size):
        
        chosen_index = indices[i:min(i+batch_size,num_examples)];
        feature_data = []; label_data = []; label_mat = [];
        for ii in chosen_index:
            file = np.load(os.path.join(data_path,Training_Data_List[ii]));
            Data = file['waveform'];
            whole_label = file['label'];
            label_mat.append(file['label_mat'])
            feature_data.append(Data);
            label_data.append(whole_label);
        feature_data = np.array(feature_data);
        label_data = np.array(label_data);
        feature_data = torch.Tensor(feature_data);
        label_data = torch.Tensor(label_data);
        yield feature_data,label_data,label_mat;

def net_init(net):
    for name,param in net.named_parameters():
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param);
        if 'bias' in name:
            torch.nn.init.constant_(param,val=0);

class train_op():
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
        
    def adjust_binary_cross_entropy(self,y_hat,y,ref_mat):
        loss_Val = torch.zeros(1,device=self.device);
        counter = 0
        for j in range(len(ref_mat)):
            for i in range(ref_mat[j].shape[0]):
                for ii in range(ref_mat[j].shape[1]):
                    for iii in range(ref_mat[j].shape[2]):
                        if ref_mat[j][i,ii,iii] == True:
                            loss_Val += torch.nn.functional.binary_cross_entropy(y_hat[j,i,ii,iii,:],y[j,i,ii,iii,:]);
                            counter += 1;
        loss_Val = loss_Val/counter;
        return loss_Val;
        
    def train_model(self,train_iter,net,epoch_now,optimizer,loss_path,SaveModelPath):
        device = self.device; net = net.to(device);
        counter = 0; CalLossInterval = 1;
        for X,y,ref_mat in train_iter:
            X = X.to(device); y = y.to(device);
            y_hat = net(X);
            l = self.adjust_binary_cross_entropy(y_hat,y,ref_mat);
            optimizer.zero_grad();
            l.backward();
            optimizer.step();
            counter += 1;
            if counter%CalLossInterval == 0:
                training_loss = l.cpu().item();
                with open(loss_path,'a') as f:
                    f.writelines('Training Loss:'+format(training_loss,'.4f')+'\n');
        torch.save(net.state_dict(),os.path.join(SaveModelPath,'model'+str(int(epoch_now))+'.pt'));

def train(net,train_info):
    
    LossLogPath = os.path.join(train_info['path_saveLossLog'],'loss_log.txt');
    
    net.train();
    net_init(net);
    # net.load_state_dict(torch.load('Para.pt')); # Transfer learning if needed
    optimizer = torch.optim.Adam(net.parameters(),lr=train_info['learning_rate']);
    
    train_tool = train_op();
    for _ in range(train_info['epoch']):
        
        with open(LossLogPath,'a') as f:
            f.writelines('-------------epoch:'+str(_+1)+'-----------\n');
        
        train_iter = Data_Gen(train_info['path_cubes'],train_info['training_batch_size']);
        train_tool.train_model(train_iter,net,epoch_now=_+1,optimizer=optimizer,
                               loss_path=LossLogPath,SaveModelPath=train_info['path_saveModels']);