import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from TCN_Model import TCN
from dataset import load_all_csv, Workload_dataset
import time, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from sklearn.metrics import mean_squared_error
import pickle


def data_split(Time_series_data):

    test_set_size = int(np.round(0.2 * Time_series_data.shape[0]))
    train_set_size = Time_series_data.shape[0] - (test_set_size)

    train_data = Time_series_data[:train_set_size, :-1, :]
    train_label = Time_series_data[:train_set_size, -1, :]

    test_data = Time_series_data[train_set_size:, :-1]
    test_label = Time_series_data[train_set_size:, -1, :]

    ##Transpose data into [Batch, input_channel, seq_len]
    train_data = np.transpose(train_data, (0,2,1))
    test_data = np.transpose(test_data, (0,2,1))

    print("x_train's shape: ", train_data.shape)
    print("y_train's shape: ", train_label.shape)
    print("x_test's shape: ", test_data.shape)
    print("y_test's shape: ", test_label.shape)

    ### Put in torch dataloader
    train_dataset = Workload_dataset(train_data, train_label)
    test_dataset = Workload_dataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

    return train_loader, test_loader


def data_scaler(data):
    scaler_cpu = MinMaxScaler(feature_range=(0, 1))
    cpu_series = data[:, :, 0]
    cpu_series = scaler_cpu.fit_transform(cpu_series.reshape(-1,1))
    cpu_series = cpu_series.reshape(-1, 100)

    scaler_mem = MinMaxScaler(feature_range=(0, 1))
    mem_series = data[:, :, 1]
    mem_series = scaler_mem.fit_transform(mem_series.reshape(-1,1))
    mem_series = mem_series.reshape(-1, 100)

    scaler_disk = MinMaxScaler(feature_range=(0, 1))
    disk_series = data[:, :, 2]
    disk_series = scaler_disk.fit_transform(disk_series.reshape(-1,1))
    disk_series = disk_series.reshape(-1, 100)
    print("disk_series's shape: ", disk_series.shape)



    All_Series = np.stack((cpu_series, mem_series, disk_series), axis=2)

    print("All_Series's shape: ", All_Series.shape)

    return All_Series, scaler_cpu, scaler_mem, scaler_disk


def build_model(num_inputs, output_size, num_channels):
    model = TCN(num_inputs, output_size, num_channels)

    return model

def train(model, num_epochs, train_loader, test_loader, criterion, optimiser, device):

    model = model.to(device)
    
    train_loss_list=[]
    test_loss_list = []
    for i in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0

        for data in train_loader:
            x_train, y_train = data[0].to(device), data[1].to(device)
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train)
            train_loss_epoch += loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        avg_train_loss = train_loss_epoch / len(train_loader)
        train_loss_list.append(avg_train_loss.item())
        
        ##Evaluation
        model.eval()
        eval_loss_epoch = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                x_test, y_test = test_data[0].to(device), test_data[1].to(device)
                y_test_pred = model(x_test)
                loss = criterion(y_test_pred, y_test)
                eval_loss_epoch += loss
        avg_test_loss = eval_loss_epoch / len(test_loader)
        test_loss_list.append(avg_test_loss.item())
        print("Epoch ", i, "Train Loss: ", round(avg_train_loss.item(), 7), "Test Loss: ", round(avg_test_loss.item(), 7))
            
    ## Save Model
    torch.save(model, "./checkpoint/model_lastest.pth")
    return train_loss_list, test_loss_list

def evaluation(model, test_loader, device):
    model = model.eval()
    test_pred = []
    test_label = []
    with torch.no_grad():
        for test_data in test_loader:
            x_test, y_test = test_data[0].to(device), test_data[1].to(device)
            y_test_pred = model(x_test)

            y_test_pred = y_test_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()

            test_pred.append(y_test_pred)
            test_label.append(y_test)

    #print("check test_pred: ", len(test_pred), test_pred[-1].shape)
    #print("check test_label: ", len(test_label), test_label[-1].shape)
    test_label = np.concatenate(test_label, axis=0)
    test_pred = np.concatenate(test_pred, axis=0)
    print("predcition is: ", test_pred.shape, " | ", test_label.shape)
    rme_value = math.sqrt(mean_squared_error(test_label, test_pred))
    print('RME value: %.2f RMSE' % (rme_value))

    return test_pred, test_label


if __name__ == '__main__':


    #root_data_path = r"/Users/zhouz/Project/VMmonitoring"
    if os.path.exists("./data/workload_series.npy") and os.path.exists("./data/scalers_dict.pkl"):
        print("Load Normalized series data from ./data/workload_series.npy")
        print("Load saved scaler from ./data/scalers_dict.pkl")
        All_Series = np.load('./data/workload_series.npy')
        with open('./data/scalers_dict.pkl', 'rb') as f:
            scalers = pickle.load(f)
    else:
        print("Load data from original *.CSV file")
        root_data_path = r"/proj/zhou-cognit/users/x_zhozh/project/faststorage/VMmonitoring"
        All_Series, scalers = load_all_csv(root_data_path, 100)
    print("Check All Series shape: ", All_Series.shape)###
    print("Check All scalers: ", scalers.keys())
    np.random.shuffle(All_Series)


    ##TEST
    #test_array = np.array([0.9, 0.8, 0.6, 0.5])
    #test_array = test_array.reshape(-1, 1)
    #original = scalers["cpu_usage_percent"].inverse_transform(test_array)
    #print("Finished")

    

    train_data, test_data = data_split(All_Series)

    num_inputs=4
    output_size=4
    num_channels=[32, 32, 32]

    num_epochs = 200

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = build_model(num_inputs, output_size, num_channels).to(device)
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_loss_list, test_loss_list = train(model, num_epochs, train_data, test_data, criterion, optimiser, device)


    plt.figure()
    plt.plot(train_loss_list, color='b', label='train loss curve')
    plt.savefig('CPU_train_loss_curve.jpg')

    plt.figure()
    plt.plot(test_loss_list, color='b', label='test loss curve')
    plt.savefig('CPU_test_loss_curve.jpg')

    test_pred, test_label = evaluation(model, test_data, device)
    test_pred = np.clip(test_pred, 0, None)
    test_label = np.clip(test_label, 0, None)

    ##Get random index for easier visualization
    indexs = np.random.randint(1, test_pred.shape[0], size=200).tolist()

    fig, axs = plt.subplots(2, 2)

    '''
    axs[0, 0].plot(scalers["cpu_usage_percent"].transform(test_pred[:,0].reshape(-1,1))[indexs], color='b')
    axs[0, 0].plot(scalers["cpu_usage_percent"].transform(test_label[:,0].reshape(-1,1))[indexs], color='g')
    axs[0, 0].set_title('cpu_usage_percent')

    axs[0, 1].plot(scalers["memory_usage"].transform(test_pred[:,1].reshape(-1,1))[indexs], color='b')
    axs[0, 1].plot(scalers["memory_usage"].transform(test_label[:,1].reshape(-1,1))[indexs], color='g')
    axs[0, 1].set_title('memory_usage')

    axs[1, 0].plot(scalers["disk_write"].transform(test_pred[:,2].reshape(-1,1))[indexs], color='b')
    axs[1, 0].plot(scalers["disk_write"].transform(test_label[:,2].reshape(-1,1))[indexs], color='g')
    axs[1, 0].set_title('disk_write')

    axs[1, 1].plot(scalers["net_transmit"].transform(test_pred[:,3].reshape(-1,1))[indexs], color='b')
    axs[1, 1].plot(scalers["net_transmit"].transform(test_label[:,3].reshape(-1,1))[indexs], color='g')
    axs[1, 1].set_title('net_transmit')
    '''

    axs[0, 0].plot(test_pred[:,0][indexs], color='b')
    axs[0, 0].plot(test_label[:,0][indexs], color='g')
    axs[0, 0].set_title('cpu_usage_percent')

    axs[0, 1].plot(test_pred[:,1][indexs], color='b')
    axs[0, 1].plot(test_label[:,1][indexs], color='g')
    axs[0, 1].set_title('memory_usage')

    axs[1, 0].plot(test_pred[:,2][indexs], color='b')
    axs[1, 0].plot(test_label[:,2][indexs], color='g')
    axs[1, 0].set_title('disk_write')

    axs[1, 1].plot(test_pred[:,3][indexs], color='b')
    axs[1, 1].plot(test_label[:,3][indexs], color='g')
    axs[1, 1].set_title('net_transmit')

    plt.tight_layout()
    plt.savefig('comparision.jpg', dpi=300)



















