import numpy as np
import torch
import pandas as pd
from model import LSTM, LSTMAttention


df_raw = pd.read_csv("/Users/zhouz/Project/VM_Workload_Predictor/fastStorage/VMmonitoring/882.csv")
df = df_raw.set_index('Timestamp [ms]')
