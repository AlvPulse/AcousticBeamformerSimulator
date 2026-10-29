import torch
try:
    model = torch.hub.load('nttcslab/msm-mae', 'msm_mae')
    print("Success")
except Exception as e:
    print(e)
