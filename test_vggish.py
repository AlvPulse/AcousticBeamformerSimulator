import torch
import numpy as np
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

audio = np.random.randn(16000)
import librosa
try:
    print(model.forward(audio, 16000).shape)
except Exception as e:
    print(e)
