from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch

try:
    extractor = AutoFeatureExtractor.from_pretrained("google/yamnet")
    model = AutoModelForAudioClassification.from_pretrained("google/yamnet")
    print("Success")
except Exception as e:
    print(e)
