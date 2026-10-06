import tensorflow_hub as hub
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
print("Loading YAMNet model from TF Hub...")
model = hub.load('https://tfhub.dev/google/yamnet/1')
print("Model loaded.")
