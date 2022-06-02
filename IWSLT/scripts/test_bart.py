import torch
from fairseq.models.bart import BARTModel


bart = BARTModel.from_pretrained("/data/wangguitao/IWSLT/Pre-trained_models/BART-large",
                                 checkpoint_file='model.pt')
bart.eval()
tokens = bart.encode('Hello world!')
last_layer_features = bart.extract_features(tokens)
print(last_layer_features.shape)