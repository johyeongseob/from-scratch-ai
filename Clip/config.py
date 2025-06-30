import torch

debug = False
image_path = "./flickr30k_images"
captions_path = "./captions.csv"

batch_size = 32
num_workers = 4
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 1
factor = 0.8
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 40  # covers 99.5% of Flickr30k captions, max=88

pretrained = True
trainable = True
temperature = 1.0

size = 224

num_projection_layers = 1
projection_dim = 256
dropout = 0.1
