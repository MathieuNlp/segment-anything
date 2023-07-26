from modeling.image_encoder import ImageEncoderViT


model = ImageEncoderViT()

for name, param in model.named_parameters():
    print(name, param.shape)