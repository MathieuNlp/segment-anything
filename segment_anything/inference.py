from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder

model = ImageEncoderViT()

for name, param in model.named_parameters():
    print(name, param.shape)


