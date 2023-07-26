from segment_anything.build_sam import build_sam_vit_h
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
from PIL import Image
import requests

# load model and load weights
sam = build_sam_vit_h(checkpoint="sam_vit_h_4b8939.pth")
sam_generator = SamAutomaticMaskGenerator(sam) 


# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)


sam_generator.set_image(image)

sam_generator.predict(image)