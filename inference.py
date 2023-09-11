import torch
import numpy as np
import utils as utils
from segment_anything import build_sam_vit_b, SamPredictor, sam_model_registry
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

dataset_path = "./bottle_glass_dataset"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"

image_path = "./bottle_glass_dataset/images/glass2.jpg"
mask_path = "./bottle_glass_dataset/masks/glass2.tiff"

def inference_standard_sam(image_path, mask_path):

    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    ground_truth_mask =  np.array(mask)
    box = utils.get_bounding_box(ground_truth_mask)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(image))

    masks, _, _ = predictor.predict(
        box=np.array(box),
        multimask_output=False,
    )
    
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline ="red")
    image.save("image+bbox.jpg")

    plt.imshow(masks[0])


inference_standard_sam(image_path, mask_path)

    

