from segment_anything.build_sam import build_sam_vit_h
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
import requests
import matplotlib.pyplot as plt
import cv2
import urllib.request
import numpy as np
import torch

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# load model and load weights
sam = build_sam_vit_h(checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda")
sam_generator = SamAutomaticMaskGenerator(sam) 

# load image
url = 'https://upload.wikimedia.org/wikipedia/commons/6/6a/PS3-Fat-Console-Set.jpg'
url_response = urllib.request.urlopen(url)
image = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = sam_generator.generate(image)

# plots masks

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig("./test_images/results/glass_water.png")