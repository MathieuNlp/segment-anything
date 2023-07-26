import build_sam

sam = build_sam.build_sam_vit_h(checkpoint="./sam_vit_h_4b8939.pth")

for name, param in sam.named_parameters():
    print(name, param.shape)