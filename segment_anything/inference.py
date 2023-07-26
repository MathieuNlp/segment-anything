from build_sam import build_sam_vit_b

sam = build_sam_vit_b(checkpoint="build_sam_vit_b")

for name, param in sam.named_parameters():
    print(name, param.shape)