from build_sam import sam_model_registry, build_sam_vit_b

sam = build_sam_vit_b(checkpoint="build_sam_vit_b")

for name, param in sam.named_parameters():
    print(name, param.shape)