import datasets
import segmentation_models_pytorch as smp
import torch
import transformers
from huggingface_hub import HfApi


def test_load_segformer_models():
    for model_id in [
        "restor/tcd-segformer-mit-b0",
        "restor/tcd-segformer-mit-b1",
        "restor/tcd-segformer-mit-b2",
        "restor/tcd-segformer-mit-b3",
        "restor/tcd-segformer-mit-b4",
        "restor/tcd-segformer-mit-b5",
    ]:
        model = transformers.AutoModelForSemanticSegmentation.from_pretrained(model_id)
        processor = transformers.AutoImageProcessor.from_pretrained(model_id)

        dummy_input = torch.rand((3, 512, 512))
        inputs = processor(images=dummy_input, do_rescale=False, return_tensors="pt")
        _ = model(pixel_values=inputs.pixel_values)


def test_load_unet_models():
    for s in [34, 50, 101]:
        api = HfApi()
        path = api.hf_hub_download(repo_id=f"restor/tcd-unet-r{s}", filename="model.pt")
        model = smp.Unet(encoder_name="resnet" + str(s), classes=2, in_channels=3)
        model.load_state_dict(torch.load(path), strict=True)
        model.eval()
        dummy_input = torch.rand((1, 3, 512, 512))
        _ = model(dummy_input)


def test_load_maskrcnn_models():
    pass
