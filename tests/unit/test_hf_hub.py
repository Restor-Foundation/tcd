import datasets
import torch
import transformers


def test_load_models():
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


def test_download_dataset():
    ds = datasets.load_dataset("restor/tcd")

    assert len(ds) > 0
