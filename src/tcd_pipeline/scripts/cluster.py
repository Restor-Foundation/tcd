import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import sklearn.utils
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from tcd_pipeline.scripts._instance import Instance, instances_from_geo
from tcd_pipeline.scripts.extract import extract_crops


def generate_clip_embeddings(
    model: torch.nn.Module, processor: Any, instances: list[Instance], device: str
):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []

    with torch.no_grad():
        try:
            for _, instance in tqdm(enumerate(instances), total=len(instances)):
                emb = model(
                    processor(Image.fromarray(instance.raster)).unsqueeze(0).to(device)
                )[0].cpu()
                embeddings.append(emb)
        except KeyboardInterrupt:
            pass

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings


def plot_images_scatter(projections, crops, spread=2, image_size=(64, 64), n=20):
    fig, ax = plt.subplots(figsize=(10, 10))

    idx = list(range(len(projections)))
    random.shuffle(idx)
    idx = idx[:n]

    for i, (x, y) in enumerate(projections):
        if i not in idx:
            continue

        image = Image.fromarray(crops[i].raster)
        image = image.resize(image_size, Image.Resampling.BILINEAR)
        img_array = np.array(image)
        ax.imshow(
            img_array,
            aspect="equal",
            extent=(
                spread * x - 0.5,
                spread * x + 0.5,
                spread * y - 0.5,
                spread * y + 0.5,
            ),
            zorder=1,
        )

    plt.ylim(
        0.95 * min(spread * projections[:, 1]), max(spread * projections[:, 1]) * 1.05
    )
    plt.xlim(
        0.95 * min(spread * projections[:, 0]), max(spread * projections[:, 0]) * 1.05
    )

    plt.axis("off")


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("geometry", help="Geometry file, fiona compatible", type=str)
    parser.add_argument("image", help="Path to image", type=str)
    parser.add_argument("output", help="Output path", type=str)
    parser.add_argument(
        "--project", help="Perform a UMAP projection", action="store_true"
    )
    parser.add_argument("--plot", help="Plot a projection", action="store_true")
    parser.add_argument("--class_id", help="Class filter", type=str, default="tree")
    parser.add_argument(
        "--examples", help="Number of examples to plot", type=int, default=100
    )
    parser.add_argument(
        "--model", help="Open CLIP model to use", default="hf-hub:imageomics/bioclip"
    )
    parser.add_argument(
        "--device", help="Compute device e.g. cpu, cuda, mps", default="cuda"
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    repo_id = args.model.split("/")[-1]
    embedding_path = os.path.join(args.output, f"embeddings_{repo_id}.npz")

    instances = instances_from_geo(args.geometry)
    crops = extract_crops(args.image, instances)

    instances = list(filter(lambda x: x.class_idx == args.class_id, crops))

    if not os.path.exists(embedding_path) or args.overwrite:
        model, _, processor = open_clip.create_model_and_transforms(args.model)

        device = "cpu"
        if args.device:
            try:
                torch.tensor(1).to(args.device)
                device = args.device
            except:
                print("Failed to construct tensor on device, using CPU")

        model.to(device)

        embeddings = generate_clip_embeddings(model, processor, instances, device)

        np.savez_compressed(embedding_path, x=embeddings)
    else:
        embeddings = np.load(embedding_path)["x"]
        print("Using existing embeddings")

    if args.project:
        print("Training UMAP predictor")
        import umap

        mapper = umap.UMAP()
        mapper.fit(embeddings)
        projection = mapper.transform(embeddings)
        np.savez_compressed(
            os.path.join(args.output, f"projection_{repo_id}.npz"), x=projection
        )
        # save mapper

        if args.plot:
            plot_images_scatter(
                projection, instances, spread=2, image_size=(64, 64), n=args.examples
            )
            plt.savefig(
                os.path.join(args.output, "umap.jpg"), dpi=450, bbox_inches="tight"
            )


if __name__ == "__main__":
    main()
