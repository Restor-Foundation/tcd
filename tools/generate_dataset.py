import json
import logging
import os
from typing import Union

import datasets
from pycocotools.coco import COCO
from tqdm.auto import tqdm

logger = logging.getLogger("Dataset generator")
logging.basicConfig(level=logging.INFO)


def save_geotiff(metadata, filename):
    """
    Convert a HF record into a GeoTIFF image
    """
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    image = metadata["image"]
    bounds = metadata["bounds"]
    crs = metadata["crs"]

    image_array = np.array(image)

    transform = from_bounds(*bounds, image_array.shape[1], image_array.shape[0])

    meta = {
        "driver": "GTiff",
        "height": image_array.shape[0],
        "width": image_array.shape[1],
        "count": image_array.shape[2] if len(image_array.shape) > 2 else 1,
        "dtype": image_array.dtype,
        "crs": CRS.from_string(crs),
        "transform": transform,
    }

    # Save the image as a GeoTIFF
    with rasterio.open(filename, "w", **meta) as dst:
        if len(image_array.shape) == 2:
            dst.write(image_array, 1)
        else:
            for i in range(image_array.shape[2]):
                dst.write(image_array[:, :, i], i + 1)


def dataset_to_coco(
    dataset: datasets.Dataset,
    output_root: str,
    split: str,
    limit: int = None,
    store_images: bool = True,
    use_jpeg: bool = True,
):
    images = []
    annotations = []

    # TODO more flexible licenses per-image, for example
    # if creating merged datasets.
    licenses = []
    license2id = {}
    for idx, license in enumerate(set(dataset["license"])):
        licenses.append(
            {
                "id": idx + 1,
                "name": license,
            }
        )
        print(f"Adding license: {license}")
        license2id[license] = idx + 1

    # Hardcode for now, should probably get this from somewhere else though.
    # TODO: Add metadata to dataset on HF
    categories = [
        {"id": 2, "name": "tree", "supercategory": "root"},
        {"id": 1, "name": "canopy", "supercategory": "root"},
    ]

    image_root = os.path.join(output_root, "images")
    mask_root = os.path.join(output_root, "masks")

    os.makedirs(output_root, exist_ok=True)
    if store_images:
        os.makedirs(image_root, exist_ok=True)
        os.makedirs(mask_root, exist_ok=True)

    # Lists, so by reference is OK
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "licenses": licenses,
        "categories": categories,
    }

    # Load annotations efficiently
    for ann in dataset["coco_annotations"]:
        annotations.extend(json.loads(ann))

    pbar = tqdm(enumerate(dataset), desc=f"Split: {split}", total=len(dataset))
    for idx, row in pbar:
        if limit and limit < idx:
            break

        oam_id = row["oam_id"]
        image_id = row["image_id"]

        if not use_jpeg:
            image_name = f"{oam_id}_{image_id}.tif"
        else:
            image_name = f"{oam_id}_{image_id}.jpg"

        mask_name = f"{oam_id}_{image_id}.png"
        image_path = os.path.join(image_root, image_name)
        mask_path = os.path.join(mask_root, mask_name)

        # Save image + mask
        if store_images:
            if not os.path.exists(image_path):
                if not use_jpeg:
                    save_geotiff(row, image_path)
                else:
                    row["image"].save(image_path)

            if not os.path.exists(mask_path):
                row["annotation"].save(mask_path)

        # Image dict
        images.append(
            {
                "file_name": image_name,
                "width": row["width"],
                "height": row["height"],
                "id": image_id,
                "license": license2id[row["license"]],
            }
        )

    annotation_path = os.path.join(output_root, f"{split}.json")
    with open(annotation_path, "w") as fp:
        json.dump(coco_dict, fp, indent=1)

    _ = COCO(annotation_path)

    return coco_dict


def generate_coco_dataset(
    dataset: Union[str, list],
    output_folder: str,
    use_jpeg: bool = True,
    generate_folds: bool = True,
):
    """
    Generates the full OAM-TCD dataset from a HF Parquet table.

    Args:
        dataset :
        output_folder: Path to store dataset
        use_jpeg: Convert TIFF files to JPEG (can speed up dataloading)
    """

    if isinstance(dataset, str):
        logger.info("Loading dataset")
        dataset = datasets.load_dataset(dataset, writer_batch_size=10)
    elif isinstance(dataset, list):
        dataset = [datasets.load_dataset(d) for d in dataset if isinstance(d, str)]

        splits = set.union(*[set(d.keys()) for d in dataset])

        print(f"Found splits: {splits}")

        combined_dataset = {}
        for split in splits:
            combined_dataset[split] = datasets.concatenate_datasets(
                [d[split] for d in dataset if split in d]
            )

        dataset = datasets.DatasetDict(combined_dataset)

    # Generate full dataset for holdout/release:
    logger.info("Generating full/holdout dataset")

    for split in dataset:
        dataset_to_coco(
            dataset[split],
            output_root=os.path.join(output_folder, "holdout"),
            split=split,
            use_jpeg=use_jpeg,
        )

    # Generate k-fold splits:
    if generate_folds:
        for fold in range(5):
            fold_folder = os.path.join(output_folder, f"kfold_{fold}")

            logger.info(f"Generating k-fold split {fold} in {fold_folder}")

            val_fold_idx = dataset["train"]["validation_fold"]

            # Do not use filter. It's extremely slow and uses an enormous amount of RAM.
            # It's faster to select the column, filter on that and then use the indices:
            train_indices = [
                i for (i, v) in enumerate(val_fold_idx) if v != fold and v >= 0
            ]

            if len(train_indices) > 0:
                dataset_to_coco(
                    dataset["train"].select(train_indices),
                    output_root=fold_folder,
                    split="train",
                    store_images=False,
                )

            val_indices = [
                i for (i, v) in enumerate(val_fold_idx) if v == fold and v >= 0
            ]
            if len(val_indices) > 0:
                dataset_to_coco(
                    dataset["train"].select(val_indices),
                    output_root=fold_folder,
                    split="val",
                    store_images=False,
                )

            # Symlink images/mask folders to save space
            os.symlink(
                os.path.join(output_folder, "holdout", "images"),
                os.path.join(fold_folder, "images"),
                target_is_directory=True,
            )
            os.symlink(
                os.path.join(output_folder, "holdout", "masks"),
                os.path.join(fold_folder, "masks"),
                target_is_directory=True,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a Restor TCD dataset on HuggingFace to MS-COCO format."
    )
    parser.add_argument(
        "dataset",
        nargs="+",
        type=str,
        help="List of dataset names, to be downloaded from HF Hub",
    )
    parser.add_argument(
        "output", type=str, help="Output path where the dataset will be created"
    )
    parser.add_argument(
        "--tiffs",
        action="store_true",
        help="Output images as GeoTIFF instead of JPEG (may slow down dataloading)",
    )
    parser.add_argument(
        "--folds",
        action="store_true",
        help="Generate cross-validation folds based on indices in the dataset",
    )
    args = parser.parse_args()

    generate_coco_dataset(
        args.dataset, args.output, use_jpeg=not args.tiffs, generate_folds=args.folds
    )
