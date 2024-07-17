import dataclasses
import os
from collections import defaultdict
from typing import Union

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
import rtree
import shapely
from rasterio import mask
from shapely.plotting import plot_polygon

from tcd_pipeline.scripts._instance import Instance, instances_from_geo


def plot_instances(instances):
    """
    Plot a list of instances
    """
    plt.figure()

    for instance in instances:
        plot_polygon(
            instance.polygon,
            add_points=False,
            edgecolor=(1, 0, 0, 0.5) if instance.class_idx > 1 else (0, 1, 0, 0.5),
            alpha=0.5 * np.median(instance.score),
        )


def split(instances: list[Instance], iou_threshold=0.3) -> list[Instance]:
    """
    Split a list of instances based on heuristics.

    First, instancs are filtered based on whether they
    contain centroids of other instances. Then, any
    instances which have a small overlap are considered
    separate. Instances with a largeer overlap are merged.

    This process continues iteratively until all instances
    have been either merged or split out.

    Returns a list of merged instances.
    """
    instances = list(instances)
    merged = []

    instances = filter_centroids(instances)

    while len(instances) > 0:
        instance_a = instances.pop()
        a_overlaps = False
        ab_union = None

        for idx, instance_b in enumerate(instances):
            if instance_a == instance_b:
                continue

            ab_intersection = instance_a.polygon.intersection(instance_b.polygon)
            ab_union = instance_a.polygon.union(instance_b.polygon)
            iou = ab_intersection.area / ab_union.area

            if iou < iou_threshold:
                continue
            else:
                a_overlaps = True
                break

        # Base case, no significant overlap
        if not a_overlaps:
            merged.append(instance_a)
        # Otherwise, we should combine a, b
        # and add that polygon back to the
        # instance list.
        else:
            instances.pop(idx)
            instances.append(instance_a + instance_b)

    return merged


def dissolve(instances: list[Instance]) -> dict:
    """
    Perform a dissolve operation and return a dictionary of merged
    polygons and their sub-polygons (i.e. from the source list).
    """

    # Unary union gives a single geometry - split it into polygons:
    union = shapely.unary_union([i.polygon for i in instances])
    # Case when the union is a single polygon
    if isinstance(union, shapely.geometry.Polygon):
        merged = [union]
    else:
        merged = [g for g in union.geoms]

    out = defaultdict(list)

    # Add the merged/dissolved polygons to a spatial index
    idx = rtree.index.Index()

    for i, m_poly in enumerate(merged):
        idx.insert(i, m_poly.bounds, obj=m_poly)

    used_geoms = set()

    # Iterate over the source instances and associate source <> merged
    for instance in instances:
        poly = instance.polygon
        if np.any(np.isnan(poly.bounds)):
            continue
        # BBOX intersection
        potential_candidates = idx.intersection(poly.bounds, objects=True)
        for n in potential_candidates:
            # Polygon intersection
            if poly.intersects(n.object) and instance.polygon not in used_geoms:
                out[n.object].append(instance)
                used_geoms.add(instance.polygon)

    return out


def merge(
    instances: list[Instance],
    class_idx: int,
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.5,
) -> list[Instance]:
    """Merge a list of instances.

    1) Filter by confidence threshold
    2) Dissovle instances to separate overlapping groups
    3) For each group, split into polygons following simple heuristics

    Args:
        instances (list[ProcessedInstance]): Instance list to consider merging
        class_index (int): Class filter
        confidence_threshold (float): Confidence threshold
        iou_threshold(float): Threshold above which to consider a polygon as overlapping
    Returns:
        list[ProcessedInstance]: List of merged instances
    """

    import shapely

    merged_instances = []
    instances = list(filter(lambda x: x.score > confidence_threshold, instances))

    for poly, instance_group in dissolve(instances).items():
        if len(instance_group) == 1:
            instance = instance_group[0]
            merged_instances.append(instance)
        else:
            pre = len(instance_group)
            split_instances = split(instance_group, iou_threshold=iou_threshold)
            for instance in split_instances:
                if isinstance(instance.score, list):
                    instance.score = np.median(instance.score)

                merged_instances.append(instance)

    return merged_instances


def filter_centroids(instances, max_overlaps=1):
    """
    Filter objects to remove those which contain the centroids
    of others - this is a simple but fairly effective heuristic
    to remove "dumbell" shaped predictions which contain multiple
    individuals.
    """
    instances = list(instances)

    overlaps = defaultdict(int)

    for a in instances:
        for idx, b in enumerate(instances):
            if a == b:
                continue
            if a.polygon.centroid.within(b.polygon):
                overlaps[idx] += 1

    out = []
    for idx, a in enumerate(instances):
        if overlaps[idx] > max_overlaps:
            continue

        out.append(a)

    return out


def save(instances, image, filename):
    schema = {
        "properties": {
            "id": "str",
            "area": "float",
            "perimeter": "float",
            "score": "float",
            "class": "int",
        },
        "geometry": "Polygon",
    }

    with rasterio.open(image) as src:
        with fiona.open(filename, schema=schema, crs=src.crs, mode="w") as dst:
            features = []
            for idx, instance in enumerate(instances):
                feature = defaultdict(dict)
                feature["geometry"] = instance.polygon
                feature["properties"]["class"] = instance.class_idx
                feature["properties"]["score"] = instance.score
                feature["properties"]["area"] = instance.polygon.area
                feature["properties"]["perimeter"] = instance.polygon.length
                feature["properties"]["id"] = str(idx)
                features.append(feature)

            dst.writerecords(features)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", help="Geometry file, fiona compatible")
    parser.add_argument("image", help="Source image (GeoTIFF")
    parser.add_argument("--confidence", help="Confidence threshold")
    parser.add_argument("--iou", help="IoU merge threshold")
    args = parser.parse_args()

    instances = instances_from_geo(args.geometry, args.image)

    merged = merge(instances, 0, confidence_threshold=0.35, iou_threshold=0.1)

    base, ext = os.path.splitext(os.path.basename(args.geometry))
    save(merged, args.image, base + "_merge" + ext)


if __name__ == "__main__":
    main()
