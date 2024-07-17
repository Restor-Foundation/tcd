import dataclasses
import warnings
from typing import Union

import fiona
import numpy as np
import shapely
from PIL import Image


@dataclasses.dataclass
class Instance:
    polygon: shapely.geometry.Polygon = None
    score: Union[float, list[float]] = None
    class_idx: Union[float, str] = None
    raster: np.typing.NDArray = None
    properties: dict = None
    id: int = None
    crs: fiona.crs.CRS = None
    image_path: str = None

    def __add__(self, other):
        polygon = self.polygon.union(other.polygon)
        score = []
        if isinstance(self.score, list):
            score = list(self.score)

            if isinstance(other.score, list):
                score += other.score
            elif isinstance(other.score, float):
                score.append(other.score)

        elif isinstance(self.score, float) and isinstance(other.score, list):
            score = list(other.score)
            score.append(self.score)
        else:
            score = [self.score, other.score]

        return Instance(polygon, score, class_idx=self.class_idx, crs=self.crs)


def instances_from_geo(filename):
    instances = []

    with fiona.open(filename) as cxn:
        geo_crs = cxn.crs
        for feature in cxn:
            polygon = shapely.geometry.shape(feature.geometry)
            props = feature["properties"]
            score = props["score"]
            class_idx = props["class"]
            raster = None
            if "image" in props:
                raster = np.array(Image.open(props["image"]))

            instances.append(
                Instance(polygon, score, class_idx, crs=geo_crs, raster=raster)
            )

    if len(instances) == 0:
        warnings.warn("No instances found")

    return instances
