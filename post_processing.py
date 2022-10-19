import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.affinity import translate


class Bbox:
    def __init__(self, minx, miny, maxx, maxy):
        self.minx = (int)(minx)
        self.miny = (int)(miny)
        self.maxx = (int)(maxx)
        self.maxy = (int)(maxy)
        self.bbox = (self.minx, self.miny, self.maxx, self.maxy)
        self.width = self.maxx - self.minx
        self.height = self.maxy - self.miny

    def __str__(self):
        return f"Bbox(minx={self.minx:.4f}, miny={self.miny:.4f}, maxx={self.maxx:.4f}, maxy={self.maxy:.4f})"


class ProcessedInstance:
    def __init__(self, score, polygon, bbox, class_index):
        self.score = score
        self.polygon = polygon
        self.bbox = bbox
        self.class_index = class_index
        new_poly = translate(self.polygon, xoff=-self.bbox.minx, yoff=-self.bbox.miny)
        shape_local_mask = (self.bbox.width, self.bbox.height)
        self.local_mask = rasterio.features.rasterize(
            [new_poly], out_shape=shape_local_mask
        ).astype(bool)

    def get_pixels(self, image):
        return image[self.bbox.minx : self.bbox.maxx, self.bbox.miny : self.bbox.maxy][
            self.local_mask
        ]

    def __str__(self):
        return f"ProcessedInstance(score={self.score:.4f}, class={self.class_index}, {str(self.bbox)})"


class ProcessedResult:
    def __init__(self, image, trees=None, tree_mask=None, canopy_mask=None):
        self.trees = trees
        self.tree_mask = tree_mask
        self.canopy_mask = canopy_mask
        self.image = image

    def visualise(
        self, color_trees=(0.8, 0, 0), color_canopy=(0, 0, 0.8), alpha=0.8, **kwargs
    ):
        fig, ax = plt.subplots(**kwargs)
        plt.axis("off")
        ax.imshow(self.image)
        canopy_mask_image = np.zeros(
            (self.image.shape[0], self.image.shape[1], 4), dtype=float
        )
        canopy_mask_image[self.canopy_mask == 1] = list(color_canopy) + [alpha]
        ax.imshow(canopy_mask_image)
        tree_mask_image = np.zeros(
            (self.image.shape[0], self.image.shape[1], 4), dtype=float
        )
        tree_mask_image[self.tree_mask == 1] = list(color_trees) + [alpha]
        ax.imshow(tree_mask_image)
        plt.show()

    def __str__(self) -> str:
        canopy_cover = np.count_nonzero(self.canopy_mask) / np.prod(
            self.image.shape[:2]
        )
        tree_cover = np.count_nonzero(self.tree_mask) / np.prod(self.image.shape[:2])
        return f"ProcessedResult(n_trees={len(self.trees)}, canopy_cover={canopy_cover:.4f}, tree_cover={tree_cover:.4f})"


class PostProcessor:
    def __init__(self, runner) -> None:
        self.runner = runner

    def _mask_to_polygon(self, mask):
        all_polygons = []
        for shape, _ in features.shapes(mask.astype(np.int16), mask=mask):
            all_polygons.append(shapely.geometry.shape(shape))

        all_polygons = shapely.geometry.MultiPolygon(all_polygons)
        if not all_polygons.is_valid:
            all_polygons = all_polygons.buffer(0)
            # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            # need to keep it a Multi throughout
            if all_polygons.type == "Polygon":
                all_polygons = shapely.geometry.MultiPolygon([all_polygons])
        return all_polygons

    def _bbox_overlap(self, bbox1, bbox2):
        if (
            bbox1.minx >= bbox2.maxx
            or bbox1.maxx <= bbox2.minx
            or bbox1.maxy <= bbox2.miny
            or bbox1.miny >= bbox2.maxy
        ):
            return False
        return True

    def _get_proper_bbox(self, bbox, image):
        minx, miny, maxx, maxy = self.runner.bbox_to_original_image(bbox, image)
        # Sort coordinates if necessary
        if miny > maxy:
            miny, maxy = maxy, miny

        if minx > maxx:
            minx, maxx = maxx, minx

        return Bbox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def non_max_suppression(self):
        pass

    def remove_edge_predictions(self):
        pass

    def process_untiled_result(self, result, image, treshold=0.5):
        canopy_mask = np.zeros((image.height, image.width), dtype=bool)
        tree_mask = np.zeros((image.height, image.width), dtype=bool)
        trees = []

        for i in range(len(result)):
            if result.scores[i] < treshold:
                continue

            mask = result.pred_masks[i].cpu().numpy()
            polygon = self._mask_to_polygon(mask)
            class_idx = int(result.pred_classes[i])

            if self.runner.config.data.classes[class_idx] == "canopy":
                canopy_mask = np.logical_or(canopy_mask, mask)
            else:
                bbox_instance = result.pred_boxes[i].tensor[0].cpu().numpy()

                bbox = Bbox(
                    minx=bbox_instance[0],
                    miny=bbox_instance[1],
                    maxx=bbox_instance[2],
                    maxy=bbox_instance[3],
                )

                trees.append(
                    ProcessedInstance(
                        score=result.scores[i],
                        polygon=polygon,
                        class_index=class_idx,
                        bbox=bbox,
                    )
                )

                tree_mask = np.logical_or(tree_mask, mask)

        return ProcessedResult(
            image=image.read().transpose(1, 2, 0),
            trees=trees,
            tree_mask=tree_mask,
            canopy_mask=canopy_mask,
        )

    def _collect_tiled_result(self, results, image, treshold=0.5):
        untiled_instances = []

        canopy_mask = np.zeros((image.height, image.width), dtype=bool)

        tree_mask = np.zeros((image.height, image.width), dtype=bool)

        for i, result in enumerate(results):
            instances, bbox = result
            proper_bbx = self._get_proper_bbox(bbox, image)

            for i in range(len(instances)):
                if instances.scores[i] < treshold:
                    continue

                mask = instances.pred_masks[i].cpu().numpy()
                class_idx = int(instances.pred_classes[i])
                mask_height, mask_width = mask.shape
                polygon = self._mask_to_polygon(mask)
                polygon = translate(polygon, xoff=proper_bbx.minx, yoff=proper_bbx.miny)

                if (
                    self.runner.config.data.classes[class_idx] == "canopy"
                    and instances.scores[i] >= treshold
                ):
                    canopy_mask[
                        proper_bbx.miny : proper_bbx.miny + mask_height,
                        proper_bbx.minx : proper_bbx.minx + mask_width,
                    ][mask] = True
                else:
                    bbox_instance_tiled = (
                        instances.pred_boxes[i].tensor[0].cpu().numpy()
                    )
                    bbox_instance = Bbox(
                        minx=proper_bbx.minx + bbox_instance_tiled[0],
                        miny=proper_bbx.miny + bbox_instance_tiled[1],
                        maxx=proper_bbx.minx + bbox_instance_tiled[2],
                        maxy=proper_bbx.miny + bbox_instance_tiled[3],
                    )

                    new_instance = ProcessedInstance(
                        class_index=class_idx,
                        polygon=polygon,
                        bbox=bbox_instance,
                        score=instances.scores[i],
                    )

                    tree_mask[
                        proper_bbx.miny : proper_bbx.miny + mask_height,
                        proper_bbx.minx : proper_bbx.minx + mask_width,
                    ][mask] = True

                    untiled_instances.append(new_instance)

        return untiled_instances, canopy_mask, tree_mask

    def process_tiled_result(self, results, image, treshold=0.5):
        untiled_instances, canopy_mask, tree_mask = self._collect_tiled_result(
            results, image, treshold
        )

        merged_instances = []

        for i, instance in enumerate(untiled_instances):
            j = i + 1
            while j < len(untiled_instances):

                other_instance = untiled_instances[j]

                if (
                    self._bbox_overlap(instance.bbox, other_instance.bbox)
                    and instance.class_index == instance.class_index
                    and instance.polygon.intersects(other_instance.polygon)
                ):
                    bbox1 = instance.bbox
                    bbox2 = other_instance.bbox
                    size1 = instance.polygon.area
                    size2 = other_instance.polygon.area

                    instance = ProcessedInstance(
                        class_index=instance.class_index,
                        polygon=instance.polygon.union(other_instance.polygon),
                        bbox=Bbox(
                            minx=min(bbox1.minx, bbox2.minx),
                            miny=min(bbox1.miny, bbox2.miny),
                            maxx=max(bbox1.maxx, bbox2.maxx),
                            maxy=max(bbox1.maxy, bbox2.maxy),
                        ),
                        score=(size1 * instance.score + size2 * other_instance.score)
                        / (size1 + size2),
                    )
                    del untiled_instances[j]
                    j = i + 1
                else:
                    j = j + 1

            merged_instances.append(instance)

        return ProcessedResult(
            image=image.read().transpose(1, 2, 0),
            trees=merged_instances,
            tree_mask=tree_mask,
            canopy_mask=canopy_mask,
        )
