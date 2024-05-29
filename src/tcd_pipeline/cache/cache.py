import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import rasterio
import rasterio.windows
from natsort import natsorted
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class ResultsCache:
    """
    Convenience class for storing intermediate results from models. In the case
    of huge images, it is quite possible that the intermediate results are too
    large to store in memory. We therefore (by default) cache tiled results as they
    are generated.

    A cache should contain:

    - some reference to the source image and the location of the stored results with
    respect to that image. This is a bounding box in global image coordinates.
    - if instance segmentation data, polygons that have been predicted by the model along
    with classes, detection scores, etc.
    - if semantic segmentation data, semantic (confidence) masks corresponding to the predicted tile

    Instance segmentation caches return results as ProcessedInstance objects which
    are then used by the post-processor (or can be inspected directly for debugging).

    """

    def __init__(self, cache_folder, image_path: str, classes=None, cache_suffix=None):
        self.cache_folder = cache_folder
        self.cache_suffix = f"_{cache_suffix}" if cache_suffix is not None else ""
        self.image_path = os.path.abspath(image_path)
        self.tile_count = 0
        self.classes = classes
        self._results = []

    def initialise(self) -> None:
        """
        Create the cache folder. Optionally clear if this folder
        is being reused.
        """
        os.makedirs(self.cache_folder, exist_ok=True)
        logger.debug(f"Caching to {self.cache_folder}")
        self.tile_count = 0

    def clear(self) -> None:
        """
        Clears the current cache, deleting the contents of the folder.
        Does not warn, so be careful about calling this manually or
        if you've altered the cache folder path.
        """
        if self.cache_folder is None:
            logger.debug("Cache folder not set")
            return

        if not os.path.exists(self.cache_folder):
            logger.debug("Cache folder doesn't exist")
        else:
            shutil.rmtree(self.cache_folder)
            logger.debug(f"Removed existing cache folder {self.cache_folder}")

    @abstractmethod
    def save(self) -> None:
        """
        Save the output from a single model pass to the cache.
        """

    @property
    def cache_files(self) -> list[str]:
        """
        Files stored in the cache folder.
        """
        _cache_files = natsorted(self._find_cache_files())

        if len(_cache_files) == 0:
            logger.warning("No cached files found.")

        return _cache_files

    @abstractmethod
    def _find_cache_files(self) -> list[str]:
        """
        Locate cache files matching the particular type of cache (e.g.
        Numpy, pickle, COCO).
        """

    @abstractmethod
    def _load_file(self, path) -> list[Dict]:
        """
        Internal method for loading a cached file. Assumes that the cached
        file contains a single or list of results. Each individual result is
        returned as a dictionary.
        """

    def load(self) -> None:
        """
        (Re)load the cache, clears the internal results list first.
        """

        self.results.clear()

        for cache_file in tqdm(self.cache_files):
            self.results.append(self._load_file(cache_file))

        self.tile_count = len(self)

    def cache_image(self, image, window) -> None:
        """
        Stores a geotiff for the tile
        """

        kwargs = image.meta.copy()

        kwargs.update(
            {
                "height": window.height,
                "width": window.width,
                "transform": rasterio.windows.transform(window, image.transform),
                "compress": "jpeg",
            }
        )

        with rasterio.open(
            os.path.join(self.cache_folder, f"{self.tile_count}_tile.tif"),
            "w",
            **kwargs,
        ) as dst:
            dst.write(image.read(window=window, boundless=True))

    @property
    def results(self):
        """
        A list of results that are stored in the cache
        """
        return self._results

    def __len__(self):
        return len(self._results)

    def __getitem__(self, idx):
        return self._results[idx]


import weakref


class CloudCache(ResultsCache):
    """
    Cache helper for uploading to cloud storage (GCP). Wraps a normal ResultsCache object
    and adds methods for uploading/downloading to a bucket directly.
    """

    def __init__(
        self,
        cache_bucket,
        cache_folder,
        image_path: str,
        local_folder: str = None,
        classes=None,
        cache_suffix=None,
    ):
        self.cache_bucket = cache_bucket
        self.cache_folder = cache_folder
        self.cache_suffix = cache_suffix
        self.image_path = os.path.abspath(image_path)
        self.tile_count = 0
        self.classes = classes
        self._results = []

        from google.cloud import storage

        storage_client = storage.Client()
        self.bucket = storage_client.bucket(cache_bucket)

        if local_folder is None:
            logger.debug(f"Caching to temporary local folder: {self.local_folder}")
            self.local_folder = tempfile.mkdtemp(prefix="tcd")
            self._finalizer = weakref.finalize(self)

    def _finalizer(self):
        logger.debug("Clearing temporary local cache folder")
        shutil.rmtree(self.local_folder)

    def upload_file(self, local_file, destination_blob):
        blob = self.bucket.blob(destination_blob)
        blob.upload_from_filename(local_file)

    def get_files(self, prefix, delimiter):
        return self.storage_client.list_blobs(
            self.bucket_name, prefix=prefix, delimiter=delimiter
        )

    def initialise(self) -> None:
        """
        Create the cache folder. Optionally clear if this folder
        is being reused.
        """
        self.tile_count = 0
        # Create bucket

    def clear(self) -> None:
        """
        Clears the current cache, deleting the contents of the folder.
        Does not warn, so be careful about calling this manually or
        if you've altered the cache folder path.
        """
        # Attempt to remove the bucket o
        blobs = list(self.get_files(self.prefix))
        self.bucket.delete_blobs(blobs)
        logger.debug(f"Deleted: {len(blobs)}")

    @abstractmethod
    def save(self) -> None:
        """
        Save the output from a single model pass to the cache.
        """
        pass
