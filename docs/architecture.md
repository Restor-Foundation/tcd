# Pipeline architecture

This document serves as an overview of the components in the repository and how they fit together. This may be helpful for users and developers wishing to contribute to the repo or just to better understand how the pipeline works.

## Why a pipeline?

You may have noticed that the accompanying datset (OAM-TCD) and our models can be used on their own, without the need for a lot of extra code. This is by design, so that our work is as accessible as possible. If you just want to make some predictions on an image, you can use the AutoModels in `transformers` and predict away. No pipeline needed.

However, at Restor we have a need for a platform that can be used for operational purposes and that requires a few more features built on top of the models.

- Tilled inference: a problem in geospatial image processing is how to deal with really enormous images. The normal way to deal with this is _tiling_. We split the input image up into overlapping regions, process them separately and then merge the results together. This is effectively a split-apply-combine operation.
