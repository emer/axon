# LVis CU3D100 TE16deg Axon

This is the "standard" version of the LVis model, implemented using the `axon` spiking activation algorithm, with the architecture tracing back to the `cemer` C++ versions that were originally developed (`lvis_te16deg.proj` and `lvis_fix8.proj`).

The `lvix_fix8.proj` version has "blob" color filters in addition to the monochrome gabor filters, and has the capacity to fixate on different regions in the image, but this was never fully utilized.  This Go implementation has the blob color filters, but no specified fixation -- just random 2D planar transforms.

# Images: CU3D100

This [google drive folder](https://drive.google.com/drive/folders/13Mi9aUlF1A3sx3JaofX-qzKlxGoViT86?usp=sharing) has .png input files for use with this model.

By default the models looks for the images extracted from `CU3D_100_renders_lr20_u30_nb.tar.gz` in the `<repo>/sims/lvis_cu3d100_te16deg_axon/images/CU3D_100_renders_lr20_u30_nb/` folder.  This contains 18,859 images of rendered 3D objects from 100 different object categories, with roughly 8-10 3D object instances per category.

See `Config.Env.Path` for path to use for finding these files -- typically make a symlink for `images` to point to a central location having these files.

There is also a larger collection of images: `CU3D_100_plus_renders.tar.gz` which has 30,240 rendered images from the same 100 3D object categories, with 14.45 average different instances per category. However, the additional instances were of lower quality overall and performance is generally slightly worse with this set.

The original reference for these images and the LVis model is:

O'Reilly, R.C., Wyatte, D., Herd, S., Mingus, B. & Jilk, D.J. (2013). Recurrent Processing during Object Recognition. *Frontiers in Psychology, 4,* 124. [PDF](https://ccnlab.org/papers/OReillyWyatteHerdEtAl13.pdf) | [URL](http://www.ncbi.nlm.nih.gov/pubmed/23554596)

The image specs are: 320x320 color images. 100 object classes, 20 images per exemplar. Rendered with 40° depth rotation about y-axis (plus horizontal flip), 20° tilt rotation about x-axis, 80° overhead lighting rotation.

The `ImagesEnv` environment in `images_env.go` adds in-plane affine transformations (translation, scale, rotation) (now known as "data augmentation"), with the standard case being scaling in the range .7 - 1.2, rotation +/- 16 degrees, and translation using a uniform distribution of 30% of the half-width of the image, where 100% would move something in the center to be centered on the edge.  30% is about the maximum amount of translation that does not result in significant amounts of the image being off the edge.

# Benchmarking

See [bench](bench.md) for full info.

# Building

To build and link against MPI:
```bash
go build -v -mod=mod -tags mpi
```
Without the tag all the MPI calls are replaced with stubs that don't do anything.

To run with MPI (e.g.):
```bash
mpirun -np 4 ./lvis_cu3d100_te16deg_axon -no-gui -mpi
```

# TODO:

* no f8
* no cross between f8, f16
* no te

