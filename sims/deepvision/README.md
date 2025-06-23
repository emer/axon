# DeepVision

`deepvision` does deep predictive learning of 3D objects tumbling through space, with periodic saccadic eye movements, providing plenty of opportunity for prediction errors. 

The visual system simulated includes both "where" (LIP, dorsal pathway) and "what" (ventral pathway) networks, and their integration, and this model has also been named the _wwi_ (what-where-integration) model for that reason.

The _where_ (dorsal) pathway is trained first and residual prediction error trains _what_ pathway.

This is (an updated version of) the model described in:

* O’Reilly, R. C., Russin, J. L., Zolfaghar, M., & Rohrlich, J. (2021). Deep Predictive Learning in Neocortex and Pulvinar. *Journal of Cognitive Neuroscience*, ArXiv:2006.14800 [q-Bio]. http://arxiv.org/abs/2006.14800

# Install

See [cogentcore.org](https://cogentcore.org/core) for installation instructions -- basically you need install Go (e.g., `brew install go` on mac), then do `go build` in this directory, but may need to do some things with Cogent Core if it doesn't work right away.

Then, you need to get `CU3D_100_20obj8inst_8tick4sac.tar` from this [google drive folder](https://drive.google.com/drive/folders/13Mi9aUlF1A3sx3JaofX-qzKlxGoViT86?usp=sharing), which has the 3D rendered movies that the network is trained on.  Install it as `images` in the directory where this code is.  For example:

```bash
$ tar -xf CU3D_100_20obj8inst_8tick4sac.tar
$ mv CU3D_100_20obj8inst_8tick4sac images
```

(we usually have it in a centralized place and create a symbolic link, which works on the cluster too..)

# Running

Just run the deepvision executable that is built with the `go build` command. You can see how it processes processes input patterns, etc.

TODO: update for GPU:

It takes about 1 day to train across 32 processors on our older cluster (use `go build -tags mpi` to build with mpi support), so it would take about 16 days without MPI.

