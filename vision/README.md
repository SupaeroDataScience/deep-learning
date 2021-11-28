# Deep Learning for Computer Vision, practical session at ISAE-SUPAERO

## Introduction

This repository contains the code and documentation of a "Deep Learning practical session" given at ISAE-SUPAERO on December 2nd/3rd 2019 and Nov 30/Dec 1st 2020

The introduction slides can be accessed at [this URL website](https://fchouteau.github.io/isae-practical-deep-learning). It is recommended to read it first as it contains the necessary information to run this from scratch.

There are three notebooks at the root of this repository, those as the exercises

## Where to run it ?

This hands on session is based on running code & training using [Google Cloud Platform Deep Learning VMs](https://cloud.google.com/deep-learning-vm/), see `gcp/` for examples on configuring your own machine. 

See [recipes/](recipes) for tutorial on running this on Deep Learning VM

However, this is runnable everywhere since data access is based on public URLs & numpy

Should you want to do this at home you can use [Google Collaboratory instances](https://colab.research.google.com/) - it's even easier than deep learning VMs (and free)

### Run this course on GCP Deep Learning VMs

- Link to docs https://cloud.google.com/ai-platform/deep-learning-vm/docs

- Create DLVM instance using the CLI

```bash
export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="europe-west4-c"
export INSTANCE_NAME="${USER}-dlvm"

gcloud compute instances create ${INSTANCE_NAME} \
  --zone=${ZONE} \
  --image-family=${IMAGE_FAMILY} \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --scopes="storage-rw" \
  --accelerator="type=nvidia-tesla-p4,count=1" \
  --metadata="install-nvidia-driver=True"
```

Wait for the instance to boot

- Connect to jupyter lab and upload notebooks (or clone this repo)
```bash
gcloud compute ssh --project ${PROJECT_ID} --zone ${ZON}E \
  ${INSTANCE_NAME} -- -L 8080:localhost:8080
```


## Reading list

- The Bible : [Convolutional Neural Networks for Image Recognition by Stanford](http://cs231n.stanford.edu/syllabus.html), slides & course notes are very useful
- [CS231n: Details on convolutions](https://cs231n.github.io/convolutional-networks/), how to compute number of parameters & tensor sizes in a CNN...
- [Guide on convolution arithmetics](https://github.com/vdumoulin/conv_arithmetic ) and a lot of visualisations to understand convolutions better
- Two [medium blog posts](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
) that try [to explain things better](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

## Usage & Contribution

No support is guaranteed by the authors beyond the hands-on session.

This hands-on session was created by Florient Chouteau and Matthieu Le Goff.

See [`licence.md`](./licence.md) for licence information.
