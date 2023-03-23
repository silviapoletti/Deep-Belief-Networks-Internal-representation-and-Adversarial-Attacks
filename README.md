# Deep Belief Networks - Internal representation and Adversarial Attacks

Exploration of how a DBN learns some high-level features of the data. The models are implemented in MATLAB and data are represented by dendrograms and scatterplots, produced by using the t-Distributed Stochastic Neighbour Embedding technique (the code is in the python notebook). The simulation involves data augmentation and adversarial attacks to test the robustness of the models.

Each .mat file refers to a different dataset: EMNIST Digits, EMNIST Letters, EMNIST Digits with noise and data augmentation, EMNIST Digits with Adversarial examples.

# Description of the model

Generative unsupervised learning is performed by
implementing a DBN, which is a hierarchical generative
model consisting in some Restricted Boltzman
Machines (RBM) staked one over the other. These are
stochastic graphical neural networks in which the hidden
neurons model the latent statistical structure of
data observations, that are clamped to the visible neurons.

The training of the whole DBN proceeds bottom-up,
starting from the first RBM and continuing one layer
at a time, for 30 epochs; while training a layer the
weights of lower layers are freezed.

An output layer is added to the deepest layer of the
DBN to carry out a classification task. This only needs
a linear read-outs because the reconstructed images are
high-level representations of the data; in fact the layers
of the hierarchy learn increasingly complex and abstract
features of the data.

# Internal representation analysis 

For both EMNIST Digits and EMNIST Letters datasets, the network
developed center-sorround detectors at the first layer,
that become less blurred in the second and encode
even more complex features in the third: the shape of
digits/letters clearly emerges in some receptive fields
of the highest layer neurons.

The following shows the receptive fields of neurons in the layers of the hierarchy: each square represents the region of sensory space that activates the neuron. [A] EMNIST Digits [B] EMNIST Letters.

<p align="center">
  <img src="https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/b0f8096eeb6d0372ba7e3410bb26b5567373eb2d/report/Receptive_fields.jpg" width="1000"/>
</p>

This is consistent with what the next figure shows: internal
feature representation of the data is visualized with
a scatterplot that becomes less blurred and more
clustered as going up in the hierarchy.

<p align="center">
  <img src="https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/b0f8096eeb6d0372ba7e3410bb26b5567373eb2d/report/t-DistributedStochasticNeighborEmbedding.jpg" width="10000"/>
</p>

The plots are generated using t-Distributed Stochastic Neighbor Embedding: dimensionality reduction (from 784 to 2) by minimizing the divergence between the pairwise similarity distributions of the input and of the corresponding low-dimensional points in the embedding. [A] EMNIST Digits [B] EMNIST Letters.

In the next figure, notice how similar images tend to be grouped together in the scatterplot:

<p align="center">
  <img src="https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/b0f8096eeb6d0372ba7e3410bb26b5567373eb2d/report/image_similarity.jpg"/>
</p>

Indeed, some digits that are near in the scatterplots have been reported in the panels in the same row. A,B and C refer to raw images, D and E refer to DBN reconstructions at third layer. [A] At raw level some clusters of 5s, indicated with green circles and reported in [B], are clearly separated and [C] there’s a naive association of shapes. [D] At the third level of the hierarchy there’s a more complex feature representation, in fact the previous clusters of 5s are grouped togheter and [E] digits with different labels are associated only if there exists a strong similarity of shapes.

The following dendogram and scatterplot
show that digits 8-3-5 and 7-4-9 share
some visual features and their clusters are closeby. According to [M. Grissinger - "Misidentification of alphanumeric symbols plays a role in errors" (2009)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5614409/) the same numbers are also commonly confused by humans.

<p align="center">
  <img src="https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/b0f8096eeb6d0372ba7e3410bb26b5567373eb2d/report/tree.jpg"/>
</p>

Dendogram on the left considered 500 sample. The scatterplot on the right hilights 7 clusters obtained by recursively merging the pair of clusters that minimally increases a given linkage distance in the dendogram.

From the next figure, it emerges that the model makes errors
that are consistent with its internal representation: 5s
are often classified as 3s, 9s as 4s (and vice versa) and
7s ad 9s.

<p align="center">
  <img src="https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/b0f8096eeb6d0372ba7e3410bb26b5567373eb2d/report/wrong_predictions.jpg" width="66%"/>
</p>

These errors are often similar to what one
would expect from a human observer.


# Adversarial examples

It’s possible to fool a deep network by injecting a
small percentage of noise in the data such as the image
looks almost the same to the human eyes. Noise isn’t
random: the image is modified in the direction of the
gradient that maximizes the loss function with respect
to the input image.
The following adversarial example is generated using the Fast Gradient
Sign Method Attack. This example is compared to a noisy example generated by adding Gaussian noise to the original image.

<p align="center">
  <img src="https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/b0f8096eeb6d0372ba7e3410bb26b5567373eb2d/report/adversarial.jpg" width="66%"/>
</p>

The model shows a weak resistance
to Adversarial attacks: even if Gaussian
noisy images seem more perturbated than Adversarial
images according to human eyes, the model achieved much less
accuracy in classifying the latter.
