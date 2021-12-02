# Deep Belief Networks - Internal representation and Adversarial Attacks

Exploration of how a DBN learns some high-level features of the data. The models are implemented in MATLAB and data are represented by dendrograms and scatterplots, produced by using the t-Distributed Stochastic Neighbour Embedding technique (the code is in the python notebook). The simulation involves data augmentation and adversarial attacks to test the robustness of the models.

Each .mat file refers to a different dataset: EMNIST Digits, EMNIST Letters, EMNIST Digits with noise and data augmentation, EMNIST Digits with Adversarial examples.

![alt text](https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/main/Images/Receptive_fields.jpg?raw=true)

Receptive fields of neurons in the layers of the hierarchy: each square represents the region of sensory space that activates the neuron. [A] EMNIST Digits [B] EMNIST Letters.

![alt text](https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/main/Images/t-DistributedStochasticNeighborEmbedding.jpg?raw=true)

t-Distributed Stochastic Neighbor Embedding: dimensionality reduction (from 784 to 2) by minimizing the divergence between the pairwise similarity distributions of the input and of the corresponding low-dimensional points in the embedding. [A] EMNIST Digits [B] EMNIST Letters.

![alt text](https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/main/Images/image_similarity.jpg?raw=true)

Similar images tend to be grouped together in the scatterplot. To see that, some digits that are near in the scatterplots are reported in the panels in the same row. A,B and C refer to raw images, D and E refer to DBN reconstructions at third layer. [A] At raw level some clusters of 5s, indicated with green circles and reported in [B], are clearly separated and [C] there’s a naive association of shapes. [D] At the third level of the hierarchy there’s a more complex feature representation, in fact the previous clusters of 5s are grouped togheter and [E] digits with different labels are associated only if there exists a strong similarity of shapes.

![alt text](https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/main/Images/tree.jpg?raw=true)

Dendogram on the left considered 500 samples rather than 10, and the scatterplot on the right hilights 7 clusters obtained by recursively merging the pair of clusters that minimally increases a given linkage distance in the dendogram.

![alt text](https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/main/Images/wrong_predictions.jpg?raw=true)

Some wrong predictions.

![alt text](https://github.com/silviapoletti/Deep-Belief-Networks-Internal-representation-and-Adversarial-Attacks/blob/main/Images/adversarial.jpg?raw=true)

Example of adversarial and noisy images.
