# MNIST Handwriting Images Classifier

An MNIST handwriting images classifier implemented using `numpy` and Convolutional Neural Network(CNN).

## Network Architecture

![architecture-light](./images/architecture-light.png#gh-light-mode-only)
![architecture-dark](./images/architecture-dark.png#gh-dark-mode-only)

Overall architecture of the network is shown above.

## Training and Test Results

Among 60k MNIST training images, 50k images were used for training and 10k images were used for validation. The model was trained for 2 epochs, with 48 iterations for each epoch, total 96 iterations, with batch size of 1024.

The training was tried four times, and average training time was 302.5(5min 2.5sec) ± 1.5 seconds. Loss graph over iterations is shown below.

![loss-graph-light](./images/loss-graph-light.png#gh-light-mode-only)
![loss-graph-dark](./images/loss-graph-dark.png#gh-dark-mode-only)

10k images from MNIST test images were used for test. Average test accuracy was 97.27 ± 0.43%. Some prediction results are shown below.

![test-result-plot](./images/test-result-plot.png)
