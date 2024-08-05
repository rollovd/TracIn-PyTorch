# TracIn via PyTorch

This repository contains an implementation of the TracIn method using PyTorch, as described in the paper ["TracIn: A Simple Method for Assessing the Influence of Training Data on Models"](https://arxiv.org/pdf/2002.08484).

## Overview

TracIn (Tracking Influential Examples) is a method that quantifies the influence of training data points on a model's predictions. This implementation extends the original code to support batch processing, allowing you to calculate the influence of several samples simultaneously, which improves efficiency and scalability.

## Features

* **Batch Processing**: Calculate influences for multiple samples in parallel.
* **Easy Integration**: Simple to integrate with existing PyTorch models and datasets.
* **Example Included**: An example using the MNIST dataset is provided for quick setup and testing.

## Installation
To install the necessary dependencies, run:
```
pip install -r requirements.txt
```

## Usage
An example of how to use this implementation can be found in `main.py`. The example demonstrates how to calculate the influence of training samples on the predictions of a model trained on the MNIST dataset.

## Example Code Snippet
Here's a brief overview of how you can use TracIn in your project:

```python
from src.tracin import vectorized_calculate_tracin_score

model = YourModel()
criterion = YourCriterion()
weights = YourListOfWeights # currently glob.glob(os.path.join("supplementary", "weights", "*"))
train_loader = YourTrainLoader()
test_loader = YourTestLoader()
lr = YourLearningRateScale
device = YourDevice
use_nested_loop_for_dot_product = False # via einsum
float_labels = False # depends on your loss function


matrix = vectorized_calculate_tracin_score(
    model=model,
    criterion=criterion,
    weights_paths=weights,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    lr=lr,
    device=device,
    use_nested_loop_for_dot_product=use_nested_loop_for_dot_product,
    float_labels=float_labels,
)

```

For more detailed information, please refer to the `src/tracin.py` file.

## File Structure

```
├── README.md
├── main.py # Example script demonstrating the use of TracIn with the MNIST dataset.
├── requirements.txt # List of dependencies required to run the implementation.
├── src
│   ├── net.py # Your architecture 
│   └── tracin.py # Core implementation of the TracIn method.
└── supplementary
    └── weights # Your models' weights.
```

* `main.py`: Example script demonstrating the use of TracIn with the MNIST dataset.
* `src/tracin.py`: Core implementation of the TracIn method.
* `requirements.txt`: List of dependencies required to run the implementation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or bug reports.

## References

* ["TracIn: A Simple Method for Assessing the Influence of Training Data on Models"](https://arxiv.org/pdf/2002.08484).
* [Original TracIn implementation](https://github.com/frederick0329/TracIn).