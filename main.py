import os
import glob
import click
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from src.net import Net
from src.tracin import vectorized_calculate_tracin_score


@click.command()
@click.option("--train_subset", default=95)
@click.option("--test_subset", default=5)
@click.option("--train_bs", default=25)
@click.option("--test_bs", default=3)
@click.option("--lr", default=1e-3)
@click.option("--use_nested_loop_for_dot_product", is_flag=True)
@click.option("--float_labels", is_flag=True)
def main(
    train_subset: int,
    test_subset: int,
    train_bs: int,
    test_bs: int,
    lr: float,
    use_nested_loop_for_dot_product: bool,
    float_labels: bool,
) -> torch.Tensor:
    """
    Main function to execute the TracIn method on the MNIST dataset.

    Args:
        train_subset (int): Size of the training subset.
        test_subset (int): Size of the testing subset.
        train_bs (int): Training batch size.
        test_bs (int): Testing batch size.
        lr (float): Learning rate.
        use_nested_loop_for_dot_product (bool): Use nested loop for dot product calculation.
        float_labels (bool): Whether to use float labels.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_subset = torch.utils.data.Subset(train_dataset, range(train_subset))
    test_subset = torch.utils.data.Subset(test_dataset, range(test_subset))

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=train_bs, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=test_bs, shuffle=False
    )

    criterion = nn.CrossEntropyLoss(reduction="none")
    weights = glob.glob(os.path.join("supplementary", "weights", "*"))

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

    return matrix


if __name__ == "__main__":
    matrix = main()
