
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10


RANDOM_SEED = 42
VALIDATION_SIZE = 5000

DATA_DIR = Path("data")
SPLIT_DIR = Path("splits")


def count_classes(
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, int]:
    #Count the number of images belonging to each class
    counts = np.bincount(labels, minlength=len(class_names))

    return {
        name: int(count)
        for name, count in zip(class_names, counts)
    }


def main() -> None:
    #Create the directories used to store the dataset and split
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    #Download the original CIFAR-10 training and test datasets
    train_dataset = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
    )

    test_dataset = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
    )

    #Extract the labels without modifying the original images
    train_labels = np.asarray(train_dataset.targets, dtype=np.int64)
    test_labels = np.asarray(test_dataset.targets, dtype=np.int64)

    class_names = train_dataset.classes

    #Create a fixed stratified split from the original training set
    all_indices = np.arange(len(train_dataset))

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_labels,
    )

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)

    #Make sure the two subsets do not share any images
    if np.intersect1d(train_indices, val_indices).size != 0:
        raise RuntimeError("Training and validation sets overlap")

    #Make sure every original training image is assigned once
    combined_indices = np.concatenate(
        [train_indices, val_indices]
    )

    if not np.array_equal(
        np.sort(combined_indices),
        all_indices,
    ):
        raise RuntimeError("The dataset split is incomplete")

    #Check that CIFAR-10 has the expected dimensions and labels
    if train_dataset.data.shape != (50000, 32, 32, 3):
        raise RuntimeError("Unexpected CIFAR-10 training shape")

    if test_dataset.data.shape != (10000, 32, 32, 3):
        raise RuntimeError("Unexpected CIFAR-10 test shape")

    train_counts = count_classes(
        train_labels[train_indices],
        class_names,
    )

    val_counts = count_classes(
        train_labels[val_indices],
        class_names,
    )

    test_counts = count_classes(test_labels, class_names)

    #Verify that stratification preserved the expected class balance
    if any(count != 4500 for count in train_counts.values()):
        raise RuntimeError("Unexpected training class distribution")

    if any(count != 500 for count in val_counts.values()):
        raise RuntimeError("Unexpected validation class distribution")

    if any(count != 1000 for count in test_counts.values()):
        raise RuntimeError("Unexpected test class distribution")

    #Prevent accidental changes to an existing saved split
    for filename, indices in (
        ("train_indices.npy", train_indices),
        ("val_indices.npy", val_indices),
    ):
        path = SPLIT_DIR / filename

        if path.exists():
            saved_indices = np.load(path, allow_pickle=False)

            if not np.array_equal(saved_indices, indices):
                raise RuntimeError(
                    f"Existing split does not match: {path}"
                )
        else:
            np.save(path, indices)

    #Save the dataset information needed for reproducibility
    metadata = {
        "dataset": "CIFAR-10",
        "random_seed": RANDOM_SEED,
        "image_shape": [32, 32, 3],
        "pixel_range_original": [0, 255],
        "num_classes": len(class_names),
        "class_names": class_names,
        "original_train_size": len(train_dataset),
        "training_size": len(train_indices),
        "validation_size": len(val_indices),
        "test_size": len(test_dataset),
        "training_distribution": train_counts,
        "validation_distribution": val_counts,
        "test_distribution": test_counts,
    }

    metadata_path = SPLIT_DIR / "dataset_metadata.json"
    metadata_text = json.dumps(metadata, indent=2) + "\n"

    if metadata_path.exists():
        if metadata_path.read_text() != metadata_text:
            raise RuntimeError("Existing dataset metadata does not match")
    else:
        metadata_path.write_text(metadata_text)

    print("\nDataset preparation completed successfully")
    print(json.dumps(metadata, indent=2))
    print("\nDataset checks: PASSED")
    print("Training/validation overlap: NONE")
    print("Saved split indices and metadata")


if __name__ == "__main__":
    main()
