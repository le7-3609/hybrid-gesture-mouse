import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Dynamically append the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.logger import get_logger
from config.settings import CLASSES, DATASET_PATH, MODEL_PATH

logger = get_logger("train")


def generate_synthetic_data(filepath, num_samples_per_class=100):
    """
    Generates a synthetic, distinct dataset of normalized hand features to verify
    the compilation, training, and loading flow without webcam recordings.
    """
    logger.info(f"Generating synthetic dataset at '{filepath}'...")

    np.random.seed(42)
    data = []

    # 63 features representing 21 landmarks (x, y, z)
    num_features = 63

    for label in CLASSES.keys():
        # Create a unique 'base' posture for each class
        base_vector = np.zeros(num_features)

        if label == 0:  # Idle (Relaxed hand)
            base_vector[::3] = np.linspace(0.0, 0.5, 21)  # gradual x spread
            base_vector[1::3] = np.linspace(
                0.0, 0.8, 21
            )  # fingers extended straight
        elif label == 1:  # Move (Index extended, others closed)
            base_vector[8 * 3 + 1] = -1.0  # index finger tip high up
            base_vector[12 * 3 + 1] = 0.2  # middle closed
            base_vector[16 * 3 + 1] = 0.2  # ring closed
            base_vector[20 * 3 + 1] = 0.2  # pinky closed
        elif label == 2:  # Click (Index & Thumb tip pinched close)
            # Thumb tip (4) and Index tip (8) are extremely close
            base_vector[4 * 3 : 4 * 3 + 3] = [0.1, -0.4, 0.0]
            base_vector[8 * 3 : 8 * 3 + 3] = [0.12, -0.38, 0.0]
        elif label == 3:  # Drag (Fist / Pinch & Hold)
            base_vector[4 * 3 : 4 * 3 + 3] = [0.1, -0.3, 0.0]
            base_vector[8 * 3 : 8 * 3 + 3] = [0.11, -0.29, 0.0]
            base_vector[12 * 3 : 12 * 3 + 3] = [
                0.08,
                -0.28,
                0.0,
            ]  # other fingers also close
        elif label == 4:  # Scroll Up (Index, Middle & Ring fingers extended)
            base_vector[8 * 3 + 1] = -1.0  # index up
            base_vector[12 * 3 + 1] = -0.9  # middle up
            base_vector[16 * 3 + 1] = -0.8  # ring up
            base_vector[20 * 3 + 1] = 0.2  # pinky closed
        elif label == 5:  # Scroll Down (Index, Middle, Ring & Pinky fingers extended)
            base_vector[8 * 3 + 1] = -1.0  # index up
            base_vector[12 * 3 + 1] = -0.9  # middle up
            base_vector[16 * 3 + 1] = -0.8  # ring up
            base_vector[20 * 3 + 1] = -0.7  # pinky up
        elif label == 6:  # Right Click (Index & Middle finger extended close together)
            base_vector[8 * 3 + 1] = -1.0  # index up
            base_vector[12 * 3 + 1] = -0.9  # middle up
            base_vector[16 * 3 + 1] = 0.2  # ring closed
            base_vector[20 * 3 + 1] = 0.2  # pinky closed
        elif label == 7:  # Double Click (Index & Middle finger forming a "V")
            base_vector[8 * 3 + 1] = -1.0  # index up
            base_vector[8 * 3 + 0] = -0.3  # index left
            base_vector[12 * 3 + 1] = -0.9  # middle up
            base_vector[12 * 3 + 0] = 0.3  # middle right
            base_vector[16 * 3 + 1] = 0.2  # ring closed
            base_vector[20 * 3 + 1] = 0.2  # pinky closed

        # Generate samples by adding Gaussian noise to the base vector
        for _ in range(num_samples_per_class):
            noise = np.random.normal(0, 0.05, num_features)
            sample = base_vector + noise
            data.append([label] + sample.tolist())

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Write to CSV
    header = ["label"] + [f"feat_{i}" for i in range(num_features)]
    df = pd.DataFrame(data, columns=header)
    df.to_csv(filepath, index=False)
    logger.info(f"Synthetic dataset saved with {len(df)} samples.")


def train_model(csv_path, model_path):
    """
    Loads dataset, trains a Random Forest Classifier, prints evaluation reports,
    and serializes the trained model.
    """
    if not os.path.exists(csv_path):
        logger.error(f"Dataset file '{csv_path}' not found!")
        logger.info(
            "Please run 'python training/collect_data.py' to record your custom gestures first, or run 'python training/train.py --synthetic' to create a dummy test dataset."
        )
        return False

    logger.info(f"Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # Check data distribution
    class_counts = df["label"].value_counts().to_dict()
    logger.info("Data distribution by class:")
    for code, name in CLASSES.items():
        count = class_counts.get(code, 0)
        logger.info(f"  Class {code} ({name.upper()}): {count} samples")

    # Check if we have enough data to split
    min_samples = min(class_counts.values()) if class_counts else 0
    if len(class_counts) < len(CLASSES) or min_samples < 5:
        logger.warning("Some classes have very few samples.")
        logger.info(
            "For robust classification, collect at least 50+ samples per class."
        )

    # Extract features and targets
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # 80/20 train-test split with stratification to keep equal class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(class_counts) >= 2 and min_samples >= 2 else None,
    )

    logger.info(f"Training set size: {X_train.shape[0]} samples")
    logger.info(f"Testing set size: {X_test.shape[0]} samples")

    # Initialize Random Forest Classifier
    # High estimators + balanced class weights for stability
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, class_weight="balanced"
    )

    logger.info("Training Random Forest model...")
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Set Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    target_names = [CLASSES[i].upper() for i in sorted(np.unique(y))]
    logger.info(
        f"Classification Report:\n{classification_report(y_test, y_pred, target_names=target_names, zero_division=0)}"
    )

    # Confusion Matrix
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Serialize model using Pickle
    logger.info(f"Saving model to '{model_path}'...")
    if os.path.dirname(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info("Model training complete and serialized successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Hand Gesture Recognition Model"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic gesture data for testing",
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH, help="Path to gestures CSV file"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to save trained model"
    )

    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_data(args.dataset)

    train_model(args.dataset, args.model)
