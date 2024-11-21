import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="un",
    help="model and dataset to use, possible options are un, en, pt",
)

parser.add_argument(
    "--do-training",
    type=bool,
    default=False,
    help="Train the model, if not used just uses the pretrained one or fails",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size used during training",
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-4,
    help="Learning rate used during training",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of Epochs used during training",
)

parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-5,
    help="Weight decay used during training",
)

parser.add_argument(
    "--test-size",
    type=float,
    default=20,
    help="Percentage of the dataset to use for testing when evaluating the dataset",
)


args = parser.parse_args()

MODEL = args.model
DO_TRAINING = args.do_training
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
WEIGHT_DECAY = args.weight_decay
TEST_SIZE = args.test_size / 100
