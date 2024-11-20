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
    help="Batch size used during training (ignored if not training)"
)

args = parser.parse_args()

DO_TRAINING = args.do_training
BATCH_SIZE = args.batch_size
MODEL = args.model
