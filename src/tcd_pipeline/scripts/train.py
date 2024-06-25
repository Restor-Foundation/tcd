import argparse

from tcd_pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("model")
    parser.add_argument(
        "options",
        nargs=argparse.REMAINDER,
        help="Configuration options to pass to the training pipeline, formatted as <key>=<value> with spaces between options.",
    )

    args = parser.parse_args()

    pipeline = Pipeline(args.model, args.options)
    res = pipeline.train()


if __name__ == "__main__":
    main()
