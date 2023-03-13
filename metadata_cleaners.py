import argparse
import os

from text import cleaners


def main(args):
    cleaner = getattr(cleaners, args.cleaner)
    if not cleaner:
        raise Exception('Unknown cleaner: %s' % args.cleaner)

    train_path = args.train_path
    train_metadata = list()

    val_path = args.validation_path
    val_metadata = list()

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            wav_path, text, speaker = line.strip("\n").split("|")
            meta = [wav_path, cleaner(text), speaker]
            train_metadata.append(meta)

    path, ext = os.path.splitext(train_path)
    with open(path + "_cleaned" + ext, "w", encoding="utf-8") as f:
        for meta in train_metadata:
            f.write("|".join(meta) + "\n")

    if val_path is not None:
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                wav_path, text, speaker = line.strip("\n").split("|")
                meta = [wav_path, cleaner(text), speaker]
                val_metadata.append(meta)

        path, ext = os.path.splitext(val_path)
        with open(path + "_cleaned" + ext, "w", encoding="utf-8") as f:
            for meta in val_metadata:
                f.write("|".join(meta) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train_path",
        type=str,
        required=True,
        help="path to train meatadata",
    )
    parser.add_argument(
        "-v", 
        "--validation_path", 
        type=str, 
        default=None,
        help="path to validation meatadata",
    )
    parser.add_argument(
        "-c", 
        "--cleaner", 
        type=str, 
        default='korean_cleaners',
        help="cleaner for text cleaning",
    )

    args = parser.parse_args()
    main(args)
