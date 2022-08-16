#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# python local/utils/libri_labels.py data/ls_960/dev_clean.scp --output_dir data/ls_960 --output_name dev_clean

"""
Helper script to pre-compute embeddings for a flashlight (previously called wav2letter++) dataset
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}

    with open(args.text, "r") as text, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        for line in text:
            uttid, words = line.strip().split(maxsplit=1)

            print(words, file=wrd_out)
            print(
                " ".join(list(words.replace(" ", "|"))) + " |",
                file=ltr_out,
            )


if __name__ == "__main__":
    main()
