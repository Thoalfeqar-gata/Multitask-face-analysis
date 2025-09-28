#!/usr/bin/env bash

# directory containing the .tar files
INPUT_DIR="data/datasets/FaceCaption/data/images"
# directory to extract into
OUT_DIR="data/datasets/FaceCaption/data/extracted images"

mkdir -p "$OUT_DIR"

for tarfile in "$INPUT_DIR"/*.tar; do
  echo "Processing $tarfile ..."
  tar -xvf "$tarfile" --ignore-zeros --warning=no-unknown-keyword -C "$OUT_DIR" || {
    echo "Warning: failed to extract $tarfile"
  }
done

echo "Done processing all tar files."
