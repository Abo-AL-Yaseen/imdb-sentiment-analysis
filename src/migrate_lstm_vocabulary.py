"""Create a locale-independent copy of this project's saved LSTM archive.

The original Keras 3.15.1 vocabulary assets were saved as CP1252 text.
Embedding the identical ordered vocabulary in the JSON configuration uses
Keras's supported vocabulary constructor argument and avoids locale-dependent
asset reads. No model fitting, weight serialization, or TensorFlow is required.
"""

import argparse
import io
import json
from pathlib import Path
import zipfile


VOCABULARY_ASSETS = (
    "assets/layers/text_vectorization/vocabulary.txt",
    "assets/layers/text_vectorization/_lookup_layer/vocabulary.txt",
)


def migrate(source: Path, destination: Path) -> int:
    """Migrate the inspected archive layout, refusing to replace any file."""
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    buffer = io.BytesIO()
    with zipfile.ZipFile(source, "r") as original:
        config = json.loads(original.read("config.json"))
        layers = config["config"]["layers"]
        vectorizers = [layer for layer in layers
                       if layer["class_name"] == "TextVectorization"]
        if len(vectorizers) != 1:
            raise ValueError("Expected exactly one TextVectorization layer.")
        settings = vectorizers[0]["config"]
        if settings["vocabulary"] is not None or settings["output_mode"] != "int":
            raise ValueError("Expected the original integer-vectorization model.")
        assets = [original.read(name) for name in VOCABULARY_ASSETS]
        if assets[0] != assets[1]:
            raise ValueError("The two original vocabulary assets disagree.")
        # Match Keras load_assets(), including its treatment of trailing lines.
        vocabulary = assets[0].decode("cp1252", errors="strict").splitlines()
        while vocabulary and vocabulary[-1] == "":
            vocabulary.pop()
        if (len(vocabulary) != settings["vocabulary_size"]
                or vocabulary[:2] != ["", "[UNK]"]
                or len(set(vocabulary)) != len(vocabulary)):
            raise ValueError("Unexpected vocabulary size, special tokens, or duplicates.")
        settings["vocabulary"] = vocabulary
        config_bytes = json.dumps(config, ensure_ascii=True).encode("ascii")

        with zipfile.ZipFile(buffer, "w") as portable:
            portable.comment = original.comment
            for entry in original.infolist():
                if entry.filename in VOCABULARY_ASSETS:
                    continue  # Vocabulary now lives in config, as Keras supports.
                payload = (config_bytes if entry.filename == "config.json"
                           else original.read(entry.filename))
                portable.writestr(entry, payload)

    # Exclusive creation also prevents replacement if another process creates it.
    with destination.open("xb") as output:
        output.write(buffer.getvalue())
    return len(vocabulary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = migrate(args.source, args.destination)
    print(f"Created {args.destination} with {count} vocabulary entries; source unchanged.")
