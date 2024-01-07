import os
import urllib.request
import json
import numpy as np
import torch

def download(url, filename, chunk_size=10**6):
    # Skip download if file already exists
    if os.path.isfile(filename):
        return

    # Create directory if it doesn't exist
    dirname = os.path.dirname(filename)

    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Download file in chunks and display progress
    with urllib.request.urlopen(url) as r, open(filename, "wb") as f:
        total_size = int(r.headers["Content-Length"])

        size = 0
        while size < total_size:
            data = r.read(chunk_size)

            if len(data) == 0:
                break

            f.write(data)
            size += len(data)

            percent = 100.0 * size / total_size

            scale = 1e-6 if size < 10**9 else 1e-9
            unit = "MB" if size < 10**9 else "GB"

            print(f"{percent:7.2f} % of {filename} downloaded ({size * scale:.3f}/{total_size * scale:.1f} {unit})")

        assert size == total_size, f"Downloaded of file {filename} incomplete, only {size} bytes of {total_size} bytes downloaded"


def load_safetensors(filename, device, new_dtype=None):
    dtypes = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
    }

    state_dict = {}

    with open(filename, "r+b") as f:
        header_size = int.from_bytes(f.read(8), byteorder="little")
        header = f.read(header_size).decode("utf-8")
        info = json.loads(header)
        after_header = f.tell()
        m = np.memmap(f)
        for name, value in info.items():
            if name.startswith("__"): continue
            dtype = dtypes[value["dtype"]]
            shape = value["shape"]
            start, end = value["data_offsets"]
            weights = m[after_header + start:after_header + end]
            weights = torch.from_numpy(weights).view(dtype).reshape(shape)
            if new_dtype is not None:
                weights = weights.type(new_dtype)
            state_dict[name] = weights.to(device)

    return state_dict
