import os
import urllib.request

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

