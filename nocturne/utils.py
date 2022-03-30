import pathlib
import tempfile
from typing import Optional
import requests

from tqdm import tqdm


class FileLoadingContext:
    @staticmethod
    def _get_chunk_size_in_bytes(chunk_size: int) -> int:
        return int(chunk_size * 1024 * 1024)

    def __init__(self, url: str, _dir: Optional[str] = None, chunk_size_mb: int = 1):
        self._temp_dir = tempfile.TemporaryDirectory(dir=_dir)
        chunk_size_in_bytes = self._get_chunk_size_in_bytes(chunk_size_mb)
        request = requests.get(url, stream=True)
        content_length = int(request.headers.get("Content-length"))
        progress_bar = tqdm(total=content_length, unit="iB", unit_scale=True)
        chunk_iterator = request.iter_content(chunk_size=chunk_size_in_bytes)
        self._output_file = pathlib.Path(self._temp_dir.name) / "output.json.gz"
        with open(self._output_file, "wb") as f:
            for chunk in chunk_iterator:
                progress_bar.update(chunk_size_in_bytes)
                f.write(chunk)
                f.flush()
            progress_bar.close()

    def __enter__(self) -> str:
        return str(self._output_file).replace("/dbfs/", "dbfs:/")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._temp_dir.cleanup()
