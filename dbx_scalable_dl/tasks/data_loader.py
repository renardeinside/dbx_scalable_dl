import pathlib
import tempfile
from typing import Optional

import requests
from dbx_scalable_dl.common import Job
from tqdm import tqdm


class FileLoadingContext:
    @staticmethod
    def _get_chunk_size(chunk_size: int) -> int:
        return int(chunk_size * 1024 * 1024)

    def __init__(self, url: str, prefix: Optional[str] = None, chunk_size: int = 1):
        self._temp_dir = tempfile.TemporaryDirectory(prefix)
        self._chunk_size = self._get_chunk_size(chunk_size)
        request = requests.get(url, stream=True)
        chunk_iterator = request.iter_content(chunk_size=self._chunk_size)
        self._output_file = pathlib.Path(self._temp_dir.name) / "output.json.gz"
        with open(self._output_file, "wb") as f:
            for chunk in tqdm(chunk_iterator):
                f.write(chunk)
                f.flush()

    def __enter__(self) -> str:
        return str(self._output_file)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._temp_dir.cleanup()


class DataLoaderTask(Job):
    def _create_db(self):
        _db = self.conf["database"]
        if _db not in [d.name for d in self.spark.catalog.listDatabases()]:
            self.spark.sql(f"create database if not exists {_db}")

    def launch(self):
        self.logger.info("Starting the data loader job")
        self._create_db()

        with FileLoadingContext(self.conf["data_url"]) as output_file:
            _df = (
                self.spark.read.format("json")
                .option("inferSchema", True)
                .load(output_file)
                .drop("style")
            )
            _df.write.format("delta").mode("overwrite").saveAsTable(
                f"{self.conf['database']}.reviews"
            )
        self.logger.info("Data loading successfully finished")


if __name__ == "__main__":
    DataLoaderTask().launch()
