from dbx_scalable_dl.common import Job
from dbx_scalable_dl.utils import FileLoadingContext


class DataLoaderTask(Job):
    def _create_db(self):
        _db = self.conf["database"]
        if _db not in [d.name for d in self.spark.catalog.listDatabases()]:
            self.spark.sql(f"create database if not exists {_db}")

    def _save_data_to_table(self, source_url: str, output_table: str):
        with FileLoadingContext(source_url) as output_file:
            _df = (
                self.spark.read.format("json")
                .option("inferSchema", True)
                .load(output_file)
                .drop("style")
            )
            _df.write.format("delta").mode("overwrite").saveAsTable(output_table)

    def launch(self):
        self.logger.info("Starting the data loader job")
        self._create_db()
        self._save_data_to_table(
            self.conf["data_url"], f"{self.conf['database']}.reviews"
        )
        self.logger.info("Data loading successfully finished")


if __name__ == "__main__":
    DataLoaderTask().launch()
