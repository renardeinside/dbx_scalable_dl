from pyspark.sql import DataFrame, SparkSession

from dbx_scalable_dl.common import Job
from dbx_scalable_dl.utils import FileLoadingContext


class DataLoaderTask(Job):
    def _create_db(self):
        self.spark.sql(f"create database if not exists {self.conf['database']}")

    @staticmethod
    def _extract(spark: SparkSession, path: str) -> DataFrame:
        _df = spark.read.format("json").option("inferSchema", True).load(path)
        return _df

    @staticmethod
    def _transform(df: DataFrame) -> DataFrame:
        return df.drop("style").selectExpr(
            "cast(asin as string) as product_id",
            "cast(reviewerID as string) as user_id",
            "cast(overall as double) as rating",
        )

    def _save_data_to_table(self, source_url: str, output_table: str):
        with FileLoadingContext(
            source_url, self.conf["temp_directory_prefix"]
        ) as output_path:
            raw = self._extract(self.spark, output_path)
            transformed = self._transform(raw)
            transformed.write.format("delta").mode("overwrite").saveAsTable(
                output_table
            )

    def launch(self):
        self.logger.info("Starting the data loader job")
        self._create_db()
        self._save_data_to_table(
            self.conf["data_url"], f"{self.conf['database']}.{self.conf['table']}"
        )
        self.logger.info("Data loading successfully finished")


if __name__ == "__main__":
    DataLoaderTask().launch_with_metric_collection()
