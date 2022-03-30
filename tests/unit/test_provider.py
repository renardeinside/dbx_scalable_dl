from pyspark.sql import SparkSession, DataFrame
from nocturne.api.providers import ConverterProvider


def test_provider(
    spark: SparkSession, sample_ratings_dataset: DataFrame, petastorm_cache_dir: str
):
    provider = ConverterProvider(spark, cache_dir=f"file://{petastorm_cache_dir}")
    partitioned_converter = provider.get_dataset_converter(sample_ratings_dataset)

    assert len(partitioned_converter.file_urls) == provider.DEFAULT_NUM_REPARTITIONS

    non_partitioned_converter = provider.get_dataset_converter(
        sample_ratings_dataset, with_repartition=False
    )
    assert len(non_partitioned_converter) == sample_ratings_dataset.count()
