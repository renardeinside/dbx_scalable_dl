from pyspark.sql import SparkSession


def get_num_executors(spark: SparkSession) -> int:
    """
    Returns number of available executors.
    For cases when driver and executor are on the same machine, it returns 1
    :return: num_executors, int >=1
    """
    tracker = spark.sparkContext._jsc.sc().statusTracker()  # noqa
    num_executors = max(1, len(tracker.getExecutorInfos()) - 1)
    return num_executors
