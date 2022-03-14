import time

from pytest_httpserver import HTTPServer

from dbx_scalable_dl.metrics_collector import MetricsCollector


def test_ganglia(ganglia_server: HTTPServer):
    collector = MetricsCollector(
        endpoint_url=ganglia_server.url_for("/"), request_interval=1, total_requests=3
    )
    collector.start()
    time.sleep(5)
    collector.finish()
    metrics = collector.metrics
    assert len(metrics) > 1
