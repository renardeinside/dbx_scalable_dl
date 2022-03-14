import datetime as dt
import logging
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from http import HTTPStatus
from typing import List, Optional

import pandas as pd
import requests


@dataclass
class MetricInfo:
    metric_name: str
    metric_type: str
    metric_units: str
    metric_value: Optional[float]


@dataclass
class HostInfo:
    cluster_name: str
    cluster_id: str
    host_name: str
    host_ip: str
    reported_dttm: dt.datetime
    metrics: List[MetricInfo]


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class MetricsCollector:
    def __init__(
        self,
        endpoint_url: Optional[str] = "http://localhost:8652",
        request_interval: Optional[int] = 5,
        total_requests: Optional[int] = None,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
        cluster_name: Optional[str] = "local",
        cluster_id: Optional[str] = "local",
    ):
        self._endpoint_url: str = endpoint_url
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._table_name = "metrics"
        self._thread: Optional[StoppableThread] = None
        self._request_interval = request_interval
        self._total_requests = total_requests
        self._cluster_id = cluster_id
        self._cluster_name = cluster_name
        self._logger = logger

    @staticmethod
    def _metrics_to_df(metrics: List[HostInfo]) -> pd.DataFrame:
        framed = pd.json_normalize([asdict(h) for h in metrics])
        framed = framed.explode("metrics")
        metrics_values = pd.DataFrame(
            framed["metrics"].values.tolist(), index=framed.index
        )
        _df = pd.concat([framed.drop("metrics", axis=1), metrics_values], axis=1)
        return _df

    def _append_to_storage(self, df: pd.DataFrame):
        df.to_sql(self._table_name, if_exists="append", index=False, con=self._conn)

    def _schedule(self, request_interval: int):
        total_requests_made = 0
        while True:
            self._logger.info(f"Collecting metrics, round number {total_requests_made}")
            time.sleep(request_interval)
            metrics = self._collect_metrics()
            _df = self._metrics_to_df(metrics)
            self._append_to_storage(_df)
            total_requests_made += 1
            self._logger.info(
                f"Collecting metrics, round number {total_requests_made} - done"
            )
            if self._total_requests and total_requests_made >= self._total_requests:
                self._logger.info(
                    "Total requests property is met, stopping metrics collection process"
                )
                break

    def start(self):
        self._logger.info("Starting the metrics collection process in a thread")
        self._thread = StoppableThread(
            target=self._schedule, args=(self._request_interval,)
        )
        self._thread.start()

    def finish(self):
        if self._thread:
            self._thread.stop()

    @staticmethod
    def _collect_metric_info(metric_payload: ET.Element) -> MetricInfo:
        try:
            metric_value = float(metric_payload.get("VAL"))
        except ValueError:
            metric_value = None

        return MetricInfo(
            metric_payload.get("NAME"),
            metric_payload.get("TYPE"),
            metric_payload.get("UNITS"),
            metric_value,
        )

    def _collect_host_info(self, host_payload: ET.Element):
        return HostInfo(
            cluster_name=self._cluster_name,
            cluster_id=self._cluster_id,
            host_name=host_payload.get("NAME"),
            host_ip=host_payload.get("IP"),
            reported_dttm=dt.datetime.fromtimestamp(int(host_payload.get("REPORTED"))),
            metrics=[
                MetricsCollector._collect_metric_info(metric)
                for metric in host_payload.findall("METRIC")
            ],
        )

    def _collect_metrics(self) -> List[HostInfo]:
        resp = requests.get(self._endpoint_url)
        if resp.status_code == HTTPStatus.OK:
            tree: ET.Element = ET.fromstring(resp.text)
            hosts_info = [
                self._collect_host_info(host) for host in tree.findall(".//HOST")
            ]
            return hosts_info

    @property
    def metrics(self) -> pd.DataFrame:
        return pd.read_sql(f"select * from {self._table_name}", con=self._conn)
