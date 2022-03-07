#!/bin/bash

set -ex

echo 'running init script for AWS Cloud Watch integration'

function download_json_layout_jar() {
  echo 'downloading the json layout jar'
  wget -q -O /mnt/driver-daemon/jars/log4j12-json-layout-1.0.0.jar https://sa-iot.s3.ca-central-1.amazonaws.com/collateral/log4j12-json-layout-1.0.0.jar
  echo 'jar downloaded'
}

function install_cloudwatch_agent() {
  echo 'installing cloudwatch agent'
  cd /tmp

  # download cloudwatch agent
  wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/debian/amd64/latest/amazon-cloudwatch-agent.deb
  wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/debian/amd64/latest/amazon-cloudwatch-agent.deb.sig
  KEY=$(curl https://s3.amazonaws.com/amazoncloudwatch-agent/assets/amazon-cloudwatch-agent.gpg 2>/dev/null| gpg --import 2>&1 | cut -d: -f2 | grep 'key' | sed -r 's/\s*|key//g')
  FINGERPRINT=$(echo "9376 16F3 450B 7D80 6CBD 9725 D581 6730 3B78 9C72" | sed 's/\s//g')
  # verify signature
  if ! gpg --fingerprint $KEY| sed -r 's/\s//g' | grep -q "${FINGERPRINT}"; then
  echo "cloudwatch agent deb gpg key fingerprint is invalid"
  exit 1
  fi
  if ! gpg --verify ./amazon-cloudwatch-agent.deb.sig ./amazon-cloudwatch-agent.deb; then
  echo "cloudwatch agent signature does not match deb"
  exit 1
  fi
  sudo apt-get install ./amazon-cloudwatch-agent.deb
  echo 'cloudwatch agent installed successfully'
}

function configure_spark_metrics_namespace() {
  echo 'configuring spark metrics namespace'
  sudo bash -c "cat <<EOF >> /databricks/driver/conf/custom-spark-metrics-name-conf.conf
[driver] {
  spark.metrics.namespace = metrics
}
EOF"
  echo 'configuring spark metrics namespace - done'
}

function configure_statsd_export() {
  echo 'configuring statsd spark props'
  sudo bash -c "cat <<EOF >> /databricks/spark/conf/metrics.properties
*.sink.statsd.class=org.apache.spark.metrics.sink.StatsdSink
*.sink.statsd.host=localhost
*.sink.statsd.port=8125
*.sink.statsd.prefix=spark
master.source.jvm.class=org.apache.spark.metrics.source.JvmSource
worker.source.jvm.class=org.apache.spark.metrics.source.JvmSource
driver.source.jvm.class=org.apache.spark.metrics.source.JvmSource
executor.source.jvm.class=org.apache.spark.metrics.source.JvmSource
EOF"
  echo 'configuring statsd spark props - done'
}

function configure_cloudwatch_agent() {
  echo 'configuring cloudwatch agent'
  pip install j2cli

  if [ ! -z $DB_IS_DRIVER ] && [ $DB_IS_DRIVER = TRUE ] ; then
    j2 /dbfs/init_scripts/cloud-watch/driver-agent.j2 > /tmp/amazon-cloudwatch-agent.json
    sed -i '/^log4j.appender.publicFile.layout/ s/^/#/g' /home/ubuntu/databricks/spark/dbconf/log4j/driver/log4j.properties
    sed -i '/log4j.appender.publicFile=com.databricks.logging.RedactionRollingFileAppender/a log4j.appender.publicFile.layout=com.databricks.labs.log.appenders.JsonLayout' /home/ubuntu/databricks/spark/dbconf/log4j/driver/log4j.properties
  else
    j2 /dbfs/init_scripts/cloud-watch/executor-agent.j2 > /tmp/amazon-cloudwatch-agent.json
  fi
  echo 'configuring cloudwatch agent - done'
}

function start_cloudwatch_agent() {
  echo 'starting the cloudwatch agent'
  sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/tmp/amazon-cloudwatch-agent.json -s
  sudo systemctl enable amazon-cloudwatch-agent
  sleep 10
  /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a status -m ec2
  cat /opt/aws/amazon-cloudwatch-agent/logs/amazon-cloudwatch-agent.log
  cat /opt/aws/amazon-cloudwatch-agent/logs/configuration-validation.log
  echo 'starting the cloudwatch agent - done'

}

if [ $DB_IS_DRIVER = TRUE ] ; then
  echo 'running init script on the driver'

  # this is logging setup
  download_json_layout_jar

  # this is metrics setup
  configure_spark_metrics_namespace

  # cloudwatch-agent related settings
  configure_cloudwatch_agent
  install_cloudwatch_agent
  start_cloudwatch_agent

  echo 'running init script on the driver - done'
fi

