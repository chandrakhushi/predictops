-- Storage. Logs dominate volume: ZSTD on the message column plus LowCardinality
-- on host/level keeps them cheap, and ORDER BY (host, ts) makes per-host window reads fast.
CREATE TABLE metrics (
    ts DateTime, host LowCardinality(String),
    cpu Float32, mem Float32, disk Float32, p99_ms Float32
) ENGINE = MergeTree ORDER BY (host, ts) TTL ts + INTERVAL 3 DAY;

CREATE TABLE logs (
    ts DateTime, host LowCardinality(String), level LowCardinality(String), msg String CODEC(ZSTD(3))
) ENGINE = MergeTree ORDER BY (host, ts) TTL ts + INTERVAL 3 DAY;

CREATE TABLE incidents (ts DateTime, host LowCardinality(String), kind LowCardinality(String))
ENGINE = MergeTree ORDER BY ts;

CREATE TABLE predictions (ts DateTime, host LowCardinality(String), risk Float32, alert UInt8, reason String)
ENGINE = MergeTree ORDER BY (host, ts) TTL ts + INTERVAL 3 DAY;

-- Ingestion: ClickHouse consumes the Kafka topics itself (Kafka engine + materialized view),
-- so there is no consumer service to write, scale or babysit.
CREATE TABLE metrics_queue AS metrics ENGINE = Kafka SETTINGS
    kafka_broker_list = 'redpanda:9092', kafka_topic_list = 'metrics', kafka_group_name = 'clickhouse',
    kafka_format = 'JSONEachRow', kafka_flush_interval_ms = 1000, kafka_skip_broken_messages = 100;
CREATE TABLE logs_queue AS logs ENGINE = Kafka SETTINGS
    kafka_broker_list = 'redpanda:9092', kafka_topic_list = 'logs', kafka_group_name = 'clickhouse',
    kafka_format = 'JSONEachRow', kafka_flush_interval_ms = 1000, kafka_skip_broken_messages = 100;
CREATE TABLE incidents_queue AS incidents ENGINE = Kafka SETTINGS
    kafka_broker_list = 'redpanda:9092', kafka_topic_list = 'incidents', kafka_group_name = 'clickhouse',
    kafka_format = 'JSONEachRow', kafka_flush_interval_ms = 1000, kafka_skip_broken_messages = 100;

CREATE MATERIALIZED VIEW metrics_mv TO metrics AS SELECT * FROM metrics_queue;
CREATE MATERIALIZED VIEW logs_mv TO logs AS SELECT * FROM logs_queue;
CREATE MATERIALIZED VIEW incidents_mv TO incidents AS SELECT * FROM incidents_queue;
