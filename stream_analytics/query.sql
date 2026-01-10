-- 全量 raw
SELECT *
INTO [RawOutput]
FROM [TelemetryInput];

-- 过滤/警告
SELECT *
INTO [stream-output]
FROM [TelemetryInput]
WHERE maxenergy > 8000;
