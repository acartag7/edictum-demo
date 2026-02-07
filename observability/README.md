# Observability

OpenTelemetry configuration and Grafana dashboard for Edictum-governed agents.

## Files

| File | Description |
|------|-------------|
| `otel_config.py` | OTel setup — auto-detects OTLP, console, or disabled mode |
| `grafana/edictum-dashboard.json` | Pre-built Grafana dashboard |

## OTel modes

1. **Grafana Cloud / OTLP** — set `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS`
2. **Local collector** — set `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`
3. **Console** — set `EDICTUM_OTEL_CONSOLE=1` (prints spans/metrics to terminal)
4. **Disabled** — if nothing is set, OTel is silently skipped

## Grafana dashboard

Import `grafana/edictum-dashboard.json` into Grafana. Requires:
- **Tempo** datasource (traces)
- **Prometheus** datasource (metrics via OTLP)

Panels:
- Governance decisions (allowed vs denied over time)
- Denial rate gauge
- Denials by tool (bar chart)
- Recent governance traces
- Denied traces only
- Tool execution latency histogram
