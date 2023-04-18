from prometheus_api_client import PrometheusConnect

# Connect to Prometheus
prom = PrometheusConnect(url='http://localhost:9090', disable_ssl=True)



def get_5m_cpu_avg():
    # Measure utilization of CPU
    query = 'sum(rate(container_cpu_usage_seconds_total{container_name!="POD"}[5m])) by (kubernetes_io_hostname) / sum(machine_cpu_cores) by (kubernetes_io_hostname) * 100'


    # get query result
    results = prom.custom_query(query)

    print(results)

    return float(results[0]['value'][1])

