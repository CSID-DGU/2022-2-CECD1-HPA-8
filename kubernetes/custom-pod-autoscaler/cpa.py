from kubernetes import client, config, watch
from prometheus import get_5m_cpu_avg
import time

# Connect to Kubernetes API
config.load_kube_config()  # change to load_incluster_config() in cluster
v1 = client.CoreV1Api()
api = client.AppsV1Api()


def scale_deployment(name, replicas):
    deployment = api.read_namespaced_deployment(name=name, namespace='example')
    deployment.spec.replicas = replicas
    api.patch_namespaced_deployment_scale(name = name, namespace = 'example', body = deployment)
    print("Deployment {} scaled to {} replicas".format(name, replicas))


if __name__ == '__main__':
    while True:
        pods = v1.list_namespaced_pod(namespace = "example")
        for pod in pods.items:
            if pod.metadata.labels.get('app') == 'nginx': 
                cpu = get_5m_cpu_avg()
                if cpu > 50:
                    scale_deployment('nginx', 3)
                elif cpu < 10:
                    scale_deployment('nginx', 1)
        time.sleep(10)
                 