build:
	docker build -t my-flask-app:latest ./flask

invoke:
	curl http://0.0.0.0/change/1/34

namespace:
	kubectl create namespace example

ingress-controller:
	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.0.0/deploy/static/provider/cloud/deploy.yaml

ingress:
	kubectl apply -f ./kubernetes/ingress.yaml

config:
	kubectl apply -f ./kubernetes/config.yaml

run-kube:
	kubectl apply -f ./kubernetes/service_deploy.yaml

metrics:
	kubectl apply -f ./kubernetes/components.yaml

state-metric:
	kubectl apply -f ./kubernetes/kube-state-metrics/.

node-exporter:
	kubectl apply -f ./kubernetes/node-exporter/.

prometheus:
	kubectl create -f ./kubernetes/prometheus/bundle.yaml 
	kubectl apply -f ./kubernetes/prometheus/prometheus.yaml
	kubectl apply -f ./kubernetes/prometheus/clusterRole.yaml
	kubectl apply -f ./kubernetes/prometheus/clusterRoleBinding.yaml
	kubectl apply -f ./kubernetes/prometheus/serviceAccount.yaml

serviceMonitor:
	kubectl apply -f ./kubernetes/prometheus/serviceMonitor/.

grafana:
	kubectl apply -f ./kubernetes/prometheus/grafana/grafana.yaml 

delete:
	kubectl delete -f ./kubernetes/ingress.yaml
	kubectl delete -f ./kubernetes/service_deploy.yaml
	kubectl delete -f ./kubernetes/config.yaml
	kubectl delete -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.0.0/deploy/static/provider/cloud/deploy.yaml
	kubectl delete -f ./kubernetes/components.yaml
	kubectl delete -f ./kubernetes/node-exporter/.
	kubectl delete -f ./kubernetes/prometheus/bundle.yaml 
	kubectl delete -f ./kubernetes/prometheus/prometheus.yaml
	kubectl delete -f ./kubernetes/prometheus/clusterRole.yaml
	kubectl delete -f ./kubernetes/prometheus/clusterRoleBinding.yaml
	kubectl delete -f ./kubernetes/prometheus/serviceAccount.yaml
	kubectl delete -f ./kubernetes/prometheus/serviceMonitor/.
	kubectl delete -f ./kubernetes/prometheus/grafana/grafana.yaml 
