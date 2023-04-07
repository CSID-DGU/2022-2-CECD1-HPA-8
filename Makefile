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

delete:
	kubectl delete -f ./kubernetes/ingress.yaml
	kubectl delete -f ./kubernetes/service_deploy.yaml
	kubectl delete -f ./kubernetes/config.yaml
	kubectl delete -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.0.0/deploy/static/provider/cloud/deploy.yaml
	
	
	
