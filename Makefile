build ::
	docker build -t pytorch:dev .
run ::
	docker run -it --rm --name pytorch -p 8097:8097 pytorch:dev 
start-visdom-server ::
	python -m visdom.server