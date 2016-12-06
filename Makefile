COMPOSE = docker-compose
TODAY = `date +%Y%m%d`

init:
	mkdir -p checkpoints
	mkdir -p tensorboard

train:
	python -m lstm.train

docker-train:
	bash gen_env.sh
	$(COMPOSE) up --build -d train tensorboard

# docker-cpu-train:
# 	$(COMPOSE) -f docker-compose-cpu.yml up --build -d train

docker-retrain:
	bash gen_env.sh
	$(COMPOSE) up --build --force-recreate -d train tensorboard

log:
	$(COMPOSE) logs -f --tail 10 train

bash:
	$(COMPOSE) exec train /bin/bash

stop:
	$(COMPOSE) stop train tensorboard

down:
	$(COMPOSE) down -v