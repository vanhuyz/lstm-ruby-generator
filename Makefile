COMPOSE = docker-compose

init:
	mkdir -p checkpoints
	mkdir -p tensorboard

train:
	python -m lstm.train

docker-train:
	$(COMPOSE) up --build -d train tensorboard

# docker-cpu-train:
# 	$(COMPOSE) -f docker-compose-cpu.yml up --build -d train

docker-retrain:
	$(COMPOSE) up --build --force-recreate -d train tensorboard

log:
	$(COMPOSE) logs -f --tail 10 train

stop:
	$(COMPOSE) stop train tensorboard

down:
	$(COMPOSE) down -v