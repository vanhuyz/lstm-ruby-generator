COMPOSE = docker-compose

init:
	mkdir -p checkpoints
	mkdir -p tensorboard

train:
	python -m lstm.train

docker-train:
	$(COMPOSE) up --build -d train

# docker-cpu-train:
# 	$(COMPOSE) -f docker-compose-cpu.yml up --build -d train

docker-retrain:
	$(COMPOSE) up --build --force-recreate -d train

log:
	$(COMPOSE) logs -f --tail 10 train

stop:
	$(COMPOSE) stop train

down:
	$(COMPOSE) down -v