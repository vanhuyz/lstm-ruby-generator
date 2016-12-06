FROM docker-registry.glpgs-dev.com/tensorflow/tensorflow-gpu

COPY . /tmp/lstm-generator
WORKDIR /tmp/lstm-generator

ENTRYPOINT ["python", "-m", "lstm.train"]
