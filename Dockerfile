FROM docker-registry.glpgs-dev.com/tensorflow/tensorflow-gpu

COPY . /tmp/lstm-ruby-generator
WORKDIR /tmp/lstm-ruby-generator

ENTRYPOINT ["python", "-m", "lstm.train"]
