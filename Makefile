SHELL=/bin/zsh

HOME_DIR:=$${HOME}
include .env
export $(shell sed 's/=.*//' .env)

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it \
		--env HDF5_USE_FILE_LOCKING='FALSE' \
		--shm-size=16g \
		--mount type=bind,source=$(HOME_DIR)$(MOUNT_PATH),target=/ml \
		--mount type=bind,source=$(HOME_DIR)$(AUDIO_PATH),target=/ml/dataset/audio \
		--mount type=bind,source=$(HOME_DIR)$(MODEL_PATH),target=/ml/models \
		--name $(CONTAINER_NAME) \
		--gpus all \
		$(IMAGE_NAME) /bin/bash

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash
