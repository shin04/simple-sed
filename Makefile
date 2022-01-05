SHELL=/bin/zsh

HOME_DIR:=$${HOME}
include .env
export $(shell sed 's/=.*//' .env)

build:
	docker build -t $(IMAGE_NAME) .

build-finetune:
	docker build -t sed-finetune -f Dockerfile.finetune .

build-cuml:
	docker build -t sed-cuml -f Dockerfile.cuml .

run:
	docker run -it \
		--env HDF5_USE_FILE_LOCKING='FALSE' \
		--shm-size=16g \
		--mount type=bind,source=$(HOME_DIR)$(MOUNT_PATH),target=/ml \
		--mount type=bind,source=$(HOME_DIR)$(AUDIO_PATH),target=/ml/dataset/audio \
		--mount type=bind,source=$(HOME_DIR)$(MODEL_PATH),target=/ml/models \
		--mount type=bind,source=$(HOME_DIR)$(RESULT_PATH),target=/ml/results \
		--mount type=bind,source=$(HOME_DIR)$(FEAT_PATH),target=/ml/dataset/feat \
		--name $(CONTAINER_NAME) \
		--gpus all \
		$(IMAGE_NAME) /bin/bash

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash

run-finetune:
	docker run -it \
		--env HDF5_USE_FILE_LOCKING='FALSE' \
		--shm-size=16g \
		--mount type=bind,source=$(HOME_DIR)$(MOUNT_PATH),target=/ml \
		--mount type=bind,source=$(HOME_DIR)$(AUDIO_PATH),target=/ml/dataset/audio \
		--mount type=bind,source=$(HOME_DIR)$(MODEL_PATH),target=/ml/models \
		--mount type=bind,source=$(HOME_DIR)$(RESULT_PATH),target=/ml/results \
		--mount type=bind,source=$(HOME_DIR)$(FEAT_PATH),target=/ml/dataset/feat \
		--name sed-finetune \
		--gpus all \
		sed-finetune /bin/bash

attach-finetune:
	docker exec -it sed-finetune /bin/bash

feature_visualize:
	docker run -it \
		--shm-size=16g \
		--mount type=bind,source=$(HOME_DIR)/work/sed/src/analyze,target=/work/analyze \
		--mount type=bind,source=$(HOME_DIR)$(FEAT_PATH),target=/work/dataset/feat \
		--mount type=bind,source=$(HOME_DIR)$(VISUALIZE_RESULT_PATH),target=/work/visualize_result \
		--name sed-feature-visualize \
		--gpus all \
		sed-cuml /bin/bash

attach_vis:
	docker exec -it sed-feature-visualize /bin/bash

pretrained_feature_vis:
	docker run -it \
		--shm-size=16g \
		--mount type=bind,source=$(HOME_DIR)/work/sed/src/analyze,target=/work/analyze \
		--mount type=bind,source=$(HOME_DIR)/mrnas02/home/models/hubert/nmf_act_ite1,target=/work/dataset/feat \
		--mount type=bind,source=$(HOME_DIR)$(VISUALIZE_RESULT_PATH),target=/work/visualize_result \
		--name sed-pretrained-feature-visualize \
		--gpus all \
		sed-cuml /bin/bash

attach_pretrained_vis:
	docker exec -it sed-pretrained-feature-visualize /bin/bash