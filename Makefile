SHELL=/bin/zsh

include .env
export $(shell sed 's/=.*//' .env)

build:
	docker build -t $(IMAGE_NAME) .

build-finetune:
	docker build -t $(IMAGE_NAME)-finetune -f Dockerfile.finetune .

build-cuml:
	docker build -t $(IMAGE_NAME)-cuml -f Dockerfile.cuml .

run:
	docker run -it \
		--env HDF5_USE_FILE_LOCKING='FALSE' \
		--shm-size=16g \
		--mount type=bind,source=$(WORK_DIR),target=/ml \
		--mount type=bind,source=$(AUDIO_PATH),target=/ml/dataset/audio \
		--mount type=bind,source=$(FEAT_PATH),target=/ml/dataset/feat \
		--mount type=bind,source=$(MODEL_PATH),target=/ml/models \
		--mount type=bind,source=$(RESULT_PATH),target=/ml/results \
		--name $(CONTAINER_NAME) \
		--gpus all \
		$(IMAGE_NAME) /bin/bash

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash

train:
	docker run -it \
		--env HDF5_USE_FILE_LOCKING='FALSE' \
		--shm-size=16g \
		--mount type=bind,source=$(WORK_DIR),target=/ml \
		--mount type=bind,source=$(AUDIO_PATH),target=/ml/dataset/audio \
		--mount type=bind,source=$(FEAT_PATH),target=/ml/dataset/feat \
		--mount type=bind,source=$(MODEL_PATH),target=/ml/models \
		--mount type=bind,source=$(RESULT_PATH),target=/ml/results \
		--name $(CONTAINER_NAME)-train \
		--gpus all \
		--workdir /ml/src \
		$(IMAGE_NAME) /bin/bash -c "python3 run.py"

run-finetune:
	docker run -it \
		--env HDF5_USE_FILE_LOCKING='FALSE' \
		--shm-size=16g \
		--mount type=bind,source=$(MOUNT_PATH),target=/ml \
		--mount type=bind,source=$(AUDIO_PATH),target=/ml/dataset/audio \
		--mount type=bind,source=$(FEAT_PATH),target=/ml/dataset/feat \
		--mount type=bind,source=$(MODEL_PATH),target=/ml/models \
		--mount type=bind,source=$(RESULT_PATH),target=/ml/results \
		--name $(CONTAINER_NAME)-finetune \
		--gpus all \
		$(IMAGE_NAME)-finetune /bin/bash

attach-finetune:
	docker exec -it $(CONTAINER_NAME)-finetune /bin/bash

feature_visualize:
	docker run -it \
		--shm-size=16g \
		--mount type=bind,source=/work/sed/src/analyze,target=/work/analyze \
		--mount type=bind,source=$(FEAT_PATH),target=/work/dataset/feat \
		--mount type=bind,source=$(RESULT_PATH),target=/work/visualize_result \
		--name $(CONTAINER_NAME)-feature-visualize \
		--gpus all \
		$(IMAGE_NAME)-cuml /bin/bash

attach_vis:
	docker exec -it $(CONTAINER_NAME)-feature-visualize /bin/bash

pretrained_feature_vis:
	docker run -it \
		--shm-size=16g \
		--mount type=bind,source=/work/sed/src/analyze,target=/work/analyze \
		--mount type=bind,source=/mrnas02/home/models/hubert/nmf_act_ite1,target=/work/dataset/feat \
		--mount type=bind,source=$(VISUALIZE_RESULT_PATH),target=/work/visualize_result \
		--name $(CONTAINER_NAME)-pretrained-feature-visualize \
		--gpus all \
		$(IMAGE_NAME)-cuml /bin/bash

attach_pretrained_vis:
	docker exec -it $(CONTAINER_NAME)-pretrained-feature-visualize /bin/bash