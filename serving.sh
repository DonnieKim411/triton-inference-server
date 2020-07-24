#!/bin/sh

home_path="/home/donnie/Documents/Repos/triton-inference-server"
docker_img="nvcr.io/nvidia/tritonserver:20.06-py3" 
#docker_img="tritonserver:latest"

# for server debugging purpose
docker run --rm --shm-size=1g --ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-p 8000:8000 -p 8001:8001 -p 8002:8002 \
	-v $home_path/docs/examples/model_repository:/models \
	-v $home_path/trace:/tmp\
	$docker_img tritonserver --model-repository=/models --trace-file=/tmp/trace.json --trace-rate=1 --trace-level=MAX


#docker run --rm --shm-size=1g --ulimit memlock=-1 \
#	--ulimit stack=67108864 \
#	-p 8000:8000 -p 8001:8001 -p 8002:8002 \
#	-v $home_path/docs/examples/model_repository:/models \
#	$docker_img tritonserver --model-repository=/models

#docker run --gpus=1 --rm --shm-size=1g --ulimit memlock=-1 \
#	--ulimit stack=67108864 \
#	-p 8000:8000 -p 8001:8001 -p 8002:8002 \
#	-v $home_path/docs/examples/model_repository:/models \
#	$docker_img tritonserver --model-repository=/models


