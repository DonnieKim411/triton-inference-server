#!/bin/sh

home_path="/home/donnie/Documents/Repos/triton-inference-server"
docker_img="nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk" 

docker run -it --rm --net=host \
	-v $home_path:/mnt\
	--name triton_client\
	$docker_img


