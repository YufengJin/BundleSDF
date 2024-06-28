docker rm -f 6dsplats 
DIR=$(pwd)
xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name 6dsplats  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v /home:/home -v /tmp:/tmp -v /mnt:/mnt -v $DIR:$DIR -w $DIR --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE pearl/6dsplats:latest bash
