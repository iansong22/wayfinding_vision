# wayfinding_vision

To fix noetic-gpu dockerfile (make sure you are on noetic branch):
```
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
```
Also add `: public QObject` to `class _AdditionalTopicSubscriber` in `additional_topic_subscriber.h` from `spencer_tracking_rviz_plugin` to fix build error.

To build with no cached docker container layers, add `--no-cache` to the Makefile.

Changes to `run.sh` (not sure if necessary?):
```
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist :1 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod a+r $XAUTH

# Run the Docker container in interactive mode
echo "starting container..."
            # --env="QT_X11_NO_MITSHM=1" \
            # --env="XAUTHORITY=$XAUTH" \
            # --volume="$XAUTH:$XAUTH" \
docker run --rm -it \
            --env="DISPLAY=$DISPLAY" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
            $RUNTIME_ARGS \
            $NETWORK_ARGS \
            $EXTRA_DOCKER_ARGS \
            --env="QT_X11_NO_MITSHM=1" \
            --env="XAUTHORITY=$XAUTH" \
            --volume="$XAUTH:$XAUTH" \
            spencer/spencer_people_tracking:$IMAGE_VARIANT \
            $COMMAND
```