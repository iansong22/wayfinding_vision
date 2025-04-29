all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build            - build images"
	@echo "   1. make build_drspaam    - build dr_spaam image"
	# @echo "   1. make pull             - pull all images"
	# @echo "   1. make clean            - remove all images"
	@echo ""

build:
	cd .. && \
	docker build --tag=spencer/spencer_people_tracking:noetic --file=.docker/noetic/Dockerfile .

build_drspaam:
	cd .. && \
	docker build --tag=dr_spaam --file=.docker/drspaam/Dockerfile .

