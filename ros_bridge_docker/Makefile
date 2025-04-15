all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build-noetic          - build with base of noetic"
	@echo "   2. make build-foxy            - build with base of foxy"
	# @echo "   1. make pull             - pull all images"
	# @echo "   1. make clean            - remove all images"
	@echo ""

build-noetic:
	docker build --tag=wayfinding/ros1_bridge:noetic --file=./noetic-base/Dockerfile .
build-foxy:
	docker build --tag=wayfinding/ros1_bridge:foxy --file=./foxy-base/Dockerfile .