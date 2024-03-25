image := image-fluidsim
tag := $(shell date -I'date'| tr -d "[:punct:]")

define STR_HELP
This makefile can be used to build images and start containers.

Examples
--------
$ make start  # starts Docker service
$ make pull   # pulls a Docker image fron Docker hub
$ make build  # by default builds python3-stable image
$ make run
$ make build image=python3-stable
$ make cleanall  # clean (unnecessary, not all) containers and images

endef

export STR_HELP

.PHONY: help

help:
	@echo "$$STR_HELP"

start:
	systemctl start docker

build:
	docker build -f Dockerfile -t fluiddyn/$(image) ..
	docker tag fluiddyn/$(image) fluiddyn/$(image):$(tag)

list:
	@printf "\nImages: "
	docker images
	@printf "\nContainers: "
	docker ps

cleancontainers:
	@echo "Clean exited containers."
	for cont in $$(docker ps -a | awk 'NR>1{print $$1}'); do docker stop $$cont; docker rm $$cont; done

cleanimages:
	@echo "Clean dangling images with no tag."
	for img in $$(docker images -qf "dangling=true"); do docker rmi -f $$img; done

cleanall: cleancontainers cleanimages cleanmako

run:
	docker run --name $(image) --restart always -it fluiddyn/$(image) bash
