# Basic Go makefile

GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

# exclude python from std builds
DIRS=`go list ./... | grep -v python`

all: build

build:
	@echo "GO111MODULE = $(value GO111MODULE)"
	$(GOBUILD) -v $(DIRS)

build_exe: ./examples/*
	@echo "GO111MODULE = $(value GO111MODULE)"
	@for f in $^; do  \
		DIR=$$(basename $${f}) && echo "" && echo "###########################" && \
		echo "example: $${DIR} in path: $${f}"; \
		cd $${f}; go build -v; cd ../../; \
    done

params_good: ./examples/*
	@echo "GO111MODULE = $(value GO111MODULE)"
	@for f in $^; do  \
		DIR=$$(basename $${f}) && echo "" && echo "###########################" && \
		echo "example: $${DIR} in path: $${f}"; \
		cd $${f}; go build -v; ./$${DIR} -nogui -Params.SaveAll -Params.Good; cd ../../; \
    done
	
test:
	@echo "GO111MODULE = $(value GO111MODULE)"
	export TEST_GPU=false; $(GOTEST) -v -tags multinet $(DIRS)

test_race:
	@echo "GO111MODULE = $(value GO111MODULE)"
	export TEST_GPU=false; $(GOTEST) -v -race -tags multinet $(DIRS)

test_gpu:
	@echo "GO111MODULE = $(value GO111MODULE)"
	export TEST_GPU=true; $(GOTEST) -v -tags multinet $(DIRS)

test_long:
	@echo "GO111MODULE = $(value GO111MODULE)"
	export TEST_LONG=true; $(GOTEST) -v -tags multinet $(DIRS)

test_long_gpu:
	@echo "GO111MODULE = $(value GO111MODULE)"
	export TEST_LONG=true TEST_GPU=true; $(GOTEST) -v -tags multinet $(DIRS)

clean:
	@echo "GO111MODULE = $(value GO111MODULE)"
	$(GOCLEAN) ./...

fmts:
	gofmt -s -w .

vet:
	@echo "GO111MODULE = $(value GO111MODULE)"
	$(GOCMD) vet $(DIRS) | grep -v unkeyed

tidy: export GO111MODULE = on
tidy:
	@echo "GO111MODULE = $(value GO111MODULE)"
	go mod tidy

# updates go.mod to master for all of the goki dependencies
# note: must somehow remember to do this for any other depend
# that we need to update at any point!
master: export GO111MODULE = on
master:
	@echo "GO111MODULE = $(value GO111MODULE)"
	go get -u github.com/emer/etable@master
	go get -u github.com/emer/emergent@master
	go get -u github.com/emer/empi@master
	go list -m all | grep emer
	go mod tidy

old:
	@echo "GO111MODULE = $(value GO111MODULE)"
	go list -u -m all | grep '\['

mod-update: export GO111MODULE = on
mod-update:
	@echo "GO111MODULE = $(value GO111MODULE)"
	go get -u ./...
	go mod tidy

# gopath-update is for GOPATH to get most things updated.
# need to call it in a target executable directory
gopath-update: export GO111MODULE = off
gopath-update:
	@echo "GO111MODULE = $(value GO111MODULE)"
	cd examples/ra25; go get -u ./...

release:
	$(MAKE) -C axon release

