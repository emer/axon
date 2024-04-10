# Go testing

# note: have to prevent regular tests from running using -run not
bench:
	go test -v -bench Benchmark -run not

bench-gpu:
	export TEST_GPU=true; go test -v -bench Benchmark -run not
	
test:
	go test -v

test-gpu:
	export TEST_GPU=true;  go test -v
	
race:
	go test -race

debug:
	export TEST_DEBUG=true; go test -v
	
	
