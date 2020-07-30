#!/bin/bash

# usage: bash concurrent_call_testing.sh num_batch num_calls --no-save_logging
# num_batch: number of images to be fed per call
# num_calls: number of times to call the python script
# --no-save_logging: a flag to be used for concurrent call testing. Make sure to call it.

# repeat python calls as many time as requested
concurrent_call=""

for i in $(seq "$2");do
	command='python http_call_tester.py --url=localhost:8000 --batch_size '"$1 & "
	concurrent_call+=$command
done

# echo $concurrent_call

# evluate the expression
eval $concurrent_call

wait
echo jobs done!
