#!/bin/bash

tasks=(
    "abalone"
    "breastw"
    "spambase"
    "heloc"
    "beans"
    "ionosphere"
    "breastcancer"
    "btsc"
    "spf"
    "wine"
    "diggle"
    "iris"
    "vehicle"
    "autouniv"
    )

echo "Starting script with ${#tasks[@]} tasks"

# Function to run both commands for a task
run_task() {
    local task=$1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting task: $task"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Running $task with target: model, testset"
    python 2_compute_explanations_torch.py --task "$task" --target model --expltarget testset
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed $task with target: model, testset"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Running $task with target: model, validationset"
    python 2_compute_explanations_torch.py --task "$task" --target model --expltarget validationset
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed $task with target: model, validationset"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Finished task: $task"
}

# Function to run tasks in parallel
run_parallel() {
    local -a pids=()
    for task in "$@"; do
        run_task "$task" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid
    done
}

# Loop through tasks, running two at a time
for ((i=0; i<${#tasks[@]}; i+=2)); do
    if [ $((i+1)) -lt ${#tasks[@]} ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Launching tasks in parallel: ${tasks[i]} and ${tasks[i+1]}"
        run_parallel "${tasks[i]}" "${tasks[i+1]}"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Launching final task: ${tasks[i]}"
        run_parallel "${tasks[i]}"
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - All tasks completed"