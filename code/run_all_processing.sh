#!/bin/bash
# --------------------------
# User parameters
# --------------------------
# Default values
PATH_DATA="$PATH_DATA" #Defaults from environment
PATH_CODE="$PATH_CODE" #Defaults from environment
IDs=() # empty  → process all participants
TASKS=() # empty → process all tasks
RUN_PREPROSS=false
RUN_DENOISING=false
RUN_FIRSTLEVEL=false
RUN_SECONDLEVEL=false
RUN_FIGURES=false
RUN_SHIMMING=false
REDO=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --path-data) PATH_DATA="$2"; shift 2 ;;
        --path-code) PATH_CODE="$2"; shift 2 ;;
        --ids) shift; while [[ $# -gt 0 && "$1" != --* ]]; do IDs+=("$1"); shift; done ;;
        --tasks) shift; while [[ $# -gt 0 && "$1" != --* ]]; do TASKS+=("$1"); shift; done ;;
        --preprocess) RUN_PREPROSS=true; shift;;
        --denoising) RUN_DENOISING=true; shift;;
        --firstlevel) RUN_FIRSTLEVEL=true; shift;;
        --secondlevel) RUN_SECONDLEVEL=true; shift;;
        --shimming) RUN_SHIMMING=true; shift;;
        --redo) REDO=true; shift;;
      *) echo "Unknown argument $1"; exit 1 ;;
    esac
done

if [ "${RUN_PREPROSS}" = false ] && \
   [ "${RUN_DENOISING}" = false ] && \
   [ "${RUN_FIRSTLEVEL}" = false ] && \
   [ "${RUN_SECONDLEVEL}" = false ] && \
   [ "${RUN_SHIMMING}" = false ] && \
   [ "${RUN_FIGURES}" = false ]; then
    echo "ERROR: No processing step selected."
    echo "Use --preprocess, --denoising, --firstlevel and/or  --secondlevel"
    exit 1
fi

# Show participants
[ ${#IDs[@]} -eq 0 ] && echo "No specific IDs provided: processing all participants" \
                     || echo "Processing participants: ${IDs[*]}"
[ ${#IDs[@]} -eq 0 ] && IDs=("") # If no IDs were provided, set to empty string


if [ ${#TASKS[@]} -eq 0 ]; then
    echo "No task specified: processing all tasks"
    TASKS_ARG=()        # do not pass --task
else
    echo "Processing tasks: ${TASKS[*]}"
    TASKS_ARG=(--tasks "${TASKS[@]}")
fi


# --------------------------
# Prepare log folder
# --------------------------
cd "${PATH_CODE}" || exit 1
mkdir -p log
cd log || exit 1

timestamp=$(date +"%Y%m%d_%H%M%S")

# --------------------------
# Run preprocessing
# --------------------------

if [ "${RUN_PREPROSS}" = true ]; then
    echo "Starting preprocessing..."
    nohup python -u ../code/preprocessing_workflow.py --path-data "${PATH_DATA}"  --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}" \
    > "nohup_preprocessing_${timestamp}.txt" 2>&1 &

    PID=$!
    echo "Preprocessing launched in background."
    echo "Log file: log/nohup_preprocessing_${timestamp}.txt"
    echo "To stop the process, run:"
    echo "kill ${PID}"
    wait ${PID}
    echo "Finished preprocessing!"
fi

# --------------------------
# Run shimming
# --------------------------

if [ "${RUN_SHIMMING}" = true ]; then
    echo "Starting shimming..."
    nohup python -u ../code/shimming.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}" \
    > "nohup_shimming_${timestamp}.txt" 2>&1 &

    PID=$!
    echo "Shimming launched in background."
    echo "Log file: log/nohup_shimming_${timestamp}.txt"
    echo "To stop the process, run:"
    echo "kill ${PID}"
    wait ${PID}
    echo "Finished shimming!"
fi

# --------------------------
# Run denoising
# --------------------------

if [ "${RUN_DENOISING}" = true ]; then
    echo "Starting denoising..."
    nohup python -u ../code/denoising_workflow.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}" \
    > "nohup_denoising_${timestamp}.txt" 2>&1 &
    
    PID=$!
    echo "Denoising launched in background."
    echo "Log file: log/nohup_denoising_${timestamp}.txt"
    echo "To stop the process, run:"
    echo "kill ${PID}"
    wait ${PID}
    echo "Finished denoising!"
fi

# --------------------------
# Run first level analysis
# --------------------------

if [ "${RUN_FIRSTLEVEL}" = true ]; then
    echo "Starting first level analysis..."
    nohup python -u ../code/firstlevel_workflow.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}" \
    > "nohup_firstlevel_${timestamp}.txt" 2>&1 &

    PID=$!
    echo "First level analysis launched in background."
    echo "Log file: log/nohup_firstlevel_${timestamp}.txt"
    echo "To stop the process, run:"
    echo "kill ${PID}"
    wait ${PID}
    echo "Finished first level analysis!"
fi

# --------------------------
# Run second level analysis
# --------------------------
if [ "${RUN_SECONDLEVEL}" = true ]; then
    echo "Starting second level analysis..."
    nohup python -u ../code/secondlevel_workflow.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}" \
    > "nohup_secondlevel_${timestamp}.txt" 2>&1 &

    PID=$!
    echo "second level analysis launched in background."
    echo "Log file: log/nohup_secondlevel_${timestamp}.txt"
    echo "To stop the process, run:"
    echo "kill ${PID}"
    wait ${PID}
    echo "Finished second level analysis!"
fi
