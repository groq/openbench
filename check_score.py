import sys
import os

# Suppress stdout/stderr from library calls if they are chatty
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

from inspect_ai.log import read_eval_log

# Restore stdout for our output
sys.stdout = sys.__stdout__

log_file = sys.argv[1]
try:
    log = read_eval_log(log_file)
    
    if log.status:
        print(f"Status: {log.status}")

    if log.results:
        for score in log.results.scores:
            print(f"Score: {score.value}")
    else:
        print("No results found.")
        
    if log.error:
        # Print error but handle if it's an object
        print(f"Error: {log.error}")

except Exception as e:
    print(f"Failed to read log: {e}")
