#!/usr/bin/env python3
# This script can be run from the command line to perform a specified task.
# Usage: python script.py [options]

import sys

# Define a function for the main logic of the script.
def main(*args):
    # Your main script logic goes here.
    # For example, let's just print the arguments passed.
    for arg in args:
        print(f"Argument received: {arg}")

# Check if this script is being run directly and not imported.
if __name__ == "__main__":
    # Call the main function with command line arguments, excluding the script name.
    main(*sys.argv[1:])