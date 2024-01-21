#!/usr/bin/env python3
import sys

# This function checks if a given number is prime
def is_prime(num):
    """Return True if the input number is a prime; otherwise, return False."""
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

# This function takes an integer argument from the command line
def main():
    if len(sys.argv) != 2:
        print("Usage: python prime_checker.py <number>")
        return
    try:
        number = int(sys.argv[1])
    except ValueError:
        print("The argument must be an integer.")
        return
    
    if is_prime(number):
        print(f"{number} is a prime number.")
    else:
        print(f"{number} is not a prime number.")

# Python's convention to check if the run command was called on this script
if __name__ == "__main__":
    main()
```

In the script:

- `sys.argv` is used to access command-line arguments.
- The `is_prime` function is defined to determine if a number is prime.
- The `main` function checks for the correct number of command-line arguments and prints usage information if necessary.
- `ValueError` is caught in case a non-integer value is passed as a command-line argument.
- It checks whether the given number is a prime and prints the result. 

To run this script, the following should be entered on the command line in a terminal:

```bash
python prime_checker.py <number>
```

Replace `<number>` with the integer you want to check. No additional script commands are needed to install dependencies since it only uses the standard Python library.
