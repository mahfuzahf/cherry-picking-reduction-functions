# Taken from https://researchdatapod.com/radix-sort-python/


# Helper function: Counting sort to sort array by the digit represented by exp
def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n  # Output array to store sorted numbers
    count = [0] * 10  # There are 10 possible digits (0-9)
    
    # Store count of occurrences of each digit
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1
        
    # Modify count to store actual positions of digits in output array
    for i in range(1, 10):
        count[i] += count[i - 1]
        
    # Build the output array by placing numbers in their correct position
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        
    # Copy the sorted numbers back to the original array
    for i in range(n):
        arr[i] = output[i]


# Main radix sort function
def radix_sort(arr):
    # Find the maximum number to determine the number of digits
    max_val = max(arr)
    
    # Apply counting sort to sort based on each digit, from least significant to most significant
    exp = 1  # Initial exponent (1s place)
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10  # Move to the next digit place (10s, 100s, etc.)