def find_numbers():
    # Find and display numbers divisible by 7 and not multiples of 5 between 1000 and 2000
    result = []
    for num in range(1000, 2001):
        if num % 7 == 0 and num % 5 != 0:
            result.append(num)
    return result

# Call the function and print the result
numbers = find_numbers()
print("Numbers divisible by 7 and not multiples of 5 between 1000 and 2000:")
print(numbers)
