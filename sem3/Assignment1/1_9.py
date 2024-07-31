# Function to check if a number is a perfect number
def is_perfect_number(num):
    # Sum of divisors of num
    sum_divisors = sum(i for i in range(1, num) if num % i == 0)
    return sum_divisors == num

# Function to check if a number is an Armstrong number
def is_armstrong_number(num):
    # Number of digits in num
    num_str = str(num)
    num_digits = len(num_str)
    # Sum of digits raised to the power of num_digits
    sum_of_powers = sum(int(digit) ** num_digits for digit in num_str)
    return sum_of_powers == num

# Input a number
number = int(input("Enter a number: "))

# Check if the number is a perfect number
if is_perfect_number(number):
    print(f"{number} is a perfect number.")
else:
    print(f"{number} is not a perfect number.")

# Check if the number is an Armstrong number
if is_armstrong_number(number):
    print(f"{number} is an Armstrong number.")
else:
    print(f"{number} is not an Armstrong number.")
