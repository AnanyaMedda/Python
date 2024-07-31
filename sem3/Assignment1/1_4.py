# Input number
num = int(input("Enter a number: "))

# Reverse the number
reversed_num = 0
while num != 0:
    digit = num % 10
    reversed_num = reversed_num * 10 + digit
    num //= 10

# Print the reversed number
print(f"Reversed number is: {reversed_num}")
