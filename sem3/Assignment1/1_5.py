# Get user input
num = int(input("Enter a number: "))

# Initialize an empty list to store factors
factors = []

# Loop from 1 to the number itself
for i in range(1, num + 1):
    # Check if i is a factor of num
    if num % i == 0:
        factors.append(i)

# Print the factors
print(f"The factors of {num} are:", factors)
