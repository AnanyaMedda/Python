# Input the number of terms
n = int(input("Enter the number of terms: "))

# Initialize the first two terms
a, b = 0, 1
count = 0

# Print the Fibonacci series
print("Fibonacci Series:")
while count < n:
    print(a, end=' ')
    a, b = b, a + b
    count += 1
