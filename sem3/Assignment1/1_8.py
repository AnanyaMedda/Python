# Input a number
number = input("Enter a number: ")

# Check if the number is a palindrome
if number == number[::-1]:
    print(f"{number} is a palindrome.")
else:
    print(f"{number} is not a palindrome.")
