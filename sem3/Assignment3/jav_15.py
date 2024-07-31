number = int(input("Enter a number: "))

if number%10==7 or number%7==0 :
    print(f"{number} is a Buzz number.")
else:
    print(f"{number} is not a Buzz number.")