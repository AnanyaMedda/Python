num = int(input("Enter a number: "))
copy = num
digit = 0
rev = 0

while copy != 0 :
    digit = copy%10
    rev = (rev*10) + digit
    copy//=10
    
print(f"Reverse of {num} is {rev}")