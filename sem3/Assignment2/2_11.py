def reverseNum(num):
    
    copy = num
    sum = 0
    rev = 0
    
    while(copy>0) :
        digit = copy%10
        rev = (rev * 10) + digit
        copy //= 10
        
    return rev


number = int(input("Enter a number: "))

print(f"Reverse of {number} is : {reverseNum(number)}")
