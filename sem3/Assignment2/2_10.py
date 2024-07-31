def fact(n) :
    result=1
    for i in range(1,n+1) :
        result *= i
    return result
    

def isKrish(num):
    
    copy = num
    sum = 0
    
    while(copy!=0) :
        digit = copy%10
        copy //= 10
        sum += fact(digit)
        
    return sum


number = int(input("Enter a number: "))

if number == isKrish(number):
    print(f"{number} is an Krishnamurthy number.")
else:
    print(f"{number} is not an Krishnamurthy number.")
