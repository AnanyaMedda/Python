count = int(input("Enter no of terms : "))

i = 1
result = 1

while i<=count :
    result *= i
    print(f"{result} ", end = '')
    i+=1
    