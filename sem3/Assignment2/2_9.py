#function to calculate the binary equivalent of n
def toBinary(num):
    return format(int(num),'b')
    
n = int(input("Enter a number : "))
# Print the Fibonacci series
print(f"Binary equivalent of {n} = {toBinary(n)}")
