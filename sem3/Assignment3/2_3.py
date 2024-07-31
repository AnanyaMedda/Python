num1 = int(input("Enter 1st number : "))
num2 = int(input("Enter 2nd number : "))

for i in range(min(num1,num2),0,-1) :
    if(num1 % i == 0 and num2 % i ==0) :
        hcf = i
        print(f"\nGCD/HCF of {num1} and {num2} = {hcf}")
        break
    
lcm = (num1 * num2) / hcf
print(f"\nLCM of {num1} and {num2} = {lcm}")