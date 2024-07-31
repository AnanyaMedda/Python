
def powerNumber (base,exp) :
    if exp == 0 :
        return 1
    else :
        return (base * powerNumber(base,exp-1))

base = int(input("Enter a number: "))
exp = int(input("Enter the power : "))

result = powerNumber(base,exp)
print(f"{base} ^ {exp} = {result}")
    
