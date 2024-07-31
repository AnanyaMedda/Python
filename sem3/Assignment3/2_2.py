import math

a = int(input("Enter the A coeffecient : "))
b = int(input("Enter the B coeffecient : "))
c = int(input("Enter the C coeffecient : "))

print(f"Quadratic Equation : {a}x^2 + {b}x + {c}")

discriminant = b**2 - (4 * a * c)

if discriminant >= 0 :
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
    print(f"Root 1 : {root1}")
    print(f"Root 2 : {root2}")
else :
    print("Equation has no real roots")
