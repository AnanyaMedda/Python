import math

# Input radius of the circle
radius = float(input("Enter the radius of the circle: "))

# Calculate area and perimeter (circumference)
area = math.pi * radius ** 2
perimeter = 2 * math.pi * radius

# Print the results
print(f"Area of the circle is {area}")
print(f"Perimeter (circumference) of the circle is {perimeter}")
