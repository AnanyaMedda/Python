year = int(input("Enter year to check: "))

if year % 100 == 0:
    if year % 400 == 0:
        print(f"{year} is a LEAP YEAR")
    else:
        print(f"{year} is not a LEAP YEAR")
elif year % 4 == 0:
    print(f"{year} is a LEAP YEAR")
else:
    print(f"{year} is not a LEAP YEAR")
