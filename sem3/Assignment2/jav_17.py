count = int(input("Enter the number of terms : "))
a = 0
b = 1
c = a + b
print(f"{a} {b} {c}", end='')
i = 1

while i <= count-3 :
    a = b
    b = c
    c = a+b
    print(f" {c} ", end='')
    i+=1

