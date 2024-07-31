math = int(input("Enter your marks in Math : "))
physics = int(input("Enter your marks in Physics : "))
chemistry = int(input("Enter your marks in Chemistry : "))

if math >= 60 and physics >= 50 and chemistry >= 40 and ( ((math+physics+chemistry) >= 200) or ((math+physics) >= 150) ) :
    print("You are eligible for admission")
else :
    print("You are not eligible for admission")

