str1 = input("Enter a string : ")

def reverseString (str1) :
        rev = str1[::-1] #slicing in reverse, start, stop defaults to begining and start of string
        return rev


print(f"Reverse of {str1} is {reverseString(str1)}")