#!/usr/bin/env python
# coding: utf-8

# In[1]:


input_str = "India is my motherland. I love my country. Capital of India is New Delhi."

# Calculate the length of the string
length = len(input_str)
print("Length of the string:", length)

# Find the substring "country"
substring = "country"
if substring in input_str:
    print(f"'{substring}' is found in the string.")
else:
    print(f"'{substring}' is not found in the string.")

# Count occurrences of each word
words = input_str.split()
word_count = {}
for word in words:
    word = word.strip(".").lower()  # Remove punctuation and make case-insensitive
    word_count[word] = word_count.get(word, 0) + 1

print("Word occurrences:")
for word, count in word_count.items():
    print(f"{word}: {count}")


# In[2]:


input_str = "without,hello,bag,world"

# Split the input string by commas and sort
words = input_str.split(',')
words.sort()

# Join the sorted list back into a comma-separated string
output_str = ','.join(words)
print("Sorted words:", output_str)


# In[4]:


input_lines = ["Hello world", "Practice makes perfect"]

# Convert each line to uppercase
output_lines = [line.upper() for line in input_lines]

# Print the capitalized lines
for line in output_lines:
    print(line)

    


# In[5]:


input_str = "hello world and practice makes perfect and hello world again"

# Split the input string into words
words = input_str.split()

# Remove duplicates by converting to a set and then back to a sorted list
unique_words = sorted(set(words))

# Join the sorted words back into a string
output_str = ' '.join(unique_words)
print("Unique sorted words:", output_str)


# In[6]:


input_str = "hello world! 123"

# Initialize counters
letter_count = 0
digit_count = 0

# Count letters and digits
for char in input_str:
    if char.isalpha():
        letter_count += 1
    elif char.isdigit():
        digit_count += 1

print("LETTERS", letter_count)
print("DIGITS", digit_count)


# In[7]:


input_str = input("Enter a string: ")

if input_str in ["yes", "YES", "Yes"]:
    print("Yes")
else:
    print("No")


# In[8]:


input_str = "2 cats and 3 dogs."

# Split the input string into words
words = input_str.split()

# Filter out the words that are composed of digits only
digit_words = [word for word in words if word.isdigit()]

print(digit_words)


# In[9]:


input_str = "abcdefgabc"

# Initialize an empty dictionary to count characters
char_count = {}

# Count occurrences of each character
for char in input_str:
    char_count[char] = char_count.get(char, 0) + 1

# Print the count of each character
for char, count in char_count.items():
    print(f"{char},{count}")


# In[10]:


input_str = "example"
reversed_str = input_str[::-1]
print("Reversed string:", reversed_str)


# In[11]:


input_str = "madam"
is_palindrome = input_str == input_str[::-1]
print("Is palindrome:", is_palindrome)


# In[12]:


input_str = "hello world"
substring = "world"
ends_with = input_str.endswith(substring)
print(f"Ends with '{substring}':", ends_with)


# In[13]:


input_str = "hello world"
capitalized_str = input_str.title()
print("Capitalized:", capitalized_str)


# In[14]:


str1 = "listen"
str2 = "silent"

# Check if sorted characters of both strings are equal
is_anagram = sorted(str1) == sorted(str2)
print("Is anagram:", is_anagram)


# In[15]:


input_str = "hello world"
vowels = "aeiouAEIOU"
no_vowels_str = ''.join([char for char in input_str if char not in vowels])
print("String without vowels:", no_vowels_str)


# In[16]:


input_str = "Practice makes perfect"
words = input_str.split()
longest_word_length = max(len(word) for word in words)
print("Length of the longest word:", longest_word_length)


# In[ ]:




