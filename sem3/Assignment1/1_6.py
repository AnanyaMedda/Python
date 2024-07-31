def generate_primes(n):
    # Generate prime numbers up to the nth term
    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    return primes

# Input the number of prime terms to generate
n = int(input("Enter the number of prime terms to generate: "))

# Generate and print the prime numbers
primes = generate_primes(n)
print(f"The first {n} prime numbers are: {primes}")
