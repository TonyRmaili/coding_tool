def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b

def power(a, b):
    return a ** b

def square_root(a):
    return a ** 0.5

def cube_root(a):
    return a ** (1/3)

def factorial(a):
    if a == 0:
        return 1
    else:
        return a * factorial(a-1)
    
