"""
In Python, control statements are used to control the flow of a program's execution. There are three main types of control statements in Python: conditional statements, loops, and functions.

Conditional statements:
Conditional statements are used to make decisions based on the value of a condition. In Python, the most common conditional statement is the "if" statement. The basic syntax of the "if" statement is as follows:
"""
if condition:
   statement(s)
"""
The "if" statement checks if the condition is true or false. If the condition is true, the statement(s) inside the if block are executed. Otherwise, they are skipped.

You can also use the "elif" and "else" statements to create more complex conditional statements. The "elif" statement is used to check additional conditions, while the "else" statement is used to execute code if none of the previous conditions were met. Here's an example:
"""

x = 10

if x > 10:
    print("x is greater than 10")
elif x < 10:
    print("x is less than 10")
else:
    print("x is equal to 10")

"""
Loops:
Loops are used to execute a block of code repeatedly. In Python, there are two main types of loops: "for" loops and "while" loops.
The "for" loop is used to iterate over a sequence of items. Here's an example:
"""
for i in range(5):
    print(i)

"""
This will print the numbers 0 through 4.

The "while" loop is used to execute a block of code as long as a condition is true. Here's an example:
"""
i = 0

while i < 5:
    print(i)
    i += 1
"""
This will print the numbers 0 through 4.

Functions:
Functions are used to group a set of related statements together that perform a specific task. Functions make it easier to reuse code and make the code more modular. Here's an example of a simple function:
"""

def square(x):
    return x * x

"""
This function takes a parameter "x" and returns its square.

In summary, control statements in Python are essential for controlling the flow of a program's execution. By using conditional statements, loops, and functions, you can create more complex programs that perform a variety of tasks.
"""
