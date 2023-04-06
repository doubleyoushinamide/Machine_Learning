"""
1. Classes in Python:
A class is a blueprint for creating objects that have certain properties and methods. The properties are called attributes, and the methods are functions that are associated with the object. Here is an example of a simple class:
"""
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print("Hello, my name is", self.name, "and I am", self.age, "years old.")
"""
The __init__() method is a special method that is called when an object of the class is created. It initializes the attributes of the object with the values that are passed to it as arguments.

The greet() method is a simple method that prints a greeting message with the name and age of the person.

To create an object of the class, you simply need to call the class with the required arguments. Here's an example:
"""
person1 = Person("John", 25)
person1.greet()   # Output: Hello, my name is John and I am 25 years old.

"""
2. Functions in Python:
Functions are reusable blocks of code that perform a specific task. They take input parameters and return a value. Here's an example of a simple function:
"""
def add_numbers(a, b):
    result = a + b
    return result
"""
This function takes two input parameters a and b, adds them together, and returns the result.

To call the function, you simply need to pass the required arguments. Here's an example:
"""
sum = add_numbers(5, 7)
print(sum)   # Output: 12

"""
You can also define default values for the input parameters. Here's an example:
"""
def greet(name, greeting="Hello"):
    print(greeting, name)

greet("John")   # Output: Hello John
greet("Mary", "Hi")   # Output: Hi Mary

"""
In this example, the greeting parameter has a default value of "Hello". If you don't pass a value for greeting, it will use the default value. If you pass a value for greeting, it will use the passed value.

In summary, classes and functions are essential concepts in Python that allow you to create reusable and modular code. By using classes, you can define objects with specific properties and methods, while functions allow you to perform specific tasks with input parameters and return values.
"""

"""
####### Importance of classes and functions

Classes and functions are both important concepts in Python, but they serve different purposes and have different use cases.

1. Classes in Python:
A class is a blueprint for creating objects that have certain properties and methods. The properties are called attributes, and the methods are functions that are associated with the object. Classes are used to define complex data structures and behaviors, and to create multiple instances of objects with the same properties and methods. You can use classes when you want to create objects that have specific attributes and methods that are common across multiple instances.

2. Functions in Python:
A function is a block of code that performs a specific task. Functions are used to perform specific operations on data, and to modularize code so that it can be reused and maintained more easily. Functions take input parameters and can return a value, making them versatile and flexible. You can use functions when you want to perform a specific task on some data or when you want to modularize code for reuse.

Here are some key differences between classes and functions:

Classes define objects with specific properties and methods, while functions perform specific tasks on data.
Classes are used to create multiple instances of objects with the same properties and methods, while functions are typically used for one-off operations.
Classes are more complex and can have multiple attributes and methods, while functions are simpler and typically have a single purpose.
Classes can be used to define custom data structures and behaviors, while functions are typically used to manipulate existing data.
In summary, you should use classes when you want to define complex data structures and behaviors that can be reused across multiple instances of objects, and functions when you want to perform specific operations on data or when you want to modularize code for reuse. Classes and functions are both important concepts in Python, and choosing the right one for the task at hand can help you write more effective and efficient code.
"""
# A simple game of rock-paper-scissors

import random

class Game:
    def __init__(self):
        self.choices = ["rock", "paper", "scissors"]
        self.cpu_choice = None
        self.player_choice = None
    
    def play(self):
        print("Welcome to the Rock-Paper-Scissors game!")
        self.cpu_choice = random.choice(self.choices)
        self.player_choice = input("Enter your choice (rock/paper/scissors): ")
        print("CPU chose:", self.cpu_choice)
        self.determine_winner()
    
    def determine_winner(self):
        if self.player_choice == self.cpu_choice:
            print("It's a tie!")
        elif self.player_choice == "rock" and self.cpu_choice == "scissors":
            print("You win!")
        elif self.player_choice == "paper" and self.cpu_choice == "rock":
            print("You win!")
        elif self.player_choice == "scissors" and self.cpu_choice == "paper":
            print("You win!")
        else:
            print("CPU wins!")

game = Game()
game.play()
# This version of the game is more object-oriented and modular, making it easier to modify or extend in the future.
"""
In this version of the game, we define a Game class with an __init__ method that initializes the game's choices and sets the player's and CPU's choices to None. We then define a play method that prompts the player for their choice, randomly selects a choice for the CPU, and then calls the determine_winner method to determine the outcome of the game.

The determine_winner method uses conditional statements to determine the winner based on the player's and CPU's choices. If the player and CPU chose the same option, it's a tie. Otherwise, we check if the player's choice beats the CPU's choice based on the traditional Rock-Paper-Scissors rules.
"""
