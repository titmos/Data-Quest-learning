#A "phantom" argument passes methods when we call them.
#The "phantom" argument is actually the object itself.
#We need to include that in our method definition.
#The convention is to call the "phantom" argument self



class MyClass:
    
    def first_method(self):
        return "This is my first method"
my_instance = MyClass()
result = my_instance.first_method()
print(result, "\n", my_instance)


#Another try
class MyClass:    
    def first_method(self):
        return "This is my first method"
    
    # Add method here
    def return_list(self, input_list):
        return input_list
my_instance = MyClass()
result = my_instance.return_list([1, 2, 3])

#Attributes and the Init Method
class MyList:
    def __init__(self, initial_data):
        self.data = initial_data

my_list = MyList([1, 2, 3, 4, 5])
print(my_list.data)


#create a append method (similar to list append in PY)
class MyList:

    def __init__(self, initial_data):
        self.data = initial_data
        
    # Add method here
    def append(self, new_item):
        self.data = self.data + [new_item]
my_list = MyList([1, 2, 3, 4, 5])
#test
print(my_list.data)
my_list.append(6)
print(my_list.data)


#Creating and Updating an Attribute
class MyList:

    def __init__(self, initial_data):
        self.data = initial_data
        # Calculate the initial length
        self.length = 0
        for item in self.data:
            self.length += 1
##Because the code we added that defined the length attribute was only in the init method, if the list becomes longer using the append() method, our length attribute is no longer accurate.

#To address this, we need to run the code that calculates the length after any operation that modifies the data, which, in our case, is just the append() method. More precisely, we need to increment the length of the list each time we append a value to it.
    def append(self, new_item):
        self.data = self.data + [new_item]
        # Update the length here
        self.length += 1
my_list = MyList([1, 1, 2, 3, 5])
print(my_list.length)
my_list.append(8)
print(my_list.length)
