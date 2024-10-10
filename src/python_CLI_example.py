#!/usr/bin/env -S python3
#^ What the hell is this? 
# That is called a "shebang" line. 
# The ! tells the python interpreter that it would like to execute some code in the command line real quick before running the code. 
# This line in particular will find whatever python version you are currently running and execute the code with that python version.
# It allows any of us to execute python code on the command line even if we have different version of python.  


# Argeparse library that with creating arguemnts for command line code. 
# Sys is used to find out how many arguemnts the user passed to the code
import argparse 
import sys

 # class that helps with printing help message. Leave this here for now.
class Formatter(argparse.RawDescriptionHelpFormatter): 
	pass

# This is where "helper" functions should be written
def print_message(message): 
	
    print("Are you ready for this awesome message to print?")
    print(message)
    print("Dude, you totally just printed that message. Crazy!")



# This is where the magic happens. Main function that executes your helper functions to achieve code goal
def main():

    # parser object allos your to set arguments for command line and grab input from user. 
    parser = argparse.ArgumentParser(description="%(prog)s prints a message inputed by the code") 
    parser.add_argument("message", help = "filepath to rpjb files") # copy and paste this line to make more arguemnts. 
    # To make unique arguments that only take intergers, or do something fancy, go and read argparse.ArguementParser.add_argument documentation


    # check that there are at least the required arguments are inputed by user. If not, give user an error message
    if len(sys.argv) < 2:
        print(f"{len(sys.argv)-1} parameters passed. At least 2 required. Please provide a message\n")
        parser.print_help()
        sys.exit(1)
		
    # grab the argument. message field comes from the parser.add_argument("message" ...) line. 
    args = parser.parse_args()
    message = args.message

    # grab the messsage and input it into the the helper function
    print("Printing message...")
    print_message(message)
    print("Done")
	

if __name__ == "__main__":
	 	main()