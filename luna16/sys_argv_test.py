#!/usr/bin/env/python

# Information about the first line:
# If you have several versions of Python installed, /usr/bin/env will ensure the interpreter
# used is the first one on your environment's $PATH.
# The alternative would be to hardcode something like #!/usr/bin/python; that's ok, but less flexible.
# In Unix, an executable file that's meant to be interpreted can indicate what interpreter to use
# by having a #! at the start of the first line, followed by the interpreter (and any flags it may need).
import argparse
import sys


# print('Command line arguments are:')
# for i in sys.argv:
#     print(i)
# print('\n\nThe python path is:', sys.path, '\n')


parser = argparse.ArgumentParser(description='Process some integers')  # the information of this object

# set input variables. One variable, one line, more variables, more line.

# optional argument: must have name, then the value of argument, could be unordered
# command line recognizes positional argument or optional argument based on prefix_chars (default is '-')
# prefix_chars: abbreviation is '-h', full name is '--help'

parser.add_argument('integer', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulation')
# about the above line:
# 'integer': positional argument name
# metavar: other name of this argument and use it
# nargs='+': all the variables in command line are accumulated in one list


parser.add_argument('--sum', action='store_const', const=sum,
                    dest='accumulation', default=max,
                    help='sum the integers (default: find the max)')
# about the above line:
# '--sum': optional argument name
# action: store_const (store the value of 'const'); store (store the value of argument, default)
# const: store a value
# dest: other name of this argument, and transfer it to upper class

# args = parser.parse_args()  # default is to get arguments from sys.argv (from command line)
args = parser.parse_args(['--sum', '1', '2', '3'])  # no need to get value from command line
print(args.accumulation(args.integer))
