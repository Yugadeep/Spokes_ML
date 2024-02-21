#!/usr/bin/env -S python3

import argparse
import sys

class Formatter(argparse.RawDescriptionHelpFormatter):
	pass

def main():
    parser = argparse.ArgumentParser(description="%(prog)s adds a filepath to every line in the list of files")
	
    parser.add_argument("path_to_list", help = "filepath to list of rpjbs")
    parser.add_argument("path_to_rpjbs", help = "filepath to rpjb files")

    if len(sys.argv) < 2:
        print(f"{len(sys.argv)-1} parameters passed. At least 3 required. A valid file list is needed if you want to apply.\n")
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    path_to_list = args.path_to_list
    path_to_rpjbs = args.path_to_rpjbs

    print("Rewriting file to include path ...")
    f = open(path_to_list, "r").read()

    list_of_paths = f.split("\n")

    new_list = [path_to_rpjbs+x+"\n" for x in list_of_paths]
    new_list[-1] = new_list[-1].split("\n")[0]
    new_file_content = "".join(new_list)

    f = open(path_to_list, "w").write(new_file_content)
    
    print("done")
    

	

if __name__ == "__main__":
 	main()