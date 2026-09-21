import argparse
import re
import sys

def tag_id(input_file):
    line_num = 0
    
    try:
        # Open file with UTF-8 encoding (equivalent to Perl's "<:encoding(utf8)" and "use utf8")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Equivalent to: if (($line ne '') & ($line !~ /^ *$/))
                # line.strip() checks if the line is empty or contains only whitespace
                if line.strip(): 
                    # Equivalent to: chomp($line); $line =~ s/^\s+|\s+$//g;
                    line = line.strip()
                    
                    # Equivalent to: $line =~ s/ +/ /g;
                    # (Note: " ".join(line.split()) is a more Pythonic way to do this, 
                    # but re.sub is used here to exactly match your Perl regex)
                    line = re.sub(r' +', ' ', line)
                    
                    # Increment line number ONLY for valid lines
                    line_num += 1
                    
                    # Equivalent to: print "$line (ye_$lineNum)\n";
                    # (Python's print adds the newline automatically)
                    print(f"{line} (ye_{line_num})")
                    
    except FileNotFoundError:
        # Equivalent to: or die "Couldn't open input file $ARGV[0]!, $!\n"
        print(f"Error: Couldn't open input file '{input_file}'!", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Read a text file, clean whitespace, and append a unique ID (ye_<num>) to each line."
    )
    parser.add_argument(
        '-i', '--input_file', 
        required=True, 
        help="Path to the input text file."
    )
    
    args = parser.parse_args()
    tag_id(args.input_file)

if __name__ == "__main__":
    main()