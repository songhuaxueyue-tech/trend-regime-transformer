import argparse

def main():
   args =  argparse.ArgumentParser()
   args.add_argument("--c", dest="content", required=True, help="content to print")

   content = args.parse_args().content
   print(content)


if __name__ == "__main__":
    main()
    





