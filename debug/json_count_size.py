"""Get size of parts of a JSON file.

Handy to debug why ray tune checkpoints are so big.

Created because a search did not yield any ready-to-use solutions...

Example:
    $ echo '{"aba": [1, 2, 3], "caba": {"1": "10", "2": 20}}' > a.json  # create a json file
    $ python json_count_size.py --file a.json --level 1
        {
        "aba": 9,
        "caba": 20
        }
        # This means that the key "aba" has length 9 in the json format, and the key "caba" 20
        # set level to 2 to get size of individual items of 'aba' and 'caba', or to 0 to get
        # the length of the whole file

Interactive mode example:
    $ python json_count_size.py --file a.json --cli
      # prints the size with default level 5, sizes of all individual elements
      > size
       {
          "aba": [
            1,
            1,
            1
          ],
          "caba": {
            "1": 4,
            "2": 2
          }
        }

     # set recursion level to 1
     > level 1

     # give top-level sizes
     > size
      {
        "aba": 9,
        "caba": 20
      }

     # select a part of the file
     # locator is a list of items separated by a comma (,)
     # empty locator means the whole file
     # locator elements are integers or strings
     # if a corresponding element is an array, an integer is required

     > loc aba
     > level 1
     > size
      [
        1,
        1,
        1
      ]

    > loc caba 2
    > level 1
     2

"""

import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Print a 'schema' of a json file in terms of length")
parser.add_argument('--file', help="JSON file to load", type=str, required=True)
parser.add_argument('--level', help="Maximal recursion level", type=int, default=20)
parser.add_argument('--cli', help="Run a command-line interface", action='store_true')


def json_recursive_size(data, level=0, maxlevel=None):
    """Get size of a JSON object recursively."""

    ### HELPERS

    # how many levels remaining?
    remaining_levels = maxlevel - level if maxlevel is not None else None

    def length_now(d):
        """Get length of a json string."""
        return len(json.dumps(d))

    def subcall(**kwargs):
        """Call myself recursively."""
        return json_recursive_size(level=level + 1,
                                   maxlevel=maxlevel,
                                   **kwargs)

    if maxlevel is not None and level >= maxlevel:
        return length_now(d=data)

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = subcall(data=value)
    elif isinstance(data, list):
        result = []
        for val in data:
            result.append(subcall(data=val))
    else:
        result = length_now(d=data)

    return result


def by_locator(data, locator):
    """Given a locator, get the data from json."""
    # a quick search didn't yield any existing libs...

    if locator is None:
        locator = []
    elif isinstance(locator, str):
        locator = locator.split(',')

    for i, key in enumerate(locator):
        key = key.strip()
        if isinstance(data, list):
            try:
                key = int(key)
            except ValueError:
                raise ValueError(f"Data is of array form but index is invalid: {key} {i}")
        elif isinstance(data, dict):
            key = str(key)

        try:
            data = data[key]
        except IndexError:
            raise ValueError(f"Array key [{key}] not found in data at {i}")
        except KeyError:
            raise ValueError(f"Dictionary key [{key}] not found in data at {i}")

    return data


if __name__ == '__main__':
    args = parser.parse_args()

    print("Loading file...")
    with open(args.file, 'r') as f:
        data = json.load(f)

    if args.cli:
        cmd = None
        level = 5
        locator = None

        while str(cmd) != 'exit':
            print("Type size to compute the size and print it")
            print("Type loc [dict_key], [arr_key], [...]' to"
                  f" locate a part of the JSON [now {locator}]")
            print(f"Type level [int] to set the level [now {level}]")
            print("Type exit to quit the program")
            cmd = input("> ").strip()

            if cmd == 'size':
                try:
                    subdata = by_locator(data=data, locator=locator)
                except ValueError as e:
                    print(f"Invalid locator: {e}\n")
                    continue
                sizes = json_recursive_size(data=subdata, maxlevel=level)
                print(json.dumps(sizes, indent=2))
            elif cmd.startswith('loc'):
                locator = cmd.split(' ')[1:]
            elif cmd.startswith('level'):
                try:
                    level = int(cmd.split(' ')[1])
                except ValueError:
                    print(f"Please provide an integer for the level: {cmd}\n")
                    continue
            print("")
        sys.exit(0)

    print("Computing size...")
    sizes = json_recursive_size(data=data, maxlevel=args.level)

    print(json.dumps(sizes, indent=4))
