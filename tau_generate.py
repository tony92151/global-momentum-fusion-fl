import math
import argparse

def liner_get(l=10, max_value=0.2):
    data = []
    c = max_value/(l-1)
    for i in range(l-1):
        data.append(math.ceil(c*i*1000)/1000)
    data.append(max_value)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max', type=float, required=True)
    args = parser.parse_args()

    print(liner_get(l=args.steps, max_value=args.max))
