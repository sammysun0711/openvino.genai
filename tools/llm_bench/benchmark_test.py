import subprocess
import sys
from benchmark import get_argprser
import os

def parser_text(path, n):
    # Open the file in read mode
    with open(path, "r") as file:
        # Read all lines into a list
        lines = file.readlines()
        # Iterate over each line in the list
        get_text = False
        max_rss_mem = 0
        for line in lines:
            if "input_token_count:" in line:
                input_token_count = line.strip().split(" ")[-1]
            elif "output_token_count" in line:
                output_token_count = line.strip().split(" ")[-1]
            elif "avg_1st_token_latency" in line:
                avg_1st_token_latency = line.split(" ")[-2]
            elif "avg_2nd_tokens_latency" in line:
                avg_2nd_tokens_latency = line.split(" ")[-2]
            elif "avg_2nd_token_throughput" in line:
                avg_2nd_token_throughput = line.split(" ")[-2]
            elif "max_rss_mem" in line:
                max_rss_mem = float(line.strip().split(" ")[-1]) if float(line.strip().split(" ")[-1]) > max_rss_mem else max_rss_mem
            elif not get_text and line.strip() != "":
                text = line
                get_text = True

        print("==Generated output: ")
        print(text)
        print(" ")
        print(f"== Performance metrics from {n} times run:")
        print(f"input_token_count: {input_token_count}")
        print(f"output_token_count: {output_token_count}")
        print(f"avg_1st_token_latency: {avg_1st_token_latency} ms/token")
        print(f"avg_2nd_tokens_latency: {avg_2nd_tokens_latency} ms/token")
        print(f"avg_2nd_token_throughput: {avg_2nd_token_throughput} tokens/sec")
        print(f"max_rss_mem: {max_rss_mem} MB")

def main():
    # Define the path to the Python script you want to run
    script_path = "benchmark.py"
    args = get_argprser()

    # Define the options and arguments for the script
    options = [
        "-m", str(args.model),
        "-d", str(args.device),
        "-n", str(args.num_iters),
        "-ic", str(args.infer_count),
        "-pf", str(args.prompt_file[0]),
        "--genai",
        "-mc", "2"
    ]

    # Construct the full command
    command = [sys.executable, script_path] + options

    # Run the command and wait for it to complete
    result = subprocess.run(command, capture_output=True, text=True)
    # print("Output:", result.stdout)
    parser_text("log.txt", args.num_iters)
    os.remove("log.txt")


if __name__ == '__main__':
    main()
