"""

This script calculates pass@k. It receives a list of directories as its
argument, and calculates the mean pass@k for the set of problems in each
directory. It checks that all results in a directory were generated at the same
temperature. It calculates pass@1 for temperature 0.2 and both pass@10 and
pass@100 for temperature 0.8.

The output has the following columns:

- Dataset: the name of a directory
- Pass@k: the value of k
- Estimate: the mean pass@k for the problems in the directory
- NumProblems: the number of problems in the directory
- MinCompletions: the minimum number of completions for any problem in the 
  directory
- MaxCompletions: the maximum number of completions for any problem in the
  directory
"""
import numpy as np
from pathlib import Path
import itertools
import argparse
import json
import pprint
import glob
from os import listdir
from os.path import isfile, join
import os

def read_json_res(path):
    """
    Reads a JSON file with results.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def for_file(path):
    # print(f"Reading {path}...")
    data = read_json_res(path)
    if data is None:
      return None
    n = len(data["results"])
    c = len([True for r in data["results"] if r["status"] == "OK" and r["exit_code"] == 0])
    if c > 0 :
        print('='*20)
        print(f'For file {path} n = {n} and c = {c}')
        print('='*20)
    return {
        "pass@1": estimator(n, c, 1),
        "pass@10": estimator(n, c, 10),
        "pass@100": estimator(n, c, 100),
        "n": n,
        "temperature": data["temperature"] if "temperature" in data else 0.2
    }



def main(dirs = None, output = None):
    res_holder= {
        "dataset": None,
        "pass@k": None,
        "estimate": None,
        "num_problems": None,
        "min_completions": None,
        "max_completions": None,
    }
    
    results_dict = {}
    if dirs is None and output is None :
        parser = argparse.ArgumentParser()
        parser.add_argument("dirs", type=str,  help="Directories with results. ", nargs="+")
        parser.add_argument("--output", type=str, help="Output file")
        args = parser.parse_args()
        dirs = args.dirs
        output = args.output

    print("Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions")
    if type(dirs) == list:
        d = dirs[0]
    else:
        d = dirs
    
    # results = [ for_file(p) for p in itertools.chain(Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz")) ]
    results = []

    print(f"Reading results from {d}...")
    print(f"And saving results to {output}...")

    for p in itertools.chain(Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz")):
        res = for_file(p)
        if res is not None:
            results.append(res)

    print(f"Read {len(results)} results")
    print(f"Sample result: {results[0]}")
    
    results = [ r for r in results if r is not None ]
    name = d.split("/")[-1] if d.split("/")[-1] != "" else d.split("/")[-2]
    temperatures = set(r["temperature"] for r in results)
    if len(temperatures) != 1:
        raise ValueError(f"Different Value temperatures: {temperatures}")
    temperature = list(temperatures)[0]
    num_problems = len(results)
    min_completions = np.min([r["n"] for r in results])
    max_completions = np.max([r["n"] for r in results])
    if temperature == 0.2:
        pass_1 = np.mean([r["pass@1"] for r in results])
        print(f"{name},1,{pass_1},{num_problems},{min_completions},{max_completions}")
        res_holder["dataset"] = name
        res_holder["pass@k"] = 1
        res_holder["estimate"] = float(pass_1)
        res_holder["num_problems"] = int(num_problems)
        res_holder["min_completions"] = int(min_completions)
        res_holder["max_completions"] = int(max_completions)
        results_dict["pass@1"] = res_holder
    elif temperature == 0.8:
        pass_10 = np.mean([r["pass@10"] for r in results])
        pass_100 = np.mean([r["pass@100"] for r in results])
        print(f"{name},10,{pass_10},{num_problems},{min_completions},{max_completions}")
        print(f"{name},100,{pass_100},{num_problems},{min_completions},{max_completions}")
        res_holder["dataset"] = name
        res_holder["pass@k"] = 10
        res_holder["estimate"] = float(pass_10)
        res_holder["num_problems"] = int(num_problems)
        res_holder["min_completions"] = int(min_completions)
        res_holder["max_completions"] = int(max_completions)
        results_dict["pass@10"] = res_holder
        
        res_holder2 = {}
        res_holder2["dataset"] = name
        res_holder2["pass@k"] = 100
        res_holder2["estimate"] = float(pass_100)
        res_holder2["num_problems"] = int(num_problems)
        res_holder2["min_completions"] = int(min_completions)
        res_holder2["max_completions"] = int(max_completions)
        results_dict["pass@100"] = res_holder2
        
    else:
        raise ValueError(f"Unexpected temperature: {temperature}")
        
    
    #pretty print results_dict
    pprint.pprint(results_dict)
    
    output_file = "results_dict.json"
    if output is not None:
        output_file = output

    with open(output_file, "w") as f:
        json.dump(results_dict, f)
    print(f"Saved results to {output_file}")
    
    


if __name__ == "__main__":
    exp_name = 'codegen-350M-mon_pass100x200_py_bs50'
    exp_name = f'{exp_name.replace("-", "")}'
    target_dir  = f'{exp_name}'
    output_file = f'{exp_name}_results.json'

    if len(os.sys.argv) > 1:
        main()

    else:
        main(target_dir, output_file)
