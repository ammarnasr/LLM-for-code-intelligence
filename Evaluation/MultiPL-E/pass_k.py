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
from multipl_e.util import gunzip_json, eprint
import json

def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def for_file(path):
    data = gunzip_json(path)
    if data is None:
      return None   
    n = len(data["results"])
    c = len([True for r in data["results"] if r["status"] == "OK" and r["exit_code"] == 0])
    return {
        "pass@1": estimator(n, c, 1),
        "pass@10": estimator(n, c, 10),
        "pass@100": estimator(n, c, 100),
        "n": n,
        "temperature": data["temperature"] if "temperature" in data else 0.2
    }

def main():
    res_holder= {
        "dataset": None,
        "pass@k": None,
        "estimate": None,
        "num_problems": None,
        "min_completions": None,
        "max_completions": None,
    }
    results_dict = {
        "pass@1": res_holder,
        "pass@10": res_holder,
        "pass@100": res_holder,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--suppress-header", action="store_true", help="Suppress the header")
    parser.add_argument("dirs", type=str,  help="Directories with results. ", nargs="+")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()
    if not args.suppress_header:
        print("Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions")
    for d in args.dirs:
        results = [ for_file(p) for p in itertools.chain(Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz")) ]
        results = [ r for r in results if r is not None ]
        name = d.split("/")[-1] if d.split("/")[-1] != "" else d.split("/")[-2]
        temperatures = set(r["temperature"] for r in results)
        if len(temperatures) != 1:
            eprint(f"Found multiple temperatures {temperatures} in {d} {results}")
            continue
        temperature = list(temperatures)[0]
        num_problems = len(results)
        min_completions = np.min([r["n"] for r in results])
        max_completions = np.max([r["n"] for r in results])
        if temperature == 0.2:
            pass_1 = np.mean([r["pass@1"] for r in results])
            print(f"{name},1,{pass_1},{num_problems},{min_completions},{max_completions}")
            res_holder["dataset"] = name
            res_holder["pass@k"] = 1
            res_holder["estimate"] = pass_1
            res_holder["num_problems"] = num_problems
            res_holder["min_completions"] = min_completions
            res_holder["max_completions"] = max_completions
            results_dict["pass@1"] = res_holder
        elif temperature == 0.8:
            pass_10 = np.mean([r["pass@10"] for r in results])
            pass_100 = np.mean([r["pass@100"] for r in results])
            print(f"{name},10,{pass_10},{num_problems},{min_completions},{max_completions}")
            print(f"{name},100,{pass_100},{num_problems},{min_completions},{max_completions}")
            res_holder["dataset"] = name
            res_holder["pass@k"] = 10
            res_holder["estimate"] = pass_10
            res_holder["num_problems"] = num_problems
            res_holder["min_completions"] = min_completions
            res_holder["max_completions"] = max_completions
            results_dict["pass@10"] = res_holder
            res_holder["pass@k"] = 100
            res_holder["estimate"] = pass_100
            results_dict["pass@100"] = res_holder
        else:
            raise ValueError(f"Unexpected temperature: {temperature}")
        
    print(results_dict)
    
    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(results_dict, f)
        print(f"Saved results to {args.output}")
    else:    
        with open("results_dict.json", "w") as f:
            json.dump(results_dict, f)
        
        print("Saved results to results_dict.json")
    

if __name__ == "__main__":
    main()
