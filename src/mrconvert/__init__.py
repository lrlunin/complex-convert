
import argparse
import pathlib
import logging
from functools import reduce
from itertools import chain
import torch
from mrconvert.data_proc import convert_data

def main():
    parser = argparse.ArgumentParser(
                        prog='mrconvert',
                        description='convert multicoil to complex data')
    parser.add_argument("in_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--file_glob", default=["file_brain_AXT2_*.h5", "file_brain_AXT1*.h5"])
    parser.add_argument("--proc_func", default="walsh")
    parser.add_argument("--smooth_width", default=1)
    parser.add_argument("--first_n", default=None)
    args = parser.parse_args()
    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)
    glob_patterns = args.file_glob
    proc_func = None
    import mrpro.algorithms.csm
    match args.proc_func:
        case "walsh":
            proc_func = mrpro.algorithms.csm.walsh
        case "inati":
            proc_func = mrpro.algorithms.csm.inati
        case _:
            raise TypeError(f"There is no function: {args.proc_func}")
    smooth_width = args.smooth_width
    first_n = None if args.first_n is None else int(args.first_n)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(args.out_dir) / "csm_conversion.log"
    use_cuda = torch.cuda.is_available()

    logging.basicConfig(handlers=[logging.FileHandler(filename=log_path), logging.StreamHandler()], format='%(asctime)s, %(name)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger("main()")
    logger.info(f"in_dir: {in_dir}, out_dir: {out_dir}")
    logger.info(f"proc function: {proc_func.__name__}")
    logger.info(f"glob patterns: {glob_patterns}")
    logger.info(f"using CUDA: {use_cuda}")
    files = list(reduce(lambda a,b: chain(a, b), map(lambda pattern: in_dir.glob(pattern), glob_patterns)))
    files = files[:first_n]
    for i, file in enumerate(files):
        convert_data(file, 
                     i,
                     len(files),
                     out_dir,
                     smooth_width,
                     proc_func,
                     use_cuda)