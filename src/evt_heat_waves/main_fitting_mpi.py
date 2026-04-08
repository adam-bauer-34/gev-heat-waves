"""Main file for fitting GEV distribution to data in a fully
parallelized way using MPI.

Adam Bauer
UChicago
Apr 8, 2026
"""

import shutil
import time

from mpi4py import MPI

from evt_heat_waves.logging import setup_logger
from evt_heat_waves.config import parse_args_fitting, get_git_hash
from evt_heat_waves.mpi.mostmems import runner
from evt_heat_waves.mpi.primmem import runner

width = shutil.get_terminal_size(fallback=(80, 20)).columns

def main():
    """Main function for GEV fitting.
    """

    # parse arguments
    args = parse_args_fitting()

    # setup logging
    logger = setup_logger(args.debug)

    t0 = time.time()

    # print statement for logging and reproducibility
    logger.info("-" * width)
    logger.info(f"Git hash: {get_git_hash()}")

    

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get command line arguments
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: srun -n <nprocs> python main_cmip_allmembers_mpi.py STAT")
            print("where STAT is 'stat' or 'nonstat'")
        sys.exit(1)
    
    STAT = sys.argv[1]
    
    if STAT not in ['stat', 'nonstat']:
        if rank == 0:
            print("⚠️ Invalid entry for command line argument `STAT` (supports 'stat' or 'nonstat').")
        sys.exit(1)
    
    
    # Only rank 0 does initial setup
    if rank == 0:
        start_time = time.time()
        print('='*width)
        print(f"🚀 Starting MPI parallel processing with {size} processes")
        print(f"🎯 Strategy: Parallelizing over variables, fit types, and members")
        print('='*width)
        
        # Setup CMIP config object
        CMIPConfig = CMIP6EnsembleConfig.from_yaml("config/meta.yaml", 
                                                    "config/qc.yaml")
        
        # Define variables and fit types
        vars = ['tas_annual_max', 'tas_annual_min']
        fit_types = ['raw', 'annmean', 'trend']
        
        # Collect all tasks (each variable-fit-member combination is a task)
        all_tasks = []
        
        for var in vars:
            print('='*width)
            print(f"🏋🏼‍♀️ Setting up tasks for: {var}")
            print('='*width)
            
            # Make data directory if it doesn't exist
            os.makedirs(DATA_ROOT / 'CMIP6' / var / 'gev', exist_ok=True)
            data_path = DATA_ROOT / 'CMIP6' / var / 'landonly'
            
            # Find model with most members
            model_with_most, fpath, n_members, tied_models = find_model_with_most_members(
                var, CMIPConfig, data_path
            )
            
            message = (f"🪛 Identified {model_with_most} as model with most ensemble "
                      f"members (has {n_members} members).")
            print(message)
            
            if tied_models:
                print(f"Note: This model was tied with {tied_models}!")
            
            # Create one task for each fit type and member for this variable
            for fit_type in fit_types:
                for mem in CMIPConfig.ensemble_config[model_with_most].ensemble_members:
                    all_tasks.append({
                        'var': var,
                        'model_with_most': model_with_most,
                        'mem': mem,
                        'fpath': fpath,
                        'STAT': STAT,
                        'fit_type': fit_type,
                        'width': width
                    })
        
        print(f"\n📋 Total tasks to process: {len(all_tasks)}")
        print(f"   ({len(vars)} variables × {len(fit_types)} fit types × ~{len(all_tasks)//(len(vars)*len(fit_types))} members)")
        print(f"🖥️  Number of MPI processes: {size}")
        
        if len(all_tasks) <= size:
            print(f"✨ All {len(all_tasks)} tasks can run simultaneously!")
        else:
            print(f"📊 Tasks per process: ~{len(all_tasks) / size:.1f}")
        
        print('='*width)
    else:
        all_tasks = None
    
    # Broadcast tasks to all processes
    all_tasks = comm.bcast(all_tasks, root=0)
    
    # Distribute tasks using round-robin distribution
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    if len(my_tasks) > 0:
        print(f"[Rank {rank}] Processing {len(my_tasks)} tasks")
    else:
        print(f"[Rank {rank}] No tasks assigned (more processes than tasks)")
    
    # Process assigned tasks
    my_results = []
    for task_idx, task in enumerate(my_tasks):
        print(f"[Rank {rank}] Processing task {task_idx+1}/{len(my_tasks)}: "
              f"{task['var']}:{task['fit_type']}:{task['mem']}")
        
        result = process_single_fit(
            var=task['var'],
            model_with_most=task['model_with_most'],
            mem=task['mem'],
            fpath=task['fpath'],
            STAT=task['STAT'],
            fit_type=task['fit_type'],
            width=task['width'],
            rank=rank
        )
        my_results.append(result)
    
    # Gather all results to rank 0
    all_results = comm.gather(my_results, root=0)
    
    # Rank 0 combines results and prints summary
    if rank == 0:
        # Flatten results
        flat_results = [item for sublist in all_results for item in sublist]
        
        successes = sum(1 for r in flat_results if r[0])
        failures = sum(1 for r in flat_results if not r[0])
        
        # Combine successful results into datasets
        if successes > 0:
            output_paths = combine_results_into_datasets(
                flat_results, model_with_most, STAT, DATA_ROOT, width
            )
        else:
            output_paths = {}
        
        # Count by variable and fit type
        breakdown = {}
        for r in flat_results:
            var = r[1]
            fit_type = r[2]
            key = f"{var}:{fit_type}"
            if key not in breakdown:
                breakdown[key] = {'success': 0, 'failure': 0}
            if r[0]:
                breakdown[key]['success'] += 1
            else:
                breakdown[key]['failure'] += 1
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print('='*width)
        print("📊 SUMMARY")
        print('='*width)
        print(f"✅ Successful: {successes}/{len(flat_results)}")
        print(f"❌ Failed: {failures}/{len(flat_results)}")
        print(f"\nBreakdown by variable and fit type:")
        for key, counts in sorted(breakdown.items()):
            total = counts['success'] + counts['failure']
            status = "✅" if counts['success'] == total else "⚠️"
            print(f"  {status} {key:30s}: {counts['success']}/{total}")
            if key in [(k[0] + ':' + k[1]) for k in output_paths.keys()]:
                var_key, fit_key = key.split(':')
                path = output_paths.get((var_key, fit_key), 'N/A')
                print(f"      📁 Output: {path}")
        
        print(f"\n⏱️  Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"⚡ Average time per task: {elapsed/len(flat_results):.2f} seconds")
        
        if failures > 0:
            print("\n❌ Failed tasks:")
            for r in flat_results:
                if not r[0]:
                    print(f"   - {r[1]}:{r[2]}:{r[3]}")
                    if r[7]:  # error message
                        print(f"     Error: {r[7][:200]}...")  # Truncate long errors
        
        print('='*width)
        print("🥳 All done! 🥳")
        print('='*width)

if __name__ == "__main__":
    main()  # R U N   I T