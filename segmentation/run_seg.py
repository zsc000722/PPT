import random
import subprocess
import multiprocessing


def run_experiment(gpu_id, bs, lr, seed):
    script_path = "./finetune_segmentation.sh"
    command = f"bash {script_path} {gpu_id} {seed} {bs} {lr}"
    subprocess.run(command, shell=True)


if __name__ == "__main__":


    # GPU list
    gpu_ids = list([0])
    # gpu_ids = list([0,1,2,3,5,6,7,8])
    # gpu_ids = list(range(8))
    num_experiments = 3
    
    # pool = multiprocessing.Pool(processes=len(gpu_ids))
    seed = []
    for i in range(num_experiments):
        seed.append(random.randint(0, 65535))
    for i in range(num_experiments):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        if i == 0: 
            bs, lr = 32, 2e-4
        elif i == 0: 
            bs, lr = 64, 1e-4
        elif i == 0: 
            bs, lr = 32, 1e-4
        # pool.apply_async(run_experiment, args=(gpu_id, bs, lr, 0))
        # print(1)
        run_experiment(gpu_id, bs, lr, 0)

    # pool.close()
    # pool.join()
