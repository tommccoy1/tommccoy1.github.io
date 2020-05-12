

fi = open("evals_to_run.txt", "r")
fo_all = open("all_scripts.sh", "w")

for line in fi:
	parts = line.strip().split("\t")
	key = parts[0]
	command = parts[1]

	fo_all.write("sbatch " + key + ".scr\n")
	fo = open(key + ".scr", "w")

	fo.write("#!/bin/bash\n")
	fo.write("#SBATCH --job-name=" + key + "\n")
	fo.write("#SBATCH --time=72:0:0\n")
	fo.write("#SBATCH --partition=shared\n")
	fo.write("#SBATCH --nodes=1\n")
	fo.write("#SBATCH --ntasks-per-node=2\n")
	fo.write("#SBATCH --mail-type=end\n")
	fo.write("#SBATCH --mail-user=rmccoy20@jhu.edu\n")
	fo.write("#SBATCH --output=" + key + ".log\n")
	fo.write("#SBATCH --error=" + key + ".err\n\n")
	fo.write("module load pytorch\n\n")
	fo.write(command + "\n\n")







