#!/bin/bash
#
#SBATCH --job-name=agmlRarray
#SBATCH --output=output_agmlR%j.txt
#SBATCH --error=error_agmlR%j.txt
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-7
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --time=7-24:00:00
#SBATCH --constraint=gen3

config=/path/to/agml/predictors/config.txt
indicator=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

# module load 2023
module load R/4.2.0 gdal geos proj
# terra package installation works after this

Rscript predictor_data_prep.r -c wheat -r EU -i ${indicator} -p 8
