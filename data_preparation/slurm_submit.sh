#!/bin/bash
#
#SBATCH --job-name=dilli_agmlR%j
#SBATCH --output=output_agmlR%j.txt
#SBATCH --error=error_agmlR%j.txt
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --time=1-24:00:00
#SBATCH --constraint=gen3

# module load 2023
# module load R-bundle-CRAN
export MODULEPATH=/shared/eb_modules/all

module load R/4.3.2-foss-2022b

Rscript data_prep.r
