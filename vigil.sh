#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --job-name=refseg

module load python/3.6.8
module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2

set -e
#dataset="referit"
dataset=$1

if [[ $dataset == "flickr30k" ]]
then
    if [ -d "/ssd_scratch/cvit/kanishk/flickr30k_files/IMAGES_.lmdb" ]
    then 
        echo "folder exists, proceed with training"
    else
        mkdir -p /ssd_scratch/cvit/kanishk/
        rm -rf /ssd_scratch/cvit/kanishk/*

        echo "copying features from share3 to ssd_scratch"
        scp -r kanishk@ada:/share3/kanishk/flickr30k_files /ssd_scratch/cvit/kanishk/
        echo "copied features from scratch3"

        tar xf /ssd_scratch/cvit/kanishk/flickr30k_files/Annotations.tar -C /ssd_scratch/cvit/kanishk/flickr30k_files/
        tar xf /ssd_scratch/cvit/kanishk/flickr30k_files/Sentences.tar -C /ssd_scratch/cvit/kanishk/flickr30k_files/

        rm /ssd_scratch/cvit/kanishk/flickr30k_files/Annotations.tar
        rm /ssd_scratch/cvit/kanishk/flickr30k_files/Sentences.tar

        echo "Extracted Annotations and Sentences folders"
    fi
elif [[ $dataset == "referit" ]]
then
    if [ -d "/ssd_scratch/cvit/kanishk/referit/referit_data" ]; then
	    echo "initiate training on referit"
    else
        mkdir -p /ssd_scratch/cvit/kanishk/referit/cache

        echo "copying features from share3 to ssd_scratch"
        scp -r kanishk@ada:/share3/kanishk/referit.tar /ssd_scratch/cvit/kanishk/referit/
        echo "copied features from scratch3"

        tar xf /ssd_scratch/cvit/kanishk/referit/referit.tar -C /ssd_scratch/cvit/kanishk/referit/
        rm /ssd_scratch/cvit/kanishk/referit/referit.tar

        echo "Extracted Annotations and Sentences folders"
    fi
elif [[ $dataset == "genome" ]]
then
    if [ -f "/ssd_scratch/cvit/kanishk/visual_genome/image_data.json" ]; then
        echo "Folder exists, proceed with training"
    else
        mkdir -p /ssd_scratch/cvit/kanishk/visual_genome/cache

        echo "copying features from share3 to ssd_scratch"
        scp -r kanishk@ada:/share3/kanishk/visual_genome /ssd_scratch/cvit/kanishk/
        echo "copied features from scratch3"

        unzip -qq /ssd_scratch/cvit/kanishk/visual_genome/\*.zip -d /ssd_scratch/cvit/kanishk/visual_genome/
        rm /ssd_scratch/cvit/kanishk/visual_genome/*.zip
        echo "Extracted files from zip"
    fi
else
    echo "Incorrect dataset provided: $dataset"
    exit 1
fi

scp -r kanishk@ada:/share3/kanishk/models.tar /ssd_scratch/cvit/kanishk/
