#!/bin/bash
celebs=('IU' 'Michael' 'Harry' 'Sundar' 'Margot')
experiments=('kmeans_3' 'kmeans_5' 'kmeans_10')
gpus=(1 2 3 4 5)

# # for each index, run the command for the celeb at that index and for gpu of that index
# for i in "${!celebs[@]}"
#     do
#         celeb=${celebs[$i]}
#         (
#             for experiment in ${experiments[@]}
#             do
#                 # device is index + 2
#                 device=$((i+2))
# for each index, run the command for the celeb at that index and for gpu of that index
for i in "${!celebs[@]}"
    do
        celeb=${celebs[$i]}
        device=${gpus[$i]}
        (
            for experiment in ${experiments[@]}
            do
                CUDA_VISIBLE_DEVICES=$device python eval/batch_synthesize_celeb.py --celeb $celeb --experiment $experiment
            done
        ) &
    done

# wait until all training is done

    
#             done
#         ) &
#     done
#     wait
