#!/bin/bash

celebs=('Harry' 'IU' 'Margot' 'Michael' 'Sundar')

for celeb in ${celebs[@]};
do
  mv /playpen-nas-ssd/awang/mystyle_original/out/$celeb/synthesis /playpen-nas-ssd/awang/mystyle_original/out/$celeb/synthesis_old
done