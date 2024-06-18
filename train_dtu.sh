for i in 65 118 122 114 110 106 105 97 83 69 63 55 40 37 24
do
    python train.py -s /media/super/data/dataset/dtu/DTU_mask/scan$i/ -m output_decoupled2/scan$i -r 2 --use_decoupled_appearance
done