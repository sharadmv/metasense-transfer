set -x
python scripts/train_split_model.py splits-500-3relu 4_elcajon_11 --round 4  --location elcajon --board 11 --dim 3 --hidden-size 100 --num-iter 1000000
python scripts/train_split_model.py splits-500-3relu 4_elcajon_12 --round 4  --location elcajon --board 12 --dim 3 --hidden-size 100 --num-iter 1000000
python scripts/train_split_model.py splits-500-3relu 4_elcajon_13 --round 4  --location elcajon --board 13 --dim 3 --hidden-size 100 --num-iter 1000000
python scripts/train_split_model.py splits-500-3relu 4_donovan_17 --round 4  --location donovan --board 17 --dim 3 --hidden-size 100 --num-iter 1000000
#python scripts/train_split_model.py splits-500-100 4_donovan_19 --round 4  --location donovan --board 19 --dim 100 --hidden-size 500
#python scripts/train_split_model.py splits-500-100 4_donovan_21 --round 4  --location donovan --board 21 --dim 100 --hidden-size 500
#python scripts/train_split_model.py splits-500-100 4_shafter_15 --round 4  --location shafter --board 15 --dim 100 --hidden-size 500
#python scripts/train_split_model.py splits-500-100 4_shafter_18 --round 4  --location shafter --board 18 --dim 100 --hidden-size 500
#python scripts/train_split_model.py splits-500-100 4_shafter_20 --round 4  --location shafter --board 20 --dim 100 --hidden-size 500

#python scripts/train_split_model.py splits-500-100 2_elcajon_17-3_shafter_17 --round 2,3  --location elcajon,shafter --board 17,17
#python scripts/train_split_model.py splits-500-100 2_elcajon_19-3_shafter_19 --round 2,3  --location elcajon,shafter --board 19,19
#python scripts/train_split_model.py splits-500-100 2_elcajon_21-3_shafter_21 --round 2,3  --location elcajon,shafter --board 21,21
#python scripts/train_split_model.py splits-500-100 2_donovan_15-3_elcajon_15 --round 2,3  --location donovan,elcajon --board 15,15
#python scripts/train_split_model.py splits-500-100 2_donovan_18-3_elcajon_18 --round 2,3  --location donovan,elcajon --board 18,18
#python scripts/train_split_model.py splits-500-100 2_donovan_20-3_elcajon_20 --round 2,3  --location donovan,elcajon --board 20,20
#python scripts/train_split_model.py splits-500-100 2_shafter_11-3_donovan_11 --round 2,3  --location shafter,donovan --board 11,11
#python scripts/train_split_model.py splits-500-100 2_shafter_12-3_donovan_12 --round 2,3  --location shafter,donovan --board 12,12
#python scripts/train_split_model.py splits-500-100 2_shafter_13-3_donovan_13 --round 2,3  --location shafter,donovan --board 13,13

#python scripts/train_split_model.py splits-500-100 3_elcajon_15-4_shafter_15 --round 3,4  --location elcajon,shafter --board 15,15
#python scripts/train_split_model.py splits-500-100 3_elcajon_18-4_shafter_18 --round 3,4  --location elcajon,shafter --board 18,18
#python scripts/train_split_model.py splits-500-100 3_elcajon_20-4_shafter_20 --round 3,4  --location elcajon,shafter --board 20,20
#python scripts/train_split_model.py splits-500-100 3_donovan_11-4_elcajon_11 --round 3,4  --location donovan,elcajon --board 11,11
#python scripts/train_split_model.py splits-500-100 3_donovan_12-4_elcajon_12 --round 3,4  --location donovan,elcajon --board 12,12
#python scripts/train_split_model.py splits-500-100 3_donovan_13-4_elcajon-13 --round 3,4  --location donovan,elcajon --board 13,13
#python scripts/train_split_model.py splits-500-100 3_shafter_17-4_donovan_17 --round 3,4  --location shafter,donovan --board 17,17
#python scripts/train_split_model.py splits-500-100 3_shafter_19-4_donovan_19 --round 3,4  --location shafter,donovan --board 19,19
#python scripts/train_split_model.py splits-500-100 3_shafter_21-4_donovan_21 --round 3,4  --location shafter,donovan --board 21,21
