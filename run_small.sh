#!/bin/sh

# hacky: max epochs needs to be high as to not collide with training-length
# model params indicates neurons per layer -> [16, 16] -> 2 layers with 16 neurons each

META_PARAMS="--directory ./data/cfire  --model-seed 666 --num-runs 50 --max-epochs 500  --data-seed 11880"
echo "Using the following params for all: $META_PARAMS"

WINE_PARAMS="--modelparams [128,128] --training-length 80 --batch-sizes [32,36]"
python 1_train_models.py $META_PARAMS $WINE_PARAMS --dataset wine  &
IRIS_PARAMS="--modelparams [128,128] --training-length 80 --batch-sizes [32,38]"
python 1_train_models.py $META_PARAMS $IRIS_PARAMS --dataset iris  &

DIGGLE_PARAMS="--modelparams [128,128,128] --training-length 200 --batch-sizes [32,124]"
python 1_train_torch.py $META_PARAMS $DIGGLE_PARAMS --dataset diggle  &

AUTOUNIV_PARAMS="--modelparams [256,256] --training-length 1000 --batch-sizes [32,140]"
python 1_train_torch.py $META_PARAMS $AUTOUNIV_PARAMS --dataset autouniv  &

VEHILCE_PARAMS="--modelparams [128,128,128] --training-length 350 --batch-sizes [32,170]"
python 1_train_torch.py $META_PARAMS $VEHILCE_PARAMS --dataset vehicle  &

ABALONE_PARAMS="--modelparams [32,32] --training-length 1000 --batch-sizes [32,627]"
python 1_train_torch.py $META_PARAMS $ABALONE_PARAMS --dataset abalone  &

wait

META_PARAMS="--directory ./data/cfire --training-length 500 --model-seed 666 --num-runs 50 --max-epochs 500 --modelparams [32,32,32,32]  --data-seed 11880"
echo "Using the following params for all: $META_PARAMS"

echo "starting spf"
python 1_train_torch.py $META_PARAMS --dataset spf --batch-sizes [32,389] &

echo "starting btsc"
python 1_train_torch.py $META_PARAMS --dataset btsc --batch-sizes [32,150] &

echo "starting breastw"
python 1_train_torch.py $META_PARAMS --dataset breastw --batch-sizes [32,1000] &

echo "starting spambase"
python 1_train_torch.py $META_PARAMS --dataset spambase --batch-sizes [32,921] &
echo "done"


## BEANS
PARAMS="--dataset beans --training-length 800 --modelparams [16,16,16] --batch-sizes [32,1050] --data-seed 11880 --num-runs 10 --max-epochs 1000"
echo "Using the following params for  beans: $PARAMS"
echo "starting beans"
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 666 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 745616 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 615645 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 154665 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 532465 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 724357 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 268423 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 964732 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 251332 &
python 1_train_torch.py $META_PARAMS $PARAMS --model-seed 672532 &

wait

## HELOC
PARAMS="--training-length 5000 --num-runs 20 --max-epochs 10000 --modelparams [8,8,8,8] --batch-sizes [32,732]"
echo $META_PARAMS
echo $PARAMS
python 1_train_torch.py $META_PARAMS --dataset heloc $PARAMS --model-seed 666 &
python 1_train_torch.py $META_PARAMS --dataset heloc $PARAMS --model-seed 745616 &
python 1_train_torch.py $META_PARAMS --dataset heloc $PARAMS --model-seed 615645 &
python 1_train_torch.py $META_PARAMS --dataset heloc $PARAMS --model-seed 154665 &
python 1_train_torch.py $META_PARAMS --dataset heloc $PARAMS --model-seed 532465 &

wait


PARAMS=" --data-seed 11880 --num-runs 100 --max-epochs 1000 --batch-sizes [16,300]"
## BREAST CANCER
echo "starting breast cancer"
python 1_train_torch.py $META_PARAMS --dataset breastcancer --training-length 200 --modelparams [16] $PARAMS &
##
### IONOSPHERE
echo "starting ionosphere"
python 1_train_torch.py $META_PARAMS --dataset ionosphere --training-length 400 --modelparams [8] $PARAMS &
