# Used to train a model on AWS
ENV=$1
ID=$2

echo Training model...

python a3c_trainer.py --env-name $ENV --reset-delay 1 --non-sample 1 --model-id $ID --record
