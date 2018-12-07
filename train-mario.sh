# Used to train a model on AWS

ID=$1

echo Training model...

python a3c_trainer.py --env-name SuperMarioBrosNoFrameskip-1-1-v0 --reset-delay 1 --non-sample 1 --model-id $ID --record
