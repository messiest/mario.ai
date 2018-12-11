ENV=$1
CHECKPOINTS=$2

echo "Fetching logs..."
bash fetch-records.sh $ENV $CHECKPOINTS

echo "Plotting..."
python utils/generate_plots.py --env-name $ENV

echo "Done"
