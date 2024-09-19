#for datasetname in Cora Citeseer
#do
#  for power in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#  do
#      python training.py --dataset $datasetname --net FracGCN --frac_power $power
#  done
#done

#for datasetname in Cornell Chameleon Squirrel
#do
#  python training.py --dataset $datasetname --net GCN
#done

for rate in 0.05
do
  python training.py --dataset Texas --net GCN --change_labels $rate
  for power in 0.5 0.6 0.7 0.8 0.9
  do
    python training.py --dataset Texas --net FracGCN --frac_power $power --change_labels $rate
  done
done
