python model.py \
	-lr 0.0001 \
	-l /Users/ctang/dev/CarND-Behavioral-Cloning-P3/data_udacity/driving_log.csv \
	-i /Users/ctang/dev/CarND-Behavioral-Cloning-P3/data_udacity/IMG \
	--epochs 10 \
	--droprate 0.5 \
	--activation 'relu' \
	--conv1filters 24 \
	--conv2filters 32
