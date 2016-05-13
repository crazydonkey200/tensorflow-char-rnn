python train.py \
       --data_file=data/eecs349-data.txt \
       --num_epochs=50 \
       --hidden_size=256 \
       --num_layers=1 \
       --model="rnn" \
       --batch_size=64 \
       --output_dir=large \

tensorboard --logdir=large/tensorboard_log/ --port=6007
