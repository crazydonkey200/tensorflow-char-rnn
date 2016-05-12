python train.py \
       --data_file=data/eecs349-data.txt \
       --num_epochs=100 \
       --hidden_size=8 \
       --num_layers=1 \
       --model="rnn" \
       --batch_size=64 \
       --output_dir=small

python sample.py --init_dir=small
screen -X quit
