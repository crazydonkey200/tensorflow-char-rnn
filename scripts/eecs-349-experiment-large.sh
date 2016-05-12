python train.py \
       --data_file=data/eecs349-data.txt \
       --num_epochs=50 \
       --hidden_size=256 \
       --num_layers=1 \
       --model="rnn" \
       --batch_size=100 \
       --output_dir=large \

python sample.py --init_dir=large
screen -X quit
