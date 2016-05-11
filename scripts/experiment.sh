python train.py \
       --data_file=data/tiny_shakespeare.txt \
       --num_epochs=50 \
       --hidden_size=8 \
       --num_layers=1 \
       --log_to_file \
       --output_dir=small

python sample.py --init_dir=small
screen -X
