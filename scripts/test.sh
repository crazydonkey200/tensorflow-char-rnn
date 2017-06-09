python train.py --data_file=data/tiny_shakespeare.txt --num_epochs=5 --test --n_save=2
python train.py --data_file=data/tiny_shakespeare.txt --num_epochs=5 --test --init_dir=output --n_save=2
python sample.py --init_dir=output

