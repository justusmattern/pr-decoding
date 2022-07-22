import os

for name in ['ur0937743', 'ur17571044', 'ur17825945', 'ur23055365', 'ur6216723.txt', 'ur0806494', 'ur1293485', 'ur1773414', 'ur2206551', 'ur2496397']:
    os.command(f"python train_lm.py --model-name gpt2 --train-file data/{name}.txt --test-file data/{name}.txt --model-save-name {name} --batch-size 4 --num-epochs 5")