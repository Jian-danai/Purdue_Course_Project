# Final Project ECE 50024

# Full Run: Train & Test
'CUDA_VISIBLE_DEVICES=0 python train.py --log_name "horse2zebra_run1"`

# Test
'CUDA_VISIBLE_DEVICES=0 python train.py --mode "test" --log_name "horse2zebra_run1" --epoch 200`

Replace 'horse2zebra' to any other dataset name from CycleGAN. You can add any suffix after the first '_' in the log_name.

Rememeber the dataset structure must be: 

```
dataset_name
            --trainA

            |
            
            --trainB
            
            |
            
            --testA
            
            |
            
            --testB
            
           ```
