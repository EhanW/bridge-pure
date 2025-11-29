##### Get clean baseline and LSP-attacked baseline #####
python prepare_imgs.py --dataset cifar100 --attack lsp --num-purifying-data 40000 --get-baseline
python train.py --dataset cifar100 --data-path images/cifar100/clean/40000_val_data_targets/data_targets.npz
python train.py --dataset cifar100 --data-path images/cifar100/lsp/40000_val_data_targets/data_targets.npz

##### Use BridgePure to purify LSP-attacked CIFAR-100 #####
# prepare paired data with gaussian noise intensity beta = 0 and 0.02
python prepare_imgs.py --dataset cifar100 --attack lsp --num-training-data 500  --num-purifying-data 40000 --beta-max 0
# train ddbm on the paired data
bash train_ddbm.sh cifar100 ve lsp_gaussian_0.0 500 40000
# sample from the trained ddbm model, with different random level s = 0.33 and 0.8
bash sample_ddbm.sh cifar100 ve lsp_gaussian_0.0 500 40000 0.33 0.5 test 100000
bash sample_ddbm.sh cifar100 ve lsp_gaussian_0.0 500 40000 0.8 0.5 test 100000
# train classifiers on the purified data
python train.py --dataset cifar100 --data-path workdir/cifar100_lsp_gaussian_0_500_40000_32_192d_ve/sample_100000/w=0.5_churn=0.33_test_40/data_targets.npz
python train.py --dataset cifar100 --data-path workdir/cifar100_lsp_gaussian_0_500_40000_32_192d_ve/sample_100000/w=0.5_churn=0.8_test_40/data_targets.npz




python prepare_imgs.py --dataset cifar100 --attack lsp --num-training-data 500  --num-purifying-data 40000 --beta-max 0.02
bash train_ddbm.sh cifar100 ve lsp_gaussian_0.02 500 40000
bash sample_ddbm.sh cifar100 ve lsp_gaussian_0.02 500 40000 0.33 0.5 test 100000
bash sample_ddbm.sh cifar100 ve lsp_gaussian_0.02 500 40000 0.8 0.5 test 100000
python train.py --dataset cifar100 --data-path workdir/cifar100_lsp_gaussian_0.02_500_40000_32_192d_ve/sample_100000/w=0.5_churn=0.33_test_40/data_targets.npz
python train.py --dataset cifar100 --data-path workdir/cifar100_lsp_gaussian_0.02_500_40000_32_192d_ve/sample_100000/w=0.5_churn=0.8_test_40/data_targets.npz
