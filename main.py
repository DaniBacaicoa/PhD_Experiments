import os
import timm.optim
import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
from ucimlrepo import fetch_ucirepo 

from src.dataset import Data_handling
from src.weakener import Weakener
from src.model import MLP, ResNet_18, BasicBlock, ResNet18CIFAR,ResNet, ResNet50, ResNet_last, BasicBlock2 #ResNet18,ResNet32,ResNet18_old
from utils.datasets_generation import generate_dataset
import utils.losses as losses
from utils.train_test_loop import train_and_evaluate

import timm

def main(args):
    reps = args.reps
    dataset_base_path = args.dataset_base_path
    dataset = args.dataset
    corruption = args.corruption
    corr_p = args.corr_p
    corr_n = args.corr_n
    loss_type = args.loss_type
    epochs = args.epochs
    model = args.model
    learning_rate = args.lr


    for i in range(reps):

        generate_dataset(dataset=dataset,corruption=corruption,corr_p=corr_p,corr_n=corr_n,repetitions=i,loss_type=loss_type)
        
        base_dir = dataset_base_path
        if corr_n is not None:
            folder_path = os.path.join(base_dir, f'{dataset}_{corruption}_p_+{corr_p}p_-{corr_n}')
        else:
            folder_path = os.path.join(base_dir, f'{dataset}_{corruption}_p{corr_p}')
        
        # Load the dataset
        with open(folder_path + f'/Dataset_{i}.pkl', 'rb') as f:
            Data, Weak = pickle.load(f)
            
        Data.include_weak(Weak.z)
        # Select the appropriate loss function
        if loss_type == 'Supervised':
            loss_fn = losses.FwdBwdLoss(np.eye(Weak.c), np.eye(Weak.c))
        if loss_type == 'Backward':
            loss_fn = losses.FwdBwdLoss(Weak.Y, np.eye(Weak.c))
        elif loss_type == 'Forward':
            loss_fn = losses.FwdBwdLoss(np.eye(Weak.d), Weak.M)
        elif loss_type == 'FB_decomposed':
            B = np.linalg.pinv(Weak.Ml)
            loss_fn = losses.FwdBwdLoss(B, Weak.Mr)
        elif loss_type == 'Forward_opt':
            pest = torch.from_numpy(Weak.generate_wl_priors())
            tm = torch.from_numpy(Weak.M)
            B = tm @ torch.inverse(tm.T @ torch.inverse(torch.diag(pest)) @ tm) @ tm.T @ torch.inverse(torch.diag(pest))
            loss_fn = losses.FwdBwdLoss(B, Weak.M)
        elif loss_type == 'Backward_opt':
            loss_fn = losses.FwdBwdLoss(Weak.Y_opt, np.eye(Weak.c))
        elif loss_type == 'Backward_conv':
            loss_fn = losses.FwdBwdLoss(Weak.Y_conv, np.eye(Weak.c))
        elif loss_type == 'Backward_opt_conv':
            loss_fn = losses.FwdBwdLoss(Weak.Y_opt_conv, np.eye(Weak.c))

        


        # Prepare data loaders
        trainloader, testloader = Data.get_dataloader(weak_labels='weak')

        if model == 'lr':
            lr = MLP(Data.num_features, [], Weak.c, dropout_p=0, bn=False, activation='id')
            optim = torch.optim.Adam(lr.parameters(), lr=1e-3)
            lr, results = train_and_evaluate(lr, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            if dataset == 'gmm':
                results_dict = {'overall_models': lr}
                res_dir = f"Results/{dataset}_{corruption}"
                os.makedirs(res_dir, exist_ok=True)
                if corr_n is not None:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                    pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
                else:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                    pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
                file_path = os.path.join(res_dir, file_name)
                pickle_path = os.path.join(res_dir, pickle_name)
                results.to_csv(file_path, index=False)
                with open(pickle_path, "wb") as f:
                    pickle.dump(results_dict, f)
            else:
                res_dir = f"Results/{dataset}_{corruption}"
                os.makedirs(res_dir, exist_ok=True)
                if corr_n is not None:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                else:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                file_path = os.path.join(res_dir, file_name)
                results.to_csv(file_path, index=False)
        elif model == 'mlp':
            mlp = MLP(Data.num_features, [Data.num_features], Weak.c, dropout_p=0.5, bn=True, activation='tanh')
            optim = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
            mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            if dataset == 'gmm':
                results_dict = {'overall_models': mlp}
                res_dir = f"Results/{dataset}_{corruption}"
                os.makedirs(res_dir, exist_ok=True)
                if corr_n is not None:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                    pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
                else:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                    pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
                file_path = os.path.join(res_dir, file_name)
                pickle_path = os.path.join(res_dir, pickle_name)
                results.to_csv(file_path, index=False)
                with open(pickle_path, "wb") as f:
                    pickle.dump(results_dict, f)
            else:
                res_dir = f"Results/{dataset}_{corruption}"
                os.makedirs(res_dir, exist_ok=True)
                if corr_n is not None:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                else:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                file_path = os.path.join(res_dir, file_name)
                results.to_csv(file_path, index=False)
        elif model == 'resnet18':
            #mlp = ResNet18(num_classes=10)
            mlp = ResNet_18(num_classes=10)
            optim = torch.optim.SGD(mlp.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            #optim = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
            mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=5, rep=i, loss_type=loss_type)
            results_dict = {'overall_models': mlp}
            res_dir = f"Results/{dataset}_{corruption}"
            os.makedirs(res_dir, exist_ok=True)
            if corr_n is not None:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            else:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            file_path = os.path.join(res_dir, file_name)
            pickle_path = os.path.join(res_dir, pickle_name)
            results.to_csv(file_path, index=False)
            with open(pickle_path, "wb") as f:
                pickle.dump(results_dict, f)
        elif model == 'resnet18_old':
            mlp = ResNet18_old(num_classes=10)
            optim = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
            mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            results_dict = {'overall_models': mlp}
            res_dir = f"Results/{dataset}_{corruption}"
            os.makedirs(res_dir, exist_ok=True)
            if corr_n is not None:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            else:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            file_path = os.path.join(res_dir, file_name)
            pickle_path = os.path.join(res_dir, pickle_name)
            results.to_csv(file_path, index=False)
            with open(pickle_path, "wb") as f:
                pickle.dump(results_dict, f)
        elif model == 'resnet50':
            #mlp = ResNet32(num_classes=20)
            mlp = ResNet50(num_classes=14, fine_tune_all=False) # For Clothing1M
            
            #mlp = ResNet50(num_classes=20, fine_tune_all=False) #For CIFAR100
            if mlp.fine_tune_all:
                params_to_optimize = mlp.parameters()
            else:
                params_to_optimize = mlp.resnet50.fc.parameters()
            #optim = torch.optim.Adam(params_to_optimize, lr=learning_rate) #Clothing1M
            
            optim = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            #optim = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
            mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            results_dict = {'overall_models': mlp}
            res_dir = f"Results/{dataset}_{corruption}"
            os.makedirs(res_dir, exist_ok=True)
            if corr_n is not None:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            else:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            file_path = os.path.join(res_dir, file_name)
            pickle_path = os.path.join(res_dir, pickle_name)
            results.to_csv(file_path, index=False)
            with open(pickle_path, "wb") as f:
                pickle.dump(results_dict, f)
        elif model == 'efficientnet':
            mlp = timm.create_model('efficientnet_b0', pretrained=False, num_classes=20)
            mlp.conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # Adapt for 32x32
            mlp = mlp.to('cuda')
            optim = timm.optim.create_optimizer_v2(mlp, opt='adamw', lr=1e-3, weight_decay=1e-4)
            mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            results_dict = {'overall_models': mlp}
            res_dir = f"Results/{dataset}_{corruption}"
            os.makedirs(res_dir, exist_ok=True)
            if corr_n is not None:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            else:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            file_path = os.path.join(res_dir, file_name)
            pickle_path = os.path.join(res_dir, pickle_name)
            results.to_csv(file_path, index=False)
            with open(pickle_path, "wb") as f:
                pickle.dump(results_dict, f)
        elif model == 'resnet32':
            #mlp = ResNet32(num_classes=20)
            mlp = ResNet(BasicBlock, layers=[5, 5, 5], num_classes=20)
            optim = torch.optim.SGD(mlp.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            #optim = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
            mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            results_dict = {'overall_models': mlp}
            res_dir = f"Results/{dataset}_{corruption}"
            os.makedirs(res_dir, exist_ok=True)
            if corr_n is not None:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            else:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            file_path = os.path.join(res_dir, file_name)
            pickle_path = os.path.join(res_dir, pickle_name)
            results.to_csv(file_path, index=False)
            with open(pickle_path, "wb") as f:
                pickle.dump(results_dict, f)
        elif model == 'resnet34':
            mlp = ResNet_last(BasicBlock2, [3,4,6,3], num_classes=20)
            optim = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            #optim = torch.optim.SGD(mlp.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            results_dict = {'overall_models': mlp}
            res_dir = f"Results/{dataset}_{corruption}"
            os.makedirs(res_dir, exist_ok=True)
            if corr_n is not None:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            else:
                file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
            file_path = os.path.join(res_dir, file_name)
            pickle_path = os.path.join(res_dir, pickle_name)
            results.to_csv(file_path, index=False)
            with open(pickle_path, "wb") as f:
                pickle.dump(results_dict, f)

            

            




    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for dataset handling, model training and evaluation.")
    parser.add_argument("--reps", type=int, default=10, help="Number of repetitions.")
    parser.add_argument("--dataset_base_path", type=str, default='Datasets/weak_datasets', help="Base path for datasets.")
    parser.add_argument("--dataset", type=str, default='image', help="Dataset name.")
    parser.add_argument("--corruption", type=str, default='Partial', help="Corruption type.")
    parser.add_argument("--corr_p", type=float, default=0.5, help="Positive corruption probability.")
    parser.add_argument("--corr_n", type=float, default=None, help="Negative corruption probability.")
    parser.add_argument("--loss_type", type=str, default='Forward', help="Type of loss function to use.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--model", type=str, default='lr', help="Whether to use an MLP or a LR" )
    parser.add_argument("--lr",type=float,default=1e-3,help='Learning_rate')
    
    args = parser.parse_args()
    main(args)


# Clothing1M
# python main.py --reps 1 --dataset clothing1m --model resnet50 --corruption clothing --loss_type Forward --corr_p 0.2 --corr_n 0.2 --epochs 100
# python main.py --reps 1 --dataset clothing1m --model resnet50 --corruption clothing --loss_type Backward --corr_p 0.2 --corr_n 0.2 --epochs 100
# python main.py --reps 1 --dataset clothing1m --model resnet50 --corruption clothing --loss_type FB_decomposed --corr_p 0.2 --corr_n 0.2 --epochs 100

# BINARY OK
## Noisy DONE
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.2 --corr_n 0.2 --epochs 100
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.3 --corr_n 0.1 --epochs 100
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.4 --corr_n 0.4 --epochs 100

# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Backward --corr_p 0.2 --corr_n 0.2 --epochs 100
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Backward --corr_p 0.3 --corr_n 0.1 --epochs 100
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Backward --corr_p 0.4 --corr_n 0.4 --epochs 100

# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type FB_decomposed --corr_p 0.2 --corr_n 0.2 --epochs 100
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type FB_decomposed --corr_p 0.3 --corr_n 0.1 --epochs 100
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type FB_decomposed --corr_p 0.4 --corr_n 0.4 --epochs 100

# python main.py --reps 1 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Supervised --corr_p 0.2 --corr_n 0.2 --epochs 100


# MNIST
## Noisy DONE
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type FB_decomposed --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type FB_decomposed --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type FB_decomposed --corr_p 0.8 --epochs 50

# python main.py --reps 1 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Supervised --corr_p 0.2 --epochs 50


# MNIST
## Complementary DONE
# python main.py --reps 10 --dataset mnist --model mlp --corruption Complementary --loss_type Forward --corr_p 0.2 --epochs 50 --lr 1e-3
# python main.py --reps 10 --dataset mnist --model mlp --corruption Complementary --loss_type Backward --corr_p 0.2 --epochs 50 --lr 1e-3
# python main.py --reps 10 --dataset mnist --model mlp --corruption Complementary --loss_type FB_decomposed --corr_p 0.2 --epochs 50 --lr 1e-3

# python main.py --reps 1 --dataset mnist --model mlp --corruption Complementary --loss_type Supervised --corr_p 0.2 --epochs 50 --lr 1e-3


# MNIST
## pll TBD
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Forward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Forward --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Forward --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_opt --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_opt --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_opt --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_conv --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_conv --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_conv --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_opt_conv --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_opt_conv --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Backward_opt_conv --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Forward_opt --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Forward_opt --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type Forward_opt --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.2 --corr_n 2 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.5 --corr_n 2 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.8 --corr_n 2 --epochs 50 --lr 1e-2

# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.2 --corr_n 5 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.5 --corr_n 5 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.8 --corr_n 5 --epochs 50 --lr 1e-2

# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.2 --corr_n 8 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.5 --corr_n 8 --epochs 50 --lr 1e-2
# python main.py --reps 10 --dataset mnist --model mlp --corruption Partial --loss_type FB_decomposed --corr_p 0.8 --corr_n 8 --epochs 50 --lr 1e-2

# python main.py --reps 1 --dataset mnist --model mlp --corruption Partial --loss_type Supervised --corr_p 0.2 --epochs 50 --lr 1e-2


# GMM
## Noisy DONE
# python main.py --reps 10 --dataset gmm --model lr --corruption unif_noise --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption unif_noise --loss_type Forward --corr_p 0.5 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption unif_noise --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption unif_noise --loss_type Backward --corr_p 0.5 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption unif_noise --loss_type FB_decomposed --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption unif_noise --loss_type FB_decomposed --corr_p 0.5 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption unif_noise --loss_type Supervised --corr_p 0.2 --epochs 50

# GMM  
## Complementary DONE
# python main.py --reps 10 --dataset gmm --model lr --corruption Complementary --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Complementary --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Complementary --loss_type FB_decomposed --corr_p 0.2 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Complementary --loss_type Supervised --corr_p 0.2 --epochs 50

# GMM
## pll
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Forward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Forward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_opt --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_opt_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_opt_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_opt_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Backward_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Forward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Forward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Forward_opt --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type FB_decomposed --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type FB_decomposed --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type FB_decomposed --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Partial --loss_type Supervised --corr_p 0.2 --epochs 50

# CIFAR10
## pll TBD
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Forward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Forward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_opt --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_opt_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_opt_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_opt_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Backward_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Forward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Forward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Forward_opt --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type FB_decomposed --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type FB_decomposed --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type FB_decomposed --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset Cifar10 --model resnet18 --corruption Partial --loss_type Supervised --corr_p 0.2 --epochs 50


# CIFAR10
## Noisy DONE
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10 --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10 --loss_type Forward --corr_p 0.5 --epochs 50
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10 --loss_type Forward --corr_p 0.8 --epochs 50

# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10 --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10 --loss_type Backward --corr_p 0.5 --epochs 50
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10 --loss_type Backward --corr_p 0.8 --epochs 50

# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10  --loss_type FB_decomposed --corr_p 0.2 --epochs 50
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10  --loss_type FB_decomposed --corr_p 0.5 --epochs 50
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10  --loss_type FB_decomposed --corr_p 0.8 --epochs 50

# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Noisy_Patrini_CIFAR10  --loss_type Supervised --corr_p 0.2 --epochs 50


# CIFAR10
## Complementary DONE
# cd /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses/
# source .venv_fb_kumo/bin/activate
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Complementary --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Complementary --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Complementary --loss_type FB_decomposed --corr_p 0.2 --epochs 50

# python main.py --reps 5 --dataset Cifar10 --model resnet18 --corruption Complementary --loss_type Supervised --corr_p 0.2 --epochs 50


# CIFAR100
## Noisy BAD PERFORMANCE
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type Forward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type Forward --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type Forward --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type Backward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type Backward --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type Backward --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type FB_decomposed--corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type FB_decomposed --corr_p 0.5 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type FB_decomposed --corr_p 0.8 --epochs 50 --lr 1e-2

# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Noisy_CIFAR100 --loss_type Supervised --corr_p 0.2 --epochs 50 --lr 1e-2


# CIFAR100
## Complementary BAD PERFORMANCE
# cd /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses/
# source .venv_fb_kumo/bin/activate
# python main.py --reps 5 --dataset Cifar100 --model resnet50 --corruption Complementary --loss_type Forward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet50 --corruption Complementary --loss_type Backward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet50 --corruption Complementary --loss_type FB_decomposed --corr_p 0.2 --epochs 50 --lr 1e-2

# python main.py --reps 5 --dataset Cifar100 --model resnet50 --corruption Complementary --loss_type Supervised --corr_p 0.2 --epochs 50 --lr 1e-2


# CIFAR100
## Complementary
# cd /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses/
# source .venv_fb_kumo/bin/activate
# python main.py --reps 5 --dataset Cifar100 --model efficientnet --corruption Complementary --loss_type Forward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model efficientnet --corruption Complementary --loss_type Backward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model efficientnet --corruption Complementary --loss_type FB_decomposed --corr_p 0.2 --epochs 50 --lr 1e-2

# python main.py --reps 5 --dataset Cifar100 --model efficientnet --corruption Complementary --loss_type Supervised --corr_p 0.2 --epochs 50 --lr 1e-2

# CIFAR100
## Complementary
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Complementary --loss_type Forward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Complementary --loss_type Backward --corr_p 0.2 --epochs 50 --lr 1e-2
# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Complementary --loss_type FB_decomposed --corr_p 0.2 --epochs 50 --lr 1e-2

# python main.py --reps 5 --dataset Cifar100 --model resnet34 --corruption Complementary --loss_type Supervised --corr_p 0.2 --epochs 50 --lr 1e-2