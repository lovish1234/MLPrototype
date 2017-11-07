## Experiments concering Unpaired Image to Image Translation with Limited Data
Simple prototypes of common ML concepts

### Effect of Number of Instances on DC-GAN

#### FashionMNIST

Class | 10 | 100 | 1000
:---: | :---: | :---: | :---: |
T-Shirt/Top | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_0_10_1000.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_0_100_1000.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_0_1000_20.png" alt="FashionMNIST 1000 instances" width="200px"/> 
Trouser | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_1_10_1000.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_1_100_1000.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_1_1000_20.png" alt="FashionMNIST 1000 instances" width="200px"/> 
Pullover | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_2_10_1000.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_2_100_1000.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/FashionMNIST/FashionMNIST_2_1000_20.png" alt="FashionMNIST 1000 instances" width="200px"/>


#### MNIST

Class | 10 | 100 | 1000
:---: | :---: | :---: | :---: |
'1' | <img src="./DCGAN/plots/MNIST/MNIST_1_10_1000.png" alt="MNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/MNIST/MNIST_1_100_100.png" alt="MNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/MNIST/MNIST_1_1000_20.png" alt="MNIST 1000 instances" width="200px"/>
'2' | <img src="./DCGAN/plots/MNIST/MNIST_2_10_1000.png" alt="MNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/MNIST/MNIST_2_100_100.png" alt="MNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/MNIST/MNIST_2_1000_20.png" alt="MNIST 1000 instances" width="200px"/>
'3' | <img src="./DCGAN/plots/MNIST/MNIST_3_10_1000.png" alt="MNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/MNIST/MNIST_3_100_100.png" alt="MNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/MNIST/MNIST_3_1000_20.png" alt="MNIST 1000 instances" width="200px"/>


#### notMNIST
Class | 10 | 100 | 1000 |
:---: | :---: | :---: | :---: |
'B'| <img src="./DCGAN/plots/notMNIST/notMNIST_1_10_1000.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/notMNIST/notMNIST_1_100_100.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/notMNIST/notMNIST_1_1000_20.png" alt="notMNIST 1000 instances" width="200px"/> 
'C' | <img src="./DCGAN/plots/notMNIST/notMNIST_2_10_1000.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/notMNIST/notMNIST_2_100_100.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/notMNIST/notMNIST_2_1000_20.png" alt="notMNIST 1000 instances" width="200px"/> 
'D' | <img src="./DCGAN/plots/notMNIST/notMNIST_3_10_1000.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./DCGAN/plots/notMNIST/notMNIST_3_100_100.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./DCGAN/plots/notMNIST/notMNIST_3_1000_20.png" alt="notMNIST 1000 instances" width="200px"/> 


### Effect of number of Instances of Target Domain on Cycle GAN 

#### MNIST

Source | Target | 10 | 100 | 1000
:---: | :---: | :---: | :---: | :---: |
'1' | '7' | <img src="./CycleGAN/results/MNIST/1_7/MNIST_1_7_10_10_200_results/AtoB/Test_result_20.png" alt="MNIST 10 instances" width="200px"/> | <img src="./CycleGAN/results/MNIST/1_7/MNIST_1_7_100_100_200_results/AtoB/Test_result_20.png" alt="MNIST 100 instances" width="200px"/> | <img src="./CycleGAN/results/MNIST/1_7/MNIST_1_7_1000_1000_200_results/AtoB/Test_result_20.png" alt="MNIST 1000 instances" width="200px"/>
'3' | '8' |<img src="./CycleGAN/results/MNIST/3_8/MNIST_3_8_10_10_200_results/AtoB/Test_result_20.png" alt="MNIST 10 instances" width="200px"/> | <img src="./CycleGAN/results/MNIST/3_8/MNIST_3_8_100_100_200_results/AtoB/Test_result_20.png" alt="MNIST 100 instances" width="200px"/> | <img src="./CycleGAN/results/MNIST/3_8/MNIST_3_8_1000_1000_40_results/AtoB/Test_result_20.png" alt="MNIST 1000 instances" width="200px"/>
'6' | '9' |<img src="./CycleGAN/results/MNIST/6_9/MNIST_6_9_10_10_200_results/AtoB/Test_result_20.png" alt="MNIST 10 instances" width="200px"/> | <img src="./CycleGAN/results/MNIST/6_9/MNIST_6_9_100_100_200_results/AtoB/Test_result_20.png" alt="MNIST 100 instances" width="200px"/> | <img src="./CycleGAN/results/MNIST/6_9/MNIST_6_9_1000_1000_40_results/AtoB/Test_result_20.png" alt="MNIST 1000 instances" width="200px"/>

#### FashionMNIST

Source | Target | 10 | 100 | 1000
:---: | :---: | :---: | :---: | :---: |
Sneaker | Ankle Boot | <img src="./CycleGAN/results/FashionMNIST/7_9/FashionMNIST_7_9_10_10_200_results/AtoB/Test_result_20.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./CycleGAN/results/FashionMNIST/7_9/FashionMNIST_7_9_100_100_200_results/AtoB/Test_result_20.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./CycleGAN/results/FashionMNIST/7_9/FashionMNIST_7_9_1000_1000_40_results/AtoB/Test_result_20.png" alt="FashionMNIST 1000 instances" width="200px"/>
Sneaker | Bag | <img src="./CycleGAN/results/FashionMNIST/7_8/FashionMNIST_7_8_10_10_200_results/AtoB/Test_result_20.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./CycleGAN/results/FashionMNIST/7_8/FashionMNIST_7_8_100_100_200_results/AtoB/Test_result_20.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./CycleGAN/results/FashionMNIST/7_8/FashionMNIST_7_8_1000_1000_40_results/AtoB/Test_result_20.png" alt="FashionMNIST 1000 instances" width="200px"/>

#### notMNIST

Source | Target | 10 | 100 | 1000
:---: | :---: | :---: | :---: | :---: |
Sneaker | Bag |  <img src="./CycleGAN/results/notMNIST/4_5/notMNIST_4_5_10_10_200_results/AtoB/Test_result_20.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./CycleGAN/results/notMNIST/4_5/notMNIST_4_5_100_100_200_results/AtoB/Test_result_20.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./CycleGAN/results/notMNIST/4_5/notMNIST_4_5_1000_1000_40_results/AtoB/Test_result_20.png" alt="notMNIST 1000 instances" width="200px"/>








