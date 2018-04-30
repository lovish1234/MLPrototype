## Using Domain Adaptation for Few Shot Generative Modeling 

### Effect of Number of Instances on DC-GAN

#### FashionMNIST

Class | 10 | 100 | 1000
:---: | :---: | :---: | :---: |
T-Shirt/Top | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_0_10.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_0_100.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_0_1000.png" alt="FashionMNIST 1000 instances" width="200px"/> 
Trouser | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_1_10.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_1_100.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_1_1000.png" alt="FashionMNIST 1000 instances" width="200px"/> 
Pullover | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_2_10.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_2_100.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/FashionMNIST/FashionMNIST_2_1000.png" alt="FashionMNIST 1000 instances" width="200px"/>


#### MNIST

Class | 10 | 100 | 1000
:---: | :---: | :---: | :---: |
'1' | <img src="./DCGAN/results/samples/MNIST/MNIST_1_10.png" alt="MNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/MNIST/MNIST_1_100.png" alt="MNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/MNIST/MNIST_1_1000.png" alt="MNIST 1000 instances" width="200px"/>
'2' | <img src="./DCGAN/results/samples/MNIST/MNIST_2_10.png" alt="MNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/MNIST/MNIST_2_100.png" alt="MNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/MNIST/MNIST_2_1000.png" alt="MNIST 1000 instances" width="200px"/>
'3' | <img src="./DCGAN/results/samples/MNIST/MNIST_3_10.png" alt="MNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/MNIST/MNIST_3_100.png" alt="MNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/MNIST/MNIST_3_1000.png" alt="MNIST 1000 instances" width="200px"/>


#### notMNIST
Class | 10 | 100 | 1000 |
:---: | :---: | :---: | :---: |
'B'| <img src="./DCGAN/results/samples/notMNIST/notMNIST_1_10.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/notMNIST/notMNIST_1_100.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/notMNIST/notMNIST_1_1000.png" alt="notMNIST 1000 instances" width="200px"/> 
'C' | <img src="./DCGAN/results/samples/notMNIST/notMNIST_2_10.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/notMNIST/notMNIST_2_100.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/notMNIST/notMNIST_2_1000.png" alt="notMNIST 1000 instances" width="200px"/> 
'D' | <img src="./DCGAN/results/samples/notMNIST/notMNIST_3_10.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./DCGAN/results/samples/notMNIST/notMNIST_3_100.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./DCGAN/results/samples/notMNIST/notMNIST_3_1000.png" alt="notMNIST 1000 instances" width="200px"/> 


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

### Evaluation 

#### Confusion Matrix

Train Dataset | Test Dataset | 10 | 100 | 1000
:---: | :---: | :---: | :---: | :---: |
MNIST GAN | MNIST Real | <img src="./Evaluate/plots/cm/MNIST/MNIST_fake_10.png" alt="MNIST 10 instances" width="200px"/> | <img src="./Evaluate/plots/cm/MNIST/MNIST_fake_100.png" alt="MNIST 100 instances" width="200px"/> | <img src="./Evaluate/plots/cm/MNIST/MNIST_fake_1000.png" alt="MNIST 100 instances" width="200px"/>
FashionMNIST GAN  | FashionMNIST Real | <img src="./Evaluate/plots/cm/FashionMNIST/FashionMNIST_fake_10.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./Evaluate/plots/cm/FashionMNIST/FashionMNIST_fake_100.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./Evaluate/plots/cm/FashionMNIST/FashionMNIST_fake_1000.png" alt="FashionMNIST 1000 instances" width="200px"/>
notMNIST GAN  | notMNIST Real | <img src="./Evaluate/plots/cm/notMNIST/notMNIST_fake_10.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./Evaluate/plots/cm/notMNIST/notMNIST_fake_100.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./Evaluate/plots/cm/notMNIST/notMNIST_fake_1000.png" alt="notMNIST 100 instances" width="200px"/>
MNIST Real | MNIST Real | <img src="./Evaluate/plots/cm/MNIST/MNIST_real_10.png" alt="MNIST 10 instances" width="200px"/> | <img src="./Evaluate/plots/cm/MNIST/MNIST_real_100.png" alt="MNIST 100 instances" width="200px"/> | <img src="./Evaluate/plots/cm/MNIST/MNIST_real_1000.png" alt="MNIST 100 instances" width="200px"/>
FashionMNIST Real | FashionMNIST Real | <img src="./Evaluate/plots/cm/FashionMNIST/FashionMNIST_real_10.png" alt="FashionMNIST 10 instances" width="200px"/> | <img src="./Evaluate/plots/cm/FashionMNIST/FashionMNIST_real_100.png" alt="FashionMNIST 100 instances" width="200px"/> | <img src="./Evaluate/plots/cm/FashionMNIST/FashionMNIST_real_1000.png" alt="FashionMNIST 1000 instances" width="200px"/>
notMNIST Real | notMNIST Real | <img src="./Evaluate/plots/cm/notMNIST/notMNIST_real_10.png" alt="notMNIST 10 instances" width="200px"/> | <img src="./Evaluate/plots/cm/notMNIST/notMNIST_real_100.png" alt="notMNIST 100 instances" width="200px"/> | <img src="./Evaluate/plots/cm/notMNIST/notMNIST_real_1000.png" alt="notMNIST 100 instances" width="200px"/>

#### Accuracy 

Train Dataset | Test Dataset | Accuracy
:---: | :---: | :---:
MNIST GAN | MNIST Real | <img src="./Evaluate/plots/accuracy/MNIST/MNIST_fake_real.png" alt="MNIST" width="200px"/>
FashionMNIST GAN  | FashionMNIST Real | <img src="./Evaluate/plots/accuracy/FashionMNIST/FashionMNIST_fake_real.png" alt="FashionMNIST" width="200px"/> 
notMNIST GAN  | notMNIST Real | <img src="./Evaluate/plots/accuracy/notMNIST/notMNIST_fake_real.png" alt="notMNIST" width="200px"/> 
MNIST Real | MNIST Real | <img src="./Evaluate/plots/accuracy/MNIST/MNIST_real_real.png" alt="MNIST" width="200px"/>
FashionMNIST Real | FashionMNIST Real| <img src="./Evaluate/plots/accuracy/FashionMNIST/FashionMNIST_real_real.png" alt="FashionMNIST" width="200px"/> 
notMNIST Real | notMNIST Real | <img src="./Evaluate/plots/accuracy/notMNIST/notMNIST_real_real.png" alt="notMNIST" width="200px"/> 

#### Visualization

Train Dataset | 10 | 100 | 1000 
:---: | :---: | :---: | :---:
MNIST | <img src="./Visualize/embedding/scatter/images/2D/MNIST/1_7_10.jpg"/> |  <img src="./Visualize/embedding/scatter/images/2D/MNIST/1_7_100.jpg"/> |  <img src="./Visualize/embedding/scatter/images/2D/MNIST/1_7_1000.jpg"/>
FashionMNIST | <img src="./Visualize/embedding/scatter/images/2D/FashionMNIST/Ankle Boot_Sneaker_10.jpg"/> |  <img src="./Visualize/embedding/scatter/images/2D/FashionMNIST/Ankle Boot_Sneaker_100.jpg"/> |  <img src="./Visualize/embedding/scatter/images/2D/FashionMNIST/Ankle Boot_Sneaker_1000.jpg"/>
notMNIST | <img src="./Visualize/embedding/scatter/images/2D/notMNIST/E_F_10.jpg"/> |  <img src="./Visualize/embedding/scatter/images/2D/notMNIST/E_F_100.jpg"/> |  <img src="./Visualize/embedding/scatter/images/2D/notMNIST/E_F_1000.jpg"/>

## Usage

- Execute the conditional DC-GAN with `python cDCGAN_train.py`. This would create 4 directories within `DCGAN`
	- `DCGAN/animation` containing gif of generated images after every epoch
	- `DCGAN/loss` containing loss values for both generator and discriminator vs. epochs
	- `DCGAN/plots` containing a png file with generated images after last epoch
	- `DCGAN/models` containing model file with discriminator and generator weights in pytorch format

### References 

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset
- [notMNIST](https://github.com/davidflanagan/notMNIST-to-MNIST) dataset
- GAN [Tutorial](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/generative_adversarial_network/main.py)
- CycleGAN [Implementation](https://github.com/togheppi/CycleGAN)

### TODO

- [ ] Add classification results for DCGAN with/without MMD 
- [x] DCGAN with MMD [ Learning from all classes of same dataset ]
- [ ] MMD Comparison [ Cross and within domain ]
- [ ] Batches with max and min MMD
- [ ] Update .py files [presently in .ipynb format]



