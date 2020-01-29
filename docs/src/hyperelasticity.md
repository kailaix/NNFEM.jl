# Hyperelasticity



| Parameter      | Value                                   |
| -------------- | --------------------------------------- |
| `T`            | 0.2 (hyperelasticity), 200 (plasticity) |
| `NT`           | 200                                     |
| $n_x^f, n_y^f$ | (20,10)                                 |
| $n_x, n_y$     | (10,5)                                  |
| $L_x, L_y$     | (0.1,0.05)                              |
|                |                                         |
|                |                                         |



## Instruction for Reproducing Results in the Paper

Step 1: Generate Data

```bash
sh GenerateData.sh 
```

Step 2: Train the Neural Network

```bash
sh TrainNN.sh
```



Step 1: Generate Data

```bash
sh GenerateData.sh 
```

Step 2: Use small stress data to train a linear estimate (this is only done once)

```bash
julia Linear_Train_NNPlatePull.jl
```

Step 3: Prefit the neural network

```bash
sh NNPrefit.sh
```

Step 4: Train the neural network 

```bash
sh TrainNN.sh
```







