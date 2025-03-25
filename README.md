This project aims to improve the efficiency of th PreActResNet model (computational cost and accuracy) by using several techniques.


You can see the description of the "score" we used to evaluate our model :
--
The goal is to design and train a network that **achieves 90% accuracy on CIFAR10**, while having the **lowest possible score**.

$$\text{score} =\underset{param}{\underbrace{\dfrac{[1-(p_s+p_u)]\dfrac{q_w}{32}w}{5.6\cdot10^6}}} + \underset{ops}{\underbrace{\dfrac{(1-p_s)\dfrac{\max(q_w,q_a)}{32}f}{2.8\cdot10^8}}} $$

Where:
- $p_s$: structured pruning
- $p_u$: unstructured pruning
- $q_w$: quantization of weights
- $q_a$: quantization of activations
- $w$: number of weights
- $f$: number of mult-adds (MACs) operations
- $5.6\cdot10^6$ and $2.8\cdot10^8$ are the reference param and ops scores of the ResNet18 network in half precision.
