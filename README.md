# Encrypted_Control_Ring-LWE

Code for the methods proposed in the paper ['Ring-LWE based encrypted controller with unlimited number of recursive multiplications and effect of error growth'](https://arxiv.org/abs/2406.14372) by Yeongjun Jang, Joowon Lee, Seonhong Min, Hyesun Kwak, Junsoo Kim, and Yongsoo Song. 
Implemented in [Lattigo](https://github.com/tuneinsight/lattigo) library.

---

### Overview

This code is for operating linear dynamic controllers over encrypted data, using a Ring-LWE based cryptosystem. 
Two main features are emphasized. 
* Unlimited number of recursive homomorhpic multiplications is supported. More specifically, the encrypted controller state is recursively multiplied to the encrypted state matrix without decryption. The effect of error growth is suppressed by the stability of the closed-loop system. 
* A novel packing algorithm is applied, resulting in enhanced computation speed and memory efficiency.

More details on the concept and the proofs can be found in [1].  

---

### Files
There are two files. 
1. `Ring-GSW.go` (without packing. Section 3 of [1])
2. `Ring-GSW_Packed.go` (with packing. Section 4 of [1])

When running one file, please comment out the other one. 
Then run

```
go run Ring-GSW.go  
```
or
```
go run Ring-GSW_Packed.go  
```
on the terminal.

---

### Set parameters 

* `rlwe.NewParametersFromLiteral`: Ring-LWE parameters (LogN = 11 and LogQ = 54 gives $N=2^{11}$ and some prime $q$ such that $q \approx 2^{54}$)

* `s`, `L`, and `r`: Scale factors 

* `iter`: Number of iterations for simulation 

* `A`, `B`, and `C`: State space matrices of the discrete time plant written by

> $x(t+1) = Ax(t) + Bu(t), \quad y(t) = Cx(t)$

* `F`, `G`, `R` and `H`: State space matrices of the discrete time controller. 
Given a controller of the form 
> $x(t+1) = Kx(t) + Gy(t), \quad u(t) = Hx(t)$

one can regard $u(t)$ as a fed-back input and design $R$, so that the state matrix $F:=K-RH$ of
> $x(t+1) = (K-RH)x(t) + Gy(t)+Ru(t), \quad u(t) = Hx(t)$

consists of integers. More details can be found in Section 5 of [1] or Lemma 1 of [2].

* `xPlantInit`, `xContInit`: Initial conditions of the plant and the controller

* `tau`: Least power of two greater than the dimensions of the state, output, and input of the controller (Only used in `Ring-GSW_Packed.go`)

---

### References
[1] [Y. Jang, J. Lee, S. Min, H. Kwak, J. Kim, and Y. Song, "Ring-LWE based encrypted controller with unlimited number of recursive multiplications and effect of error growth", 2024, arXiv:2406.14372.](https://arxiv.org/abs/2406.14372)

[2] [J. Kim, H. Shim, and K. Han, "Dynamic controller that operates over homomorphically encrypted data for infinite time horizon," _IEEE Trans. Autom. Control_, vol. 68, no. 2, pp. 660-672, 2023.](https://ieeexplore.ieee.org/abstract/document/9678042)

