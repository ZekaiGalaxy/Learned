# MultiModal 



## **Fusion**

\#modality < #representation

#### fusion function

- additive

- multiplicative : $x_A^Tx_B$, $x_A x_B$

- tensor fusion : $W[x_A,1]^T[x_B,1]$
- low rank

<img src="https://p.ipic.vip/iuz15t.png" alt="截屏2023-01-01 11.40.24.png" style="zoom: 33%;" align=left  />

- gated fusion
  - soft (attention)
  - hard (RL)

<img src="https://p.ipic.vip/x1h6cm.png" alt="截屏2023-01-01 11.40.50.png" style="zoom: 50%;" align=left />

- early fusion
- channel exchanging!

<img src="https://p.ipic.vip/9qw9e3.png" alt="截屏2023-01-01 11.41.21.png" style="zoom: 50%;" align=left />

#### Example

- hearing to see

<img src="https://p.ipic.vip/1qgovr.png" alt="截屏2023-01-01 11.42.29.png" style="zoom:33%;" align=left />



## **Coordination**

\#modality = #representation

#### vector arithmetic

#### alignment function

- cosine
- kernel : RBF(Gaussian)
- CCA (Canonical Correlation Analysis)
- $argmax_{V,U}corr(Uz_A, Vz_B)$

<img src="https://p.ipic.vip/a0afvt.png" alt="截屏2023-01-01 11.55.39.png" style="zoom: 50%;" align=left />

- gated coordination

<img src="https://p.ipic.vip/lsdzc9.png" alt="截屏2023-01-01 11.56.36.png" style="zoom:33%;" align=left />

- contrastive learning
- simple contrastive loss
    - $ L = max\{0, \alpha + sim(z_A, z_B^+)-sim(z_A, z_B^-)\}$
- InfoNCE
    - $L = -\frac{1}{N} \sum_{N}{log\frac{sim(z_A^l, z_B^l)}{sim(z_A^i, z_B^j)}}$
  - CLIP

<img src="https://p.ipic.vip/ffk4p3.png" alt="截屏2023-01-01 11.56.54.png" style="zoom: 50%;" align=left />

#### Example

- - Multi-view latent “intact” space

<img src="https://p.ipic.vip/gcehdx.png" alt="截屏2023-01-01 11.56.00.png" style="zoom:33%;" align=left />



## **Fission**

\#modality > #representation

#### modality level fission

<img src="https://p.ipic.vip/rgy7kd.png" alt="截屏2023-01-01 11.59.40.png" style="zoom:33%;" align=left />

- Information perspective

  - maximize mutual information

    - $ H(z_A|z_B) = -E_{z_A,z_B}[log(\frac{p(z_A,z_B)}{p(z_B)})] $ 
  
  - minimize conditional entropy
  
    - $ I(X,Y) = H(X)-H(X|Y) = E_{X,Y}log(\frac{p(z_A,z_B)}{p(z_A)p(z_B)})= D_{KL}(p_{XY}(X,Y)||P_X(x)P_Y(y)) $
  
      

#### alignment	

<img src="https://p.ipic.vip/jt3qte.png" alt="截屏2023-01-01 12.07.51.png" style="zoom:33%;" align=left />

<img src="https://p.ipic.vip/b2fsf3.png" alt="截屏2023-01-01 12.08.06.png" style="zoom:33%;" align=left />

<img src="https://p.ipic.vip/sv7bq2.png" alt="截屏2023-01-01 12.08.30.png" style="zoom:33%; " align=left />

<img src="https://p.ipic.vip/co1p8c.png" alt="截屏2023-01-01 12.08.54.png" style="zoom:33%" align=left />

#### hard alignment

- Bipartite Graph (Assignment)
  - Cycle Consistency
  
  - - My nearest neighbor should be your nearest neighbor
    - “soft” nearest neighbor