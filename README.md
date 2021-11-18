# LightQNet
LightQNet: Lightweight Face Image Quality Assessment for Risk-Controlled Face Recognition

This is a demo code of the lightqnet:
+ lightqnet-dm100.pb, 1.77 MB
+ lightqnet-dm050.pb, 0.62 MB
+ lightqnet-dm025.pb, 0.29 MB

### Training Code

https://github.com/KaenChan/lightqnet-train

### Demo

``` Shell
python main.py
```

### PreProcess

The image needs be aligned and resized to 96 x 96.
Then it is normalized by img = (img - 128.) / 128. before the input of lightqnet.

#### Run-time

The results (ms) of runtime comparison on Intel i7-8700 CPU and Kirin 985 ARM. 

| Methods            | i7-8700 | Kirin985 |         |         |         |       |
| ------------------ | ------- | -------- | ------- | ------- | ------- | ----- |
|                    |         | Thread1  | Thread2 | Thread3 | Thread4 | iGPU  |
| SER-FIQ(ResFace64) | 450.72  | -        | -       | -       | -       | -     |
| FaceQNet(ResNet50) | 107.31  | -        | -       | -       | -       | -     |
| PFE(ResFace64)     | 48.27   | 421.32   | 255.85  | 196.42  | 178.73  | 59.37 |
| PCNet(ResNet18)    | 12.27   | 95.90    | 58.44   | 52.35   | 42.62   | 16.51 |
| LightQNet          | 3.98    | 4.02     | 3.30    | 3.21    | 2.57    | 5.57  |
| LightQNetx0.5      | 2.37    | 2.02     | 1.82    | 1.72    | 1.61    | 2.24  |
| LightQNetx0.25     | 1.97    | 1.65     | 1.55    | 1.53    | 1.50    | 2.16  |

#### Reference
If you find this repo useful, please consider citing:
```
@article{chen2021lightqnet,
  title={LightQNet: Lightweight Deep Face Quality Assessment for Risk-Controlled Face Recognition},
  author={Chen, Kai and Yi, Taihe and Lv, Qi},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={1878--1882},
  year={2021},
  publisher={IEEE}
}
```
