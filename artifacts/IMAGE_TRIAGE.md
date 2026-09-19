# IMAGE TRIAGE

12000 images, classes: crack, hole, normal, rust, scratch

## Per-class averages (mean_sd = spread of image brightness inside the class)
```
         mean  std   p1   p99  lap_var  mean_sd
cls                                            
crack   108.8 12.1 73.2 127.8     31.2     36.1
hole    100.0 17.4 32.4 121.4    181.5     34.0
normal  103.0 10.7 80.2 123.0     20.6     33.5
rust     97.9 17.6 60.3 128.9    136.9     30.1
scratch 102.3 13.2 63.5 140.1     96.6     35.3
```

## Shortcut test (5-fold CV, gradient boosting on global stats only)

- brightness only: 5-class accuracy 0.223 (chance 0.200); normal-vs-defect AUROC 0.531 (chance 0.500)
- all 5 global stats: 5-class accuracy 0.806 (chance 0.200); normal-vs-defect AUROC 0.959 (chance 0.500)

## File numbering per class
```
crack    n=2400 index 0..2998 missing_in_range=599  brightness-vs-index rho=-0.027 (p=0.18)
hole     n=2400 index 0..2995 missing_in_range=596  brightness-vs-index rho=-0.006 (p=0.78)
normal   n=2400 index 0..2999 missing_in_range=600  brightness-vs-index rho=+0.003 (p=0.9)
rust     n=2400 index 0..2999 missing_in_range=600  brightness-vs-index rho=+0.004 (p=0.84)
scratch  n=2400 index 0..2999 missing_in_range=600  brightness-vs-index rho=-0.040 (p=0.052)
```

## Nearest-neighbour similarity (cosine, 32x32 thumbnails)
```
class     n>=0.98  n>=0.995  of those, nearest is same class
crack         41         1  0/41
hole           1         0  0/1
normal      1695        55  1676/1695
rust          36         2  0/36
scratch        1         0  0/1
```
