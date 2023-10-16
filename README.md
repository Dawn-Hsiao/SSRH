

## Method

### General framework

![image-20231009133628218](https://github.com/Dawn-Hsiao/Screen-Shooting-Robust-Hyperlink-Based-on-Deep-Learning/blob/main/README.assets/image-20231009133628218.png)

### Moiré simulation framework

![image-20231009133852270](https://github.com/Dawn-Hsiao/Screen-Shooting-Robust-Hyperlink-Based-on-Deep-Learning/blob/main/README.assets/image-20231009133852270.png)

## Performance

### Visualization of moire pattern with different methods

![image-20231009134210954](https://github.com/Dawn-Hsiao/Screen-Shooting-Robust-Hyperlink-Based-on-Deep-Learning/blob/main/README.assets/image-20231009134210954.png)

### Extraction accuracy with different shooting distance

![image-20231009142845765](https://github.com/Dawn-Hsiao/Screen-Shooting-Robust-Hyperlink-Based-on-Deep-Learning/blob/main/README.assets/image-20231009142845765.png)

### The PSNR and SSIM values of each method

![image-20231009143135396](https://github.com/Dawn-Hsiao/Screen-Shooting-Robust-Hyperlink-Based-on-Deep-Learning/blob/main/README.assets/image-20231009143135396.png)

## Requirements

python 3.9

torch 2.0.1

torchattackes 2.12.2

opencv-python 4.7.0.72

## Test

#### test_embedding:

```python
python main.py --dataset test_embedding --mode test_embedding --image_dir 'images/original/'
```

#### test_accuracy:

```python
python main.py --dataset test_accuracy --mode test_accuracy --image_val_dir 'images/embed/'
```
