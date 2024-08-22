# ALGM

---

- This paper focuses on the image composition of transparent objects, where existing image matting methods suffer from composition errors due to the lack of accurate foreground during the composition process. We propose a foreground prediction model named ALGM, which leverages the local feature extraction capabilities of Convolutional Neural Networks (CNNs) and incorporates an attention mechanism for global information modeling. The proposed alpha-assisted foreground prediction module extracts foreground information from the original image and conveys it. The extracted foreground color information is combined with the deep structural features of the encoder and used for foreground color prediction. ALGM reduces image composition errors in the quantitative data from the Composition-1k dataset and improves the visual quality of composed images on the AIM-500 and Transparent-460 datasets.


![Exp](https://github.com/SunLi2/ALGM2/blob/master/assets/compare1.png)
![Exp](https://github.com/SunLi2/ALGM2/blob/master/assets/compare2.png)
![Exp](https://github.com/SunLi2/ALGM2/blob/master/assets/compare3.png)

---

### Requirements
The codes are tested in the following environment:
- python 3.8
- pytorch 1.10.1
- CUDA 10.2 & CuDNN 8

### Performances

![Exp](https://github.com/SunLi2/ALGM2/blob/master/assets/Table1.png)

![Exp](https://github.com/SunLi2/ALGM2/blob/master/assets/pic1.png)

---

### Data Preparation
1] Get DIM dataset on [Deep Image Matting](https://sites.google.com/view/deepimagematting).

2] For DIM dataset preparation, please refer to [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting).
- For Training, merge 'Adobe-licensed images' and 'Other' folder to use all 431 foregrounds and alphas
- For Testing, use 'Composition_code.py' and 'copy_testing_alpha.sh' in GCA-Matting.

3] For background images, Download dataset on [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](https://cocodataset.org/#home).

***If you want to download prepared test set directly : [download link](https://drive.google.com/file/d/1fS-uh2Fi0APygd0NPjqfT7jCwUu_a_Xu/view?usp=sharing)** 

### Testing on Composition-1k dataset
```
pip3 install -r requirements.txt
```

1] Run inference code (the predicted alpha will be save to **./predDIM/pred_alpha** by default)

```
python3 infer.py
```

2] Evaluate the results by the official evaluation MATLAB code **./DIM_evaluation_code/evaluate.m** (provided by [Deep Image Matting](https://sites.google.com/view/deepimagematting))

3] You can also check out the evaluation result simplicity with the python code (un-official) 
```
python3 evaluate.py
```

### Training on Composition-1k dataset
1] You can get (imagenet pretrained) swin-transformer tiny model (**'swin_tiny_patch4_window7_224.pth'**) on [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

2] modify "config/MatteFormer_Composition1k.toml"

3] run main.py
```
CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py
```

---


### Acknowledgment
- Our Codes are mainly originated from [MG-Matting](https://github.com/yucornetto/MGMatting)
- Also, we build our codes with reference as [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting) and [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)


