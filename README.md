# Classifiers
The goal of this repository is to: 
* Implement powerful classifiers and understand their architecture so they can be used as backbones for other downstream tasks
* Test and understand various components of deep learning (e.g., learning rate schedulers)
* Training techniques
  * Self-supervised pretraining
  * Classification pretraining
 
## Environment setup

__linux__
```bash
make create
```

__mac__
```bash
python3 -m venv .venv
source .venv/bin/activate
make install_reqs_mac
```

## Training a model
This project is designed to use the configuration specified in `configs/`, but for ease of use the CLI arguments specified below will overwrite the main default config parameters for quick setup.

### Training from scratch
```bash
# train ViT
python scripts/train.py configs/train-imagenet-vit.yaml configs/vit/vit-base-16.yaml

# train Swin
python scripts/train.py configs/train-imagenet-swin.yaml configs/swin/swin-base-patch-4-window-7.yaml
```

### Resume training from a checkpoint
```bash
python scripts/train.py --dataset_root "/mnt/d/datasets/imagenet" --checkpoint_path "/path/to/checkpoint_weights.pt"
```

## Datasets
### ImageNet 2012
This section describes how to download the ImageNet ILSVRC2012 dataset and prepare it for training classification models. All 1000 classes are described [here](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/).

Download the ImageNet and prepare it for training with the following commands:
```bash
cd data
bash get_imagenet_2012.sh
```

After running the download script, the ImageNet dataset will be organized as shown below. The raw dataset only has one folder in `val` but the script mimics the class directory structure of `train`, this allows us to easily create a torch dataset with `torchvision.datasets.ImageFolder`. The labels are automatically determined by the images parent folder e.g., `class_1`. The [`ILSVRC2012 development kit`](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) has a text file of validation image labels, but the way this script structures the `val` set we should not need this text file.

    	├── train                    
    	│   ├── class_1         
    	│   │   ├── img_1.jpeg        
    	│   │   ├── img_2.jpeg        
    	│   ├── class_2
    	│   │   ├── img_1.jpeg        
    	│   │   ├── img_2.jpeg  
    	│   ├── ...        
    	├── val              
    	│   ├── class_1         
    	│   │   ├── img_1.jpeg        
    	│   │   ├── img_2.jpeg        
    	│   ├── class_2
    	│   │   ├── img_1.jpeg        
    	│   │   ├── img_2.jpeg  
    	│   ├── ...                
    	└── 

## Results
| Classifier | Pretrained | Dataset            | Accuracy (top 1) / Epoch | Notes                                                               |
|------------|------------|--------------------|--------------------------|---------------------------------------------------------------------|
| ResNet50   | Scratch    | ImageNet-1k (2012) | 74.9% / 96               |                                                                     |
| ViT-B/16   | Scratch    | ImageNet-1k (2012) | 65.7% / 114              | Accuracy was still slowly increasing, but my tiny GPU wanted a break|
| Swin-B     | Scratch    | ImageNet-1k (2012) | 69.3% / 64               | Accuracy was still increasing, but my tiny GPU wanted a break       |

## Explanations
* [Swin's relative position bias]()

## Visuals
### Learning rate schedulers
Visuals of the learning rate schedulers to get an intuitive idea of how they work in practice.
TODO Warmup cosine decay

