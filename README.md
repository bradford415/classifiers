# Classifiers

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
| Classifier | Pretrained | Dataset            | Accuracy (top 1) / Epoch |
|------------|------------|--------------------|--------------------------|
| ResNet50   | Scratch    | ImageNet-1k (2012) | 74.9% / 96               |

## Visuals
# Learning rate schedulers
Visuals of the learning rate schedulers to get an intuitive idea of how they work in practice.
TODO Warmup cosine decay

