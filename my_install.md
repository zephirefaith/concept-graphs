```
 mamba create -n conceptgraph-cuda-12-1 anaconda python=3.10
 pip install tyro open_clip_torch wandb h5py openai hydra-core torch==2.0.1+cu118 --index-url 
 mamba install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
 mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
 conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py310_cu121_pyt210.tar.bz2
 ls
 cd ..
 ls
 cd ..
 ls
 cd chamferdist
 ls
 rm -rf build chamferdist.egg-info
 pip install .
 cd ..
 cd gradslam
 ls
 rm -rf build gradslam.egg-info
 pip install .
 cd ../Grounded-Segment-Anything
 python -m pip install -e segment_anything
 python -m pip install -e GroundingDINO
 pip install --upgrade 'diffusers[torch]'
 cd ..
 git clone https://github.com/xinyu1205/recognize-anything.git
 pip install -r ./recognize-anything/requirements.txt
 pip install -e ./recognize-anything/
 cd Grounded-Segment-Anything
 pwd
 export GSA_PATH=/private/home/priparashar/SIRo/Grounded-Segment-Anything
 env
 vim ~/.zshrc
 cd ..
 git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
 cd FastSAM
 pip install -r requirements.txt
 pip install git+https://github.com/openai/CLIP.git
 pip install .
 cd ..
 cd Grounded-Segment-Anything
 wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
 wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
 wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
 wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
 cd EfficientSAM
 wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth
 wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
 pip install gdown
 gdown 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv
 cd ..
 git clone git@github.com:concept-graphs/concept-graphs.git
 cd concept-graphs
 pip install e .
 pip install -e .
 git clone https://github.com/haotian-liu/LLaVA.git
 cd LLaVA
 pip install -e .
python -m pip install distinctipy
```

Installing LLaVA broke the entire environment. It needs cuda=11.8 I think. AArrgghh. Following stuff was deleted:
```
 Attempting uninstall: pydantic
    Found existing installation: pydantic 2.6.0
    Uninstalling pydantic-2.6.0:
      Successfully uninstalled pydantic-2.6.0
  Attempting uninstall: scikit-learn
    Found existing installation: scikit-learn 1.3.0
    Uninstalling scikit-learn-1.3.0:
      Successfully uninstalled scikit-learn-1.3.0
  Attempting uninstall: httpcore
    Found existing installation: httpcore 1.0.2
    Uninstalling httpcore-1.0.2:
      Successfully uninstalled httpcore-1.0.2
  Attempting uninstall: tokenizers
    Found existing installation: tokenizers 0.15.1
    Uninstalling tokenizers-0.15.1:
      Successfully uninstalled tokenizers-0.15.1
  Attempting uninstall: httpx
    Found existing installation: httpx 0.26.0
    Uninstalling httpx-0.26.0:
      Successfully uninstalled httpx-0.26.0
  Attempting uninstall: transformers
    Found existing installation: transformers 4.37.2
    Uninstalling transformers-4.37.2:
      Successfully uninstalled transformers-4.37.2
  Attempting uninstall: gradio-client
    Found existing installation: gradio_client 0.8.1
    Uninstalling gradio_client-0.8.1:
      Successfully uninstalled gradio_client-0.8.1
  Attempting uninstall: triton
    Found existing installation: triton 2.1.0
    Uninstalling triton-2.1.0:
      Successfully uninstalled triton-2.1.0
  Attempting uninstall: torch
    Found existing installation: torch 2.1.0
    Uninstalling torch-2.1.0:
      Successfully uninstalled torch-2.1.0
  Attempting uninstall: torchvision
    Found existing installation: torchvision 0.16.0
    Uninstalling torchvision-0.16.0:
      Successfully uninstalled torchvision-0.16.0
  Attempting uninstall: accelerate
    Found existing installation: accelerate 0.26.1
    Uninstalling accelerate-0.26.1:
      Successfully uninstalled accelerate-0.26.1
  Attempting uninstall: timm
    Found existing installation: timm 0.4.12
    Uninstalling timm-0.4.12:
      Successfully uninstalled timm-0.4.12
```
```
mamba install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
pip config set global.index_url https://download.pytorch.org/whl/cu118
```

The last pip config was needed so every package that is built, uses the same 11.8 CUDA version rather than pulling the latest torch from PyPi
