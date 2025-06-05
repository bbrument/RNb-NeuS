# RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction

<div align="center">
  <hr>
  <h1>✨ RNb-NeuS2 is now available! ✨</h1>
  <p>A new and improved version of this project has been released, featuring a <strong>100x faster CUDA implementation</strong>.</p>
  <a href="https://robinbruneau.github.io/publications/rnb_neus2.html"><img src="https://img.shields.io/badge/Project_Page-🌐-blue" alt="Project Page" height="30"></a>
  <a href="https://arxiv.org/abs/2506.04115"><img src="https://img.shields.io/badge/arXiv-2506.04115-b31b1b" alt="arXiv" height="30"></a>
  <a href="https://github.com/RobinBruneau/RNb-NeuS2"><img src="https://img.shields.io/badge/Code-💻-black" alt="Code" height="30"></a>
  <hr>
</div>

This is the official implementation of **RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction**.

[Baptiste Brument*](https://bbrument.github.io/),
[Robin Bruneau*](https://robinbruneau.github.io/),
[Yvain Quéau](https://sites.google.com/view/yvainqueau),
[Jean Mélou](https://www.irit.fr/~Jean.Melou/),
[François Lauze](https://loutchoa.github.io/),
[Jean-Denis Durou](https://www.irit.fr/~Jean-Denis.Durou/),
[Lilian Calvet](https://scholar.google.com/citations?user=6JewdrMAAAAJ&hl=en)

### [Project page](https://robinbruneau.github.io/publications/rnb_neus.html) | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Brument_RNb-NeuS_Reflectance_and_Normal-based_Multi-View_3D_Reconstruction_CVPR_2024_paper.pdf)

<img src="assets/pipeline.png">

----------------------------------------
## Installation

```shell
git clone https://github.com/bbrument/RNb-NeuS.git
cd RNb-NeuS
pip install -r requirements.txt
```

## Usage

#### Data Convention

Our data format is inspired from [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md) as follows:
```
CASE_NAME
|-- cameras.npz    # camera parameters
|-- normal
    |-- 000.png        # normal map for each view
    |-- 001.png
    ...
|-- albedo
    |-- 000.png        # albedo for each view (optional)
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # mask for each view
    |-- 001.png
    ...
```

One can create folders with different data in it, for instance, a normal folder for each normal estimation method.
The name of the folder must be set in the used `.conf` file.

We provide the [DiLiGenT-MV](https://drive.google.com/file/d/1TEBM6Dd7IwjRqJX0p8JwT9hLmy_vA5nU/view?usp=drive_link) data as described above with normals and reflectance maps estimated with [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/). Note that the reflectance maps were scaled over all views and uncertainty masks were generated from 100 normals estimations (see the article for further details).

### Run RNb-NeuS!

**Train with reflectance**

```shell
python exp_runner.py --mode train_rnb --conf ./confs/CONF_NAME.conf --case CASE_NAME
```

**Train without reflectance**

```shell
python exp_runner.py --mode train_rnb --conf ./confs/CONF_NAME.conf --case CASE_NAME --no_albedo
```

**Extract surface** 

```shell
python exp_runner.py --mode validate_mesh --conf ./confs/CONF_NAME.conf --case CASE_NAME --is_continue
```

Additionaly, we provide the five meshes of the DiLiGenT-MV dataset with our method [here](https://drive.google.com/file/d/1CTQW1YLWOT2sSEWznFmSY_cUUtiTXLdM/view?usp=drive_link).

## Citation
If you find our code useful for your research, please cite
```
@inproceedings{Brument24,
    title={RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction},
    author={Baptiste Brument and Robin Bruneau and Yvain Quéau and Jean Mélou and François Lauze and Jean-Denis Durou and Lilian Calvet},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```
