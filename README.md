# RNb-NeuS
This is the official implementation of **RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction**.

[Baptiste Brument*](https://bbrument.github.io/),
[Robin Bruneau*](https://robinbruneau.github.io/),
[Yvain Quéau](https://sites.google.com/view/yvainqueau),
[Jean Mélou](https://www.irit.fr/~Jean.Melou/),
[François Lauze](https://loutchoa.github.io/),
[Jean-Denis Durou](https://www.irit.fr/~Jean-Denis.Durou/),
[Lilian Calvet](https://scholar.google.com/citations?user=6JewdrMAAAAJ&hl=en)

### [Project page](https://robinbruneau.github.io/publications/rnb_neus.html) | [Paper](https://arxiv.org/abs/2312.01215)

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
|-- cameras_xxx.npz    # camera parameters
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
@inproceedings{Brument23,
    title={RNb-Neus: Reflectance and normal Based reconstruction with NeuS},
    author={Baptiste Brument and Robin Bruneau and Yvain Quéau and Jean Mélou and François Lauze and Jean-Denis Durou and Lilian Calvet},
    eprint={2312.01215},
    archivePrefix={arXiv},
    year={2023}
}
```