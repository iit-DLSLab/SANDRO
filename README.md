# SANDRO: A Robust Solver with a Splitting Strategy for Point Cloud Registration  

## Overview  

SANDRO (Splitting strategy for point cloud Alignment using Non-convex anD Robust Optimization) is a novel algorithm for point cloud registration. It integrates an **Iteratively Reweighted Least Squares (IRLS)** framework with a **Graduated Non-Convexity (GNC) approach** and a **Geman-McClure robust loss function** to handle high outlier rates and skewed outlier distributions.  

A key feature of SANDRO is its **splitting strategy**, which partitions the point cloud into smaller subsets to **reduce bias from symmetrical outliers** and improve convergence. This technique allows SANDRO to handle complex registration problems that often cause failures in traditional methods.  

Unlike traditional methods that struggle with **point cloud symmetries** and **high outlier rates**, SANDRO achieves superior accuracy and robustness.

[[Paper]](https://arxiv.org/pdf/2503.07743v1) [[arXiv]](https://arxiv.org/abs/2503.07743v1)


## 🔔 Updates  

- [12/2025] Install instructions now available!

- [05/2025] The paper has been accepted to ICRA 2025 and it is available [here](https://ieeexplore.ieee.org/document/11128360)-

## 🔧 Installation  

SANDRO depends on **Open3D**, which includes native (C++/OpenGL) components.
For this reason, **Conda is the recommended installation method**.


### Recommended Installation (Conda)

#### 1. Clone the repository
```bash
git clone https://github.com/iit-DLSLab/SANDRO.git
cd SANDRO
```

## 📖 Usage  

🚧 **Usage examples will be provided soon**

## 📌 Citation  

If you use SANDRO in your research, please cite:  

```bibtex
@INPROCEEDINGS{adlerstein2025sandro,
  author={Adlerstein, Michael and Virgolino Soares, João Carlos and Bratta, Angelo and Semini, Claudio},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={SANDRO: A Robust Solver with a Splitting Strategy for Point Cloud Registration}, 
  year={2025},
  pages={11112-11118},
  doi={10.1109/ICRA55743.2025.11128360}
}
```


