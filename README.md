# Introduction
This code is the implementation of the paper "Hierarchical prediction of irregular multivariate time series from a multi-granularity perspective".

# Requirements
The HiP-IMTS framework was developed and evaluated using Python 3.13 and CUDA 12.4.<br>
To ensure a consistent software environment and reproducible library versions, you may install all required dependencies by executing the following command:
```
pip install -r requirements.txt
```

# Usage
To initiate the training and evaluation process for HiP-IMTS on a selected dataset, execute the corresponding script.<br>
For example, to run the model on the PhysioNet dataset, use:
```
sh ./HiP-IMTS/scripts/run_physionet.sh
```

# Data
The datasets can be obtained and put into ```dataset/``` folder

- [<u> PhysioNet </u>](https://archive.physionet.org/challenge/2012)
- [<u> USHCN </u>](https://www.osti.gov/biblio/1394920)
- [<u> Human Activity </u>](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity)
- [<u> MIMIC </u>](https://mimic.mit.edu/)

# Acknowledgements and References
This work builds upon the following influential studies:
```
- Shukla, S. N., & Marlin, B. (2021). Multi-Time Attention Networks for Irregularly Sampled Time Series. In Proceedings of the 9th International Conference on Learning Representations (pp. 1-11).
- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In Proceedings of the 11th International Conference on Learning Representations (pp. 1-12).
- Zhang, W., Yin, C., Liu, H., Zhou, X., & Xiong, H. (2024). Irregular multivariate time series forecasting: A transformable patching graph neural networks approach. In Proceedings of the 41st International Conference on Machine Learning (pp. 1-12).
```
We sincerely acknowledge and appreciate the foundational contributions of these prior works.
