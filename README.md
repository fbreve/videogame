# From Pixels to Titles: Video Game Identification by Screenshots using Convolutional Neural Networks

This repository contains the Python code, experiment metadata, selected model weights, and results associated with:

> Breve, Fabricio Aparecido. **From Pixels to Titles: Video Game Identification by Screenshots using Convolutional Neural Networks.** *IEEE Transactions on Games*, 2025.

- [Preprint on arXiv](https://arxiv.org/abs/2311.15963)
- [Author-hosted paper (PDF)](https://www.fabriciobreve.com/artigos/2025/tg-videogame.pdf)

## Project overview

The project investigates identifying video games from screenshots using convolutional neural networks. The repository includes scripts for collecting game metadata, preparing screenshot lists, training and evaluating models across gaming platforms, and processing experiment results.

## Repository contents

- `videogame.py`: main CNN training and evaluation script.
- `videogame_config.py`: model, dataset, and platform configuration.
- `videogame_allplat.py`, `videogame_byplat.py`, `videogame-feat.py`: alternative experimental workflows.
- `videogame_saveweights.py`, `videogame_loadweights.py`, `videogame_loadweights_top10.py`: weight-related experiments.
- `mobygames_getgamelist.py`, `mobygames_getscreenshots.py`: MobyGames API metadata and screenshot collection scripts.
- `data/`: game lists, screenshot indexes, counts, and platform metadata; **not** the screenshot image collection.
- `results/`: recorded experimental outputs and tables.
- `weights/`: selected model weights generated during the research.

## Dataset and rights

Screenshot images were obtained using the MobyGames API for the research. **The screenshot image dataset is not distributed through this repository.** Game screenshots remain subject to the rights and terms of their respective owners and sources. The Apache 2.0 license for this repository's code does not grant permission to redistribute third-party screenshots.

The code expects images under `data/screenshots/` by default. The CSV files in `data/` describe game and screenshot records but do not contain the underlying image files. Reproducing the experiments requires obtaining images through authorized sources, subject to applicable terms and permissions.

## Setup and configuration

The project is research code, not a packaged application. It was developed with Python and TensorFlow/Keras. The main script also imports NumPy, pandas, scikit-learn, and Pillow; some alternative workflows use additional libraries. Exact historical package versions are not pinned here, so reproducing the original software environment may require additional work.

1. Set up a compatible Python environment with the dependencies required by the script you intend to run.
2. Run from the repository root, because several scripts use relative paths.
3. For MobyGames API scripts, copy `credentials_example.py` to `credentials.py` and provide your own API key. **Never commit `credentials.py` or your API key.**
4. Obtain the required screenshots lawfully and place them under `data/screenshots/`, or change `DATASET_PATH` in `videogame_config.py`.
5. Review `videogame_config.py` before running an experiment, including `MODEL_TYPE`, `WEIGHTS`, platform range, and resource requirements.

The configuration retains the original host-specific settings for `DONALD` and `PRECISION`. On other hosts it now uses general defaults instead of exiting immediately; these defaults are **not** a guarantee of reproduction of the paper's experimental setup. Some helper scripts contain machine-specific working-directory paths.

## Citation

If you use this code in academic work, please cite the paper:

Breve, F. A. (2025). *From Pixels to Titles: Video Game Identification by Screenshots Using Convolutional Neural Networks*. IEEE Transactions on Games, 17(2), 536–544. https://doi.org/10.1109/TG.2025.3528187

```bibtex
@article{breve2025pixels,
  author  = {Breve, Fabricio Aparecido},
  title   = {From Pixels to Titles: Video Game Identification by Screenshots Using Convolutional Neural Networks},
  journal = {IEEE Transactions on Games},
  volume  = {17},
  number  = {2},
  pages   = {536--544},
  year    = {2025},
  doi     = {10.1109/TG.2025.3528187}
}
```

[arXiv:2311.15963](https://arxiv.org/abs/2311.15963)

## License

The repository includes an [Apache License 2.0](LICENSE) for the project code. Third-party data, images, and other externally sourced materials may be governed by separate rights and terms.
