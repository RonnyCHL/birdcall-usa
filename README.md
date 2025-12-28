# Birdcall-USA

CNN vocalization classifier for North American birds - classifies songs, calls & alarms using spectrograms.

## Overview

This project trains deep learning models to classify bird vocalizations into three categories:
- **Song** - Territorial/mating vocalizations
- **Call** - Contact/communication calls
- **Alarm** - Warning/distress calls

Currently supports **50 common North American species** based on eBird frequency data.

## Quick Start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RonnyCHL/birdcall-usa/blob/main/notebooks/train_colab.ipynb)

1. Open the notebook in Google Colab (Pro+ recommended for A100 GPU)
2. Run all cells to train models
3. Models are saved to `data/models/`

## Local Installation

```bash
git clone https://github.com/RonnyCHL/birdcall-usa.git
cd birdcall-usa
pip install torch torchaudio librosa scikit-learn matplotlib seaborn requests
```

## Usage

### Train all species
```bash
python full_pipeline.py
```

### Train only most common species (priority 1)
```bash
python full_pipeline.py --priority 1
```

### Train a specific species
```bash
python full_pipeline.py --species "American Robin"
```

### Dry run (see what would be trained)
```bash
python full_pipeline.py --dry-run
```

## Species List

### Priority 1 - Very Common (13 species)
| Species | Scientific Name | eBird Freq |
|---------|-----------------|------------|
| Mourning Dove | *Zenaida macroura* | 35% |
| Northern Cardinal | *Cardinalis cardinalis* | 34% |
| American Robin | *Turdus migratorius* | 33% |
| American Crow | *Corvus brachyrhynchos* | 32% |
| Blue Jay | *Cyanocitta cristata* | 28% |
| Song Sparrow | *Melospiza melodia* | 25% |
| Red-winged Blackbird | *Agelaius phoeniceus* | 25% |
| European Starling | *Sturnus vulgaris* | 25% |
| American Goldfinch | *Spinus tristis* | 24% |
| Canada Goose | *Branta canadensis* | 23% |
| House Finch | *Haemorhous mexicanus* | 23% |
| Downy Woodpecker | *Dryobates pubescens* | 23% |
| Mallard | *Anas platyrhynchos* | 22% |

### Priority 2 - Common (13 species)
Red-bellied Woodpecker, House Sparrow, Turkey Vulture, Black-capped Chickadee, Tufted Titmouse, Dark-eyed Junco, White-breasted Nuthatch, Northern Flicker, Great Blue Heron, Northern Mockingbird, Carolina Wren, Red-tailed Hawk, Common Grackle

### Priority 3 - Regular (24 species)
Barn Swallow, Yellow-rumped Warbler, Ring-billed Gull, Gray Catbird, Common Yellowthroat, Brown-headed Cowbird, Chipping Sparrow, Tree Swallow, Eastern Bluebird, White-throated Sparrow, Killdeer, Eastern Phoebe, Cedar Waxwing, Ruby-throated Hummingbird, Baltimore Oriole, Indigo Bunting, Eastern Towhee, Brown Thrasher, Purple Finch, Pine Warbler, Carolina Chickadee, Eastern Meadowlark, Wood Thrush, Scarlet Tanager

## Model Architecture

- **Input:** 128x128 mel-spectrogram
- **Architecture:** 4-layer CNN with batch normalization
- **Output:** 3 classes (song, call, alarm)
- **Training:** ~150 samples per class from Xeno-canto

## Data Source

Audio recordings are automatically downloaded from [Xeno-canto](https://xeno-canto.org/), the world's largest collection of bird sounds.

## Project Structure

```
birdcall-usa/
├── data/
│   ├── raw/              # Downloaded audio files
│   ├── models/           # Trained .pt model files
│   └── spectrograms-*/   # Generated spectrograms
├── logs/                 # Training logs and plots
├── notebooks/
│   └── train_colab.ipynb # Colab training notebook
├── src/
│   ├── classifiers/      # CNN model code
│   ├── collectors/       # Xeno-canto downloader
│   └── processors/       # Spectrogram generator
├── full_pipeline.py      # Main training script
└── us_bird_species.py    # Species list
```

## Training Time Estimates

| Scope | Species | Time (A100) |
|-------|---------|-------------|
| Priority 1 | 13 | ~1 hour |
| Priority 1+2 | 26 | ~2-3 hours |
| All | 50 | ~4-6 hours |

## Related Projects

- [emsn-vocalization](https://github.com/RonnyCHL/emsn-vocalization) - Original Dutch/European version
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) - Species identification model

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Audio data from [Xeno-canto](https://xeno-canto.org/) contributors
- Species frequency data from [eBird](https://ebird.org/)
