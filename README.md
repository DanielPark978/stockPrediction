# Multi-Purpose Python Projects

This repository contains two distinct projects:

## 1. Stock Prediction (LSTM Neural Network)

`ai.py` - A snippet from a project that models a long short-term memory recurrent neural network utilizing time series data to predict stock prices.

**Features:**
- LSTM-based stock price prediction
- Time series data processing
- PyTorch implementation

## 2. Family Tree Graph Generator

A comprehensive tool that processes screenshots and text files containing family tree relationships to generate beautiful, print-ready visualizations.

**Features:**
- OCR processing of screenshots
- Smart name recognition and disambiguation
- Hierarchical graph layout (Bigs on top)
- Print-ready tabloid tiles with overlap
- Multiple output formats (PNG, DOT, GraphML, JSON, CSV)

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your screenshots/text files to inputs/
# Then run:
python3 family_tree_generator.py

# View results in outputs/
```

### Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[FAMILY_TREE_README.md](FAMILY_TREE_README.md)** - Complete documentation
- **[inputs/README.md](inputs/README.md)** - Input format guide

### Input Format

```
B: Alice -> Bob              # Bold edge (Big relationship)
AB: Charlie -> Diana         # Dotted edge (Assistant Big)
B: Eve & Frank -> George     # Multiple names
B: A -> B -> C               # Chains
```

### Outputs

- `outputs/tree_full.png` - High-resolution master image
- `outputs/preview_small.png` - Chat-viewable preview
- `outputs/tiles/` - Tabloid-sized tiles for printing
- `outputs/edges.json` - Parsed relationship data
- `outputs/aliases.csv` - Name disambiguation mappings

---

For detailed usage instructions, see the documentation files above.
