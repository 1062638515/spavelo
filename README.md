# spaVelo: Dependency-aware RNA Velocity Inference

spaVelo is a spatial dependency-aware variational autoencoder framework for RNA velocity analysis in spatial transcriptomics data. It integrates spatial information using Gaussian processes to improve velocity estimation and capture spatially coherent dynamics.

## Requirements

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/spavelo.git
cd spavelo

# Create conda environment with required dependencies
conda env create -f environment.yml
conda activate spvelo
```

See `environment.yml` for the complete list of dependencies.

## Quick Start

### Command Line Usage

**For simulated data:**

```bash
python run_spaVelo.py \
    --data_file data_simu.h5ad \
    --device cuda:0 \
    --inducing_point_steps 15 \
    --w_latent_loss 1 \
    --KL_weight 10 \
    --penalty_scale 10 \
    --save_dir logs/experiment_name
```

**For real data (with denoising):**

```bash
cd experiments
python run_spaVelo_real_deno.py \
    --data_file /path/to/your/data.h5ad \
    --inducing_point_steps 15 \
    --gene_selection spatial \
    --device cuda:0 \
    --w_latent_loss 1 \
    --KL_weight 10 \
    --penalty_scale 10
```

## Example Scripts

We provide example scripts in the `scripts/` directory for different datasets:

### Simulated Data
```bash
bash scripts/run_experiment_simu.sh
```

### OSCC Data (Oral Squamous Cell Carcinoma)
```bash
bash scripts/run_experiment_oscc.sh
```

### Axolotl Brain Development Data
```bash
# Stage57
bash scripts/run_experiment_axolotl_stage.sh

# Stage44 and Stage54
bash scripts/run_experiment_axolotl_stage_44_54.sh
```

**Note**: You need to modify the data paths in the scripts to point to your own data locations.

## Data Format

spaVelo expects AnnData objects with the following structure:

- `adata.layers['spliced']`: Spliced counts matrix
- `adata.layers['unspliced']`: Unspliced counts matrix  
- `adata.obsm['X_coord']`: Spatial coordinates ($N \times 2$)

Example data preparation:
```python
import scanpy as sc
import numpy as np

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Ensure spatial coordinates are included
# Format: numpy array of shape (n_cells, n_spatial_dims)
adata.obsm['X_coord'] = np.array([[x1, y1], [x2, y2], ...])

# Save
adata.write_h5ad("processed_data.h5ad")
```

## Key Parameters

### Model Architecture
- `--GP_dim`: Dimension of the Gaussian process latent space (default: 2)
- `--Normal_dim`: Dimension of the standard Gaussian latent space (default: 8)
- `--encoder_layers`: Encoder layer sizes (default: [256, 128])
- `--decoder_layers`: Decoder layer sizes (default: [128, 256])

### Training
- `--maxiter`: Maximum training iterations (default: 5000)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 256, or "auto" for automatic selection)
- `--patience`: Early stopping patience (default: 50)

### Regularization
- `--KL_weight`: Weight for Dirichlet KL divergence (default: 1.0)
- `--penalty_scale`: Coefficient for the penalty term (default: 0.2)
- `--w_latent_loss`: Weight for latent space loss (default: 1.0)

### Spatial Modeling
- `--inducing_point_steps`: Grid steps for inducing points (default: 15)
  - Results in $(steps+1)^2$ inducing points
- `--kernel_scale`: Spatial kernel scale parameter (default: 20.0)
- `--loc_range`: Range for spatial coordinate normalization (default: 20.0)
- `--grid_inducing_points`: Use grid (True) or k-means (False) for inducing points

### Dynamic VAE
- `--dynamicVAE`: Enable dynamic beta adjustment (default: True)
- `--init_beta`: Initial KL coefficient (default: 4.0)
- `--min_beta`: Minimum KL coefficient (default: 1.0)
- `--max_beta`: Maximum KL coefficient (default: 25.0)
- `--KL_loss`: Target KL divergence weight $\beta$ (default: 0.025)

## Output

The model saves the following outputs in the specified `--save_dir`:

- `model.pt`: Trained model checkpoint
- `args.csv`: All hyperparameters used
- `adata_result.h5ad`: Annotated data with predictions
  - `adata.layers['pred_s']`: Predicted spliced expression
  - `adata.layers['pred_u']`: Predicted unspliced expression
  - `adata.layers['pred_velocity']`: Predicted RNA velocity


## Project Structure

```
spavelo/
├── run_spaVelo.py           # Main script for simulated data
├── spaVelo.py               # Core model implementation
├── spaVeloDenoise.py        # Denoising variant
├── preprocess.py            # Data preprocessing utilities
├── SVGP.py                  # Sparse variational Gaussian process
├── kernel.py                # Kernel functions
├── VAE_utils.py             # VAE utility functions
├── I_PID.py                 # PID control for dynamic VAE
├── scripts/                 # Example running scripts
│   ├── run_experiment_simu.sh
│   ├── run_experiment_oscc.sh
│   └── run_experiment_axolotl_*.sh
├── experiments/             # Scripts for real data experiments
│   └── run_spaVelo_real_deno.py
├── tools/                   # Additional utilities
└── align/                   # Spatial alignment modules
```


## Acknowledgments

This work builds upon:
- [spavae](https://github.com/ttgump/spaVAE) for integrating spatial coordinates using Gaussian processes
