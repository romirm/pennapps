# GNN Training Guide for Airport Bottleneck Prediction

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
cd /Users/andyphu/pennapps/pennapps
pip install torch torch-geometric pandas numpy matplotlib tqdm geopy
```

### 2. Run Simple Training

```bash
# Quick test (5 minutes)
python train_model.py --config testing --epochs 5

# Development training (20 minutes)
python train_model.py --config development

# Full production training (2+ hours)
python train_model.py --config production
```

### 3. Test Your Trained Model

```bash
python train_model.py --evaluate models/best_bottleneck_model.pth
```

## What Your GNN Model Does

Your **BottleneckGNN** analyzes spatial relationships between aircraft to predict:

1. **Queue Formation** - Aircraft stacking on runway approaches
2. **Intersection Conflicts** - Traffic conflicts at taxiway crossings
3. **Gate Congestion** - Aircraft waiting for gate assignments
4. **Severity Levels** - 1-5 scale bottleneck severity
5. **Impact Predictions** - Passenger delays, fuel waste, economic cost

## Training Architecture

```
ADS-B Data → Graph Construction → GNN → KAN Predictor → Impact Analysis
     ↓              ↓               ↓         ↓            ↓
Aircraft Positions → Spatial Graph → Bottleneck → Impact → Recommendations
```

### Components:

- **BottleneckGNN**: Graph neural network with custom message passing
- **BottleneckKANPredictor**: Impact prediction using KAN architecture
- **ADSBDataProcessor**: Converts aircraft data to graphs
- **BottleneckDataset**: Synthetic training data generator

## Training Data Generation

Your training uses **synthetic airport scenarios** with:

- **800 training samples** (configurable)
- **200 validation samples**
- **Multiple airports**: KJFK, KLAX, KPHL, KMIA
- **Realistic aircraft distributions**: B737, A320, B777, etc.
- **Operational phases**: approach, departure, taxi, gate, holding
- **Bottleneck severity levels**: 0.0 to 1.0
- **Weather impact factors**: 0.8 to 1.2 multiplier

## Training Process

### 1. Data Generation

```python
# Creates synthetic scenarios with known bottleneck labels
train_dataset = BottleneckDataset(num_samples=800, airports=['KJFK', 'KLAX'])
```

### 2. Graph Construction

```python
# Converts aircraft positions to spatial graphs
graph_data = adsb_processor.construct_bottleneck_graph(aircraft_list, airport_config)
```

### 3. Model Training

```python
# Hybrid GNN-KAN training with multiple loss functions
trainer = BottleneckTrainer(model, config)
history = trainer.train(train_dataset, val_dataset)
```

### 4. Loss Functions

- **Binary Cross-Entropy**: Queue/intersection/gate probabilities
- **Cross-Entropy**: Severity classification (1-5 levels)
- **MSE**: Impact predictions (fuel, passengers, cost)

## Configuration Options

### Quick Configs:

```bash
--config testing     # 50 samples, 5 epochs (2 minutes)
--config development # 100 samples, 10 epochs (10 minutes)
--config production  # 2000 samples, 100 epochs (2+ hours)
```

### Custom Training:

```bash
python train_model.py --epochs 20 --batch-size 16
```

## Output Files

After training, you'll get:

```
models/
├── best_bottleneck_model.pth      # Best validation loss
├── final_bottleneck_model.pth     # Final epoch model
└── bottleneck_model_20240101_120000.pth  # Timestamped

logs/
├── training_history_20240101_120000.json
└── training_metrics.log

plots/
└── training_history.png
```

## Model Performance

### Expected Results:

- **Training Loss**: Should decrease to < 0.1
- **Validation Loss**: Should track training loss
- **Bottleneck Accuracy**: Target > 80%
- **Impact Prediction**: ±20% accuracy on fuel/passenger estimates

### Training Time:

- **CPU**: 2-5x slower than GPU
- **GPU**: ~1-2 minutes per epoch (development config)
- **Memory**: ~2-4GB RAM, ~1-2GB VRAM

## Troubleshooting

### Common Issues:

1. **Import Errors**

   ```bash
   # Make sure you're in the project root
   cd /Users/andyphu/pennapps/pennapps
   python -c "import torch; print(torch.__version__)"
   ```

2. **CUDA Not Available**

   ```
   ⚠️ CUDA not available - using CPU (training will be slower)
   ```

   This is normal - training will work on CPU, just slower.

3. **Memory Issues**

   ```bash
   # Reduce batch size
   python train_model.py --batch-size 4
   ```

4. **Slow Training**
   ```bash
   # Use smaller config
   python train_model.py --config testing
   ```

## Advanced Usage

### Resume Training:

```bash
python train_model.py --resume models/checkpoint.pth
```

### Custom Data:

```python
# Modify BottleneckDataset in model/train_gnn.py
# Add your own airport scenarios
```

### Hyperparameter Tuning:

```python
# Edit model/training_config.py
TRAINING_CONFIG = {
    'learning_rate': 0.001,  # Try 0.0001 - 0.01
    'gnn_hidden_dim': 64,    # Try 32, 64, 128
    'gnn_layers': 4,         # Try 2, 4, 6
    'batch_size': 8          # Try 4, 8, 16
}
```

## Integration with Your System

### After Training:

```python
# Load trained model
model = AirportBottleneckModel(BOTTLENECK_CONFIG)
model.load_state_dict(torch.load('models/best_bottleneck_model.pth'))

# Use with real ADS-B data
from model.flight_bottleneck_predictor import BottleneckPredictor
predictor = BottleneckPredictor()
results = predictor.predict_bottlenecks_near_airport('KJFK')
```

## Next Steps

1. **Train the model**: Start with `--config development`
2. **Evaluate performance**: Check accuracy on test data
3. **Integrate real data**: Connect to ADS-B.lol API
4. **Deploy**: Use trained model in your bottleneck prediction system
5. **Monitor**: Track model performance in production

## Performance Optimization

### For Faster Training:

- Use GPU if available
- Increase batch size (if memory allows)
- Reduce number of training samples for testing
- Use fewer GNN layers for simpler scenarios

### For Better Accuracy:

- Increase training samples (production config)
- Add more diverse airport scenarios
- Tune hyperparameters
- Collect real historical bottleneck data

---

**Ready to start?** Run: `python train_model.py --config development`

