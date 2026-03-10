# VSNA
VSNA (Batley et al., 2026): separable spatiotemporal-parametric modelling for advection-diffusion systems.

## Repository Structure
```text
vsna/
├── README.md
├── requirements.txt
├── .gitignore
├── output/
├── vsna/
│   ├── __init__.py
│   ├── errors.py
│   ├── initial_condition.py
│   ├── matrix_foundry.py
│   ├── solver.py
│   └── utils.py
└── main.py
```

## Quickstart (linux)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```
Output figures are written to `output/`.

## Citation
If you use this code, please cite:

@article{batley2026separable,
    title={Separable neural architectures: a primitive for unified predictive and generative intelligence},
    author={Batley, Reza T. and Sarker, Apurba and Mostakim, Rajib and Klichine, Andrew and Saha, Sourav},
    journal={},
    year={2026}
}

## Notes
- Default run settings in `main.py`: `ncells=8`, `ncellt=8`, `ncellp=8`, `rank=16`.
- Solver output image: `output/solution.png`.
