# NeedleShapeModeling

**Physics-based 3D needle deflection simulation and real-time force/torque-driven shape estimation.**

NeedleShapeModeling is a research-oriented toolkit for estimating needle-tip displacement and visualizing needle deformation under externally applied forces and moments. The repository includes a nonlinear SBCM formulation, a distributed-load Euler–Bernoulli beam model, interactive desktop and browser interfaces, and real-time integration with a Net F/T sensor through the RDT protocol.

> **Research use only.** This repository is an engineering prototype and is not a medical device or a clinically validated navigation system.

## Features

- Nonlinear 3D needle-deflection estimation solved using Newton–Raphson iteration
- Distributed-load Euler–Bernoulli beam approximation for an inserted needle segment
- Tip displacement estimation: \(U_x\), \(U_y\), and \(U_z\)
- Tip orientation estimation: \(\theta_y\) and \(\theta_z\)
- Interactive 3D needle-shape visualization
- Standalone browser demo with no Python installation
- PyQt5 desktop interfaces with real-time parameter sliders
- Live Net F/T sensor acquisition through UDP/RDT
- Moving-average filtering, packet-drop monitoring, and CSV export

## Repository Structure

| File | Description |
|---|---|
| `UI_SBCM_Dec_10.html` | Standalone browser-based SBCM simulator with interactive controls and 3D visualization |
| `SBCM.py` | Minimal command-line implementation of the nonlinear SBCM needle-tip deflection model |
| `UI_SBCM.py` | PyQt5 interface for interactive SBCM simulation and 3D shape visualization |
| `SBCM_RDT.py` | Real-time Net F/T sensor acquisition and SBCM-based deflection estimation |
| `SBCM_modified3.py` | Enhanced real-time SBCM pipeline with calibration, filtering, diagnostics, and CSV export |
| `Distributed_load.py` | Real-time distributed-load beam model driven by Net F/T measurements |
| `UI_Distributed_load.py` | PyQt5 interface for the distributed-load model |

## Quick Start

### 1. Browser Demo

The fastest way to explore the model is to open:

```text
UI_SBCM_Dec_10.html
```

Double-click the file or open it in a modern browser. The demo is self-contained and does not require Python or additional packages.

Use the sliders to modify:

- Insertion length
- Forces \(F_x\), \(F_y\), and \(F_z\)
- Moments \(M_x\), \(M_y\), and \(M_z\)

Drag the 3D view to rotate it and use the mouse wheel to zoom.

### 2. Install the Python Dependencies

Clone the repository and enter the project directory:

```bash
git clone https://github.com/PieceZhang/NeedleShapeModeling.git
cd NeedleShapeModeling
```

Creating a virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install numpy matplotlib PyQt5
```

The real-time sensor scripts additionally use Python standard-library modules for UDP communication, threading, XML parsing, and CSV export.

## Usage

### Interactive SBCM Simulation

```bash
python UI_SBCM.py
```

This interface provides sliders for the insertion length, three force components, and three moment components. It displays the estimated tip displacement, tip rotation, solver convergence status, and an interpolated 3D needle shape.

### Interactive Distributed-Load Simulation

```bash
python UI_Distributed_load.py
```

This interface adds parameters for insertion length and friction force. It visualizes the exposed and inserted needle segments separately and updates the estimated deformation in real time.

### Command-Line SBCM Evaluation

```bash
python SBCM.py
```

Enter the force and moment values when prompted. The script reports:

- Axial displacement \(U_x\)
- Transverse displacements \(U_y\) and \(U_z\)
- Tip rotations \(\theta_y\) and \(\theta_z\)

The current command-line implementation evaluates the model using the default needle length stored in `NeedleDeflectionModel`. Use `UI_SBCM.py` or call the model directly when a variable insertion length is required.

### Use the SBCM Model from Python

```python
from SBCM import NeedleDeflectionModel

model = NeedleDeflectionModel()

Ux, Uy, Uz, theta_y, theta_z = model.calculate_tip_deflection_newton(
    F_x=1.0,       # N
    F_y=0.2,       # N
    F_z=-0.1,      # N
    M_x=0.0,       # N·m
    M_y=0.01,      # N·m
    M_z=-0.01,     # N·m
    L=0.185,       # m
)

print(f"Tip displacement: ({Ux}, {Uy}, {Uz}) m")
print(f"Tip rotation: ({theta_y}, {theta_z}) rad")
```

### Use the Distributed-Load Model from Python

```python
from Distributed_load import PdfEBBeamModel

model = PdfEBBeamModel(
    needle_length_m=0.200,
    needle_diameter_m=1.27e-3,
    E=200e9,
)

model.update_user_inputs(
    f_fric_N=0.2,
    f_insert_m=0.100,
)

result = model.compute_deflection(
    Fx=0.5,
    Fy=0.2,
    Fz=1.0,
    Mx=0.0,
    My=0.01,
    Mz=0.0,
)

print(result)
```

The returned dictionary contains displacement, angular displacement, transverse-force magnitude, moment magnitude, normal-force estimate, and total transverse deflection.

## Real-Time Net F/T Sensor Integration

The following scripts receive force/torque measurements from a Net F/T sensor:

```bash
python SBCM_RDT.py
python Distributed_load.py
python SBCM_modified3.py
```

The default network configuration is:

```python
SENSOR_IP = "192.168.1.1"
RDT_PORT = 49152
```

Before running a live script:

1. Connect the computer and sensor to the same network.
2. Change `SENSOR_IP` when the sensor uses a different address.
3. Confirm that the sensor configuration endpoint is accessible.
4. Verify the force/torque unit scaling returned by the sensor.
5. Check the coordinate transformation for the physical sensor mounting.

The current RDT scripts use the following mapping:

```text
Model (x, y, z) = Sensor (z, x, y)
```

The same mapping is applied to moments. Modify `sensor_to_model_axes()` when your sensor coordinate frame or mounting orientation differs.

Press `Ctrl+C` to stop acquisition.

### Enhanced Real-Time Pipeline

`SBCM_modified3.py` additionally provides:

- Configurable displacement calibration factor
- Moving-average filtering
- Total transverse deflection monitoring
- Raw force and torque plots
- Computation-error and packet-drop statistics
- Automatic export to `needle_deflection_data.csv`

The default calibration factor and filter window are defined directly in the script and should be adjusted for the experimental setup.

## Model Parameters and Units

The models use SI units internally.

| Quantity | Unit |
|---|---|
| Force | N |
| Moment | N·m |
| Length and displacement | m |
| Displayed displacement | mm |
| Rotation | rad |
| Young's and shear modulus | Pa |

Representative default needle properties are:

| Parameter | Default |
|---|---:|
| Needle diameter | 1.27 mm |
| Young's modulus \(E\) | 200 GPa |
| Shear modulus \(G\) | 80 GPa |
| Poisson's ratio \(\mu\) | 0.28 |
| Needle length | 185 or 200 mm, depending on the script |

The distributed-load model also uses a default bevel angle of \(30^\circ\), an offset parameter of 4.8 mm, and a 151-point finite-difference grid.

## Shape Visualization

The models directly estimate tip displacement and orientation. The graphical interfaces reconstruct a smooth 3D centerline using cubic Hermite interpolation between the fixed needle base and the predicted tip state.

This visualization is intended to provide an intuitive representation of deformation. It should not be interpreted as a validated full-field reconstruction without comparison against measured needle shapes.

## Limitations

- The mechanical models use simplified beam and loading assumptions.
- Material, geometry, friction, bevel, and calibration parameters are currently defined in the source code.
- Model accuracy depends on calibration for the specific needle, sensor mounting, insertion medium, and operating range.
- The real-time scripts assume an RDT-compatible Net F/T sensor and a reachable sensor configuration endpoint.
- Automated tests, packaging metadata, and formal validation datasets are not yet included.
- The software has not been validated for clinical decision-making or autonomous intervention.

## Contributing

Issues and pull requests are welcome. Contributions may include:

- Model validation against optical or imaging-based ground truth
- Additional needle and tissue parameter configurations
- Improved numerical stability and automated testing
- Sensor abstraction and configurable coordinate transforms
- Recorded-data replay and benchmarking tools
- Documentation, examples, and visualization improvements

## Citation

When this repository supports academic work, please cite the repository and the associated publication when available.

## License

No explicit software license is currently included in this repository. Contact the repository owner before redistribution, modification, or commercial use.
