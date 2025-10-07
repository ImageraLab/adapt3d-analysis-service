# ADAPT-3D Vessel Analysis Service
Phase 1: Basic vessel detection with .ims file support
## Features
- Reads Imaris .ims files (HDF5 format)- Extracts 2D slices from 3D volumes- Detects vessel-like structures- Returns quantitative metrics
## Deployment to Render.com
See deployment instructions in the main guide.
## Local Testing (Optional)
```bashpip install -r requirements.txtpython main.py
Visit:  http://localhost:8000