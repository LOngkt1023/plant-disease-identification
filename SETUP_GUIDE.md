# Setup Guide - Environment Configuration

## Problem
Your current MSYS64 Python (3.11.4) has build/distutils issues preventing Pillow, numpy, pandas installation.

## Solution Options (Choose One)

### **Option A: Use Official Python.org Python (Recommended - 5 min)**

1. Download Python 3.11 from https://www.python.org/downloads/
2. Install with:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install pip"
   - ❌ Uncheck "Use admin privileges" if you want to install in user directory

3. Verify installation:
   ```powershell
   python --version
   pip --version
   ```

4. From project root, create virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements-minimal.txt
   ```

5. Run crawler:
   ```powershell
   python src/crawler_pure_python.py --max 50
   ```

---

### **Option B: Use Miniconda (Easier - 10 min)**

1. Download Miniconda from https://docs.conda.io/projects/miniconda/en/latest/
2. Install (use default paths)
3. Open PowerShell and activate conda:
   ```powershell
   conda init powershell
   ```

4. Create conda environment:
   ```powershell
   conda create -n crop-disease python=3.11 -y
   conda activate crop-disease
   pip install -r requirements-minimal.txt
   ```

5. Run crawler:
   ```powershell
   python src/crawler_pure_python.py --max 50
   ```

---

### **Option C: Use WSL2 Linux (Advanced - 15 min)**

If you have WSL2 installed:

```bash
wsl
cd ~/Desktop/DaiHoc/Ky6/KHDL/Plant\ disease\ identification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-minimal.txt
python src/crawler_pure_python.py --max 50
```

---

## Test Environment After Setup

```powershell
python -c "import selenium; import requests; import tqdm; print('✓ All imports OK!')"
```

Expected: `✓ All imports OK!`

---

## Then Run Full Pipeline

**After environment is ready:**

```powershell
# Phase 1-2: Scrape images (50 per class test)
python scripts/crawl.py --max-images-per-class 50

# Phase 3: Check downloaded images
dir dataset_v2\raw\Rice_Healthy\
dir dataset_v2\raw\Tomato_EarlyBlight\
```

**If Phase 1-2 works, we'll add:**
- Phase 3: Data cleaning (duplicates)
- Phase 4: Image preprocessing (resize + filter)
- Phase 5: Parquet export + metadata

---

## Troubleshooting

**Error: "ModuleNotFoundError: No module named 'selenium'"**
- You forgot to activate venv or didn't run `pip install`

**Error: "CERTIFICATE_VERIFY_FAILED"**
- Normal SSL issue on download; our crawler handles this with SSL bypass

**Error: "ChromeDriver not found"**
- Selenium will auto-download compatible ChromeDriver (needs internet)

---

## Next Steps After Setup

1. Test with `python src/crawler_pure_python.py --max 50`
2. Run full pipeline notebook: `notebooks/01_data_crawling.ipynb`
3. Check `data/raw/<ClassNameFolder>/` for downloaded images
