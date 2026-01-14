# Quick Start Guide - Family Tree Generator

## 5-Minute Setup

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr graphviz python3-pip
```

**macOS:**
```bash
brew install tesseract graphviz python3
```

### 2. Install Python Packages

```bash
pip3 install -r requirements.txt
```

### 3. Add Your Data

Put your screenshots and/or `.txt` files in the `inputs/` folder:

```bash
# Example: Copy your screenshots
cp ~/Downloads/family_tree_screenshot*.png inputs/

# Or create a text file
nano inputs/my_tree.txt
```

**Text file format:**
```
B: Alice -> Bob
AB: Charlie -> Diana
B: Eve & Frank -> George
```

### 4. Run the Generator

```bash
python3 family_tree_generator.py
```

### 5. View Results

Check the `outputs/` folder for:
- `tree_full.png` - High-res master image
- `preview_small.png` - Preview image
- `tiles/` - Print-ready tiles
- `edges.json` - Raw data
- `aliases.csv` - Name mappings (if needed)

## Common Issues

### "tesseract not found"
Install Tesseract OCR (see step 1 above)

### "graphviz not found"
Install Graphviz (see step 1 above)

### "No module named 'X'"
Run: `pip3 install -r requirements.txt`

### "No input data found"
Add files to the `inputs/` folder

### OCR quality issues
- Use high-resolution screenshots
- Ensure good contrast
- Or manually create a `.txt` file

## Example Run

```bash
$ python3 family_tree_generator.py

============================================================
FAMILY TREE GRAPH GENERATOR
============================================================

============================================================
STEP 1: INGESTING INPUTS
============================================================

Found 2 image file(s)
  📷 OCR processing: screenshot1.png
     Extracted 15 lines
  📷 OCR processing: screenshot2.png
     Extracted 22 lines

Found 1 text file(s)
  📄 Reading: family_tree.txt
     Read 48 lines

✓ Total lines collected: 85
✓ After deduplication: 78

============================================================
STEP 2: CLEANING AND PARSING
============================================================

✓ Parsed 98 edges

Auto-corrections made (3):
  • Split 'SerenaZhangndy Xu' → ['Serena Zhang', 'Andy Xu']
  • Split 'MeganFoster' → ['Megan Foster']
  • Normalized 'Alice  and  Bob' → 'Alice & Bob'

============================================================
STEP 3: NAME DISAMBIGUATION
============================================================

⚠️  Found 2 names with duplicates:

  'Jason Chen' has 2 variant(s):
    - Jason Chen '26
    - Jason Chen '27

✓ Created 2 alias mappings

============================================================
STEP 4: BUILDING GRAPH
============================================================

✓ Graph created:
  Nodes: 47
  Edges: 98
  B edges (bold): 82
  AB edges (dotted): 16

============================================================
STEP 5: GENERATING VISUALIZATIONS
============================================================

📊 Generating tree with Graphviz...
  ✓ Saved DOT file: outputs/tree.dot
  ✓ Saved full tree: outputs/tree_full.png
    Size: 4800 x 6200 pixels

📸 Generating previews...
  ✓ Small preview: outputs/preview_small.png (1600x2066)
  ✓ Generated 6 zoom sections in previews_zoom/

🖨️  Generating tabloid tiles...
  Creating 2x3 grid (6 tiles)
  ✓ Saved 6 tiles to tiles/
  ✓ Layout map: outputs/tiles/layout_map.json

============================================================
STEP 6: EXPORTING DATA
============================================================

✓ Edges exported: outputs/edges.json
✓ Aliases exported: outputs/aliases.csv
✓ GraphML exported: outputs/tree.graphml

============================================================
✅ GENERATION COMPLETE!
============================================================

Outputs saved to: outputs

Generated files:
  📁 edges.json (8.2 KB)
  📁 aliases.csv (156 B)
  📁 preview_small.png (856.3 KB)
  📁 tree_full.png (4.2 MB)
  📁 tree.dot (12.4 KB)
  📁 tree.graphml (15.8 KB)
  📁 previews_zoom/zoom_1.png (687.5 KB)
  📁 previews_zoom/zoom_2.png (692.1 KB)
  ... (and more)
```

## Next Steps

1. **Review the preview**: Open `outputs/preview_small.png`
2. **Check for errors**: Review `outputs/edges.json`
3. **Print the tree**: Use tiles in `outputs/tiles/`
4. **Share**: The preview is sized for chat/email

## Tips

- **Test with sample data first**: The `inputs/sample_family_tree.txt` file is already provided
- **Iterate**: Run multiple times as you refine your input data
- **Manual corrections**: Create a `.txt` file to fix OCR errors
- **Customize styling**: Edit the script (lines 445-490) for colors, fonts, etc.

---

**Need help?** Check `FAMILY_TREE_README.md` for detailed documentation.
