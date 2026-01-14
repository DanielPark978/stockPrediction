# 🚀 Google Colab - Copy & Paste Version

The **easiest way** to use the Family Tree Generator in Google Colab!

## ⚡ Super Quick Start (2 Steps)

### Step 1: Go to Google Colab

Open your browser and go to: **https://colab.research.google.com/**

Click **"New notebook"** (or File → New notebook)

### Step 2: Copy & Paste the Code

1. Open the file **`colab_single_cell.py`** (in this repo)

2. **Select ALL the code** (Ctrl+A or Cmd+A)

3. **Copy it** (Ctrl+C or Cmd+C)

4. **Paste into the Colab cell** (Ctrl+V or Cmd+V)

5. **Run the cell** (Click the ▶️ play button, or press Shift+Enter)

That's it! The script will:
- ✅ Install all dependencies automatically
- ✅ Prompt you to upload files
- ✅ Generate your family tree
- ✅ Show preview inline
- ✅ Download a ZIP with all outputs

---

## 📋 What Happens When You Run It

### 1. Installation (~30 seconds)
```
Installing dependencies...
✓ All dependencies installed!
✓ Libraries imported
```

### 2. File Upload
You'll see a **"Choose Files"** button. Upload your:
- Screenshots (.png, .jpg)
- Text files (.txt) with relationships

### 3. Processing
```
============================================================
FAMILY TREE GRAPH GENERATOR
============================================================

STEP 1: INGESTING INPUTS
  📷 OCR processing: screenshot1.png
     Extracted 25 lines
  📄 Reading: family_tree.txt
     Read 50 lines

STEP 2: CLEANING AND PARSING
  ✓ Parsed 75 edges

STEP 3: NAME DISAMBIGUATION
  ✓ No duplicate names

STEP 4: BUILDING GRAPH
  ✓ Graph created:
    Nodes: 42
    Edges: 75
    B (bold): 68
    AB (dotted): 7

STEP 5: GENERATING VISUALIZATIONS
  📊 Generating tree with Graphviz...
  ✓ Saved tree_full.png (4800x6200 px)
  📸 Generating previews...
  ✓ preview_small.png (1600x2066)
  ✓ Generated 6 zoom sections
  🖨️  Generating tabloid tiles...
  Creating 2x3 grid (6 tiles)
  ✓ Saved 6 tiles

STEP 6: EXPORTING DATA
  ✓ edges.json
  ✓ tree.graphml

✅ GENERATION COMPLETE!
```

### 4. Preview Display
The tree preview appears inline in the notebook!

### 5. Statistics
```
📊 Statistics:
  Total people: 42
  Total relationships: 75
  Big relationships (bold): 68
  Assistant Big (dotted): 7
```

### 6. Download
A ZIP file downloads automatically with all outputs!

---

## 📝 Input Format

Create a text file with this format:

```
B: Alice Johnson -> Bob Smith
AB: Charlie Davis -> Diana Martinez
B: Eve Thompson -> Frank Wilson & George Taylor
B: Alice Johnson -> Bob Smith -> Charlie Wilson
```

**Rules:**
- `B:` = Big relationship (bold edge)
- `AB:` = Assistant Big relationship (dotted edge)
- `->` = Arrow (also accepts →, =>, —>)
- `&` or `and` = Multiple names
- Chains work: `A -> B -> C`

---

## 📦 What You'll Get

The downloaded ZIP contains:

**Visualizations:**
- `tree_full.png` - High-res master (300 DPI)
- `preview_small.png` - Web preview
- `tiles/` - Print-ready 11×17" tiles with overlap
  - `tile_1_1.png`, `tile_1_2.png`, etc.
  - `layout_map.json` - Assembly guide
- `previews_zoom/` - 6 zoomed sections

**Data Files:**
- `edges.json` - All parsed relationships
- `aliases.csv` - Name mappings (if duplicates exist)
- `tree.dot` - Graphviz source (for editing)
- `tree.graphml` - Import to Gephi, yEd, etc.

---

## 🎨 Example: Simple Family Tree

### Input (paste into a .txt file):

```
B: Alice -> Carol
B: Bob -> Carol
B: Carol -> David
B: Carol -> Emma
AB: Frank -> Carol
B: David -> Grace
B: Emma -> Grace
```

### Output:
- 7 people, 7 relationships
- 4 generations
- Hierarchical layout (Grace at top, Alice & Bob at bottom)
- Bold edges for B:, dotted for AB:

---

## 🎨 Example: Complex Tree

### Input:

```
B: Alice & Ben -> Carol
B: Carol -> David & Emma
AB: Frank -> Carol
B: David -> Grace
B: Emma -> Grace
B: Grace -> Henry
B: Henry -> Iris & Jack
```

### Output:
- 10 people, 8 relationships
- 5 generations
- Multiple parents to same child
- Mix of bold/dotted edges

---

## 💡 Pro Tips

### 1. Test with Sample Data First

Create a simple test file:
```
B: You -> Your Big
B: Your Big -> Their Big
```

Upload and run to see how it works!

### 2. Handle Duplicate Names

If you have multiple people with the same name, add qualifiers:
```
B: Jason Chen '26 -> Alice Wang
B: Jason Chen '27 -> Bob Lee
```

The generator will create aliases automatically.

### 3. Fix OCR Errors

If OCR makes mistakes:
1. Check the `edges.json` file in the output
2. Create a new `.txt` file with corrections
3. Re-upload and run again

### 4. Adjust Spacing

If labels overlap, edit the code around line 230:
```python
dot.set_graph_defaults(
    ranksep='3.5',  # Increase for more vertical space
    nodesep='3.0',  # Increase for more horizontal space
    ...
)
```

### 5. Change Colors

Around line 255, modify colors:
```python
# For B: edges
color='#2c3e50',  # Change to any hex color

# For AB: edges
color='#7f8c8d',  # Change to any hex color
```

---

## ❓ Troubleshooting

### "Module not found" Error
- The installation cell may have failed
- Re-run the cell (it installs packages at the top)

### "No input data found"
- Make sure you uploaded files when prompted
- Check that files are in the correct format (B: or AB:)

### OCR Quality Issues
- Use high-resolution screenshots
- Ensure good contrast (dark text, light background)
- Or create a manual `.txt` file instead

### Graph Too Large
- The preview is auto-scaled to fit
- Download `tree_full.png` for full resolution
- Use the zoom previews to see details

### Labels Overlapping
- Increase `ranksep` and `nodesep` values (see Pro Tips #4)
- Re-run the cell

---

## 🔧 Advanced: Customization

The single-cell script is fully customizable. Here are common tweaks:

### Change Graph Direction

Line 229, change `rankdir`:
```python
dot = pydot.Dot(graph_type='digraph', rankdir='TB')
# TB = Top to Bottom (default)
# BT = Bottom to Top
# LR = Left to Right
# RL = Right to Left
```

### Change Node Shape

Line 235:
```python
shape='box',        # Options: box, circle, ellipse, diamond
style='rounded,filled',  # Options: rounded, filled, dashed
fillcolor='white',  # Any color name or hex
```

### Change Font

Line 237:
```python
fontname='Arial Bold',  # Try: 'Helvetica', 'Times', 'Courier'
fontsize='14',         # Adjust size: '10', '12', '16', etc.
```

### Export as PDF

Add after line 251:
```python
pdf_path = self.output_folder / "tree.pdf"
dot.write_pdf(str(pdf_path))
print(f"  ✓ Saved tree.pdf")
```

---

## 📱 Works On Any Device!

Because it runs in Google Colab, you can use this on:
- ✅ Windows
- ✅ Mac
- ✅ Linux
- ✅ Chromebook
- ✅ iPad/Tablet (with keyboard)
- ✅ Even your phone! (though keyboard recommended)

---

## 🆚 Copy-Paste vs Notebook File

**Copy-Paste (this method):**
- ✅ Faster - just copy and paste
- ✅ Easier to customize on the fly
- ✅ All in one cell
- ❌ Need to paste every time

**Notebook File (.ipynb):**
- ✅ Upload once, use many times
- ✅ Better organized (multiple cells)
- ✅ Includes documentation inline
- ❌ Need to upload file first

**Use whichever you prefer!** Both do the exact same thing.

---

## 📄 File Locations in This Repo

- **`colab_single_cell.py`** ← Copy this entire file into Colab
- **`Family_Tree_Colab.zip`** ← Contains the .ipynb notebook version
- **`COLAB_README.md`** ← Detailed documentation

---

## 🎉 You're Ready!

1. Go to https://colab.research.google.com/
2. Create new notebook
3. Copy `colab_single_cell.py` into a cell
4. Run it!
5. Upload your files
6. Download your family tree!

**It's that easy! 🌳**

---

## 🆘 Still Need Help?

- Check that you copied the **entire** `colab_single_cell.py` file
- Make sure you're in Google Colab, not Jupyter Notebook
- Try with sample data first before your real data
- Read the error messages - they usually explain what went wrong

---

**Made with ❤️ for easy family tree generation!**
