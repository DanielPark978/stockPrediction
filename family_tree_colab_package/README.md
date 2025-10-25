# 🌳 Family Tree Generator - Google Colab Version

Generate beautiful family tree visualizations from screenshots and text files directly in your browser!

## 🚀 Quick Start

### Option 1: Upload to Google Colab (Recommended)

1. **Upload the notebook to Google Drive**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Upload `Family_Tree_Generator_Colab.ipynb`

2. **Run all cells**
   - Click `Runtime` → `Run all`
   - Or use keyboard shortcut: `Ctrl+F9` (Windows/Linux) or `Cmd+F9` (Mac)

3. **Upload your files**
   - When prompted, upload your screenshots and/or text files
   - Supported formats: PNG, JPG, JPEG, TXT

4. **View and download results**
   - Preview appears inline
   - Download ZIP with all outputs
   - Or download individual files

### Option 2: Open Directly from GitHub

If you've uploaded this to a GitHub repo:

1. Go to your GitHub repository
2. Click on `Family_Tree_Generator_Colab.ipynb`
3. Click the "Open in Colab" button (or copy the URL)
4. In Colab, go to `File` → `Open notebook` → `GitHub` tab
5. Paste your repository URL

## 📋 What You'll Get

### Visualizations
- **tree_full.png** - High-resolution master image (300 DPI)
- **preview_small.png** - Web-friendly preview (~1600px wide)
- **previews_zoom/** - 6 zoomed-in sections for detail
- **tiles/** - Print-ready tabloid tiles (11"×17") with overlap

### Data Files
- **edges.json** - All parsed relationships
- **aliases.csv** - Name disambiguation mappings
- **tree.dot** - Graphviz source (for advanced editing)
- **tree.graphml** - GraphML format (import to Gephi, yEd, etc.)

## 📝 Input Format

### Text Files (.txt)

Create a text file with relationships in this format:

```
B: Alice Johnson -> Bob Smith
AB: Charlie Davis -> Diana Martinez
B: Eve Thompson -> Frank Wilson & George Taylor
B: Alice Johnson -> Bob Smith -> Charlie Wilson
```

### Format Rules

- **B:** - Big relationship (renders as **bold** edge)
- **AB:** - Assistant Big relationship (renders as **dotted** edge)
- **->** - Arrow (also accepts `→`, `=>`, `—>`)
- **&** or **and** - Multiple names (e.g., `Alice & Bob -> Charlie`)
- **Chains** - `A -> B -> C` creates edges from A→B and B→C

### Screenshots

- Take screenshots of your relationship lists
- Ensure good contrast and legibility
- OCR will extract the text automatically

## ✨ Features

### Smart Processing
- **OCR** - Extracts text from images using Tesseract
- **Auto-correction** - Fixes common OCR errors
- **Name splitting** - Detects stuck-together names (e.g., "SerenaZhangndy" → "Serena Zhang" + "Andy Xu")
- **Deduplication** - Removes duplicate lines

### Professional Layout
- **Hierarchical** - Bigs at the top, littles at the bottom
- **No overlap** - Smart spacing prevents label collisions
- **Styled edges** - Bold for B:, dotted for AB:
- **Clean design** - White rounded boxes, professional fonts

### Print-Ready Output
- **Tabloid tiles** - 11"×17" with 1" overlap for seamless taping
- **Layout map** - JSON file shows how to assemble tiles
- **High DPI** - 150-300 DPI for crisp printing

## 📖 Step-by-Step Guide

### 1. Prepare Your Data

**Option A: Screenshots**
- Screenshot your family tree data
- Save as PNG or JPG
- Keep images clear and readable

**Option B: Text File**
- Create a `.txt` file
- Use the format shown above
- One relationship per line

**Option C: Mix Both**
- Use screenshots for existing data
- Add a text file for corrections or additions

### 2. Upload to Colab

1. Open the notebook in Google Colab
2. Run the first few cells to install dependencies (~30 seconds)
3. When you reach the upload cell, click "Choose Files"
4. Select your screenshots and/or text files
5. Wait for upload to complete

### 3. Generate the Tree

- Run the generator cell
- Watch the progress output
- See statistics about your tree

### 4. View Results

- Preview appears inline in the notebook
- Scroll down to see zoomed sections
- Check statistics (total people, relationships, etc.)

### 5. Download Outputs

**Option A: Download ZIP**
- Contains all files in one package
- Easiest for most users

**Option B: Download Individual Files**
- Choose specific files you need
- Run the individual download cell

## 🔧 Customization

### Adjust Spacing

In the notebook, find the `_generate_graphviz_tree()` method and modify:

```python
dot.set_graph_defaults(
    ranksep='3.0',    # Vertical spacing (increase for more space)
    nodesep='2.5',    # Horizontal spacing (increase for wider layout)
    splines='ortho',  # Edge routing (ortho, curved, line)
    dpi='300'         # Resolution (higher = better quality, larger file)
)
```

### Change Colors

Modify edge colors:

```python
# For B: edges (bold)
edge = pydot.Edge(little, big,
    color='#2c3e50',  # Dark blue (change to any hex color)
    penwidth='2.5'
)

# For AB: edges (dotted)
edge = pydot.Edge(little, big,
    color='#7f8c8d',  # Gray (change to any hex color)
    penwidth='2.0'
)
```

### Adjust Node Styling

```python
pydot_node = pydot.Node(
    node,
    fillcolor='white',      # Background color
    color='gray40',         # Border color
    fontname='Arial Bold',  # Font
    fontsize='14'          # Font size
)
```

## ❓ Troubleshooting

### "No input data found"
- Make sure you uploaded files in the upload cell
- Check that files are in the correct format

### OCR Quality Issues
- Use higher resolution screenshots
- Ensure good contrast (dark text, light background)
- Try creating a manual text file instead

### "Module not found" Errors
- Re-run the installation cells
- Wait for all packages to install
- Restart runtime if needed: `Runtime` → `Restart runtime`

### Names Overlapping
- Increase `ranksep` or `nodesep` values
- See Customization section above

### Graph Too Large
- The notebook handles large graphs automatically
- Preview is scaled to fit
- Download tree_full.png for full resolution

## 💡 Tips

### For Best Results

1. **High-quality inputs** - Clear screenshots with good contrast
2. **Consistent formatting** - Stick to B: and AB: prefixes
3. **Check the preview** - Review edges.json for errors
4. **Manual corrections** - Create a .txt file to fix OCR mistakes
5. **Iterate** - Run multiple times as you refine data

### Example Workflow

1. Screenshot your existing tree data → Upload to Colab
2. Generate tree → Review preview
3. Check edges.json for OCR errors
4. Create corrections.txt with fixes → Upload
5. Re-run generator → Download final outputs

### Printing the Tree

1. Download the tiles/ folder from the ZIP
2. Print each tile on 11"×17" paper
3. Use the layout_map.json to see the grid arrangement
4. Tape tiles together with 1" overlap
5. The overlap ensures seamless alignment

## 🌟 Examples

### Simple Tree

**Input (family.txt):**
```
B: Alice -> Bob
B: Charlie -> Bob
B: Bob -> Diana
```

**Output:**
- 4 people, 3 relationships
- Hierarchical tree with Bob at middle level
- Alice and Charlie at bottom, Diana at top

### Complex Tree

**Input (big_family.txt):**
```
B: Alice & Ben -> Carol
B: Carol -> David & Emma
AB: Frank -> Carol
B: David -> Grace
B: Emma -> Grace
B: Grace -> Henry
```

**Output:**
- 8 people, 7 relationships
- 4 generations
- Multiple paths to top (Henry)
- Mix of bold and dotted edges

## 📦 What's Included

- `Family_Tree_Generator_Colab.ipynb` - Main notebook
- `COLAB_README.md` - This file
- Sample data and documentation (optional)

## 🆘 Need Help?

### Common Questions

**Q: Can I edit the tree after generating?**
A: Yes! Download `tree.dot` and edit it in any text editor, then regenerate with Graphviz.

**Q: Can I import to other tools?**
A: Yes! Use `tree.graphml` to import into Gephi, yEd, or Cytoscape.

**Q: How do I handle duplicate names?**
A: Add qualifiers like years: `Jason Chen '26` vs `Jason Chen '27`. The generator will create aliases automatically.

**Q: Can I use this for other types of graphs?**
A: Yes! Any hierarchical relationship works. Just use the B:/AB: format.

### Still Stuck?

1. Check the cell outputs for error messages
2. Re-run all cells from the beginning
3. Try with sample data first
4. Review the edges.json file to see what was parsed

## 🎓 Advanced Usage

### Exporting for Publications

1. Download `tree.dot`
2. Edit in a text editor for precise control
3. Regenerate with custom DPI:
   ```bash
   dot -Tpng -Gdpi=600 tree.dot -o tree_highres.png
   ```

### Using with Other Tools

- **Gephi**: Import `tree.graphml` for network analysis
- **yEd**: Import `tree.graphml` for advanced layouts
- **Adobe Illustrator**: Import `tree.pdf` (convert from PNG)
- **PowerPoint**: Insert PNG files directly

### Batch Processing

If you have multiple families to process:
1. Upload all files at once
2. The generator automatically processes everything
3. Download the combined result

## 📄 License

Free to use for personal and educational purposes!

---

**Made with ❤️ using Python, Tesseract, and Graphviz**

**Enjoy your family tree! 🌳**
