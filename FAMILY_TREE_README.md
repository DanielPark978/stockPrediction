# Family Tree Graph Generator

A powerful tool that processes screenshots and text files containing family tree relationships to generate beautiful, print-ready visualizations.

## Features

- **Multi-source Input**: OCR from images + text file parsing
- **Smart Name Recognition**: Handles chains, multiple names, and stuck-together OCR errors
- **Name Disambiguation**: Automatically creates aliases for duplicate names
- **Professional Layout**: Hierarchical top-down tree with Bigs at the top
- **Print-Ready Outputs**: Tabloid-sized tiles with overlap for seamless taping
- **Multiple Formats**: PNG, DOT, GraphML, JSON, CSV

## Installation

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr graphviz
```

**macOS:**
```bash
brew install tesseract graphviz
```

**Windows:**
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Graphviz: https://graphviz.org/download/

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Input Format

Place your screenshots and/or text files in the `inputs/` folder.

### Supported Formats

**Bold edges (Big relationships):**
```
B: Alice -> Bob
B: Charlie -> Dana & Emily
B: Frank -> George -> Henry
```

**Dotted edges (Assistant Big relationships):**
```
AB: Ian -> Julia
AB: Kevin & Lisa -> Marcus
```

### Format Rules

- **Arrows**: `->`  (or `→`, `—>`, `=>`)
- **Multiple names**: Use `&` or `and` (e.g., `Alice & Bob -> Charlie`)
- **Chains**: `A -> B -> C` creates edges A→B and B→C
- **Edge types**: `B:` for bold, `AB:` for dotted

## Usage

### Basic Usage

```bash
python family_tree_generator.py
```

This will:
1. Read all images and `.txt` files from `inputs/`
2. OCR images and parse text files
3. Clean and normalize the data
4. Generate visualizations in `outputs/`

### Custom Folders

```bash
python family_tree_generator.py --input my_data --output my_results
```

### Make Script Executable (Optional)

```bash
chmod +x family_tree_generator.py
./family_tree_generator.py
```

## Outputs

All files are saved to the `outputs/` folder:

### Visualizations

- **`tree_full.png`**: High-resolution master image (300 DPI)
  - Bigs at the top, littles at the bottom
  - Bold edges for B: relationships
  - Dotted edges for AB: relationships
  - White rounded boxes with bold text
  - No label overlap

- **`preview_small.png`**: Chat-viewable preview (~1600px wide)

- **`previews_zoom/`**: Zoomed-in sections for detailed viewing
  - `zoom_1.png` through `zoom_6.png`

- **`tiles/`**: Tabloid-sized tiles (11"×17") for printing
  - `tile_1_1.png`, `tile_1_2.png`, etc.
  - 1" overlap between tiles for seamless taping
  - `layout_map.json`: Grid layout and assembly guide

### Data Files

- **`edges.json`**: All parsed relationships
  ```json
  [
    {"little": "Alice", "big": "Bob", "type": "B"},
    {"little": "Charlie", "big": "Dana", "type": "AB"}
  ]
  ```

- **`aliases.csv`**: Name disambiguation mappings
  ```csv
  Raw Name,Canonical Name
  Jason Chen,Jason Chen
  Jason Chen,Jason Chen (2)
  ```

- **`tree.dot`**: Graphviz DOT format (for advanced editing)

- **`tree.graphml`**: GraphML format (import into Gephi, yEd, etc.)

## How It Works

### 1. Ingestion
- OCRs all images using Tesseract
- Reads all `.txt` files
- Deduplicates lines

### 2. Cleaning & Parsing
- Normalizes arrows (`->`, `→`, `=>`)
- Splits names on `&` or `and`
- Detects stuck-together names (e.g., `SerenaZhangndy Xu` → `Serena Zhang` + `Andy Xu`)
- Filters junk tokens
- Parses chains (`A -> B -> C`)

### 3. Disambiguation
- Identifies duplicate names
- Creates canonical aliases using hints (year, class, email)
- Generates `aliases.csv` mapping

### 4. Graph Building
- Constructs directed graph (little → big)
- Applies hierarchical layout
- Ensures Bigs are above Littles

### 5. Visualization
- Uses Graphviz for optimal layout
- Applies styling:
  - **B edges**: Bold, dark blue
  - **AB edges**: Dotted, gray
  - **Nodes**: White rounded boxes, bold text
- Generates high-res PNG

### 6. Post-Processing
- Creates chat-friendly preview
- Generates zoom sections
- Tiles image for printing
- Exports data files

## Examples

### Example Input File (`inputs/family_tree.txt`)

```
B: Alice Johnson -> Bob Smith
B: Charlie Davis -> Dana Lee & Emily White
AB: Frank Brown -> George Wilson
B: Bob Smith -> Henry Clark -> Iris Green
AB: Emily White & Dana Lee -> Jack Taylor
```

### Example OCR from Screenshot

```
B: Serena Zhang -> Andy Xu
B: Jason Chen '26 -> Arthur Chen
AB: Alan Chu -> Jason Chen '27
```

### Result

- 6 unique people (2 Jason Chens disambiguated)
- 8 edges (6 B, 2 AB)
- Hierarchical tree with no overlaps
- Tiles ready for printing

## Troubleshooting

### No nodes to visualize
- Check that your input files use the correct format (`B:` or `AB:`)
- Ensure arrows are `->` not `<-`

### OCR quality issues
- Use high-resolution screenshots
- Ensure good contrast (dark text on light background)
- Manually correct errors in a `.txt` file

### Labels overlap
- Adjust `ranksep` and `nodesep` in script (around line 460)
- Increase values for more spacing

### Graphviz not found
- Install system Graphviz package (see Installation)
- Script will fall back to NetworkX layout

## Advanced Usage

### Editing the DOT File

1. Open `outputs/tree.dot` in a text editor
2. Modify layout, colors, or labels
3. Regenerate PNG:
   ```bash
   dot -Tpng -o tree_custom.png outputs/tree.dot
   ```

### Importing to Other Tools

- **Gephi**: Import `tree.graphml`
- **yEd**: Import `tree.graphml`
- **Cytoscape**: Convert via NetworkX
- **GraphViz tools**: Use `tree.dot`

### Custom Styling

Edit lines 445-490 in `family_tree_generator.py` to customize:
- Colors
- Font sizes
- Box shapes
- Edge styles
- Spacing

## Tips

1. **High-quality inputs**: Use clear screenshots with good contrast
2. **Consistent formatting**: Stick to `B:` and `AB:` prefixes
3. **Check OCR output**: Review `edges.json` for errors
4. **Manual corrections**: Create a `.txt` file with corrections
5. **Iterative refinement**: Run multiple times, adjusting as needed

## Output Statistics

After running, you'll see:

```
✓ Total lines collected: 156
✓ After deduplication: 142
✓ Parsed 98 edges
✓ Graph created:
  Nodes: 47
  Edges: 98
  B edges (bold): 82
  AB edges (dotted): 16
```

## License

MIT License - Feel free to modify and distribute!

## Credits

Built with:
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Graphviz](https://graphviz.org/)
- [NetworkX](https://networkx.org/)
- [Pillow](https://python-pillow.org/)

---

**Questions?** Check the comments in `family_tree_generator.py` or open an issue.
