"""
FAMILY TREE GENERATOR - SINGLE CELL VERSION FOR GOOGLE COLAB
==============================================================

INSTRUCTIONS:
1. Go to https://colab.research.google.com/
2. Create a new notebook
3. Copy this ENTIRE file and paste into a code cell
4. Run the cell (Shift+Enter)
5. Upload your files when prompted
6. Download the generated ZIP at the end

That's it!
"""

# ============================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================
print("Installing dependencies...")
import subprocess
import sys

# Install system packages
subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'tesseract-ocr', 'graphviz'],
               check=True, capture_output=True)

# Install Python packages
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'Pillow', 'pytesseract', 'networkx', 'matplotlib', 'pydot'],
               check=True, capture_output=True)

print("✓ All dependencies installed!\n")

# ============================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================
import os
import re
import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import pytesseract
import networkx as nx
import pydot
from google.colab import files
import numpy as np
from IPython.display import display, Image as IPImage

print("✓ Libraries imported\n")

# ============================================================
# STEP 3: FAMILY TREE GENERATOR CLASS
# ============================================================
class FamilyTreeGenerator:
    """Family Tree Graph Generator"""

    def __init__(self, input_folder="inputs", output_folder="outputs"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.raw_lines = []
        self.edges = []
        self.aliases = {}
        self.graph = nx.DiGraph()

        # Create directories
        self.output_folder.mkdir(exist_ok=True)
        (self.output_folder / "tiles").mkdir(exist_ok=True)
        (self.output_folder / "previews_zoom").mkdir(exist_ok=True)

    def ingest_inputs(self):
        """OCR images and read text files"""
        print(f"\n{'='*60}")
        print("STEP 1: INGESTING INPUTS")
        print(f"{'='*60}")

        if not self.input_folder.exists():
            print(f"⚠️  No inputs folder found")
            return

        # Process images
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_files = [f for f in self.input_folder.iterdir() if f.suffix in image_extensions]

        print(f"\nFound {len(image_files)} image file(s)")
        for img_path in image_files:
            print(f"  📷 OCR processing: {img_path.name}")
            try:
                text = pytesseract.image_to_string(Image.open(img_path))
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                self.raw_lines.extend(lines)
                print(f"     Extracted {len(lines)} lines")
            except Exception as e:
                print(f"     ⚠️  Error: {e}")

        # Process text files
        txt_files = [f for f in self.input_folder.iterdir() if f.suffix == '.txt']
        print(f"\nFound {len(txt_files)} text file(s)")
        for txt_path in txt_files:
            print(f"  📄 Reading: {txt_path.name}")
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    self.raw_lines.extend(lines)
                    print(f"     Read {len(lines)} lines")
            except Exception as e:
                print(f"     ⚠️  Error: {e}")

        original_count = len(self.raw_lines)
        self.raw_lines = list(set(self.raw_lines))
        print(f"\n✓ Total lines: {original_count}")
        print(f"✓ After deduplication: {len(self.raw_lines)}")

    def clean_and_parse(self):
        """Clean and parse relationships"""
        print(f"\n{'='*60}")
        print("STEP 2: CLEANING AND PARSING")
        print(f"{'='*60}")

        edge_pattern = re.compile(r'^(B|AB)\s*:\s*(.+?)$', re.IGNORECASE)
        corrections = []

        for line in self.raw_lines:
            line = self._normalize_line(line)
            match = edge_pattern.match(line)
            if not match:
                continue

            edge_type = match.group(1).upper()
            relationship = match.group(2)
            parsed = self._parse_relationship(relationship, edge_type)

            if parsed:
                self.edges.extend(parsed['edges'])
                corrections.extend(parsed['corrections'])

        print(f"\n✓ Parsed {len(self.edges)} edges")
        if corrections:
            print(f"\nAuto-corrections ({len(corrections)}):")
            for corr in corrections[:5]:
                print(f"  • {corr}")
            if len(corrections) > 5:
                print(f"  ... and {len(corrections) - 5} more")

    def _normalize_line(self, line):
        line = re.sub(r'[-=]+>', '->', line)
        line = re.sub(r'—>', '->', line)
        line = re.sub(r'→', '->', line)
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'\s+and\s+', ' & ', line, flags=re.IGNORECASE)
        return line.strip()

    def _parse_relationship(self, relationship, edge_type):
        corrections = []
        edges = []
        parts = [p.strip() for p in relationship.split('->')]

        if len(parts) < 2:
            return None

        processed_parts = []
        for part in parts:
            names = re.split(r'\s*&\s*', part)
            cleaned_names = []
            for name in names:
                split_names = self._split_stuck_names(name)
                if len(split_names) > 1:
                    corrections.append(f"Split '{name}' → {split_names}")
                    cleaned_names.extend(split_names)
                else:
                    cleaned_names.append(name)
            cleaned_names = [n for n in cleaned_names if self._is_valid_name(n)]
            processed_parts.append(cleaned_names)

        for i in range(len(processed_parts) - 1):
            for little in processed_parts[i]:
                for big in processed_parts[i + 1]:
                    edges.append({'little': little, 'big': big, 'type': edge_type})

        return {'edges': edges, 'corrections': corrections}

    def _split_stuck_names(self, name):
        pattern = r'([a-z])([A-Z])'
        match = re.search(pattern, name)
        if match and len(name) > 10:
            pos = match.start() + 1
            name1 = name[:pos].strip()
            name2 = name[pos:].strip()
            if self._is_valid_name(name1) and self._is_valid_name(name2):
                return [name1, name2]
        return [name]

    def _is_valid_name(self, name):
        if len(name) < 2:
            return False
        if not re.search(r'[a-zA-Z]', name):
            return False
        if re.match(r'^[^a-zA-Z0-9]+$', name):
            return False
        return True

    def disambiguate_names(self):
        """Create aliases for duplicate names"""
        print(f"\n{'='*60}")
        print("STEP 3: NAME DISAMBIGUATION")
        print(f"{'='*60}")

        all_names = set()
        for edge in self.edges:
            all_names.add(edge['little'])
            all_names.add(edge['big'])

        name_groups = defaultdict(list)
        for name in all_names:
            base = self._extract_base_name(name)
            name_groups[base].append(name)

        duplicates = {k: v for k, v in name_groups.items() if len(v) > 1}

        if not duplicates:
            print("\n✓ No duplicate names")
            return

        print(f"\n⚠️  Found {len(duplicates)} names with duplicates")
        for base_name, variants in duplicates.items():
            for i, variant in enumerate(variants):
                canonical = variant if i == 0 else f"{variant} ({i+1})"
                self.aliases[variant] = canonical

        for edge in self.edges:
            if edge['little'] in self.aliases:
                edge['little'] = self.aliases[edge['little']]
            if edge['big'] in self.aliases:
                edge['big'] = self.aliases[edge['big']]

        print(f"✓ Created {len(self.aliases)} aliases")

    def _extract_base_name(self, name):
        base = re.sub(r"'?\d{2,4}", "", name)
        base = re.sub(r'[(\[].*?[)\]]', "", base)
        base = re.sub(r'@.*', "", base)
        return base.strip()

    def build_graph(self):
        """Build NetworkX graph"""
        print(f"\n{'='*60}")
        print("STEP 4: BUILDING GRAPH")
        print(f"{'='*60}")

        self.graph = nx.DiGraph()
        for edge in self.edges:
            self.graph.add_edge(edge['little'], edge['big'], edge_type=edge['type'])

        b_edges = sum(1 for e in self.edges if e['type'] == 'B')
        ab_edges = sum(1 for e in self.edges if e['type'] == 'AB')

        print(f"\n✓ Graph created:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  B (bold): {b_edges}")
        print(f"  AB (dotted): {ab_edges}")

    def generate_visualizations(self):
        """Generate all visualizations"""
        print(f"\n{'='*60}")
        print("STEP 5: GENERATING VISUALIZATIONS")
        print(f"{'='*60}")

        if self.graph.number_of_nodes() == 0:
            print("\n⚠️  No nodes to visualize")
            return

        self._generate_graphviz_tree()
        self._generate_previews()
        self._generate_tiles()

    def _generate_graphviz_tree(self):
        print("\n📊 Generating tree with Graphviz...")
        try:
            dot = pydot.Dot(graph_type='digraph', rankdir='TB')
            dot.set_graph_defaults(ranksep='3.0', nodesep='2.5', splines='ortho', dpi='300')

            for node in self.graph.nodes():
                width = max(1.5, len(node) * 0.15)
                pydot_node = pydot.Node(
                    node, label=node, shape='box', style='rounded,filled',
                    fillcolor='white', color='gray40', fontname='Arial Bold',
                    fontsize='14', width=str(width), height='0.6', margin='0.2,0.1'
                )
                dot.add_node(pydot_node)

            for little, big, data in self.graph.edges(data=True):
                edge_type = data.get('edge_type', 'B')
                if edge_type == 'B':
                    edge = pydot.Edge(little, big, style='bold', penwidth='2.5',
                                    color='#2c3e50', arrowsize='1.0')
                else:
                    edge = pydot.Edge(little, big, style='dotted', penwidth='2.0',
                                    color='#7f8c8d', arrowsize='0.8')
                dot.add_edge(edge)

            dot_path = self.output_folder / "tree.dot"
            dot.write_raw(str(dot_path))

            png_path = self.output_folder / "tree_full.png"
            dot.write_png(str(png_path))

            img = Image.open(png_path)
            print(f"  ✓ Saved tree_full.png ({img.width}x{img.height} px)")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")

    def _generate_previews(self):
        print("\n📸 Generating previews...")
        tree_path = self.output_folder / "tree_full.png"
        if not tree_path.exists():
            return

        img = Image.open(tree_path)

        # Small preview
        preview_width = 1600
        ratio = preview_width / img.width
        preview_height = int(img.height * ratio)
        small_preview = img.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        small_path = self.output_folder / "preview_small.png"
        small_preview.save(small_path, optimize=True)
        print(f"  ✓ preview_small.png ({preview_width}x{preview_height})")

        # Zoomed sections
        num_sections = 6
        section_width = img.width // 3
        section_height = img.height // 2

        for i in range(num_sections):
            row = i // 3
            col = i % 3
            left = col * section_width
            top = row * section_height
            right = min(left + section_width, img.width)
            bottom = min(top + section_height, img.height)

            crop = img.crop((left, top, right, bottom))
            zoom_path = self.output_folder / "previews_zoom" / f"zoom_{i+1}.png"
            crop.save(zoom_path)

        print(f"  ✓ Generated {num_sections} zoom sections")

    def _generate_tiles(self):
        print("\n🖨️  Generating tabloid tiles...")
        tree_path = self.output_folder / "tree_full.png"
        if not tree_path.exists():
            return

        img = Image.open(tree_path)
        tile_width = 17 * 150
        tile_height = 11 * 150
        overlap = 150

        cols = int(np.ceil(img.width / (tile_width - overlap)))
        rows = int(np.ceil(img.height / (tile_height - overlap)))

        print(f"  Creating {rows}x{cols} grid ({rows*cols} tiles)")

        layout = {
            'total_tiles': rows * cols,
            'grid': {'rows': rows, 'cols': cols},
            'tile_size': {'width': tile_width, 'height': tile_height},
            'overlap': overlap,
            'tiles': []
        }

        tile_num = 1
        for row in range(rows):
            for col in range(cols):
                left = col * (tile_width - overlap)
                top = row * (tile_height - overlap)
                right = min(left + tile_width, img.width)
                bottom = min(top + tile_height, img.height)

                tile = Image.new('RGB', (tile_width, tile_height), 'white')
                crop = img.crop((left, top, right, bottom))
                tile.paste(crop, (0, 0))

                tile_path = self.output_folder / "tiles" / f"tile_{row+1}_{col+1}.png"
                tile.save(tile_path, dpi=(150, 150))

                layout['tiles'].append({
                    'filename': tile_path.name,
                    'position': {'row': row + 1, 'col': col + 1},
                    'number': tile_num
                })
                tile_num += 1

        layout_path = self.output_folder / "tiles" / "layout_map.json"
        with open(layout_path, 'w') as f:
            json.dump(layout, f, indent=2)

        print(f"  ✓ Saved {rows*cols} tiles")

    def export_data(self):
        """Export data files"""
        print(f"\n{'='*60}")
        print("STEP 6: EXPORTING DATA")
        print(f"{'='*60}")

        edges_path = self.output_folder / "edges.json"
        with open(edges_path, 'w') as f:
            json.dump(self.edges, f, indent=2)
        print(f"\n✓ edges.json")

        if self.aliases:
            aliases_path = self.output_folder / "aliases.csv"
            with open(aliases_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Raw Name', 'Canonical Name'])
                for raw, canonical in self.aliases.items():
                    writer.writerow([raw, canonical])
            print(f"✓ aliases.csv")

        if self.graph.number_of_nodes() > 0:
            graphml_path = self.output_folder / "tree.graphml"
            nx.write_graphml(self.graph, graphml_path)
            print(f"✓ tree.graphml")

    def run(self):
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("FAMILY TREE GRAPH GENERATOR")
        print("="*60)

        self.ingest_inputs()
        if not self.raw_lines:
            print("\n⚠️  No input data found")
            return

        self.clean_and_parse()
        self.disambiguate_names()
        self.build_graph()
        self.generate_visualizations()
        self.export_data()

        print(f"\n{'='*60}")
        print("✅ GENERATION COMPLETE!")
        print(f"{'='*60}")


# ============================================================
# STEP 4: UPLOAD FILES
# ============================================================
print("\n" + "="*60)
print("UPLOAD YOUR FILES")
print("="*60)
print("\nPlease upload your screenshots (.png, .jpg) and/or text files (.txt)")
print("\nText file format:")
print("  B: Alice -> Bob")
print("  AB: Charlie -> Diana")
print("  B: Eve & Frank -> George")
print()

# Create inputs directory
os.makedirs('inputs', exist_ok=True)

# Upload files
uploaded = files.upload()

# Move to inputs folder
for filename in uploaded.keys():
    shutil.move(filename, f'inputs/{filename}')
    print(f"✓ Moved {filename} to inputs/")

print(f"\n✓ {len(uploaded)} file(s) uploaded")

# ============================================================
# STEP 5: GENERATE FAMILY TREE
# ============================================================
generator = FamilyTreeGenerator()
generator.run()

# ============================================================
# STEP 6: DISPLAY PREVIEW
# ============================================================
print("\n" + "="*60)
print("PREVIEW")
print("="*60)

preview_path = "outputs/preview_small.png"
if os.path.exists(preview_path):
    print("\n📊 Family Tree Preview:\n")
    display(IPImage(filename=preview_path))
else:
    print("⚠️  Preview not found")

# Display statistics
edges_path = "outputs/edges.json"
if os.path.exists(edges_path):
    with open(edges_path) as f:
        edges_data = json.load(f)

    b_count = sum(1 for e in edges_data if e['type'] == 'B')
    ab_count = sum(1 for e in edges_data if e['type'] == 'AB')

    all_names = set()
    for e in edges_data:
        all_names.add(e['little'])
        all_names.add(e['big'])

    print("\n📊 Statistics:")
    print(f"  Total people: {len(all_names)}")
    print(f"  Total relationships: {len(edges_data)}")
    print(f"  Big relationships (bold): {b_count}")
    print(f"  Assistant Big (dotted): {ab_count}")

# ============================================================
# STEP 7: DOWNLOAD OUTPUTS
# ============================================================
print("\n" + "="*60)
print("DOWNLOAD OUTPUTS")
print("="*60)

# Create ZIP
print("\n📦 Creating ZIP file with all outputs...")
shutil.make_archive('family_tree_outputs', 'zip', 'outputs')

print("\n✓ Downloading family_tree_outputs.zip...")
files.download('family_tree_outputs.zip')

print("\n✅ DONE!")
print("\nThe ZIP contains:")
print("  • tree_full.png - High-res master image")
print("  • preview_small.png - Web-friendly preview")
print("  • tiles/ - Print-ready tabloid tiles (11x17\")")
print("  • previews_zoom/ - Zoomed sections")
print("  • edges.json - Raw relationship data")
print("  • aliases.csv - Name mappings (if any)")
print("  • tree.dot - Graphviz source")
print("  • tree.graphml - GraphML format")
print("\n🎉 Enjoy your family tree!")
