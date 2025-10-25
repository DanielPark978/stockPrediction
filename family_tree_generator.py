#!/usr/bin/env python3
"""
Family Tree Graph Generator
============================

Processes screenshots and text files containing family tree relationships
in the format:
  B: [little] -> [big] (Bold edges for Big relationships)
  AB: [little] -> [assistant big] (Dotted edges for Assistant Big relationships)

Outputs:
  - tree_full.png: High-resolution master image
  - tiles/: Tabloid-sized tiles for printing
  - preview_small.png: Chat-viewable preview
  - previews_zoom/: Zoomed-in sections
  - edges.json: Parsed relationship data
  - aliases.csv: Name disambiguation mappings
"""

import os
import re
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# OCR and Image Processing
try:
    from PIL import Image
    import pytesseract
except ImportError:
    print("PIL/pytesseract not installed. Install with: pip install Pillow pytesseract")

# Graph libraries
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("NetworkX/matplotlib not installed. Install with: pip install networkx matplotlib")

try:
    import pydot
except ImportError:
    print("pydot not installed. Install with: pip install pydot")

import numpy as np


class FamilyTreeGenerator:
    """Main class for generating family tree visualizations"""

    def __init__(self, input_folder: str = "inputs", output_folder: str = "outputs"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.raw_lines = []
        self.edges = []
        self.aliases = {}
        self.graph = nx.DiGraph()

        # Create output directories
        self.output_folder.mkdir(exist_ok=True)
        (self.output_folder / "tiles").mkdir(exist_ok=True)
        (self.output_folder / "previews_zoom").mkdir(exist_ok=True)

    def ingest_inputs(self):
        """OCR images and read text files from input folder"""
        print(f"\n{'='*60}")
        print("STEP 1: INGESTING INPUTS")
        print(f"{'='*60}")

        if not self.input_folder.exists():
            print(f"⚠️  Input folder '{self.input_folder}' not found.")
            print(f"    Creating it now. Please add your screenshots and .txt files there.")
            self.input_folder.mkdir(exist_ok=True)
            return

        # Process images with OCR
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_files = [f for f in self.input_folder.iterdir()
                      if f.suffix in image_extensions]

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
                    lines = [line.strip() for line in f if line.strip()]
                    self.raw_lines.extend(lines)
                    print(f"     Read {len(lines)} lines")
            except Exception as e:
                print(f"     ⚠️  Error: {e}")

        # Deduplicate
        original_count = len(self.raw_lines)
        self.raw_lines = list(set(self.raw_lines))
        print(f"\n✓ Total lines collected: {original_count}")
        print(f"✓ After deduplication: {len(self.raw_lines)}")

    def clean_and_parse(self):
        """Clean OCR text and parse relationships"""
        print(f"\n{'='*60}")
        print("STEP 2: CLEANING AND PARSING")
        print(f"{'='*60}")

        edge_pattern = re.compile(
            r'^(B|AB)\s*:\s*(.+?)$',
            re.IGNORECASE
        )

        corrections = []

        for line in self.raw_lines:
            # Normalize line
            line = self._normalize_line(line)

            match = edge_pattern.match(line)
            if not match:
                continue

            edge_type = match.group(1).upper()
            relationship = match.group(2)

            # Parse the relationship chain
            parsed = self._parse_relationship(relationship, edge_type)

            if parsed:
                self.edges.extend(parsed['edges'])
                corrections.extend(parsed['corrections'])

        print(f"\n✓ Parsed {len(self.edges)} edges")

        if corrections:
            print(f"\nAuto-corrections made ({len(corrections)}):")
            for correction in corrections[:10]:  # Show first 10
                print(f"  • {correction}")
            if len(corrections) > 10:
                print(f"  ... and {len(corrections) - 10} more")

    def _normalize_line(self, line: str) -> str:
        """Normalize arrows, spaces, and common OCR errors"""
        # Fix common arrow variations
        line = re.sub(r'[-=]+>', '->', line)
        line = re.sub(r'—>', '->', line)
        line = re.sub(r'→', '->', line)

        # Normalize spaces
        line = re.sub(r'\s+', ' ', line)

        # Normalize "and" to "&"
        line = re.sub(r'\s+and\s+', ' & ', line, flags=re.IGNORECASE)

        return line.strip()

    def _parse_relationship(self, relationship: str, edge_type: str) -> Dict:
        """Parse a relationship string into edges"""
        corrections = []
        edges = []

        # Split by arrows to get chain
        parts = [p.strip() for p in relationship.split('->')]

        if len(parts) < 2:
            return None

        # Process each part for multiple names
        processed_parts = []
        for part in parts:
            names = self._split_names(part)

            # Check for stuck-together names (e.g., "SerenaZhangndy Xu")
            cleaned_names = []
            for name in names:
                split_names = self._split_stuck_names(name)
                if len(split_names) > 1:
                    corrections.append(f"Split '{name}' → {split_names}")
                    cleaned_names.extend(split_names)
                else:
                    cleaned_names.append(name)

            # Filter junk
            cleaned_names = [n for n in cleaned_names if self._is_valid_name(n)]
            processed_parts.append(cleaned_names)

        # Build edges from chain
        for i in range(len(processed_parts) - 1):
            littles = processed_parts[i]
            bigs = processed_parts[i + 1]

            for little in littles:
                for big in bigs:
                    edges.append({
                        'little': little,
                        'big': big,
                        'type': edge_type
                    })

        return {'edges': edges, 'corrections': corrections}

    def _split_names(self, text: str) -> List[str]:
        """Split on & or 'and' to get multiple names"""
        names = re.split(r'\s*&\s*', text)
        return [n.strip() for n in names if n.strip()]

    def _split_stuck_names(self, name: str) -> List[str]:
        """Detect and split incorrectly merged names like 'SerenaZhangndy Xu'"""
        # Pattern: Look for lowercase followed by uppercase in the middle
        # e.g., "aZ" or "nA" suggests stuck names
        pattern = r'([a-z])([A-Z])'

        match = re.search(pattern, name)
        if match and len(name) > 10:  # Only if name is suspiciously long
            # Split at the pattern
            pos = match.start() + 1
            name1 = name[:pos].strip()
            name2 = name[pos:].strip()

            # Validate both parts look like names
            if self._is_valid_name(name1) and self._is_valid_name(name2):
                return [name1, name2]

        return [name]

    def _is_valid_name(self, name: str) -> bool:
        """Check if a string looks like a valid name"""
        if len(name) < 2:
            return False

        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', name):
            return False

        # Filter out single punctuation or junk
        if re.match(r'^[^a-zA-Z0-9]+$', name):
            return False

        return True

    def disambiguate_names(self):
        """Create canonical aliases for duplicate names"""
        print(f"\n{'='*60}")
        print("STEP 3: NAME DISAMBIGUATION")
        print(f"{'='*60}")

        # Collect all names
        all_names = set()
        for edge in self.edges:
            all_names.add(edge['little'])
            all_names.add(edge['big'])

        # Group by base name (without qualifiers)
        name_groups = defaultdict(list)
        for name in all_names:
            base = self._extract_base_name(name)
            name_groups[base].append(name)

        # Find duplicates
        duplicates = {k: v for k, v in name_groups.items() if len(v) > 1}

        if not duplicates:
            print("\n✓ No duplicate names found")
            return

        print(f"\n⚠️  Found {len(duplicates)} names with duplicates:")

        for base_name, variants in duplicates.items():
            print(f"\n  '{base_name}' has {len(variants)} variant(s):")
            for v in variants:
                print(f"    - {v}")

            # Create canonical names
            for i, variant in enumerate(variants):
                if i == 0:
                    # First one keeps original name
                    canonical = variant
                else:
                    # Others get numbered
                    canonical = f"{variant} ({i+1})"

                self.aliases[variant] = canonical

        # Apply aliases to edges
        for edge in self.edges:
            if edge['little'] in self.aliases:
                edge['little'] = self.aliases[edge['little']]
            if edge['big'] in self.aliases:
                edge['big'] = self.aliases[edge['big']]

        print(f"\n✓ Created {len(self.aliases)} alias mappings")

    def _extract_base_name(self, name: str) -> str:
        """Extract base name without year, class, etc."""
        # Remove common qualifiers like '26, '27, email, etc.
        base = re.sub(r"'?\d{2,4}", "", name)
        base = re.sub(r'[(\[].*?[)\]]', "", base)
        base = re.sub(r'@.*', "", base)
        return base.strip()

    def build_graph(self):
        """Build NetworkX graph with proper hierarchy"""
        print(f"\n{'='*60}")
        print("STEP 4: BUILDING GRAPH")
        print(f"{'='*60}")

        self.graph = nx.DiGraph()

        # Add edges (little -> big)
        for edge in self.edges:
            self.graph.add_edge(
                edge['little'],
                edge['big'],
                edge_type=edge['type']
            )

        print(f"\n✓ Graph created:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")

        # Calculate statistics
        b_edges = sum(1 for e in self.edges if e['type'] == 'B')
        ab_edges = sum(1 for e in self.edges if e['type'] == 'AB')
        print(f"  B edges (bold): {b_edges}")
        print(f"  AB edges (dotted): {ab_edges}")

    def generate_visualizations(self):
        """Generate all visualization outputs"""
        print(f"\n{'='*60}")
        print("STEP 5: GENERATING VISUALIZATIONS")
        print(f"{'='*60}")

        if self.graph.number_of_nodes() == 0:
            print("\n⚠️  No nodes to visualize")
            return

        # Use Graphviz for best layout
        self._generate_graphviz_tree()

        # Generate previews
        self._generate_previews()

        # Generate tiles
        self._generate_tiles()

    def _generate_graphviz_tree(self):
        """Generate high-quality tree using Graphviz"""
        print("\n📊 Generating tree with Graphviz...")

        try:
            # Convert to pydot
            dot = pydot.Dot(graph_type='digraph', rankdir='TB')

            # Set graph attributes for spacing
            dot.set_graph_defaults(
                ranksep='3.0',
                nodesep='2.5',
                splines='ortho',
                dpi='300'
            )

            # Add nodes with proper styling
            for node in self.graph.nodes():
                # Calculate width based on name length
                width = max(1.5, len(node) * 0.15)

                pydot_node = pydot.Node(
                    node,
                    label=node,
                    shape='box',
                    style='rounded,filled',
                    fillcolor='white',
                    color='gray40',
                    fontname='Arial Bold',
                    fontsize='14',
                    width=str(width),
                    height='0.6',
                    margin='0.2,0.1'
                )
                dot.add_node(pydot_node)

            # Add edges with proper styling
            for little, big, data in self.graph.edges(data=True):
                edge_type = data.get('edge_type', 'B')

                if edge_type == 'B':
                    # Bold for Big relationships
                    edge = pydot.Edge(
                        little, big,
                        style='bold',
                        penwidth='2.5',
                        color='#2c3e50',
                        arrowsize='1.0'
                    )
                else:  # AB
                    # Dotted for Assistant Big
                    edge = pydot.Edge(
                        little, big,
                        style='dotted',
                        penwidth='2.0',
                        color='#7f8c8d',
                        arrowsize='0.8'
                    )

                dot.add_edge(edge)

            # Save DOT file
            dot_path = self.output_folder / "tree.dot"
            dot.write_raw(str(dot_path))
            print(f"  ✓ Saved DOT file: {dot_path}")

            # Generate PNG
            png_path = self.output_folder / "tree_full.png"
            dot.write_png(str(png_path))
            print(f"  ✓ Saved full tree: {png_path}")

            # Get image size
            img = Image.open(png_path)
            print(f"    Size: {img.width} x {img.height} pixels")

        except Exception as e:
            print(f"  ⚠️  Graphviz error: {e}")
            print("     Falling back to NetworkX layout...")
            self._generate_networkx_tree()

    def _generate_networkx_tree(self):
        """Fallback: Generate tree using NetworkX with matplotlib"""
        print("\n📊 Generating tree with NetworkX...")

        # Calculate hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        except:
            # Fallback to spring layout if graphviz not available
            pos = nx.spring_layout(self.graph, k=3, iterations=50)

        # Create figure
        fig, ax = plt.subplots(figsize=(40, 30), dpi=150)

        # Draw edges
        for little, big, data in self.graph.edges(data=True):
            x1, y1 = pos[little]
            x2, y2 = pos[big]

            edge_type = data.get('edge_type', 'B')

            if edge_type == 'B':
                ax.arrow(x1, y1, x2-x1, y2-y1,
                        head_width=15, head_length=20,
                        fc='#2c3e50', ec='#2c3e50',
                        linewidth=3, length_includes_head=True)
            else:  # AB
                ax.plot([x1, x2], [y1, y2],
                       'gray', linestyle=':', linewidth=2)
                ax.arrow(x1, y1, x2-x1, y2-y1,
                        head_width=12, head_length=15,
                        fc='gray', ec='gray',
                        linewidth=0, length_includes_head=True)

        # Draw nodes
        for node in self.graph.nodes():
            x, y = pos[node]

            # Calculate box size based on text length
            text_width = len(node) * 0.018

            # Draw rounded box
            box = FancyBboxPatch(
                (x - text_width/2, y - 0.015),
                text_width, 0.03,
                boxstyle="round,pad=0.01",
                facecolor='white',
                edgecolor='gray',
                linewidth=1.5
            )
            ax.add_patch(box)

            # Draw text
            ax.text(x, y, node,
                   fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   zorder=10)

        ax.axis('off')
        ax.set_aspect('equal')
        plt.tight_layout()

        png_path = self.output_folder / "tree_full.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved full tree: {png_path}")

    def _generate_previews(self):
        """Generate preview images"""
        print("\n📸 Generating previews...")

        tree_path = self.output_folder / "tree_full.png"
        if not tree_path.exists():
            print("  ⚠️  tree_full.png not found, skipping previews")
            return

        img = Image.open(tree_path)

        # Small preview for chat
        preview_width = 1600
        ratio = preview_width / img.width
        preview_height = int(img.height * ratio)

        small_preview = img.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        small_path = self.output_folder / "preview_small.png"
        small_preview.save(small_path, optimize=True)
        print(f"  ✓ Small preview: {small_path} ({preview_width}x{preview_height})")

        # Zoomed sections
        num_sections = 6
        section_width = img.width // 3
        section_height = img.height // 2

        zoom_paths = []
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
            zoom_paths.append(zoom_path)

        print(f"  ✓ Generated {len(zoom_paths)} zoom sections in previews_zoom/")

    def _generate_tiles(self):
        """Generate tabloid-sized tiles with overlap for printing"""
        print("\n🖨️  Generating tabloid tiles...")

        tree_path = self.output_folder / "tree_full.png"
        if not tree_path.exists():
            print("  ⚠️  tree_full.png not found, skipping tiles")
            return

        img = Image.open(tree_path)

        # Tabloid dimensions at 150 DPI
        tile_width = 17 * 150  # 11x17 landscape
        tile_height = 11 * 150
        overlap = 150  # 1 inch overlap

        # Calculate grid
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

                # Create tile with white background
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

        # Save layout map
        layout_path = self.output_folder / "tiles" / "layout_map.json"
        with open(layout_path, 'w') as f:
            json.dump(layout, f, indent=2)

        print(f"  ✓ Saved {rows*cols} tiles to tiles/")
        print(f"  ✓ Layout map: {layout_path}")

    def export_data(self):
        """Export edges.json and aliases.csv"""
        print(f"\n{'='*60}")
        print("STEP 6: EXPORTING DATA")
        print(f"{'='*60}")

        # Export edges
        edges_path = self.output_folder / "edges.json"
        with open(edges_path, 'w') as f:
            json.dump(self.edges, f, indent=2)
        print(f"\n✓ Edges exported: {edges_path}")

        # Export aliases
        if self.aliases:
            aliases_path = self.output_folder / "aliases.csv"
            with open(aliases_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Raw Name', 'Canonical Name'])
                for raw, canonical in self.aliases.items():
                    writer.writerow([raw, canonical])
            print(f"✓ Aliases exported: {aliases_path}")
        else:
            print("✓ No aliases needed")

        # Export GraphML
        if self.graph.number_of_nodes() > 0:
            graphml_path = self.output_folder / "tree.graphml"
            nx.write_graphml(self.graph, graphml_path)
            print(f"✓ GraphML exported: {graphml_path}")

    def run(self):
        """Execute the complete pipeline"""
        print("\n" + "="*60)
        print("FAMILY TREE GRAPH GENERATOR")
        print("="*60)

        self.ingest_inputs()

        if not self.raw_lines:
            print("\n⚠️  No input data found.")
            print(f"   Please add screenshots or .txt files to: {self.input_folder}")
            print("\nExpected format:")
            print("  B: Alice -> Bob")
            print("  AB: Charlie -> Diana")
            return

        self.clean_and_parse()
        self.disambiguate_names()
        self.build_graph()
        self.generate_visualizations()
        self.export_data()

        print(f"\n{'='*60}")
        print("✅ GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"\nOutputs saved to: {self.output_folder}")
        print("\nGenerated files:")

        output_files = sorted(self.output_folder.rglob("*.*"))
        for f in output_files:
            if f.is_file():
                rel_path = f.relative_to(self.output_folder)
                size = f.stat().st_size
                size_str = self._format_size(size)
                print(f"  📁 {rel_path} ({size_str})")

    def _format_size(self, size_bytes: int) -> str:
        """Format file size for display"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate family tree visualizations from screenshots and text files"
    )
    parser.add_argument(
        '--input', '-i',
        default='inputs',
        help='Input folder containing images and text files (default: inputs)'
    )
    parser.add_argument(
        '--output', '-o',
        default='outputs',
        help='Output folder for generated files (default: outputs)'
    )

    args = parser.parse_args()

    generator = FamilyTreeGenerator(
        input_folder=args.input,
        output_folder=args.output
    )

    generator.run()


if __name__ == "__main__":
    main()
