# Input Files

Place your screenshots and text files in this folder.

## Supported Formats

### Text Files (.txt)

```
B: Alice -> Bob
AB: Charlie -> Diana
B: Eve & Frank -> George
B: Alice -> Bob -> Charlie
```

### Images (.png, .jpg, .jpeg)

- Screenshots of relationship lists
- Ensure good contrast and resolution for OCR
- Text should be clear and legible

## Format Rules

- **B:** prefix for Big relationships (renders as bold edges)
- **AB:** prefix for Assistant Big relationships (renders as dotted edges)
- **Arrows:** Use `->` or `→` or `=>` or `—>`
- **Multiple names:** Separate with `&` or `and`
- **Chains:** `A -> B -> C` creates edges from A to B and B to C

## Example

See `sample_family_tree.txt` for a comprehensive example with:
- Multiple generations
- Chains
- Multiple names
- Complex relationships
- Edge cases
