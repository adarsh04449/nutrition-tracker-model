"""Join real RGB images to dish totals; reject missing or invalid labels."""
import argparse
import csv
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

TARGETS = ['calories_kcal', 'protein_g', 'carbohydrates_g', 'fat_g']
COLUMNS = ['image_path', 'dish_id', *TARGETS]


def read_metadata(root):
    labels, ingredients = {}, {}
    for file in sorted((root / 'metadata').glob('dish_metadata_cafe*.csv')):
        with file.open(newline='') as f:
            for line, row in enumerate(csv.reader(f), 1):
                if not row:
                    continue
                if len(row) < 6 or not row[0].startswith('dish_'):
                    raise ValueError(f'{file}:{line}: unexpected metadata format')
                # Source: dish, kcal, mass, fat, carb, protein. Totals, NOT per 100g.
                values = [float(row[i]) for i in (1, 5, 4, 3)]
                if not all(math.isfinite(v) and v >= 0 for v in values):
                    raise ValueError(f'Invalid nutrition: {row[0]}')
                if row[0] in labels:
                    raise ValueError(f'Duplicate dish ID: {row[0]}')
                labels[row[0]] = values
                # Actual released rows may omit the num_ingrs field described in README.
                start = 6 if len(row) > 6 and row[6].startswith('ingr_') else 7
                ingredients[row[0]] = [row[i+1] for i in range(start, len(row)-1, 7)
                                       if row[i].startswith('ingr_')]
    if not labels:
        raise ValueError('No metadata found; run download_data.py first')
    return labels, ingredients


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/raw'))
    p.add_argument('--output', type=Path, default=Path('data/images.csv'))
    p.add_argument('--visual-review', choices=('pending', 'passed'), default='pending',
                   help='Record whether label_review.jpg passed human inspection')
    a = p.parse_args()
    root = a.root.resolve()
    labels, ingredients = read_metadata(root)
    images = sorted((root / 'imagery/realsense_overhead').glob('dish_*/rgb.png'))
    if not images:
        raise ValueError('No RGB images found; run download_data.py first')
    rows = []
    for path in images:
        dish = path.parent.name
        if dish not in labels:
            raise ValueError(f'Image has no label: {path}')
        with Image.open(path) as im:
            im.verify()
        # Paths relative to the table location keep the project portable.
        import os
        rel = Path(os.path.relpath(path, a.output.resolve().parent)).as_posix()
        rows.append(dict(zip(COLUMNS, [rel, dish, *labels[dish]])))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    selected = rows[:6]
    sheet = Image.new('RGB', (1000, 380 * ((len(selected)+1)//2)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        x, y = (i % 2)*500, (i // 2)*380
        with Image.open(a.output.resolve().parent / row['image_path']) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (490, 265))
        sheet.paste(thumb, (x, y))
        text = (f"{row['dish_id']}\n{row['calories_kcal']:.1f} kcal | "
                f"P {row['protein_g']:.1f}g C {row['carbohydrates_g']:.1f}g F {row['fat_g']:.1f}g\n"
                + ', '.join(ingredients[row['dish_id']]))
        import textwrap
        lines = '\n'.join(textwrap.fill(t, width=66) for t in text.splitlines())
        draw.multiline_text((x+5, y+270), lines, fill='black')
    sheet.save(a.output.parent / 'label_review.jpg')
    report = {'images': len(rows), 'dishes': len({r['dish_id'] for r in rows}),
              'metadata_dishes': len(labels), 'scope': 'locally downloaded overhead RGB only',
              'target_order': TARGETS, 'units': ['kcal', 'g', 'g', 'g'],
              'visual_review': ('passed: no obvious photo/label mismatches'
                                if a.visual_review == 'passed'
                                else 'pending human review of label_review.jpg'),
              'note': 'Full verified table; model subsets are generated in data/splits.'}
    (a.output.parent / 'data_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
