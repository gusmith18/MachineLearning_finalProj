"""Post-image data cleaning utilities.

This script removes columns that are entirely null or entirely empty (after
string-strip) from a CSV dataset and writes a cleaned copy. It also creates
a backup of the original CSV the first time it runs.

Usage:
	python data_clean_post_image.py --in all_mtg_cards_cleaned_image.csv \
		--out all_mtg_cards_cleaned_image_cleaned.csv
"""
from pathlib import Path
import argparse
import pandas as pd


DEFAULT_DROP = [
	'variations', 'rulings', 'watermark', 'printings', 'foreign_names',
	'original_text', 'original_type', 'legalities', 'set_name', 'set'
]


def clean_csv(input_path: Path, output_path: Path, make_backup: bool = True, extra_drop: list | None = None) -> list:
	"""Read CSV, drop columns that are entirely null/empty, write cleaned CSV.

	Returns the list of dropped column names.
	"""
	# Create a backup of the original on first run. If the original file has
	# already been moved to a backup by a prior run, read from the backup.
	backup_path = input_path.with_suffix(input_path.suffix + '.bak')
	if make_backup and not backup_path.exists() and input_path.exists():
		# move original -> backup and read from backup
		input_path.replace(backup_path)
		src = backup_path
	else:
		# prefer original if present, otherwise fall back to existing backup
		if input_path.exists():
			src = input_path
		elif backup_path.exists():
			src = backup_path
		else:
			raise FileNotFoundError(f"Input file not found: {input_path}")

	# Read with low_memory=False to avoid dtype fragmentation warnings
	df = pd.read_csv(src, low_memory=False)

	cols_to_drop = []
	for c in df.columns:
		col = df[c]
		# If all values are NA, drop
		if col.isna().all():
			cols_to_drop.append(c)
			continue
		# If all values are empty after string-conversion and strip, treat as empty
		s = col.fillna('').astype(str).str.strip()
		if (s == '').all():
			cols_to_drop.append(c)

	# Add user-requested drops (default list + any extra_drop passed)
	drops = set(cols_to_drop)

	# include default columns to remove even if not empty
	for c in DEFAULT_DROP:
		if c in df.columns:
			drops.add(c)

	if extra_drop:
		for c in extra_drop:
			if c in df.columns:
				drops.add(c)

	drops = sorted(drops)

	if drops:
		df_clean = df.drop(columns=drops)
	else:
		df_clean = df

	# Ensure parent dir exists for output
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df_clean.to_csv(output_path, index=False)

	return drops


def _build_parser():
	p = argparse.ArgumentParser(description='Clean CSV by dropping fully-empty columns')
	p.add_argument('--in', dest='input', default='all_mtg_cards_cleaned_image.csv', help='Input CSV file')
	p.add_argument('--out', dest='output', default='all_mtg_cards_cleaned_image_cleaned.csv', help='Output cleaned CSV file')
	p.add_argument('--no-backup', dest='no_backup', action='store_true', help='Do not create a .bak backup')
	p.add_argument('--drop', dest='drop', default=None, help='Comma-separated additional columns to drop')
	return p


if __name__ == '__main__':
	parser = _build_parser()
	args = parser.parse_args()
	inp = Path(args.input)
	out = Path(args.output)
	extra = None
	if args.drop:
		extra = [s.strip() for s in args.drop.split(',') if s.strip()]

	try:
		dropped = clean_csv(inp, out, make_backup=not args.no_backup, extra_drop=extra)
		if dropped:
			print(f'Dropped {len(dropped)} columns:')
			for c in dropped:
				print('-', c)
		else:
			print('No fully-empty columns found; wrote cleaned copy identical to input.')
		print('Wrote cleaned CSV to', out)
	except Exception as e:
		print('Error:', e)
		raise

