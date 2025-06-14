import sys
import os
import argparse
import csv
from pathlib import Path

# Ensure src is in the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from document_processing.text_extractor import TextExtractor

SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.doc', '.xlsx', '.png', '.jpg', '.jpeg', '.vsd', '.vsdx']

def batch_extract(input_folder, output_folder, summary_csv=None):
    extractor = TextExtractor()
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    results = []

    for file_path in input_folder.glob('**/*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            print(f"Extracting: {file_path}")
            try:
                result = extractor.extract_text(str(file_path))
                out_file = output_folder / (file_path.stem + '.txt')
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(result.text)
                status = 'success'
                error = ''
            except Exception as e:
                status = 'failed'
                error = str(e)
            results.append({
                'file': str(file_path),
                'status': status,
                'error': error
            })

    if summary_csv:
        with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file', 'status', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
    print(f"\nBatch extraction complete. {len(results)} files processed.")
    if summary_csv:
        print(f"Summary written to {summary_csv}")

def main():
    parser = argparse.ArgumentParser(description='Batch extract text from documents in a folder.')
    parser.add_argument('input_folder', help='Input folder containing documents')
    parser.add_argument('output_folder', help='Output folder for extracted text files')
    parser.add_argument('--summary_csv', help='Optional: Path to summary CSV file', default=None)
    args = parser.parse_args()
    batch_extract(args.input_folder, args.output_folder, args.summary_csv)

if __name__ == '__main__':
    main() 