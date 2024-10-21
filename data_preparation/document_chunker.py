# Standard
from pathlib import Path
from typing import Iterable
import json
import time
import os

# Third Party
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConvertedDocument, DocumentConversionInput
from docling.document_converter import DocumentConverter
from utils.logger_config import setup_logger
import click
import pandas as pd

# Local
from utils.docprocessor import DocProcessor

logger = setup_logger(__name__)


def export_documents(
    converted_docs: Iterable[ConvertedDocument],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    for doc in converted_docs:
        if doc.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = doc.input.file.stem

            # Export Deep Search document JSON format:
            with (output_dir / f"{doc_filename}.json").open("w") as fp:
                fp.write(json.dumps(doc.render_as_dict()))

            # Export Markdown format:
            with (output_dir / f"{doc_filename}.md").open("w") as fp:
                fp.write(doc.render_as_markdown())
        else:
            logger.info(f"Document {doc.input.file} failed to convert.")
            failure_count += 1

    logger.info(
        f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
    )

    return doc_filename

def save_document(index, document_text, md_output_dir):
    file_name = f"document_{index+1}.md"
    file_path = os.path.join(md_output_dir, file_name)

    with open(file_path, 'w') as f:
        f.write(document_text)

    print(f"Saved {file_path}")


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Directory containing the documents to convert",
    required=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save the converted documents",
    required=True,
)
def main(input_dir: Path, output_dir: Path):
    file_paths = list(input_dir.glob("*.pdf"))
    artifacts_path = DocumentConverter.download_models_hf()
    doc_converter = DocumentConverter(artifacts_path=artifacts_path)
    inputs = DocumentConversionInput.from_paths(file_paths)

    start_time = time.time()
    converted_docs = doc_converter.convert(inputs)
    doc_filename = export_documents(converted_docs, output_dir)
    end_time = time.time()

    logger.info(f"Parsing documents took {end_time - start_time:.2f} seconds")

    dp = DocProcessor(output_dir, user_config_path=f'{input_dir}/qna.yaml')
    seed_data = dp.get_processed_dataset()

    seed_data.to_json(f'{output_dir}/seed_data.jsonl', orient='records', lines=True)

    md_output_dir = f"{output_dir}/md"
    os.makedirs(md_output_dir, exist_ok=True)

    jsonl_file_path = f"{output_dir}/seed_data.jsonl"

    with open(jsonl_file_path, 'r') as f:
        saved_hashes = set()
        i = 0
        for line in f:
            entry = json.loads(line)
            document_text = entry.get('document', '')
            h = hash(document_text)
            if h not in saved_hashes:
                saved_hashes.add(h)
                save_document(i, document_text, md_output_dir)
                i += 1

    print("Chunking finished")

if __name__ == "__main__":
    main()
